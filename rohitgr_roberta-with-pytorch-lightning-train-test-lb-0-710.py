!pip install -qU '../input/libraries/tokenizers-0.7.0-cp37-cp37m-linux_x86_64.whl'

!pip install -qU '../input/libraries/pytorch_lightning-0.7.5-py3-none-any.whl'

!pip install -qU '../input/libraries/wandb-0.8.35-py2.py3-none-any.whl'

!pip install -qU '../input/libraries/transformers-2.8.0-py3-none-any.whl'
%matplotlib inline



import random

import os

import gc

import re

from pathlib import Path

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import shutil



import wandb

from kaggle_secrets import UserSecretsClient



import torch

from torch import nn, optim

from torch.utils.data import Dataset, DataLoader



from sklearn import model_selection



from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast, get_linear_schedule_with_warmup



import pytorch_lightning as pl



import warnings

warnings.filterwarnings('ignore')
def seed_everything(seed=7):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True
class GlobalConfig:

    def __init__(self):

        self.seed = 7

        self.path = Path('../input/tweet-sentiment-extraction/')

        

        self.max_length = 128

        self.roberta_path = Path('../input/roberta-base/')

        

        self.num_workers = os.cpu_count()

        self.batch_size = 64



        self.accum_steps = 1

        self.epochs = 5

        self.warmup_steps = 0

        self.lr = 3e-5

        

        self.offline = True

        self.saved_model_path = Path('/kaggle/working/final_models')

        self.n_splits = 5

        

        self.saved_model_path.mkdir(parents=True, exist_ok=True)
GCONF = GlobalConfig()



# If you want to use wandb offline then login with a random 40 character string

if not GCONF.offline: wandb.login(key=UserSecretsClient().get_secret('WANDB_KEY'))

else: wandb.login(key='X'*40)



seed_everything(GCONF.seed)

[f.name for f in GCONF.path.iterdir()]
spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0', '\u202f']
def remove_space(text):

    for space in spaces: text = text.replace(space, ' ')

    text = text.strip()

    text = re.sub(r'\s+', ' ', text)

    return text.strip()
train_df = pd.read_csv(GCONF.path/'train.csv')

train_df = train_df.dropna().reset_index(drop=True)

print(train_df.shape)

train_df.head()
train_df['text'] = train_df['text'].apply(remove_space)

train_df['selected_text'] = train_df['selected_text'].apply(remove_space)
def get_callbacks(name):

    mc_cb = pl.callbacks.ModelCheckpoint(

        filepath='/kaggle/working/models/{epoch}',

        monitor=f'avg_valid_loss_{name}',

        mode='min',

        save_top_k=1,

        prefix=f'{name}_',

        save_weights_only=True

    )

    

    es_cb = pl.callbacks.EarlyStopping(

        monitor=f'avg_valid_loss_{name}',

        min_delta=0.,

        patience=2,

        verbose=1,

        mode='min'

    )

    return mc_cb, es_cb



def get_best_model_fn(mc_cb, fold_i):

    for k, v in mc_cb.best_k_models.items():

        if (v == mc_cb.best) and Path(k).stem.startswith(str(fold_i)):

            return k

        

def move_to_device(x, device):

    if callable(getattr(x, 'to', None)): return x.to(device)

    if isinstance(x, (tuple, list)): return [move_to_device(o, device) for o in x]

    elif isinstance(x, dict): return {k: move_to_device(v, device) for k, v in x.items()}

    return x



class AverageMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = self.avg = self.sum = self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
class TweetRobertaDataset(Dataset):

    def __init__(self, df, tokenizer, max_length, is_testing=False):

        self.df = df

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.is_testing = is_testing

        

    def __len__(self):

        return self.df.shape[0]

    

    def _improve_st_span(self, start_token, end_token, input_ids, selected_text):

        tok_st = ' '.join(self.tokenizer.tokenize(selected_text))

        tok_ids = self.tokenizer.convert_ids_to_tokens(input_ids)

        

        for new_start in range(start_token, end_token+1):

            for new_end in range(end_token, new_start-1, -1):

                text_span = ' '.join(tok_ids[new_start:new_end+1])

                if text_span == tok_st: return new_start, new_end

        

        return start_token, end_token

    

    def __getitem__(self, ix):

        text = self.df.iloc[ix]['text']

        sentiment = self.df.iloc[ix]['sentiment']

        start_ix, end_ix = 0, 0

        

        encoded_text = self.tokenizer.encode_plus(

            text,

            add_special_tokens=False,

            return_token_type_ids=False,

            return_attention_mask=False,

            return_offsets_mapping=True

        )

        

        input_ids = encoded_text['input_ids']

        offsets = encoded_text['offset_mapping']



        sentiment_map = {

            'positive': 1313,

            'negative': 2430,

            'neutral': 7974

        }

        

        init_offset = 4

        if not self.is_testing:

            selected_text = self.df.iloc[ix]['selected_text']

            

            for i in (j for j, c in enumerate(text) if c == selected_text[0]):

                if text[i:i+len(selected_text)] == selected_text:

                    start_ix = i

                    end_ix = i + len(selected_text) - 1

                    break

            

            for i, offset in enumerate(offsets):

                if start_ix < offset[1]:

                    start_token = i

                    break

                

            for i, offset in enumerate(offsets):

                if end_ix < offset[1]:

                    end_token = i

                    break

                    

            start_token, end_token = self._improve_st_span(start_token, end_token, input_ids, selected_text)        

            start_token += init_offset

            end_token += init_offset

                    

        input_ids = [0] + [sentiment_map[sentiment]] + [2]*2 + input_ids + [2]

        attn_mask = [1]*len(input_ids)

        token_type_ids = [0]*len(input_ids)

        offsets = [[0, 0]]*init_offset + offsets + [(0, 0)]

        

        pad_len = self.max_length - len(input_ids)

        input_ids += [0]*pad_len

        attn_mask += [0]*pad_len

        token_type_ids += [0]*pad_len

        offsets += [(0, 0)]*pad_len

        

        input_ids, attn_mask, token_type_ids, offsets = map(torch.LongTensor, [input_ids, attn_mask, token_type_ids, offsets])

        

        encoded_dict = {

            'input_ids': input_ids,

            'attn_mask': attn_mask,

            'token_type_ids': token_type_ids,

            'offsets': offsets,

            'orig_text': text,

            'sentiment': sentiment

        }

        

        if not self.is_testing:

            start_token = torch.tensor(start_token, dtype=torch.long)

            end_token = torch.tensor(end_token, dtype=torch.long)

            encoded_dict['start_token'] = start_token

            encoded_dict['end_token'] = end_token

            encoded_dict['orig_selected_text'] = selected_text

        

        return encoded_dict
class TweetRobertaModel(nn.Module):

    def __init__(self, roberta_path):

        super().__init__()

        

        roberta_config = RobertaConfig.from_pretrained(roberta_path)

        roberta_config.output_hidden_states = True

        self.roberta = RobertaModel.from_pretrained(roberta_path, config=roberta_config)

        self.dropout = nn.Dropout(0.1)

        self.linear = nn.Linear(roberta_config.hidden_size * 2, 2)

        

        torch.nn.init.normal_(self.linear.weight, std=0.02)

    

    def forward(self, input_ids, attn_mask, token_type_ids):

        _, _, out = self.roberta(

            input_ids=input_ids,

            attention_mask=attn_mask,

            token_type_ids=token_type_ids

        )

        out = torch.cat([out[-1], out[-2]], dim=2)

        out = self.linear(self.dropout(out))

        

        start_logits, end_logits = out.split(1, dim=2)

        start_logits = start_logits.squeeze(2)

        end_logits = end_logits.squeeze(2)

        return start_logits, end_logits
class TweetRoberta(pl.LightningModule):

    def __init__(self, roberta_path, fold_i=None, train_len=None):

        super().__init__()

        

        self.fold_i = fold_i

        self.train_len = train_len

        

        self.model = TweetRobertaModel(roberta_path)

        

        self.am_tloss = AverageMeter()

        self.am_vloss = AverageMeter()

        self.am_jscore = AverageMeter()

        

    def _compute_loss(self, start_logits, end_logits, batch):

        ignore_ix = start_logits.size(1)

        start_logits.clamp_(0., ignore_ix)

        end_logits.clamp_(0., ignore_ix)

        

        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_ix)

        start_loss = loss_fn(start_logits, batch['start_token'])

        end_loss = loss_fn(end_logits, batch['end_token'])

        return start_loss + end_loss

    

    def _jaccard_score(self, text1, text2):

        a = set(text1.lower().split())

        b = set(text2.lower().split())

        if (len(a)==0) & (len(b)==0): return 0.5

        c = a.intersection(b)

        return float(len(c)) / (len(a) + len(b) - len(c))

    

    def _convert_tokens_to_text(self, start_token, end_token, text, sentiment, offsets):

        if end_token < start_token: end_token = start_token

        filtered_text = ''

        for i in range(start_token, end_token+1):

            filtered_text += text[offsets[i, 0]:offsets[i, 1]]

            if ((i+1) < len(offsets)) and (offsets[i, 1] < offsets[i+1, 0]): filtered_text += ' '

    

        if (sentiment == 'neutral') or (len(text.split()) < 2): filtered_text = text

        return filtered_text

    

    def batch_convert_logits_to_text(self, start_logits, end_logits, batch, return_score=True):

        filtered_texts = []

        jac_scores = []

        

        start_token_preds = torch.argmax(start_logits, dim=1)

        end_tokens_preds = torch.argmax(end_logits, dim=1)

        

        for i, text in enumerate(batch['orig_text']):

            sentiment = batch['sentiment'][i]

            start_token_pred = start_token_preds[i]

            end_token_pred = end_tokens_preds[i]

            filtered_text = self._convert_tokens_to_text(

                start_token_pred, end_token_pred, text,

                sentiment, batch['offsets'][i]

            )

            

            if return_score:

                selected_text = batch['orig_selected_text'][i]

                jac = self._jaccard_score(selected_text.strip(), filtered_text.strip())

                jac_scores.append(jac)

            else:

                filtered_texts.append(filtered_text)

        

        if return_score: return torch.tensor(jac_scores).mean()

        else: return filtered_texts

    

    def forward(self, batch):

        return self.model(batch['input_ids'], batch['attn_mask'], batch['token_type_ids'])

    

    def training_step(self, batch, batch_nb):

        start_logits, end_logits = self.forward(batch)

        train_loss = self._compute_loss(start_logits, end_logits, batch)

        self.am_tloss.update(train_loss.item(), len(batch['input_ids']))

        return {'loss': train_loss, 'log': {f'train_loss_{self.fold_i}': train_loss}}

    

    def validation_step(self, batch, batch_nb):

        start_logits, end_logits = self.forward(batch)

        start_logits, end_logits = start_logits.detach(), end_logits.detach()

        

        valid_loss = self._compute_loss(start_logits, end_logits, batch)

        jac_score = self.batch_convert_logits_to_text(start_logits, end_logits, batch)



        self.am_vloss.update(valid_loss.item(), len(batch['input_ids']))

        self.am_jscore.update(jac_score, len(batch['input_ids']))

    

    def validation_epoch_end(self, _):

        tqdm_dict = {

            f'avg_train_loss_{self.fold_i}': self.am_tloss.avg,

            f'avg_valid_loss_{self.fold_i}': self.am_vloss.avg,

            f'avg_jaccard_{self.fold_i}': self.am_jscore.avg

        }

        

        self.am_tloss.reset()

        self.am_vloss.reset()

        self.am_jscore.reset()

        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    

    def predict(self, batch, device, mode='test'):

        self.eval()

        

        with torch.no_grad():

            batch = move_to_device(batch, device)

            start_logits, end_logits = self.forward(batch)

            return start_logits.detach(), end_logits.detach()

    

    def configure_optimizers(self):

        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_parameters = [

            {

                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

                'weight_decay': 1e-3

            },

            {

                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

                'weight_decay': 0.

            }

        ]

        optimizer = optim.AdamW(optimizer_parameters, lr=GCONF.lr)

        train_steps = (self.train_len * GCONF.epochs)//GCONF.accum_steps

        scheduler = get_linear_schedule_with_warmup(

            optimizer,

            num_warmup_steps=int(GCONF.warmup_steps*train_steps),

            num_training_steps=train_steps

        )

        

        scheduler_dict = {

            'scheduler': scheduler,

            'interval': 'step'

        }

        return [optimizer], [scheduler_dict]
wandb_logger = pl.loggers.WandbLogger(

    name='roberta_uncased',

    save_dir='/kaggle/working/',

    project='kaggle_tweet_extraction',

    offline=GCONF.offline

)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = RobertaTokenizerFast.from_pretrained(str(GCONF.roberta_path), do_lower_case=True, add_prefix_space=True)
skf = model_selection.StratifiedKFold(n_splits=GCONF.n_splits, shuffle=True, random_state=GCONF.seed)



for fold_i, (train_ix, valid_ix) in enumerate(skf.split(train_df, train_df['sentiment'].values)):

    train_ds = TweetRobertaDataset(train_df.iloc[train_ix], tokenizer, GCONF.max_length, is_testing=False)

    train_dl = DataLoader(train_ds, batch_size=GCONF.batch_size, shuffle=True)

    

    valid_ds = TweetRobertaDataset(train_df.iloc[valid_ix], tokenizer, GCONF.max_length, is_testing=False)

    valid_dl = DataLoader(valid_ds, batch_size=GCONF.batch_size*4, shuffle=False)



    mc_cb, es_cb = get_callbacks(fold_i)

    model = TweetRoberta(GCONF.roberta_path, fold_i, len(train_dl))

    trainer = pl.Trainer(

        checkpoint_callback=mc_cb,

        early_stop_callback=es_cb,

        logger=wandb_logger,

        gpus='0',

        max_epochs=GCONF.epochs,

        accumulate_grad_batches=GCONF.accum_steps,

#         fast_dev_run=True

    )

    trainer.fit(model, train_dl, valid_dl)

    torch.cuda.empty_cache()

    

    best_model_fn = get_best_model_fn(mc_cb, fold_i)

    

    # currently there is a bug in PytorchLightning ModelCheckpoint callback that

    # the parameter save_weights_only does not work properly so we will load

    # and save the weights somewhere else to avoid disk full error

    trainer.restore(best_model_fn, on_gpu=False)

    torch.save(model.state_dict(), GCONF.saved_model_path/Path(best_model_fn).name)

    

    model.to('cpu')

    del train_ds, valid_ds, mc_cb, es_cb, model, trainer

    gc.collect()

    torch.cuda.empty_cache()

    

    # to avoid disk full

    shutil.rmtree('/kaggle/working/models')
test_df = pd.read_csv(GCONF.path/'test.csv')

print(test_df.shape)

test_df.head()
test_df['text'] = test_df['text'].apply(remove_space)
test_ds = TweetRobertaDataset(test_df, tokenizer, GCONF.max_length, is_testing=True)

test_dl = DataLoader(test_ds, batch_size=GCONF.batch_size*4, shuffle=False)
models = []



for mf_path in tqdm(GCONF.saved_model_path.iterdir()):

    model = TweetRoberta(GCONF.roberta_path)

    model.load_state_dict(torch.load(mf_path, map_location=lambda storage, loc: storage))

    model.to(device)

    models.append(model)
test_results = []



for batch in tqdm(test_dl):

    start_logits, end_logits = 0, 0

    

    for model in models:

        tmp_start_logits, tmp_end_logits = model.predict(batch, device)

        start_logits += tmp_start_logits

        end_logits += tmp_end_logits

    

    start_logits /= len(models)

    end_logits /= len(models)

    pred_texts = model.batch_convert_logits_to_text(start_logits, end_logits, batch, return_score=False)

    test_results.extend(pred_texts)
sub_df = pd.read_csv(GCONF.path/'sample_submission.csv')

sub_df.loc[:, 'selected_text'] = test_results

sub_df.to_csv('submission.csv', index=False)

sub_df.head()