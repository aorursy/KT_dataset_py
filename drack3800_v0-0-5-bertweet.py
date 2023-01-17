# !pip install torch-lr-finder

# !pip install fairseq fastBPE

# !kaggle datasets download -d christofhenkel/bertweet-base-transformers -p transformers_data/bertweet-base-transformers --unzip



# !mkdir models

# !mkdir data



# model = transformers.RobertaModel.from_pretrained('roberta-base')

# tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')

# output_path = './transformers_data/roberta-base'



# Path(output_path).mkdir(parents=True, exist_ok=True)



# model.save_pretrained(output_path)

# tokenizer.save_pretrained(output_path)





# !ls -thora $output_path
!pip install /kaggle/input/fairseq-and-fastbpe/sacrebleu-1.4.9-py3-none-any.whl

!pip install /kaggle/input/fairseq-and-fastbpe/fairseq-0.9.0-cp37-cp37m-linux_x86_64.whl

!pip install /kaggle/input/fairseq-and-fastbpe/fastBPE-0.1.0-cp37-cp37m-linux_x86_64.whl
import os

from pathlib import Path

import torch

import pandas as pd

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torch.optim import lr_scheduler

import json



from sklearn import model_selection

from sklearn import metrics

import transformers

import tokenizers

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm

import subprocess

import random



from types import SimpleNamespace



from fairseq.data.encoders.fastbpe import fastBPE

from fairseq.data import Dictionary





import warnings

warnings.filterwarnings("ignore")
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 777

seed_everything(seed)
class BERTweetTokenizer():

    def __init__(self, path):

        self.bpe = fastBPE(SimpleNamespace(bpe_codes=os.path.join(path, 'bpe.codes')))

        self.vocab = Dictionary()

        self.vocab.add_from_file(os.path.join(path, 'dict.txt'))

        

    def encode(self, text):

        tokens = self.tokenize(text)

        input_ids = self.convert_tokens_to_ids(tokens)

        offsets = []

        

        curr_pos = 0

        for idx, token in enumerate(tokens):

            # If previous token didn't end with '@@', it means there's a space we

            # need to account for in offsets

            need_space = idx > 0 and (not tokens[idx-1].endswith('@@'))

            curr_pos += need_space

            offsets.append((curr_pos, curr_pos + len(token.replace('@@', ''))))

            curr_pos += len(token.replace('@@', ''))            

        

        return SimpleNamespace(ids=input_ids, tokens=tokens, offsets=offsets)

    

    def tokenize(self, text):

        return self.bpe.encode(text).split()

    

    def convert_tokens_to_ids(self, tokens):

        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids

    

    def decode_tokens(self, tokens):

        decoded = ' '.join(tokens).replace('@@ ', '').strip()

        return decoded

class config:

    VERSION = 'v0-0-5-bertweet'



    INPUT_DATASET = 'tweet-sentiment-extraction'

    MODEL_DATASET = f'model-{VERSION}'

    ROBERTA_DATASET = 'bertweet-base-transformers'

    

    # If False, only inference on test data will be performed using TEST_FILE, MODELS_DIR, ROBERTA_DIR

    TRAIN = False

    IS_KAGGLE = True

    

    DATA_DIR = 'data' if not IS_KAGGLE else f'../input/{INPUT_DATASET}'

    MODELS_DIR = os.path.join(

        'models' if not IS_KAGGLE else '../input',

        MODEL_DATASET

    )

    ROBERTA_DIR = os.path.join(

        'transformers_data' if not IS_KAGGLE else '../input',

        ROBERTA_DATASET

    )



    FOLD = None

    N_FOLDS = 5

#     LEARNING_RATE = 0.2 * 3e-5

    LEARNING_RATE = 3e-5

    N_LAST_LAYERS = 3

    MAX_LEN = 120

    TRAIN_BATCH_SIZE = 32

    VALID_BATCH_SIZE = 32

    EPOCHS = 5





    TRAINING_FILE = os.path.join(DATA_DIR, 'train.csv')

    TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

    SAMPLE_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

    

    TOKENIZER = BERTweetTokenizer(ROBERTA_DIR)



    @classmethod

    def get_model_paths(cls):

        res = []

        for filename in os.listdir(cls.MODELS_DIR):

            if filename.endswith('.bin'):

                res.append(os.path.join(cls.MODELS_DIR, filename))

        return res

        

# !rm -rf $config.MODELS_DIR
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

        

def upload_kaggle_dataset(dir_path, id_, user='drack3800', title=None):

    title = title or id_

    metadata = {

      "title": title,

      "id": f"{user}/{id_}",

      "licenses": [

        {

          "name": "CC0-1.0"

        }

      ]

    }

    with open(os.path.join(dir_path, 'dataset-metadata.json'), 'w') as fout:

        fout.write(json.dumps(metadata) + '\n')

    

    proc = subprocess.run(['kaggle', 'datasets', 'create', '-p' , dir_path], capture_output=True)

    print(proc.stdout.decode())

class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, df, max_len=config.MAX_LEN):

        self.df = df

        self.max_len = max_len

        self.labeled = 'selected_text' in df

        self.tokenizer = config.TOKENIZER



    def __getitem__(self, index):

        data = {}

        row = self.df.iloc[index]

        

        ids, masks, tweet, offsets, token_type_ids = self.get_input_data(row)

        data['ids'] = ids

        data['mask'] = masks

        data['tweet'] = tweet

        data['offsets'] = offsets

        data['token_type_ids'] = token_type_ids

        data['sentiment'] = row.sentiment

        

        if self.labeled:

            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)

            data['start_idx'] = start_idx

            data['end_idx'] = end_idx

            data['selected_tweet'] = " ".join(str(row.selected_text).lower().split())

        

        return data



    def __len__(self):

        return len(self.df)

    

    def get_input_data(self, row):

        tweet = " ".join(str(row.text).lower().split())

        encoding = self.tokenizer.encode(tweet)

        sentiment_id = self.tokenizer.encode(row.sentiment).ids

        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]

        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        

        pad_len = self.max_len - len(ids)

        if pad_len > 0:

            ids += [1] * pad_len

            offsets += [(0, 0)] * pad_len

        

        ids = torch.tensor(ids)

        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))

        offsets = torch.tensor(offsets)

        token_type_ids = torch.zeros(len(ids), dtype=torch.int)

        

        return ids, masks, tweet, offsets, token_type_ids

        

    def get_target_idx(self, row, tweet, offsets):

        selected_text = " ".join(str(row.selected_text).lower().split())



        len_st = len(selected_text)

        idx0 = None

        idx1 = None



        for ind in range(len(tweet)):

            if tweet[ind: ind + len_st] == selected_text:

                idx0 = ind

                idx1 = ind + len_st - 1

                break



        char_targets = [0] * len(tweet)

        if idx0 != None and idx1 != None:

            for ct in range(idx0, idx1 + 1):

                char_targets[ct] = 1



        target_idx = []

        for j, (offset1, offset2) in enumerate(offsets):

            if sum(char_targets[offset1: offset2]) > 0:

                target_idx.append(j)



        start_idx = target_idx[0]

        end_idx = target_idx[-1]

        

        return start_idx, end_idx
class TweetModel(nn.Module):

    def __init__(self):

        super().__init__()

        conf = transformers.RobertaConfig.from_pretrained(

            config.ROBERTA_DIR,

            output_hidden_states=True,

        )

        self.roberta = transformers.RobertaModel.from_pretrained(

            os.path.join(config.ROBERTA_DIR, 'model.bin'),

            config=conf

        )

        self.drop_out = nn.Dropout(0.5)

#         self.l0 = nn.Linear(768 * config.N_LAST_LAYERS, 2)

        self.l0 = nn.Linear(768, 2)

        nn.init.normal_(self.l0.weight, std=0.02)

        nn.init.normal_(self.l0.bias, 0)

        self.layer_weights = nn.Parameter(torch.ones(config.N_LAST_LAYERS) / config.N_LAST_LAYERS)

    

    def forward(self, ids, mask, token_type_ids=None):

        _, _, out = self.roberta(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

        

        last_layers_out = torch.stack([out[-i] for i in range(1, config.N_LAST_LAYERS + 1)], dim=0)

        layer_weights = torch.softmax(self.layer_weights, dim=0)

        out = torch.sum(layer_weights[:, None, None, None] * last_layers_out, dim=0)

#         out = torch.mean(torch.stack(last_layers_out), dim=0)

#         out = torch.cat(last_layers_out, dim=-1)

        out = self.drop_out(out)

        logits = self.l0(out)



        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

    



def build_model(state_dict_path=None, train=True, device=None):

    model = TweetModel()

    if device:

        model.to(device)

    if state_dict_path:

        print(f"Loaded model from {state_dict_path}")

        model.load_state_dict(torch.load(state_dict_path))

    model.train(train)

    return model
def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    return total_loss
def train_fn(data_loader, model, optimizer, device, num_batches, scheduler=None):

    ewma_loss = None

    model.train()

    tk0 = tqdm(data_loader, total=num_batches, desc="Training")

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets_start = d['start_idx']

        targets_end = d['end_idx']



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        model.zero_grad()

        outputs_start, outputs_end = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids,

        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        loss.backward()

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

            

        loss_item = loss.item()

        if ewma_loss is None:

            ewma_loss = loss_item

        else:

            ewma_loss = 0.8 * ewma_loss + 0.2 * loss_item



        tk0.set_postfix(loss=loss_item, ewma=ewma_loss)

    return ewma_loss
def get_best_start_end_idxs(start_logits, end_logits):

    max_len = len(start_logits)

    a = np.tile(start_logits, (max_len, 1))

    b = np.tile(end_logits, (max_len, 1))

    c = np.tril(a + b.T, k=0).T

    c[c == 0] = -1000

    return np.unravel_index(c.argmax(), c.shape)



def get_selected_text(text, start_idx, end_idx, offsets):

    selected_text = ""

    for ix in range(start_idx, end_idx + 1):

        selected_text += text[offsets[ix][0]: offsets[ix][1]]

        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:

            selected_text += " "

    return selected_text



def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):

    start_pred = np.argmax(start_logits)

    end_pred = np.argmax(end_logits)

    if start_pred > end_pred:

        pred = text

    else:

        pred = get_selected_text(text, start_pred, end_pred, offsets)

        

    true = get_selected_text(text, start_idx, end_idx, offsets)

    

    return jaccard(true, pred)



def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    jaccards = AverageMeter()

    

    with torch.no_grad():

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            token_type_ids = d["token_type_ids"]

            mask = d["mask"]

            sentiment = d["sentiment"]

            tweet = d["tweet"]

            targets_start = d['start_idx']

            targets_end = d['end_idx']

            offsets = d["offsets"].cpu().numpy()



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets_start = targets_start.to(device, dtype=torch.long)

            targets_end = targets_end.to(device, dtype=torch.long)



            outputs_start, outputs_end = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )

            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []

            for px in range(len(tweet)):

                jaccard_score = compute_jaccard_score(

                    tweet[px],

                    targets_start[px],

                    targets_end[px],

                    outputs_start[px], 

                    outputs_end[px], 

                    offsets[px]

                )

    

                jaccard_scores.append(jaccard_score)



            jaccards.update(np.mean(jaccard_scores), ids.size(0))

            losses.update(loss.item(), ids.size(0))



    return jaccards.avg
def run(model, df_train, df_valid, model_name, device='cuda'):

    model = model.to(device)



    train_dataset = TweetDataset(df_train)

    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        shuffle=True,

        drop_last=True,

        num_workers=2

    )



    valid_dataset = TweetDataset(df_valid)

    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=config.VALID_BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        num_workers=1

    )



    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    param_optimizer = list(model.named_parameters())

#     no_decay = [

#         "bias",

#         "LayerNorm.bias",

#         "LayerNorm.weight"

#     ]

#     optimizer_parameters = [

#         {

#             'params': [

#                 p for n, p in param_optimizer if not any(

#                     nd in n for nd in no_decay

#                 )

#             ], 

#          'weight_decay': 0.001

#         },

#         {

#             'params': [

#                 p for n, p in param_optimizer if any(

#                     nd in n for nd in no_decay

#                 )

#             ], 

#             'weight_decay': 0.0

#         },

#     ]

    optimizer = AdamW(

#         optimizer_parameters, 

        model.parameters(),

        lr=config.LEARNING_RATE

    )

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=int(0.1 * num_train_steps),

        num_training_steps=num_train_steps

    )



    best_jac = 0

    num_batches = int(len(df_train) / config.TRAIN_BATCH_SIZE)

    

    print("Training is Starting....")



    for epoch in range(config.EPOCHS):

        ewma_loss = train_fn(

            train_data_loader, 

            model, 

            optimizer, 

            device,

            num_batches,

            scheduler

        )



        jac = eval_fn(

            valid_data_loader, 

            model, 

            device

        )

        print(f'Epoch={epoch}, Jaccard={jac:.4f}, Train Loss={ewma_loss:.4f}')

        if jac > best_jac:

            model_path = os.path.join(

                config.MODELS_DIR,

                f'{model_name}.bin'

            )

            os.makedirs(config.MODELS_DIR, exist_ok=True)

            print(f"Model Improved! Saving model to {model_path}")

            torch.save(model.state_dict(), model_path)

            best_jac = jac

            

    return best_jac





def generate_folds(dfx):

    skf = model_selection.StratifiedKFold(n_splits=config.N_FOLDS, random_state=777)

    dfx = dfx.copy()

    dfx['kfold'] = 0

    for n_fold, (_, val_idx) in enumerate(skf.split(dfx.values, dfx.sentiment.values)):

        dfx.loc[val_idx, 'kfold'] = n_fold

    return dfx
# df_train = pd.read_csv(config.TRAINING_FILE)



# train_dataset = TweetDataset(df_train)

    

# train_data_loader = torch.utils.data.DataLoader(

#     train_dataset,

#     batch_size=config.TRAIN_BATCH_SIZE,

#     shuffle=True,

#     drop_last=True,

#     num_workers=2

# )
# d = next(iter(train_data_loader))

# device = 'cuda'





# ids = d["ids"]

# token_type_ids = d["token_type_ids"]

# mask = d["mask"]

# sentiment = d["sentiment"]

# orig_selected = d["orig_selected"]

# orig_tweet = d["orig_tweet"]

# targets_start = d["targets_start"]

# targets_end = d["targets_end"]

# offsets = d["offsets"].cpu().numpy()



# ids = ids.to(device, dtype=torch.long)

# token_type_ids = token_type_ids.to(device, dtype=torch.long)

# mask = mask.to(device, dtype=torch.long)

# targets_start = targets_start.to(device, dtype=torch.long)

# targets_end = targets_end.to(device, dtype=torch.long)
# outputs_start, outputs_end = MX(

#     ids=ids,

#     mask=mask,

#     token_type_ids=token_type_ids

# )
if config.TRAIN:

#     from hyperdash import Experiment

#     exp = Experiment(config.VERSION)

    fold_metrics = []

    dfx = pd.read_csv(config.TRAINING_FILE)

    dfx = generate_folds(dfx)

#     dfx = dfx.sample(1000).reset_index(drop=True)

    

    if config.FOLD is not None:

        folds = [config.FOLD]

        print("Going to train in single-fold mode")

    else:

        folds = list(range(config.N_FOLDS))

        print("Going to train in all-folds mode")



    for n_fold in folds:

        model_name = f'model_{n_fold}'



        MX = build_model()

        df_train = dfx[dfx.kfold != n_fold].reset_index(drop=True)

        df_valid = dfx[dfx.kfold == n_fold].reset_index(drop=True)



        print(f'\n\n>>> Running training for fold={n_fold}, model_name={model_name}')

        jac = run(MX, df_train, df_valid,  model_name, device='cuda')

        

        print(f'>>> Finished training for fold={n_fold}. Jaccard = {jac:.4f}')

        fold_metrics.append(jac)

#         exp.metric(f'fold-jaccard', jac)

    

    print('\n\n\n*********** REPORT **********')

    print(f'CV Jaccard = {np.mean(fold_metrics):.4f} +- {np.std(fold_metrics):.4f}')

#     exp.end()
if config.TRAIN:

    upload_kaggle_dataset(config.MODELS_DIR, config.MODEL_DATASET)
df_test = pd.read_csv(config.TEST_FILE)

device = torch.device('cuda')
final_output_per_model = []



test_dataset = TweetDataset(df_test)

data_loader = torch.utils.data.DataLoader(

    test_dataset,

    batch_size=config.VALID_BATCH_SIZE,

    shuffle=False,

    num_workers=1

)



for model_path in config.get_model_paths():

    model = build_model(model_path, train=False, device=device)

    final_output = []

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d["ids"]

            token_type_ids = d["token_type_ids"]

            mask = d["mask"]



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)



            output_start, output_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)



            output_start = torch.softmax(output_start, dim=1).cpu().detach().numpy()

            output_end = torch.softmax(output_end, dim=1).cpu().detach().numpy()



            final_output.append((

                output_start, 

                output_end,

            ))



    final_output_per_model.append(final_output)
final_output = []

with torch.no_grad():

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        sentiment = d["sentiment"]

        tweet = d["tweet"]

        offsets = d["offsets"].numpy()



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        

        outputs_start = [x[bi][0] for x in final_output_per_model]

        outputs_end = [x[bi][1] for x in final_output_per_model]

        

        output_start = sum(outputs_start) / len(outputs_start)

        output_end = sum(outputs_end) / len(outputs_end)

        

        for i in range(len(ids)):    

            start_pred = np.argmax(output_start[i])

            end_pred = np.argmax(output_end[i])

            if start_pred > end_pred or sentiment[i] == 'neural' or len(tweet[i].split()) < 3:

                pred = tweet[i]

            else:

                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])

            final_output.append(pred)

sample = pd.read_csv(config.SAMPLE_FILE)

sample.loc[:, 'selected_text'] = final_output
sample
if config.TRAIN:

    subm_file = f'{config.VERSION}_submission.csv'

else:

    subm_file = 'submission.csv'



sample.to_csv(subm_file, index=False)

print("saved to", subm_file)