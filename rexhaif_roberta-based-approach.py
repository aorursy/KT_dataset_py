# installing requirements

!pip install transformers
# so the imports

from tqdm.auto import tqdm

from torch import nn

from torch.nn import functional as F

from torch import optim

from torch.utils import data as torchdata

from sklearn import metrics as ms

from sklearn import model_selection as md

from matplotlib import pyplot as plt



import torch

import random

import os



import argparse as ap

import pandas as pd

import numpy as np

import transformers as tr
def seed_everything(seed = 42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    

seed_everything()
import re

import emoji

import unicodedata



def process_tweet(s, save_text_formatting=False, keep_emoji=False, keep_usernames=False):

        # NOTE: will sometimes need to use Windows encoding here, depending on how CSV is generated.

        # All results saved in UTF-8

        # TODO: Try to get input data in UTF-8 and don't let it touch windows (Excel). That loses emoji, among other things



        # Clean up the text before tokenizing.

        # Why is this necessary?

        # Unsupervised training (and tokenization) is usually on clean, unformatted text.

        # Supervised training/classification may be on tweets -- with non-ASCII, hashtags, emoji, URLs.

        # Not obvious what to do. Two options:

        # A. Rewrite formatting to something in ASCII, then finetune.

        # B. Remove all formatting, keep only the text.

        

            

        EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')

        HASHTAG_BEFORE = re.compile(r'#(\S+)')

        BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')

        FIND_MENTIONS = re.compile(r'@(\S+)')

        LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')

        TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

        

        def remove_accents(input_str):

            nfkd_form = unicodedata.normalize('NFKD', input_str)

            return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

        

        if save_text_formatting:

            s = re.sub(r'https\S+', r'xxxx', str(s))

        else:

            s = re.sub(r'https\S+', r' ', str(s))

            s = re.sub(r'x{3,5}', r' ', str(s))



        # Try to rewrite all non-ASCII if known printable equivalent

        s = re.sub(r'\\n', ' ', s)

        s = re.sub(r'\s', ' ', s)

        s = re.sub(r'<br>', ' ', s)

        s = re.sub(r'&amp;', '&', s)

        s = re.sub(r'&#039;', "'", s)

        s = re.sub(r'&gt;', '>', s)

        s = re.sub(r'&lt;', '<', s)

        s = re.sub(r'\'', "'", s)



        # Rewrite emoji as words? Need to import a function for that.

        # If no formatting, just get the raw words -- else save formatting so model can "learn" emoji

        # TODO: Debug to show differences?

        if save_text_formatting:

            s = emoji.demojize(s)

        elif keep_emoji:

            s = emoji.demojize(s)

            # Transliterating directly is ineffective w/o emoji training. Try to shorten & fix

            s = s.replace('face_with', '')

            s = s.replace('face_', '')

            s = s.replace('_face', '')

            # remove emjoi formatting (: and _)

            # TODO: A/B test -- better to put emoji in parens, or just print to screen?

            #s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' (\1) ', s)

            s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)

            # TODO -- better to replace '_' within the emoji only...

            s = s.replace('(_', '(')

            s = s.replace('_', ' ')



        # Remove all non-printable and non-ASCII characters, including unparsed emoji

        s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)

        # NOTE: We can't use "remove accents" as long as foreign text and emoji gets parsed as characters. Better to delete it.

        # Replace accents with non-accented English letter, if possible.

        # WARNING: Will try to parse corrupted text... (as aAAAa_A)

        s = remove_accents(s)

        # Rewrite or remove hashtag symbols -- important text, but not included in ASCII unsupervised set

        if save_text_formatting:

            s = re.sub(HASHTAG_BEFORE, r'\1!!', s)

        else:

            s = re.sub(HASHTAG_BEFORE, r'\1', s)

            # bad logic in case ^^ done already

            s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)

        # Keep user names -- or delete them if not saving formatting.

        # NOTE: This is not an obvious choice -- we could also treat mentions vs replies differently. Or we could sub xxx for user name

        # The question is, does name in the @mention matter for category prediction? For emotion, it should not, most likely.

        if save_text_formatting:

            # TODO -- should we keep but anonymize mentions? Same as we rewrite URLs.

            pass

        else:

            # If removing formatting, either remove all mentions, or just the @ sign.

            if keep_usernames:

                # quick cleanup extra spaces

                s = ' '.join(s.split())



                # If keep usernames, *still* remove leading and trailing names in @ mentions (or tail mentions)

                # Why? These are not part of the text -- and should not change sentiment

                s = re.sub(LEADING_NAMES, r' ', s)

                s = re.sub(TAIL_NAMES, r' ', s)



                # Keep remaining mentions, as in "what I like about @nvidia drivers"

                s = re.sub(FIND_MENTIONS, r'\1', s)

            else:

                s = re.sub(FIND_MENTIONS, r' ', s)

        #s = re.sub(re.compile(r'@(\S+)'), r'@', s)

        # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace

        s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))

        # Final cleanup -- remove extra spaces created by rewrite.

        s = ' '.join(s.split())

        return s
cfg = ap.Namespace()

cfg.train_file = "../input/nlp-getting-started/train.csv"

cfg.test_file = "../input/nlp-getting-started/test.csv"

cfg.x_column = "text"

cfg.y_column = "target"



cfg.device = "cuda:0"



cfg.model_name = 'roberta-base'

cfg.num_classes = 1

cfg.max_length = 128

cfg.batch_size = 32

cfg.lr = 2e-5

cfg.weight_decay = 5e-5

cfg.n_epochs = 3

cfg.warmup_ratio = 0.1

cfg.max_grad_norm = 1.0



device = torch.device(cfg.device)
class DataframeDataset(torchdata.Dataset):

    def __init__(self, dataframe: pd.DataFrame, x_column: str, y_column: str, tokenizer: tr.PreTrainedTokenizer, max_length: int):

        super(DataframeDataset, self).__init__()

        self.X = dataframe.loc[:, [x_column]].values

        if y_column is not None:

            self.Y = dataframe.loc[:, [y_column]].values

        else:

            self.Y = None

        self.tokenizer = tokenizer

        self.max_length = max_length

        

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx):

        x = process_tweet(self.X[idx])



        x = self.tokenizer.encode_plus(x)['input_ids']

        x_tensor = np.full((self.max_length), fill_value=self.tokenizer.pad_token_id) # i prefer padding all the texts to the same length

        attn_mask = np.zeros_like(x_tensor) # we'd like not to attend to padding zeros, so we need attention mask which tells where the padding and where the tokens

        for i, tok in enumerate(x):

            x_tensor[i] = tok

            attn_mask[i] = 1

            

        if self.Y is None:

            return x_tensor, attn_mask

        else:

            y = self.Y[idx]

            return x_tensor, attn_mask, y.astype(np.float32)
tokenizer = tr.RobertaTokenizer.from_pretrained(cfg.model_name)

model = tr.RobertaForSequenceClassification.from_pretrained(

    cfg.model_name,

    num_labels=cfg.num_classes,

    #hidden_act='gelu_new' ## fixing for consistency with google implementation

)
train_df = pd.read_csv(cfg.train_file)

train, valid = md.train_test_split(train_df, test_size=0.2, random_state=42)



test = pd.read_csv(cfg.test_file)
train_dataset = DataframeDataset(train, cfg.x_column, cfg.y_column, tokenizer, cfg.max_length)

valid_dataset = DataframeDataset(valid, cfg.x_column, cfg.y_column, tokenizer, cfg.max_length)

test_dataset  = DataframeDataset(test,  cfg.x_column, None,         tokenizer, cfg.max_length)
train_dl = torchdata.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=8, sampler=torchdata.RandomSampler(train_dataset))

valid_dl = torchdata.DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=8, sampler=torchdata.SequentialSampler(valid_dataset))
total_steps = len(train_dl) * cfg.n_epochs

n_warmup_steps = int(total_steps * cfg.warmup_ratio) # it is required to use learning rate warmup
def model_step(model, x, attn_mask, y, compute_metrics=False, return_true_pred=False):

    x = x.to(device)

    attn_mask = attn_mask.to(device)

    y = y.to(device)

    

    logits, = model(x, attention_mask=attn_mask) # for some reason transformers models are returning one-element tuples

    loss = F.binary_cross_entropy_with_logits(logits, y)

    result_dict = {

        'loss': loss

    }

    if compute_metrics:

        y_pred = torch.sigmoid(logits).detach().cpu().numpy() > 0.5 # here and below we 

        y_true = y.detach().cpu().numpy()

        f1 = ms.f1_score(y_true, y_pred)

        accuracy = ms.accuracy_score(y_true, y_pred)

        result_dict['f1'] = f1

        result_dict['accuracy'] = accuracy

        

    if return_true_pred:

        y_pred = np.squeeze(torch.sigmoid(logits).detach().cpu().numpy() > 0.5)

        y_true = np.squeeze(y.detach().cpu().numpy())

        result_dict['data'] = (y_true, y_pred)

        

    return result_dict
def evaluate(model, test_dataset):

    test_dl = torchdata.DataLoader(test_dataset, batch_size=1, num_workers=1, sampler=torchdata.SequentialSampler(test_dataset))

    predicts = []

    model.eval()

    with torch.no_grad():

        for x, attn in tqdm(test_dl):

            x = x.to(device)

            attn = attn.to(device)

            logits, = model(x, attention_mask=attn)

            pred = torch.sigmoid(logits) > 0.5

            pred = pred.cpu().squeeze().item()

            predicts.append(pred)

            

    return np.array(predicts)
model = model.to(device)

optimizer = tr.optimization.AdamW(params=model.parameters(), lr=cfg.lr)

scheduler = tr.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=total_steps)
global_train_loss = []

global_val_acc    = []

global_val_f1     = []

global_val_loss   = []

for i in range(cfg.n_epochs):

    model.train()

    train_loss = []

    for x, attn, y in tqdm(train_dl, unit='batch', desc=f"Train {i}"):

        result = model_step(model, x, attn, y)

        loss = result['loss']

        loss.backward()

            

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        optimizer.step()

        scheduler.step()

        train_loss.append(loss.item())

        optimizer.zero_grad()

            

    global_train_loss += train_loss

    

    model.eval()

    valid_loss  = []

    valid_f1    = []

    valid_acc   = []

    with torch.no_grad():

        for x, attn, y in tqdm(valid_dl, unit='batch', desc=f"Valid {i}"):

            result = model_step(model, x, attn, y, compute_metrics=True)

            valid_loss.append(result['loss'].item())

            valid_f1.append(result['f1'])

            valid_acc.append(result['accuracy'])

            

    global_val_acc  += valid_acc

    global_val_f1   += valid_f1

    global_val_loss += valid_loss



    print(f"Epoch {i}: Train Loss = {np.mean(train_loss):.6f}, Val Loss = {np.mean(valid_loss):.6f}, F1 = {np.mean(valid_f1):.4f}, Accuracy = {np.mean(valid_acc):.4f}")
plt.plot(global_train_loss)
plt.plot(global_val_f1)
plt.plot(global_val_loss)
true, pred = [], []

for x, attn, y in tqdm(valid_dl):

    with torch.no_grad():

        y_true, y_pred = model_step(model, x, attn, y, return_true_pred=True)['data']

        true += list(y_true)

        pred += list(y_pred)

        

true = np.vstack(true)

pred = np.vstack(pred)

f1  = ms.f1_score(y_true=true, y_pred=pred)

acc = ms.accuracy_score(y_true=true, y_pred=pred)
print(f"F1 - {f1:.4f}")

print(f"Acc - {acc:.4f}")
predicts = evaluate(model, test_dataset)
submission = pd.DataFrame({

    'id': test['id'],

    'target': predicts.astype(np.int32)

})
submission.to_csv("./submission.csv", index=False, encoding='utf-8')