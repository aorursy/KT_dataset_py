import os

os.environ["WANDB_API_KEY"] = "0" # silence the warning
import transformers

import torch

from tqdm import tqdm

import torch.nn as nn

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import classification_report,confusion_matrix

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

import seaborn as sns

import matplotlib.pyplot as plt

import textwrap as wrap

from torch.utils import data

%matplotlib inline

%config InlineBackend.figure_format='retina'

from collections import defaultdict

import re, random, html
# constants - config params

RANDOM_SEED = 42

MAX_LEN = 84

TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

EPOCHS=6

LR=5e-5

TEST_SIZE = 0.15

TRAINING_FILE = "../input/nlp-getting-started/train.csv"

SAMPLE_FILE = "..//nlp-getting-started/sample_submission.csv"

TEST_FILE = "../input/nlp-getting-started/test.csv"



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DEVICE)

MODEL_TYPE = "bert-large-uncased"

MODEL_FILENAME = 'model.bin'

DO_LOWER = True #False When the classification is sentiment then CASED pre-trained model really helps improving Accuracy score

CLEANSE_DATA = True



# Seed it - it can be a function or part of config when packaged

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

torch.cuda.manual_seed(RANDOM_SEED)

torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
# Download tokenizerfrom S3 and cache.



TOKENIZER = transformers.BertTokenizer.from_pretrained(

    MODEL_TYPE,

    do_lower_case=DO_LOWER,

)
#clean the data - some basic cleaning of commonly known characters, words and symbols 

def cleanse_text(t):

    _re_rep = re.compile(r'(\S)(\1{2,})')

    _re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)')



    def replace_rep(t):

        "Replace repetitions at the character level, however there are exceptions"

        def _replace_rep(m):

            c,cc = m.groups()

            return f'{c}{c}'

        w = _re_rep.sub(_replace_rep, t)

        w = w.replace('LOOL','LOL').replace('lool','lol').replace('LooL','LoL').replace('Lool','Lol')

        return w



    def fix_html(x):

        "messy characters"

        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace('.',' ').replace('"','').replace('~',' ').replace(

            '#36;', '$').replace('\\n', "").replace('quot;', "'").replace('<br />', "").replace('<br>', "").replace('</br>', "").replace('&gt;','').replace('&lt;','').replace('|','').replace(

            '\\"', '"').replace('<unk>','').replace(' @.@ ','.').replace(' @-@ ','-').replace('...',' …').replace('=>','').replace('..','').replace('Û_','').replace(

            'gooaal','goal').replace('!','').replace(']','').replace('ÛÊ','').replace(' - ','').replace('åÊ',' ').replace('åÊÛÒ',' ').replace('Ûª',' ').replace(

            '[','').replace('_','').replace(';)', ':)').replace('?','').replace('+','').replace('ÛÒ',' ').replace('{','').replace('}','').replace('%20',' ').replace('*',' ')

        return html.unescape(x)



    def replace_wrep(t):

        "Replace word repetitions: w1 w1 w1 w1"

        def _replace_wrep(m):

            c,cc,e = m.groups()

            return f'{c} {e}'

        return _re_wrep.sub(_replace_wrep, t)

    

    def remove_url(t):

        return re.split('https?:\/\/.*', str(t))[0]

        

    return remove_url(fix_html(replace_wrep(replace_rep(t))))
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

abbreviations = {

    "$" : " dollar ",

    "€" : " euro ",

    "4ao" : "for adults only",

    "a.m" : "before midday",

    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",

    "acct" : "account",

    "adih" : "another day in hell",

    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",

    "afaik" : "as far as i know",

    "afair" : "as far as i remember",

    "afk" : "away from keyboard",

    "app" : "application",

    "approx" : "approximately",

    "apps" : "applications",

    "asap" : "as soon as possible",

    "asl" : "age, sex, location",

    "atk" : "at the keyboard",

    "ave." : "avenue",

    "aymm" : "are you my mother",

    "ayor" : "at your own risk", 

    "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",

    "b.c" : "before christ",

    "b2b" : "business to business",

    "b2c" : "business to customer",

    "b4" : "before",

    "b4n" : "bye for now",

    "b@u" : "back at you",

    "bae" : "before anyone else",

    "bak" : "back at keyboard",

    "bbbg" : "bye bye be good",

    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",

    "bbl" : "be back later",

    "bbs" : "be back soon",

    "be4" : "before",

    "bfn" : "bye for now",

    "blvd" : "boulevard",

    "bout" : "about",

    "brb" : "be right back",

    "bros" : "brothers",

    "brt" : "be right there",

    "bsaaw" : "big smile and a wink",

    "btw" : "by the way",

    "bwl" : "bursting with laughter",

    "c/o" : "care of",

    "cet" : "central european time",

    "cf" : "compare",

    "cia" : "central intelligence agency",

    "csl" : "can not stop laughing",

    "cu" : "see you",

    "cul8r" : "see you later",

    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",

    "cya" : "see you",

    "cyt" : "see you tomorrow",

    "dae" : "does anyone else",

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",

    "dm" : "direct message",

    "dwh" : "during work hours",

    "e123" : "easy as one two three",

    "eet" : "eastern european time",

    "eg" : "example",

    "embm" : "early morning business meeting",

    "encl" : "enclosed",

    "encl." : "enclosed",

    "etc" : "and so on",

    "faq" : "frequently asked questions",

    "fawc" : "for anyone who cares",

    "fb" : "facebook",

    "fc" : "fingers crossed",

    "fig" : "figure",

    "fimh" : "forever in my heart", 

    "ft." : "feet",

    "ft" : "featuring",

    "ftl" : "for the loss",

    "ftw" : "for the win",

    "fwiw" : "for what it is worth",

    "fyi" : "for your information",

    "g9" : "genius",

    "gahoy" : "get a hold of yourself",

    "gal" : "get a life",

    "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",

    "gg" : "good game",

    "gl" : "good luck",

    "glhf" : "good luck have fun",

    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",

    "gn" : "good night",

    "g.o.a.t" : "greatest of all time",

    "goat" : "greatest of all time",

    "goi" : "get over it",

    "gps" : "global positioning system",

    "gr8" : "great",

    "gratz" : "congratulations",

    "gyal" : "girl",

    "h&c" : "hot and cold",

    "hp" : "horsepower",

    "hr" : "hour",

    "hrh" : "his royal highness",

    "ht" : "height",

    "ibrb" : "i will be right back",

    "ic" : "i see",

    "icq" : "i seek you",

    "icymi" : "in case you missed it",

    "idc" : "i do not care",

    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",

    "idk" : "i do not know",

    "ie" : "that is",

    "i.e" : "that is",

    "ifyp" : "i feel your pain",

    "IG" : "instagram",

    "iirc" : "if i remember correctly",

    "ilu" : "i love you",

    "ily" : "i love you",

    "imho" : "in my humble opinion",

    "imo" : "in my opinion",

    "imu" : "i miss you",

    "iow" : "in other words",

    "irl" : "in real life",

    "j4f" : "just for fun",

    "jic" : "just in case",

    "jk" : "just kidding",

    "jsyk" : "just so you know",

    "l8r" : "later",

    "lb" : "pound",

    "lbs" : "pounds",

    "ldr" : "long distance relationship",

    "lmao" : "laugh my ass off",

    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",

    "ltd" : "limited",

    "ltns" : "long time no see",

    "m8" : "mate",

    "mf" : "motherfucker",

    "mfs" : "motherfuckers",

    "mfw" : "my face when",

    "mofo" : "motherfucker",

    "mph" : "miles per hour",

    "mr" : "mister",

    "mrw" : "my reaction when",

    "ms" : "miss",

    "mte" : "my thoughts exactly",

    "nagi" : "not a good idea",

    "nbc" : "national broadcasting company",

    "nbd" : "not big deal",

    "nfs" : "not for sale",

    "ngl" : "not going to lie",

    "nhs" : "national health service",

    "nrn" : "no reply necessary",

    "nsfl" : "not safe for life",

    "nsfw" : "not safe for work",

    "nth" : "nice to have",

    "nvr" : "never",

    "nyc" : "new york city",

    "oc" : "original content",

    "og" : "original",

    "ohp" : "overhead projector",

    "oic" : "oh i see",

    "omdb" : "over my dead body",

    "omg" : "oh my god",

    "omw" : "on my way",

    "p.a" : "per annum",

    "p.m" : "after midday",

    "pm" : "prime minister",

    "poc" : "people of color",

    "pov" : "point of view",

    "pp" : "pages",

    "ppl" : "people",

    "prw" : "parents are watching",

    "ps" : "postscript",

    "pt" : "point",

    "ptb" : "please text back",

    "pto" : "please turn over",

    "qpsa" : "what happens", #"que pasa",

    "ratchet" : "rude",

    "rbtl" : "read between the lines",

    "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing",

    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",

    "rt" : "retweet",

    "ruok" : "are you ok",

    "sfw" : "safe for work",

    "sk8" : "skate",

    "smh" : "shake my head",

    "sq" : "square",

    "srsly" : "seriously", 

    "ssdd" : "same stuff different day",

    "tbh" : "to be honest",

    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",

    "tfw" : "that feeling when",

    "thks" : "thank you",

    "tho" : "though",

    "thx" : "thank you",

    "tia" : "thanks in advance",

    "til" : "today i learned",

    "tl;dr" : "too long i did not read",

    "tldr" : "too long i did not read",

    "tmb" : "tweet me back",

    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",

    "u" : "you",

    "u2" : "you too",

    "u4e" : "yours for ever",

    "utc" : "coordinated universal time",

    "w/" : "with",

    "w/o" : "without",

    "w8" : "wait",

    "wassup" : "what is up",

    "wb" : "welcome back",

    "wtf" : "what the fuck",

    "wtg" : "way to go",

    "wtpa" : "where the party at",

    "wuf" : "where are you from",

    "wuzup" : "what is up",

    "wywh" : "wish you were here",

    "yd" : "yard",

    "ygtr" : "you got that right",

    "ynk" : "you never know",

    "zzz" : "sleeping bored and tired",

    # Typos

    "kno" : "know",

    "fab" : "fabulous",

    "oli" : "oil",

    "tren" : "trend", 

    "swea" : "swear", 

    "stil" : "still",

    "diff" : "different",

    "appx" : "approximately",

    "srsly" : "seriously",

    "epicente" : "epicenter",

    "evng" : "evening",

    "lookg" : "looking",

    "sayin" : "saying",

    "tryin" : "trying",

    "comin" : "Coming",  

    "jumpin" : "jumping",

    "nothin" : "nothing", 

    "burnin" : "burning", 

    "killin" : "killing",

    "thinkin" : "thinking",

    "throwin" : "throwing",

    "newss" : "news",

    "memez" : "memes",

    "fforecast" : "Forecast",

}



def convert_abbrev(w):

    return abbreviations[w.lower()] if w.lower() in abbreviations.keys() else w

print(cleanse_text('good good good good looooooool :)'))

print(cleanse_text('@bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C'))

print(" ".join(cleanse_text("Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh... **** *").split()))

print(cleanse_text('#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires.. Crash_______,] chemical%20emergency'))

print(convert_abbrev('lookg'))
df_train = pd.read_csv(TRAINING_FILE)

df_test = pd.read_csv(TEST_FILE)

df_train['classes'] = np.where(df_train['target'] == 0, 'Not Disaster', 'Disaster')
if CLEANSE_DATA:

    df_train['text'] = df_train['text'].apply(lambda x: convert_abbrev(cleanse_text(x)))

df_train['tlen'] = df_train['text'].apply(len)

df_train.head()
if CLEANSE_DATA:

    df_test['text'] = df_test['text'].apply(lambda x: convert_abbrev(cleanse_text(x)))

df_test['tlen'] = df_test['text'].apply(len)

df_test.head()
df_train.describe()
df_test.describe()
sns.set(font_scale=1.25)

plt.rcParams["figure.figsize"]=(8,6)

sns.countplot(df_train.classes);

sns.set(style="dark")

class_names = ['Not Disaster', 'Disaster'];
plt.rcParams["figure.figsize"]=(8,6)

sns.distplot(df_train.tlen,kde=True, rug=False,color='red');
plt.rcParams["figure.figsize"]=(8,6)

sns.distplot(df_test.tlen,kde=True, rug=False, color='green');
#combine text and tokenize

labels = df_train['target'].values

idx = len(labels)

df = pd.concat([df_train, df_test])

df = df.text.values



#### Splitting the train/test data after tokenizing.

train= df[:idx]

test = df[idx:]

train.shape, test.shape
#find max_len. This is a small dataset and keeping it 96/80/72/68 really not making much difference.

max_len=0

for t in df:

    # Tokenize the text and add special tokens - `[CLS]` and `[SEP]` 

    input_ids = TOKENIZER.encode(t, add_special_tokens=True)



    # Update the maximum input_ids length.

    max_len = max(max_len, len(input_ids))



print('Max input_ids length = MAX_LEN: ', max_len)

MAX_LEN = max_len
train[11:25]
df_train[11:22]
class DisasterTweetsDataset(data.Dataset):

    def __init__(self,text,target,tokenizer,max_len):

        self.text = text

        self.target = target

        self.tokenizer = tokenizer

        self.max_len = max_len



    def __len__(self):

        return len(self.text)



    def __getitem__(self, item):

        text = str(self.text[item])

        text = " ".join(text.split()) # basic cleansing to remove unwanted spaces



        # Encoding

        encoding = TOKENIZER.encode_plus(

            text, 

            add_special_tokens = True,

            max_length = self.max_len,

            truncation='longest_first',

            return_token_type_ids=True,

            pad_to_max_length=True,

            return_attention_mask = True,

            return_tensors='pt',

        )

        return {

            'text': text,

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'token_type_ids': encoding['token_type_ids'].flatten(),

            'targets': torch.tensor(self.target[item],dtype=torch.long)

        }
class DisasterTweetsModel(nn.Module):

    def __init__(self, n_classes):

        super(DisasterTweetsModel,self).__init__()

        self.bert = transformers.BertModel.from_pretrained(MODEL_TYPE)

        self.bert_drop_1 = nn.Dropout(0.15)

        self.bn = nn.BatchNorm1d(self.bert.config.hidden_size)

        self.relu = nn.ReLU()

        self.out1 =  nn.utils.weight_norm(nn.Linear(self.bert.config.hidden_size,2048))

        self.out2 =  nn.utils.weight_norm(nn.Linear(2048,self.bert.config.hidden_size))

        self.out3 =  nn.utils.weight_norm(nn.Linear(self.bert.config.hidden_size,n_classes)) # (768,2) or (1024,2)

        self.softmax = nn.Softmax(dim=1)



    def forward(self, input_ids, attention_mask,token_type_ids):

        _,pooled_output = self.bert(

            input_ids = input_ids,

            attention_mask = attention_mask,

            token_type_ids = token_type_ids,

        )

        output = self.bert_drop_1(pooled_output)

        output = self.relu(output)

        output = self.out1(output)

        output = self.out2(output)

        output = self.bn(output)

        output = self.relu(output)

        output = self.out3(output)

        return self.softmax(output)
df_train, df_val = train_test_split(df_train, test_size=TEST_SIZE,random_state=RANDOM_SEED)

df_train = df_train.reset_index(drop=True)

df_val = df_val.reset_index(drop=True)

df_train.shape, df_val.shape
def create_data_loader(df, tokenizer, max_len, bsz, shuffle):

    ds = DisasterTweetsDataset(

        text=df.text.to_numpy(),

        target=df.target.to_numpy(),

        tokenizer=TOKENIZER,

        max_len = MAX_LEN,

    )

    return data.DataLoader(

        ds,

        batch_size=bsz,

        num_workers=4,

        shuffle=shuffle

    )



train_dl = create_data_loader(df_train, tokenizer=TOKENIZER, max_len=MAX_LEN, bsz=TRAIN_BATCH_SIZE,shuffle=True)

val_dl = create_data_loader(df_val, tokenizer=TOKENIZER, max_len=MAX_LEN, bsz=VALID_BATCH_SIZE,shuffle=False)
model = DisasterTweetsModel(len(class_names))

model = model.to(DEVICE)



#optimizer parameters

param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.001},

                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}]



#optimizer 

optimizer = AdamW(optimizer_parameters, lr=LR)

steps = len(train_dl) * EPOCHS

scheduler = get_linear_schedule_with_warmup(

    optimizer,

    num_warmup_steps = 0,

    num_training_steps = steps

)



#loss function

loss_fn = nn.CrossEntropyLoss().to(DEVICE)

def train_fn(model, dl, loss_fn, optimizer, device, scheduler, n_examples):

    model.train()

    losses = []

    predictions = 0



    #iterate each from dl

    for d in tqdm(dl, total=len(dl), position=0, leave=True):

        input_ids = d['input_ids'].to(DEVICE)

        attention_mask = d['attention_mask'].to(DEVICE)

        token_type_ids = d['token_type_ids'].to(DEVICE, dtype=torch.long)

        targets = d['targets'].to(DEVICE)



        outputs = model(

            input_ids = input_ids,

            attention_mask = attention_mask,

            token_type_ids = token_type_ids

        )



        _,preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        predictions += torch.sum(preds==targets)

        losses.append(loss.item())



        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

    return predictions.double() / n_examples, np.mean(losses)



def eval_fn(model, dl, loss_fn, device, n_examples):

    model.eval()

    losses = []

    predictions = 0



    with torch.no_grad():

        for d in tqdm(dl, total=len(dl), position=0, leave=True):

            input_ids = d['input_ids'].to(DEVICE)

            attention_mask = d['attention_mask'].to(DEVICE)

            token_type_ids = d['token_type_ids'].to(DEVICE, dtype=torch.long)

            targets = d['targets'].to(DEVICE)



            outputs = model(

                input_ids = input_ids,

                attention_mask = attention_mask,

                token_type_ids = token_type_ids

            )



            _,preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)



            predictions += torch.sum(preds==targets)

            losses.append(loss.item())

    return predictions.double() / n_examples, np.mean(losses)
%%time

hist = defaultdict(list)

best_acc = 0

for epoch in range(EPOCHS):

    print(f'\nEpoch {epoch + 1} / {EPOCHS}')

    train_acc, train_loss = train_fn(model,train_dl,loss_fn,optimizer,DEVICE,scheduler,len(df_train))

    print(f'Train loss {train_loss} Accuracy {train_acc}')



    val_acc, val_loss = eval_fn(model,val_dl,loss_fn,DEVICE,len(df_val))

    print(f'Validation loss {val_loss} Accuracy {val_acc}')

    print()



    hist['train_acc'].append(train_acc)

    hist['train_loss'].append(train_loss)

    hist['val_acc'].append(val_acc)

    hist['val_loss'].append(val_loss)



    if val_acc > best_acc:

        torch.save(model.state_dict(),MODEL_FILENAME)

        best_acc = val_acc
plt.figure(figsize=(8,6))

plt.gca().title.set_text(f'Accuracy Chart')

plt.plot(np.arange(EPOCHS),hist['train_acc'],label='Training')

plt.plot(np.arange(EPOCHS),hist['val_acc'],label='Validation')

plt.legend();
plt.figure(figsize=(8,6))

plt.gca().title.set_text(f'Loss Chart')

plt.plot(np.arange(EPOCHS),hist['train_loss'],label='Training')

plt.plot(np.arange(EPOCHS),hist['val_loss'],label='Validation')

plt.legend();
def get_preds(model, data_loader):

    model.eval()

    predictions = []

    prediction_proba = []

 

    with torch.no_grad():

        for d in tqdm(data_loader, total=len(data_loader)):

            input_ids = d['input_ids'].to(DEVICE)

            attention_mask = d['attention_mask'].to(DEVICE)

            token_type_ids = d['token_type_ids'].to(DEVICE, dtype=torch.long)

            outputs = model(

                input_ids = input_ids,

                attention_mask = attention_mask,

                token_type_ids = token_type_ids

            )



            _,preds = torch.max(outputs, dim=1)

            predictions.extend(preds)

            prediction_proba.extend(outputs)

    predictions = torch.stack(predictions).cpu()

    prediction_proba = torch.stack(prediction_proba).cpu()



    return predictions, prediction_proba
model = DisasterTweetsModel(len(class_names))

model.load_state_dict(torch.load(MODEL_FILENAME))

model = model.to(DEVICE)

df_test['target']=-1

test_dl = create_data_loader(df_test, tokenizer=TOKENIZER, max_len=MAX_LEN, bsz=TRAIN_BATCH_SIZE,shuffle=False)
preds, proba = get_preds(model,test_dl)

len(preds)
df_sample = pd.read_csv(SAMPLE_FILE)
preds[:10], proba[:10]
df_sample['target']=preds;df_sample.head()
df_sample.to_csv("submission_final.csv",index=False)