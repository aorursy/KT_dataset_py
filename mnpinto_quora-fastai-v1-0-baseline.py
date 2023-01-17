import pandas as pd
import numpy as np
from pathlib import Path 
from tqdm import tqdm_notebook
from fastai import *
from fastai.text import *
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)
lm = True
if lm: path = Path('../input'); list(path.iterdir())
train_df = pd.read_csv(path/'train.csv')
train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
if lm:
    bs = 48
    data_lm = (TextList.from_df(train_df, path, cols='question_text')
                .random_split_by_pct(0.1)
                .label_for_lm()           
                .databunch(path='.', bs=bs))
if lm:
    data_lm.save('tmp_lm')
    data_lm = TextLMDataBunch.load('.', 'tmp_lm', bs=bs)
if lm:
    learn = language_model_learner(data_lm, drop_mult=0.3, emb_sz=300)
    learn.unfreeze()
if lm: learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
if lm: learn.save('language_model')
if lm: learn.save_encoder('lm_encoder')
class TextClasDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs=64, pad_idx=1, pad_first=True,
               no_check:bool=False, shuffle=[True, True, False], **kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification."
        datasets = [train_ds, valid_ds, test_ds]
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs//2, sampler=train_sampler, drop_last=True, **kwargs)
        dataloaders = [train_dl]
        dataloaders.append(DataLoader(datasets[1], batch_size=bs, **kwargs))
        dataloaders.append(DataLoader(datasets[2], batch_size=bs, **kwargs))
        return cls(*dataloaders, path=path, collate_fn=collate_fn)
    
TextList._bunch = TextClasDataBunch
path = Path('../input'); list(path.iterdir())
if lm:
    train_df = pd.read_csv(path/'train.csv').sample(frac=0.3, random_state=42)
else:
    train_df = pd.read_csv(path/'train.csv').sample(frac=0.9, random_state=42)

train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
#train0 = train_df[train_df.target==0].sample(n=100000, random_state=42)
#train1 = train_df[train_df.target==1].sample(n=100000, random_state=42, replace=True)
#train = pd.concat((train0, train1)); len(train)
#train.reset_index(inplace=True, drop=True)
test_df = pd.read_csv(path/'test.csv')
test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))
test = TextList.from_df(test_df, path, cols='question_text')
if lm:
    data = (TextList.from_df(train_df, path, cols='question_text', vocab=data_lm.vocab)
                    .random_split_by_pct(0.1)
                    .label_from_df(cols=2)
                    .add_test(test)
                    .databunch(path='.')) 
else:
    data = (TextList.from_df(train_df, path, cols='question_text')
                .random_split_by_pct(0.1)
                .label_from_df(cols=2)
                .add_test(test)
                .databunch(path='.')) 
f_score = Fbeta_binary(beta2=1,clas = 1)
learn = text_classifier_learner(data, drop_mult=0.5, metrics=[accuracy, f_score], emb_sz=300)
if lm: learn.load_encoder('lm_encoder')
learn.freeze()
gc.collect();
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
preds = learn.get_preds(DatasetType.Valid)
proba = to_np(preds[0][:,1])
ytrue = to_np(preds[1])
from sklearn.metrics import roc_curve, precision_recall_curve
def threshold_search(y_true, y_proba, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_th], [best_score], '*r')
        plt.show()
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result 
thr = threshold_search(ytrue, proba, plot=True); thr
preds = learn.get_preds(DatasetType.Test)
proba = to_np(preds[0][:,1])
predsC = (proba > thr['threshold']).astype(int)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = predsC
sub.to_csv("submission.csv", index=False)
sub.head()
