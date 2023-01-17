# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train1.head(3)
valid.head(3)
import re

from nltk.tokenize.treebank import TreebankWordTokenizer



tokenizer = TreebankWordTokenizer()




def clean(text):

    text = text.fillna("fillna").str.lower()

    text = text.map(lambda x: re.sub('\\n',' ',str(x)))

    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))

    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))

    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))

    return text



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_punc(x):

    x = str(x).replace("\n","")

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '', x)

    x = re.sub('[0-9]{4}', '', x)

    x = re.sub('[0-9]{3}', '', x)

    x = re.sub('[0-9]{2}', '', x)

    return x
mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

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

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

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



def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)

%%time

train1['comment_text'] = clean(train1["comment_text"])

train1['comment_text'] = train1["comment_text"].apply(lambda x: clean_punc(x))

train1['comment_text'] = train1['comment_text'].apply(lambda x: replace_typical_misspell(x))
train1['comment_text'] = train1['comment_text'].apply(lambda x: clean_numbers(x))
%%time

train2['comment_text'] = clean(train2["comment_text"])

train2['comment_text'] = train2["comment_text"].apply(lambda x: clean_punc(x))

train2['comment_text'] = train2['comment_text'].apply(lambda x: replace_typical_misspell(x))
train2['comment_text'] = train2['comment_text'].apply(lambda x: clean_numbers(x))
valid.head(4)
valid.head(4)
valid.toxic.value_counts()
valid.lang.value_counts()
train1.toxic.value_counts()
train1.head(3)
test.lang.value_counts()
train2.toxic.value_counts()
train_es = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')

train_es.head(3)
train_tr = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')

train_tr.head(3)
train_pt = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')

train_pt.head(3)
train_ru = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')

train_ru.head(3)
train_fr = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')

train_fr.head(3)
train_fr.toxic.value_counts()
train_it = pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')

train_it.head(3)
train_it.toxic.value_counts()
train = pd.concat([

    train1[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=0),

     train1[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

])
train.toxic.value_counts()
train_mix = pd.concat([

    train[['comment_text', 'toxic']].query('toxic==0'),

     train[['comment_text', 'toxic']].query('toxic==1'),

     train_tr[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=202),

    train_tr[['comment_text', 'toxic']].query('toxic==1'),

    train_pt[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=302),

    train_pt[['comment_text', 'toxic']].query('toxic==1'),

    train_ru[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=402),

    train_ru[['comment_text', 'toxic']].query('toxic==1'),

    train_fr[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=502),

    train_fr[['comment_text', 'toxic']].query('toxic==1'),

    train_it[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=602),

    train_it[['comment_text', 'toxic']].query('toxic==1'),

    train_es[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=702),

    train_es[['comment_text', 'toxic']].query('toxic==1')

])
train_mix.toxic.value_counts()
valid.lang.value_counts()
val_mix = pd.concat([

    valid[['comment_text', 'toxic']].query('toxic==0'),

     valid[['comment_text', 'toxic']].query('toxic==1'),

    train_pt[['comment_text', 'toxic']].query('toxic==0').sample(n=1500, random_state=102),

    train_pt[['comment_text', 'toxic']].query('toxic==1').sample(n=1500, random_state=5102),

    train_ru[['comment_text', 'toxic']].query('toxic==0').sample(n=1500, random_state=402),

    train_ru[['comment_text', 'toxic']].query('toxic==1').sample(n=1500, random_state=902),

    train_fr[['comment_text', 'toxic']].query('toxic==0').sample(n=1500, random_state=5092),

    train_fr[['comment_text', 'toxic']].query('toxic==1').sample(n=1500, random_state=5020),])

    
val_mix.head(5)
val_mix['comment_text'] = clean(val_mix["comment_text"])

val_mix['comment_text'] = val_mix["comment_text"].apply(lambda x: clean_punc(x))

val_mix['comment_text'] = val_mix['comment_text'].apply(lambda x: replace_typical_misspell(x))

val_mix['comment_text'] = val_mix['comment_text'].apply(lambda x: clean_numbers(x))
test.head(2)
%%time

test['content'] = clean(test['content'])

test['content'] = test['content'].apply(lambda x: clean_punc(x))

test['content'] = test['content'].apply(lambda x: replace_typical_misspell(x))

test['content'] = test['content'].apply(lambda x: clean_numbers(x))
test.to_csv('test_clean.csv',index = False)
val_mix.to_csv('valid_clean_mix.csv',index = False)
train_mix.to_csv('clean_train_mix.csv',index = False)