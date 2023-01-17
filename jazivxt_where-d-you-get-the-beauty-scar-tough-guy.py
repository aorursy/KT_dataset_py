%%time
from scipy.sparse import coo_matrix, hstack
import numpy as np
import pandas as pd
from sklearn import *
import lightgbm as lgb
import string

train = pd.read_csv('../input/train.csv', encoding='latin-1').fillna('') 
test = pd.read_csv('../input/test.csv', encoding='latin-1').fillna('')
print(train.shape, test.shape)
%%time
trainn = train[train['target']==1]
trainp = train[train['target']==0].tail(trainn.shape[0] * 3)
print(trainn.shape, trainp.shape)
train = pd.concat((trainn, trainp), axis=0).reset_index(drop=True)
trainn = []; trainp = [];
print(train.shape, test.shape)
#https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
puncts = list(set([',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ] + list(string.punctuation)))
#https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
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
def clean_str(df):
    df['question_text'] = df['question_text'].str.lower()
    df['question_text'] = df['question_text'].map(lambda x: (' ').join([mispell_dict[w] if w in mispell_dict else w for w in x.split(" ")]))
    return df
%%time
train = clean_str(train)
test = clean_str(test)

def starndard_text_features(df):
    col = [c for c in df.columns]
    df['len'] = df['question_text'].map(lambda x: len(str(x)))
    df['wc'] = df['question_text'].map(lambda x: len(str(x).split(' ')))
    df['wcu'] = df['question_text'].map(lambda x: len(set(str(x).split(' '))))
    df["pc"] = df['question_text'].map(lambda x: len([c for c in str(x) if c in puncts]))
    df["mwl"] = df['question_text'].map(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['wcu%'] = df['wcu'] / df['wc']
    df['pc%'] = df['pc'] / df['len']
    col = [c for c in df.columns if c not in col]
    return df[col]
%%time
tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(2, 5), max_features=10000)
cvect = feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(2, 5), max_features=10000)
tfidf_char = feature_extraction.text.TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char_wb', token_pattern=r'\w{1,}', stop_words='english', ngram_range=(2, 5), max_features=10000)
%%time
tfidf.fit(pd.concat((train['question_text'], test['question_text'])).values.astype(str))
trainf = hstack([starndard_text_features(train), tfidf.transform(train['question_text'])]); print(trainf.shape)
testf = hstack([starndard_text_features(test), tfidf.transform(test['question_text'])]); print(testf.shape)

cvect.fit(pd.concat((train['question_text'], test['question_text'])).values.astype(str))
trainf = hstack([trainf, cvect.transform(train['question_text'])]); print(trainf.shape)
testf = hstack([testf, cvect.transform(test['question_text'])]); print(testf.shape)

tfidf_char.fit(pd.concat((train['question_text'], test['question_text'])).values.astype(str))
trainf = hstack([trainf, tfidf_char.transform(train['question_text'])]); print(trainf.shape)
testf = hstack([testf, tfidf_char.transform(test['question_text'])]); print(testf.shape)
%%time
x1, x2, y1, y2 = model_selection.train_test_split(trainf, train['target'], test_size=0.2, random_state=3)

def lgb_f1(preds, y):
    y = y.get_label()
    score = metrics.f1_score(y, (preds>0.5).astype(int))
    return 'f1', score, True

params = {'learning_rate': 0.2, 'max_depth': 8, 'objective': 'binary', 'boosting_type': 'gbdt', 'metric': 'binary_error', 'num_leaves': 32, 'feature_fraction': 0.9,'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), early_stopping_rounds=50,  feval=lgb_f1,  verbose_eval=20)
test['prediction'] = (model.predict(testf, num_iteration=model.best_iteration) > 0.5).astype(int)
test[['qid', 'prediction']].to_csv('submission.csv', index=False)
print(test.prediction.value_counts())

#Early stopping, best iteration is:
#[517]	valid_0's binary_error: 0.098116	valid_0's f1: 0.795103