import pandas as pd
from fastai.imports import *
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import re, string #import regular expression 
#PATH = "/media/cuberti/My Book/datascience/data/quora/"
PATH = '../input/' #file Input for running the Kaggle Kernel
train = pd.read_csv(f"{PATH}train.csv")
test = pd.read_csv(f"{PATH}test.csv")
train.head()
#Only run cell when training the data
#train = train.sample(frac = 0.2, random_state = 42)
#train,test,y_trn,y_tst = train_test_split(train.drop('target', axis = 1), train.target, test_size=0.2, random_state=42)
#Otherwise run this cell for final submission
y_trn = train.target
train = train.drop('target', axis = 1)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])') #look for weird punctuaition to remove
def tokenize(s): return re_tok.sub(r' \1 ', s).lower().split() #split on spaces and substitue punctuation
%%time
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize, min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1)
trn_tdm = vec.fit_transform(train['question_text'])
tst_tdm = vec.transform(test['question_text'])
trn_tdm, tst_tdm
def pr(x,y_i, y):
    p = x[(y==y_i)].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
%%time
y=y_trn.values
r = np.log(pr(trn_tdm, 1,y) / pr(trn_tdm, 0,y))
m=LogisticRegression(C=4, dual=True)
m.fit(trn_tdm.multiply(r), y)
preds=m.predict_proba(tst_tdm.multiply(r))
#skip this when submitting a final model
n = 50
i_list = []; f1_list=[];
for i in np.linspace(0,1, num = n+1):
    i_list.append(i)
    f1=f1_score(y_tst, preds[:,1]>i)
    f1_list.append(f1)
df = pd.DataFrame({'i': i_list, 'f1':f1_list})
import matplotlib.pyplot as plt
plt.scatter(x='i', y='f1', data = df)
print('Maximum Threshold \n', df.iloc[df['f1'].idxmax()])
final_pred=(preds[:,1]>0.2).astype(int)
my_sub = pd.DataFrame({'qid':test.qid.values, 'prediction':final_pred})
my_sub.to_csv('submission.csv', index = False)
my_sub.head()
