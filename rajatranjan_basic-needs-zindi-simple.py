# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import gc

import re

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



# import nlpaug.augmenter.char as nac

# import nlpaug.augmenter.word as naw

# import nlpaug.augmenter.sentence as nas

# import nlpaug.flow as nafc

# from nlpaug.util import Action
def random_seed(seed_value):

    import random 

    random.seed(seed_value)  

    import numpy as np

    np.random.seed(seed_value) 

    import torch

    torch.manual_seed(seed_value) 

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)  

        torch.backends.cudnn.deterministic = True   

        torch.backends.cudnn.benchmark = False

        

random_seed(42)
train = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/Train.csv')

test = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/Test.csv')

sub = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/SampleSubmission.csv')
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
train
# train[train['label']=='Depression']['text'].apply(lambda x : " ".join([c for c in x.split(" ") if c.lower() not in stopwords.words('english')]))
# from collections import Counter





# trainCount = Counter(" ".join(set(train[train['label']=='Alcohol']['text'].apply(lambda x : " ".join([c for c in x.split() if c.lower() not in stopwords.words('english')])).values)).split(" "))

# trainCount.most_common(10)
train[train['text'].str.contains('class')]
test[test['ID']=='DWV43DRV']['text'].values
# test.loc[test['ID']=='255YNCPV','text']='Why me?, why so?, why now? I am depressed'

# test.loc[test['ID']=='2SFRZEYJ','text']='How can i overcome suicide from heartbreak?'

# test.loc[test['ID']=='X72WL59G','text']='I want to end my life'

# test.loc[test['ID']=='N4MHVE6D','text']='Lack of financial funds'

# test.loc[test['ID']=='64MWIS95','text']='How to deal with depression and child support'

# test.loc[test['ID']=='7GV9GMHR','text']='do I mean anything in this world?'

# test.loc[test['ID']=='7HYJWJTB','text']='Nobody cares?'



# test.loc[test['ID']=='96Q2X9PY','text']='Why do I suffer?'

# test.loc[test['ID']=='DT2E8Z0Y','text']='Why is it I am going through much in life compared to others. I am depressed'



# test.loc[test['ID']=='DWV43DRV','text']='Why do I suffer?'

# test.loc[test['ID']=='DT2E8Z0Y','text']='Why do I have to attend all classes?'

tarMap={'Depression':0,'Alcohol':1,'Suicide':2,'Drugs':3}

train.label=train.label.map(tarMap)

train
df=train.append(test,ignore_index=True)

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import string

punctuation=string.punctuation

from bs4 import BeautifulSoup

# !pip install emoji

import emoji

import string

punctuation=string.punctuation
from wordcloud import WordCloud, STOPWORDS

stopwords1 = set(STOPWORDS)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.utils import shuffle



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords1,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(train[train.label=='Suicide']['text'])
show_wordcloud(test['text'])
train[train['text'].str.startswith('I feel')]['label'].value_counts()
train[train['text'].str.startswith('How ')]['label'].value_counts()
train[train['text'].str.startswith('I am')]['label'].value_counts()
train[train['text'].str.startswith('I have')]['label'].value_counts()

train[train['text'].str.startswith('What are the effects of')]['label'].value_counts()



train = train[~train['text'].duplicated()].reset_index(drop=True)
train[train['text'].isin(test['text'])]
# set(train['text'].values)
# test[test['text'].str.startswith('I feel')]
# [x.split() for x in train['text'].values]
# set(test['text'].values)
# train['text'] =train['text'].apply(url_to_words).values

train
train['text'][0]


test[test['text'].str.contains(' -possible')]['text'].values


train[train['text'].str.contains(" -possible")]['text'].values
train.loc[train['ID']=='CN5UZZA0','text']='I feel like I am nothing in the world'
rep={'issolated':'isolated','dieAm':'die. I am','frorge':'forget','lonelyNow':'lonely. Now','isolatedNow':'isolated. Now','whn':'when','…':' ',

    'moderatly':'moderately','deferrin':'defered in','after a lac':'for a lack','stresseed':'stressed','drugsNow':'drugs Now','mediataton':'meditation','hatredNow':'hatred Now',

    'how do I cope with ta difficult situation?what will I d to avoid it?':'how do I cope with a difficult situation? What will I do to avoid it?','cornerFeeling':'corner. Feeling',

    'avoiod':'avoid','issueS':'issues','sucidal':'suicidal','is there harm fo me when I take alcohol':'Is there harm for me when I take alcohol',

    'I feel indescribable sadness':'I feel low and extreme sadness','confusedNow':'confused. Now',

     ' It feels as if my head is exploding Lots of burnout Am feeling much better':'It feels as if my head is exploding. Lots of burnout. I am feeling much better',

    'hopelessFor':'hopeless. For','nott':'not','dizzines':'dizziness','Insomia,headache,social problems':'Insomnia, headache, social problems',

     'my own world I feel much bette':'my own world I feel much better','everythingI':'everything. I','hw':'how','oftenly':'often',' im feeling stressed':'I am feeling stressed'

    ,'whren':'when','whn':'when','lifeRight':'life. Right','npt':'not',"feel very lonelyI'm quite okay":'feel very lonely. I am quite okay','depressionI':'depression. I'

    ,'addidcted':'addicted','alccohol':'alcohol','ahd':'had','drinnking':'drinking','lowI':'low. I','SadI':'Sad. I','I felt sad,was stressed,lowi am now better':'I felt sad. I was stressed and low. I am now better'

    ,'schoolfee':'school fees','depresses':'depressed','diserted':'deserted','lonelyCurrently':'lonely. Currently','existd':'exist','GF':'girlfriend','ed results, -dissatisfied,':'ed results, dissatisfied'

    ,'negativecurrently':'negative currently','messNow':'mess. Now','FGM':'girlfriend','Feelings of defeat(post exams depression)Motivated to do better':'Feelings of defeat from post exams depression motivated to do better'

    ,'doto':'do to','finacial':'financial','worhtless':'worthless','frustratedi':'frustrated. I','benefitto':'benefit to','weatherNow':'weather. Now','doI':'do. I','incidencesof':'incidences of'

    ,' -How':'. How','Hopelessnesss,':'Hopeless ','includingmy':'including my','motivationsuicidal':'motivation suicidal','birthNow':'birth. Now','helplessStill':'helpless. Still',

    'lowt':'low','downrecovering':'down recovering',' -possible':' possible','':''}
def replc(vl):

    for k,v in rep.items():

        vl=vl.replace(k,v)

    return vl



test['text'] = test['text'].apply(replc)

train['text'] = train['text'].apply(replc)
train['text']
train.to_csv('train_spell_corrected.csv',index=False)

test.to_csv('test_spell_corrected.csv',index=False)
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()

# train['len']=train['text'].apply(len)

# # train['len']

# v= sc.fit_transform(train['len'].values.reshape(-1,1))

# train['len']=v
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.tokenize import TreebankWordTokenizer

# norm = 'l1', lowercase=True, smooth_idf=False, sublinear_tf=False, ngram_range=(1,4), tokenizer=TreebankWordTokenizer().tokenize

v_1 = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word',norm = 'l1', lowercase=True,

                      smooth_idf=False, sublinear_tf=False, tokenizer=TreebankWordTokenizer().tokenize)

typ_tr =v_1.fit_transform(train['text'])

typ_ts =v_1.transform(test['text'])





v_1c = TfidfVectorizer(ngram_range=(2,5),stop_words="english", analyzer='char',lowercase=False,

                      smooth_idf=False, sublinear_tf=False)

typ_trc =v_1c.fit_transform(train['text'])

typ_tsc =v_1c.transform(test['text'])
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((typ_tr,typ_trc)).tocsr()

final_featurest = sparse.hstack((typ_ts ,typ_tsc)).tocsr()
typ_tsc
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss,mean_squared_log_error



X_trn, X_val, y_trn, y_val = train_test_split(typ_trc, train['label'], test_size=0.2, random_state=1994)

X_test = typ_tsc
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sklearn.metrics import make_scorer

from scipy import sparse

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,IsolationForest

from sklearn.svm import SVC



m=SVC(probability=True,C=1, kernel='sigmoid', degree=1, gamma='scale', coef0=1, 

      shrinking=True,  tol=0.0001, cache_size=500, class_weight='balanced', 

      verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=1994)

m.fit(X_trn, y_trn)

p = m.predict_proba(X_val)

print(f"log_loss is: {log_loss(y_val,p)}")
final_featurest[0]
m.predict(X_test[-1])
from lightgbm import LGBMRegressor,LGBMClassifier



# clf = LGBMClassifier(learning_rate=0.05, colsample_bytree=0.3, reg_alpha=3, reg_lambda=3, max_depth=-1, n_estimators=2000, min_child_samples=15, num_leaves=141)

# clf = LGBMClassifier(learning_rate=0.03, n_estimators=4000,random_state=1994)

# _ = clf.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], verbose=100, early_stopping_rounds=100)

# predictions_val_lgb = clf.predict_proba(X_val)

# print(f"log_loss is: {log_loss(y_val,predictions_val_lgb)}")
X=typ_trc

y=train['label']
y_pred_tot=[]

err=[]

feature_importance_df = pd.DataFrame()



from sklearn.model_selection import KFold,StratifiedKFold

fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

i=1

for train_index, test_index in fold.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=SVC(probability=True,C=1, kernel='sigmoid', degree=4, gamma='scale', coef0=1, 

      shrinking=True,  tol=0.0001, cache_size=500, class_weight='balanced', 

      verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=1994)

    m.fit(X_train,y_train)

    preds=m.predict_proba(X_test)

    print(f"log_loss is: {log_loss(y_test,preds)}")

    err.append(log_loss(y_test,preds))

    p = m.predict_proba(typ_tsc)

    i=i+1

    y_pred_tot.append(p)
# y_pred_tot=[]

# err=[]

# feature_importance_df = pd.DataFrame()



# from sklearn.model_selection import KFold,StratifiedKFold

# fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

# i=1

# for train_index, test_index in fold.split(X,y):

#     X_train, X_test = X[train_index], X[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     m=LGBMClassifier(learning_rate=0.03, n_estimators=4000,random_state=1994)

#     m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200,eval_metric='multi_logloss')

#     preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)

#     print(f"log_loss is: {log_loss(y_test,preds)}")

#     err.append(log_loss(y_test,preds))

#     p = m.predict_proba(final_featurest)

#     i=i+1

#     y_pred_tot.append(p)
np.mean(err)
train.label.unique()
# 'Depression':0,'Alcohol':1,'Suicide':2,'Drugs':3
sub[['Depression','Alcohol','Suicide','Drugs']]=np.mean(y_pred_tot,0)

sub.head()
# s['ID']=test['ID']
sub.to_csv('svm_sub_9_17.csv',index=False)