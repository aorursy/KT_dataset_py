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
train=pd.read_csv('/kaggle/input/zindi-weekend-vaccine/Train.csv')

test=pd.read_csv('/kaggle/input/zindi-weekend-vaccine/Test.csv')

s=pd.read_csv('/kaggle/input/zindi-weekend-vaccine/SampleSubmission.csv')
train
# train.agreement.fillna(1,inplace=True)

# train.append(['RQMQ0L2A','I cannot believe in this day and age some parents could be so oblivious to reality as to not #vaccinate their child. #lawandorderSVU',1,0.666667])
train=train[train.label.isin([0,1,-1])]

train.head()
# pd.DataFrame(,columns=train.columns)
train.append(pd.Series(['RQMQ0L2A','I cannot believe in this day and age some parents could be so oblivious to reality as to not #vaccinate their child. #lawandorderSVU',1,0.666667], index=train.columns ), ignore_index=True)

train['label']=train['label']*train['agreement']

train.isnull().sum()
test.iloc[2024]['tweet_id']='E0GRUEO2'

test.iloc[2024]['safe_text']='Dr. JAMES SHANNON'
train.agreement.value_counts()
test.isnull().sum()
import re

import nltk

from nltk.corpus import stopwords







from bs4 import BeautifulSoup







def url_to_words(raw_text):

    raw_text=str(raw_text).strip()

    soup = BeautifulSoup(raw_text, 'html.parser')

    raw_text = soup.text

    no_coms=re.sub(r'\.com','',raw_text)

    no_urls=re.sub('https?://www','',no_coms)

    no_urls1=re.sub('https?://','',no_urls)

    try:

        no_encoding=no_urls1.decode("utf-8-sig").replace(u"\ufffd", "?")

    except:

        no_encoding = no_urls1

    letters_only = re.sub("[^a-zA-Z]", " ",no_encoding) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words ))


df=train[['tweet_id','safe_text','label']].append(test,ignore_index=True)
df
df['safe_text'].apply(lambda x: "".join(_ for _ in x if _ in ['VACCINES']))
# import re

# re.sub(r"vaccin\w+", '', df['safe_text'][2].lower())
# df['safe_text'][2].lower()
# re.findall(r"vaccin\w+", df['safe_text'][2].lower())
import string

punctuation=string.punctuation

df['safe_text']=df['safe_text'].astype('category')

df['word_count']=df['safe_text'].apply(lambda x: len(str(x).split(" ")))

df['char_count'] = df['safe_text'].str.len()

def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words)/len(words))



df['avg_word'] = df['safe_text'].apply(lambda x: avg_word(x))

from nltk.corpus import stopwords

stop = stopwords.words('english')



df['stopwords'] = df['safe_text'].apply(lambda x: len([x for x in x.split() if x in stop]))

df['numerics'] = df['safe_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df['upper'] = df['safe_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

df['word_density'] = df['char_count'] / (df['word_count']+1)

df['punctuation_count'] = df['safe_text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 

df['hastag_count'] = df['safe_text'].apply(lambda x: len("".join(_ for _ in x if _=="#"))) 
df.head()
df.groupby('label').mean()
col=['word_count', 'char_count',

       'avg_word', 'stopwords', 'numerics', 'upper', 'word_density',

       'punctuation_count', 'hastag_count']
df['safe_text'] = df['safe_text'].apply(lambda x: re.sub(r"vaccin\w+", '',x.lower()))
df_train=df[df.label.isnull()==False]

df_test=df[df.label.isnull()==True]

print(df_train.shape,df_test.shape)
print(df_train['safe_text'].sample(20,random_state=20).values)

df_test['safe_text'].sample(20,random_state=20).values
from tqdm import tqdm

tqdm.pandas()

df_train['safe_text']=df_train['safe_text'].progress_apply(url_to_words)

df_test['safe_text']=df_test['safe_text'].progress_apply(url_to_words)
print(df_train['safe_text'].sample(20,random_state=20).values)

df_test['safe_text'].sample(20,random_state=20).values
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



show_wordcloud(df_train[df_train.label==1]['safe_text'])
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



show_wordcloud(df_train[df_train.label==0]['safe_text'])
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



show_wordcloud(df_train[df_train.label==-1]['safe_text'])
# test[test.safe_text.isnull()==True]
s.iloc[2024]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix,classification_report,f1_score,mean_squared_error

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

v_name = TfidfVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

name_tr =v_name.fit_transform(df_train['safe_text'])

name_ts =v_name.transform(df_test['safe_text'])
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((df_train[col],name_tr )).tocsr()

final_featurest = sparse.hstack((df_test[col],name_ts)).tocsr()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import math

from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_squared_log_error,log_loss

X=final_features

# y=pd.get_dummies(train_df['Labels'].values)

y=df_train.label

X_train,X_val,y_train,y_val = train_test_split(X, y, stratify=y, 

                                                  random_state=1994, 

                                                  test_size=0.25, shuffle=True)



y_bin=pd.qcut(df_train.label,3,labels=False)
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

m=LGBMRegressor(n_estimators=6000,random_state=1994,learning_rate=0.03)

# ,colsample_bytree=0.4,max_depth=-1,boosting='gbdt'

m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val, y_val.values)], early_stopping_rounds=100,verbose=300,eval_metric='rmse')

p=m.predict(X_val)

print(np.sqrt(mean_squared_error(y_val.values,p)))
from sklearn.model_selection import KFold,StratifiedKFold



errcat = []

y_pred_totcat = []



fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1994)



for train_index, test_index in fold.split(X,y_bin):

    X_train, X_test = X[train_index], X[test_index]

#     X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    cat = LGBMRegressor(n_estimators=6000,random_state=1994,learning_rate=0.03)

    cat.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], verbose=300, early_stopping_rounds=200,eval_metric='rmse')



    y_pred_cat = cat.predict(X_test)

    print("RMSE: ", np.sqrt(mean_squared_error(y_test.values,y_pred_cat)))



    errcat.append(np.sqrt(mean_squared_error(y_test.values,y_pred_cat)))

    p = cat.predict(final_featurest)

    y_pred_totcat.append(p)
np.mean(errcat)
s['label']=np.mean(y_pred_totcat,0)
s['label'].describe()
s.to_csv('kv4.csv',index=False)