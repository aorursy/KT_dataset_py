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
input_file="/kaggle/input/nlp-getting-started/train.csv"
data=pd.read_csv(input_file)
data.head()
data.isnull().sum()
data.info()
#ignoring that the feature location doesnot have any impact on the target 
train_data=data[['id','text','target']]
train_data.head()
#Data preprocessing
#to lowercase

def to_lowercase(row):
    return(row.lower())
data['text_lower']=data['text'].apply(to_lowercase)
#Removing Punctuations using Regex
train_data['text_punc']=data['text_lower'].str.replace('[^\w\s]','')
    
# tokensising

def tokensie(row):
    _row=row.split()
    return _row
    

train_data['text_tokens']=train_data['text_punc'].apply(tokensie)

# removing the stopwords
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

def remove_stopwords(row):
    _row=[x for x in row if x not in stop_words]
    return _row

train_data['text_stopwords']=train_data['text_tokens'].apply(remove_stopwords)
#Stemming the words
from nltk.stem import PorterStemmer
ps=PorterStemmer()

def stemming(row):
    _row=[ps.stem(x) for x in row]
    return _row

train_data['text_stemmed']=train_data['text_stopwords'].apply(stemming)
def detokenise(row):
    _row=" ".join([x for x in row])
    return _row

train_data['ready_text']=train_data['text_stemmed'].apply(detokenise)
#my Test Train Split

original_data=train_data

#val_data=train_data[5000:]
#train_data=train_data[:5000]
from sklearn.model_selection import train_test_split
X=original_data['ready_text']
y=original_data['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(original_data['ready_text'])
X_train_vec = vectorizer.transform(X_train)
X_val_vec=vectorizer.transform(X_val)
#y_train=train_data['target'][0:5000]
#y_val=original_data['target'][5000:]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

model=gnb.fit(X_train_vec.todense(), y_train)
y_hat=model.predict(X_val_vec.todense())
Y_VAL=[]
for x in y_val:
    Y_VAL.append(x)
results=pd.DataFrame()
results['y_hat']=y_hat
results['y_val']=Y_VAL
results.head()
#Validation Results
from sklearn.metrics import confusion_matrix,f1_score

confusion_matrix(results['y_val'],results['y_hat'])
f1_score(results['y_val'],results['y_hat'])
from sklearn.metrics import log_loss  
log_loss(results['y_val'],results['y_hat'],eps=1e-15)
from sklearn.metrics import accuracy_score,recall_score,precision_score
accuracy_score(results['y_val'],results['y_hat'])
recall_score(results['y_val'],results['y_hat'])
precision_score(results['y_val'],results['y_hat'])
test_data_file="/kaggle/input/nlp-getting-started/test.csv"

test_data=pd.read_csv(test_data_file)
test_data.columns
test_data['text_lower']=test_data['text'].apply(to_lowercase)
test_data['text_punc']=test_data['text_lower'].str.replace('[^\w\s]','')
test_data['text_tokens']=test_data['text_punc'].apply(tokensie)
test_data['text_stopwords']=test_data['text_tokens'].apply(remove_stopwords)
test_data['text_stemmed']=test_data['text_stopwords'].apply(stemming)
test_data['ready_text']=test_data['text_stemmed'].apply(detokenise)
vectorizer = CountVectorizer()
vectorizer.fit(test_data['ready_text'])
X_test = vectorizer.transform(test_data['ready_text'])
