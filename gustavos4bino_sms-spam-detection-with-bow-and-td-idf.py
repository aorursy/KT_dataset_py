import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sms = pd.read_csv('../input/spam.csv',encoding='latin-1')
sms.head()
sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
sms.columns = ['Class','Text']
sms.head()
sms['text_len'] = sms['Text'].map(len)
sns.factorplot('Class',data=sms,kind='count')
sms.groupby(['Class']).mean()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def pre_processing(document):
    tokens = word_tokenize(document)
    return [word for word in tokens if word not in stopwords.words('english')]
x_train, x_test, y_train, y_test = train_test_split(
    np.ravel(sms['Text']),
    np.ravel(sms['Class']),
    random_state=3
)
pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=pre_processing)),
    ('classifier',MultinomialNB())    
])
pipe.fit(x_train,y_train)
predicts = pipe.predict(x_test)
print(classification_report(predicts,y_test))
from sklearn.feature_extraction.text import TfidfVectorizer
pipe2 = Pipeline([
    ('tdidf',TfidfVectorizer(analyzer=pre_processing)),
    ('classifier',MultinomialNB()),
])
pipe2.fit(x_train,y_train)
predict2 = pipe2.predict(x_test)
print(classification_report(predict2,y_test))