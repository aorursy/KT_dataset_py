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
fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
print(fake.head(2))
print(real.head(2))
#cehcking shape for both files
print(fake.shape)
print(real.shape)
print('FAKE',fake.isnull().sum())
print('REAL',real.isnull().sum())
print(list(fake.columns))
print(list(real.columns))
#adiing label to fake
fake['label'] = 'fake'
fake.head(5)
#adiing label to Real
real['label'] = 'real'
real.head(5)
# let's concatenate the dataframes
frames = [fake, real]
news_dataset = pd.concat(frames)
news_dataset
news_dataset.describe()
news_dataset.info()
final_data = news_dataset.dropna()
final_data.isnull().sum()
import seaborn as sns
final_data['length'] = final_data['title'].apply(len)
sns.countplot(final_data['length'], hue='label', data=final_data)
final_data.hist(column='length', by='label', figsize=(20,5), bins=50)
import copy
from nltk.corpus import stopwords
## removing punctuations from title
import string

def text_process(title):
    
    nop = [char for char in title if char not in string.punctuation]
    
    nop = ''.join(nop)
    
    return [word for word in nop.split() if word in word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Naive model with hyper parameters
piplineTitle = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])
X_train, X_test, y_train, y_test = train_test_split(final_data['title'], final_data['label'], test_size=0.2, random_state=123)
print(piplineTitle.fit(X_train, y_train))
y_pred = piplineTitle.fit(X_train, y_train).predict(X_test)
y_pred
clf_report = classification_report(y_test, y_pred)
print('Classification_Report',clf_report)
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Cnfusion Matrix',cnf_matrix)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sns.heatmap(cnf_matrix, annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')