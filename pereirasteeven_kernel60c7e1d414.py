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



# Any results you write to the current directory are saved as output.|
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(train_data.isnull().sum())

print(test_data.isnull().sum())
from scipy import stats

train_data.groupby(['target']).size()
import seaborn as sns

import matplotlib.pyplot as plt

dis = pd.concat([train_data, test_data], ignore_index=True)

graph = dis.target.value_counts().values

graph_values = [(graph[1] / sum(graph)),  (graph[0] / sum(graph))]

sns.barplot(x=['Disaster', 'Not Disaster'], y=graph_values, palette="Paired").set_title('Target distribution data')

plt.show()
import emoji

import re

def clean_data(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    emoji_pattern = re.compile("["

    u"\U0001F600-\U0001F64F"  

    u"\U0001F300-\U0001F5FF"  

    u"\U0001F680-\U0001F6FF"  

    u"\U0001F1E0-\U0001F1FF"  

    u"\U00002702-\U000027B0"

    u"\U000024C2-\U0001F251"

    "]+", flags=re.UNICODE)

    text = re.sub(r'\n',' ', text) 

    text = re.sub('\s+', ' ', text).strip() 

    return emoji_pattern.sub(r'', text)

    return url.sub(r'',text)
train_data['text'] = train_data['text'].apply(clean_data)

test_data['text'] = test_data['text'].apply(clean_data)
sns.barplot(y=train_data[dis.target == 1].location.value_counts()[:10].index, 

            x=train_data[dis.target == 1].location.value_counts()[:10].values,

            palette="Paired").set_title('Top 10 Locations in Real Disaster')

plt.show()
sns.barplot(y=train_data[dis.target == 0].location.value_counts()[:10].index, 

            x=train_data[dis.target == 0].location.value_counts()[:10].values,

            palette="Paired").set_title('Top 10 Locations in Non Disaster')

plt.show()
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer



count_vectorizer = feature_extraction.text.TfidfVectorizer()

train_vectors = count_vectorizer.fit_transform(train_data["text"])

test_vectors = count_vectorizer.transform(test_data["text"])
clf =  linear_model.RidgeClassifier()
result = model_selection.cross_val_score(clf, train_vectors, train_data["target"], cv=4, scoring="f1")

result
clf.fit(train_vectors, train_data["target"])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)