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



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
df = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', encoding = 'latin')

print(df.shape)

df.head()
df['launched']= pd.to_datetime(df['launched'])

df['deadline']= pd.to_datetime(df['deadline'])

df = df.dropna()



df.info()
df['duration_days'] = (df['deadline'] - df['launched']) / np.timedelta64(1, 's') / 86400

df.head()
df["name"].fillna("", inplace=True)

most_occur = pd.Series(' '.join(df[df["state"] == 'successful']['name']).lower().split()).value_counts()[:100]

most_occur = most_occur.to_dict()

stopwords_included = []

for i in most_occur.keys():

    if i in stopwords.words('english'):

        stopwords_included.append(i)

    if i in '''!()-[]{};:'"+1234567890\,<>./?@#$%^&*_~''':

        stopwords_included.append(i)

for i in stopwords_included:

    del most_occur[i]

most_occur['album'] += most_occur['album!']

del most_occur['album!']

print(most_occur)
def get_name_score(name, most_occur):

    name = name.lower()

    score = 0

    for i in name.split():

        for j in most_occur.keys():

            if j in i:

                score += most_occur[j] / 100

    return score
df["name_score"] = df["name"].apply(lambda x: get_name_score(x, most_occur))
df["name_score"].head()
df = df[(df["state"] == "successful") | (df["state"] == "failed")]

X = pd.get_dummies(df[["category", "main_category", "usd_goal_real","name_score"]])

y = df[["state"]]
#Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
X_train.head()
#Prediction Using Random Forest Classifier 

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train.values.ravel())

pred_train = rfc.predict(X_test)

report = classification_report(y_test, pred_train)

print(report)