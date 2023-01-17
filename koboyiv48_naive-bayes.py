import numpy as np

import pandas as pd 

import glob
# import datasets

files = glob.glob("../input/*.csv")

list = []

for f in files:

    df = pd.read_csv(f,index_col=None)

    list.append(df)

df = pd.concat(list)
# check count missing value

df.isnull().sum()
# drop row missing value from index

index_df = df[df.to_address.isna()].index

df = df.drop(index_df, axis=0)
# Convert to label

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['labels'] = le.fit_transform(df['status'])
df = df.drop(['status', 'date', 'from_address', 'to_address'], axis=1)
df.columns
cols = ['value', 'balance', 'open', 'high', 'low', 'close', 'volumefrom']
ss = preprocessing.StandardScaler()

df[cols] = ss.fit_transform(df[cols])
X = df[cols]

y = df['labels']
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   test_size=0.3)
model = GaussianNB()

model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, pred))