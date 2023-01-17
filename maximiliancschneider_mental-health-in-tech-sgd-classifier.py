# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/unemployment-and-mental-illness-survey/Data_All_190402/CSV/Mental Illness Survey 1.csv")
df
df.isna().sum()
df1 = df.iloc[1:, [9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,35,36,37,38]]
df1 = df1.dropna()
df1.isna().sum()
df1.dtypes
df1.shape
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

yn_encoder = LabelEncoder()

for i in range(0,1):
    yn_labels = yn_encoder.fit_transform(df1.iloc[:,i])
    df1['Label_'+str(i)] = y_labels
for i in range(2,8):
    yn_labels = yn_encoder.fit_transform(df1.iloc[:,i])
    df1['Label_'+str(i)] = y_labels
for i in range(10,12):
    yn_labels = yn_encoder.fit_transform(df1.iloc[:,i])
    df1['Label_'+str(i)] = y_labels
for i in range(13,15):
    yn_labels = yn_encoder.fit_transform(df1.iloc[:,i])
    df1['Label_'+str(i)] = y_labels

ed_encoder = LabelEncoder()
df1["Label_1"] = ed_encoder.fit_transform(df1.iloc[:,1])

gaps_encoder = LabelEncoder()
df1["Label_8"] = gaps_encoder.fit_transform(df1.iloc[:,8])

income_encoder = LabelEncoder()
df1["Label_9"] = income_encoder.fit_transform(df1.iloc[:,9])

welfare_encoder = LabelEncoder()
df1["Label_12"] = welfare_encoder.fit_transform(df1.iloc[:,12])

age_encoder = LabelEncoder()
df1["Label_15"] = age_encoder.fit_transform(df1.iloc[:,15])

gender_encoder = LabelEncoder()
df1["Label_16"] = gender_encoder.fit_transform(df1.iloc[:,16])

household_encoder = LabelEncoder()
df1["Label_17"] = household_encoder.fit_transform(df1.iloc[:,17])

region_encoder = LabelEncoder()
df1["Label_18"] = region_encoder.fit_transform(df1.iloc[:,18])

df2 = df1.iloc[:,19:]
df2
from sklearn.model_selection import train_test_split
X = df2.iloc[:,1:]
y = df2.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, clf.predict(X_test))