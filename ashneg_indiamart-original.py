import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import seaborn as sns
df = pd.read_excel("../input/dataset_phase_1/dataset_phase_1.xlsx")

df['S_No'] = np.arange(len(df)) 

df = df.iloc[:126] #126 obtained after analizing graphs to remove outliers
X = df['S_No']

Y = df['Price']

df.describe()
plt.scatter(X ,Y)
sns.boxplot(x=df['Price'])
df['Unit'].unique()
df['Unit'].replace(['Pair', 'Piece', 'pack', 'Unit', 'Pack', 'Unit/Onwards', 'Pair(s)'],['Pair','Unit','Pack','Unit','Pack','Unit','Pack'] , inplace = True)
df
import category_encoders as ce

X = df.drop('S_No', axis = 1)

y = df.drop('Unit', axis = 1)

from sklearn.preprocessing import OneHotEncoder

one_ht_enc = ce.OneHotEncoder(cols = ['Unit', 'Category Name'])

df = one_ht_enc.fit_transform(X ,y)
#here the values with -1 after underscore are for dummy value correction

df
df.rename(columns={'Unit_1': 'Pair', 'Unit_2': 'Unit', 'Unit_3': 'Pack'}, inplace=True)
#Dataset splitting

from sklearn.model_selection import train_test_split

X_train ,X_test ,Y_train ,Y_test = train_test_split(X ,y ,test_size=0.25)
df
#Feature Scaling helpful in case of big datasets

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

Classifier = RandomForestClassifier(n_estimators = 10 ,criterion='entropy')