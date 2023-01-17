import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

df.head()

df.shape
df.isnull().sum(axis = 0)
df_clean = df.dropna()

df_clean.shape
df_clean.describe(include = 'all')
plt.pie(df_clean["sex"].value_counts(), labels=df_clean["sex"].unique(),autopct='%1.0f%%')

plt.hist(df_clean['age'])

plt.xlabel('age')
plt.hist(df_clean['education.num'])

plt.xlabel('education num')
plt.hist(df_clean['hours.per.week'])

plt.xlabel('hours per week')
df_clean['native.country'].value_counts()

df_clean = df.drop(['fnlwgt', 'native.country','education'], axis=1)
df_prep = pd.get_dummies(df_clean, columns=['race', 'sex','marital.status','occupation','relationship','workclass'])

df_prep.head()
Y_train = df_prep.pop('income')



X_train = df_prep
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=20)
from sklearn.model_selection import cross_val_score



score = cross_val_score(knn, X_train, Y_train, cv=10)

score
score.mean()
best_score = 0



for k in range (10, 31, 2):

    knn = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(knn, X_train, Y_train, cv=10)

    score_mean = round(score.mean(), 5)

    if best_score < score_mean:

        best_score = score_mean

        best_k = k

print("Melhor k:", best_k)

print("Melhor acurácia:", best_score)
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

df_test_prep = pd.get_dummies(df_test, columns=['race', 'sex','marital.status','occupation','relationship','workclass'])

X_test = df_test_prep.drop(['fnlwgt', 'native.country','education'], axis=1)



knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train, Y_train)



data_pred = knn.predict(X_test)
submission = pd.DataFrame()
submission[0] = X_test.index

submission[1] = data_pred

submission.columns = ["Id", "Income"]

submission.head()
submission.to_csv('submission.csv',index = False)