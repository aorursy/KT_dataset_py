import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

df.head()

# df.isnull()
col_names = list(df.columns.values)

col_names.pop()

import seaborn as sns
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

features_mean= list(df.columns[1:11])

features_se= list(df.columns[11:20])

features_worst=list(df.columns[21:31])
corr = df[features_mean].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, annot=True)
corr = df[features_se].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, annot=True)
corr = df[features_worst].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, annot=True)
drop_list1 = ['Unnamed: 32','id','perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

df_1 = df.drop(drop_list1,axis = 1 )  

df_1.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(df_1, df_1['diagnosis'], test_size=0.3, random_state=2)







rfclassfier = RandomForestClassifier(random_state=2,n_estimators=12)

rfclassfier = rfclassfier.fit(x_train,y_train)



accuracy = accuracy_score(y_test,rfclassfier.predict(x_test))

accuracy
cm = confusion_matrix(y_test,rfclassfier.predict(x_test))

sns.heatmap(cm,annot=True,fmt="d")
