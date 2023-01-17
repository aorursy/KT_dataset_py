import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random
df = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.head()
plt.scatter(df['age'],df['target'])
cross_sex = pd.crosstab(df['sex'],df['target'])

plt.bar(['0','1'],cross_sex[0])

plt.bar(['0','1'],cross_sex[1])
cross_cp = pd.crosstab(df['cp'],df['target'])

print(cross_cp)

plt.bar(['0','1','2','3'],cross_cp[0])

plt.bar(['0','1','2','3'],cross_cp[1])
fig,ax = plt.subplots(figsize=(12,8))

sns.violinplot(x='target',y='trestbps',data=df,ax=ax)
fig,ax = plt.subplots(figsize=(12,8))

sns.violinplot(x='target',y='chol',data=df,ax=ax)
fig,ax = plt.subplots(figsize=(12,8))

sns.violinplot(x='target',y='thalach',data=df,ax=ax)
fig,ax = plt.subplots(figsize=(12,8))

sns.violinplot(x='target',y='oldpeak',data=df,ax=ax)
ax = plt.axes(projection='3d')

ax.view_init(40, 20)

plt.figure(figsize=(10,8))

ax.scatter3D(df['age'],df['trestbps'],df['target'])
cross_fbs = pd.crosstab(df['fbs'],df['target'])

plt.bar(['0','1'],cross_fbs[1])

plt.bar(['0','1'],cross_fbs[0])
cross_restecg = pd.crosstab(df['restecg'],df['target'])

print(cross_restecg)

plt.bar(['0','1','2'],cross_restecg[0])

plt.bar(['0','1','2'],cross_restecg[1])
cross_exang = pd.crosstab(df['exang'],df['target'])

print(cross_exang)

plt.bar(["0",'1'],cross_exang[0])

plt.bar(["0",'1'],cross_exang[1])
cross_slope = pd.crosstab(df['slope'],df['target'])

print(cross_slope)

plt.bar(['0','1','2'],cross_slope[0])

plt.bar(['0','1','2'],cross_slope[1])
df['ca'].loc[df['ca'] == 4] = 0

cross_ca = pd.crosstab(df['ca'],df['target'])

print(cross_ca)

plt.bar(['0','1','2','3'],cross_ca[0])

plt.bar(['0','1','2','3'],cross_ca[1])
thal_unique,thal_count = np.unique(df['thal'],return_counts=True)

print(thal_unique,thal_count)
df['thal'].loc[(df['thal'] != 0) & (len(df['thal'].unique()) == 4)] -=1
thal_unique,thal_count = np.unique(df['thal'],return_counts=True)

print(thal_unique,thal_count)


cross_thal = pd.crosstab(df['thal'],df['target'])

print(cross_thal)

plt.bar(['0','1','2'],cross_thal[0])

plt.bar(['0','1','2'],cross_thal[1])
fig,ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.astype(int).corr(),ax=ax,annot=True,robust=True)
y = df['target']

df.drop('target',inplace=True,axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
X = df
X.drop(['age'],inplace=True,axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y)
Lr = LogisticRegression(C=100,max_iter=1000)

Lr.fit(X_train,y_train)

Lr_predict = Lr.predict(X_test)

Lr_score = Lr.score(X_test,y_test)

Lr_report = classification_report(y_test,Lr_predict)

Lr_conf_mat = confusion_matrix(y_test,Lr_predict)

print(Lr_conf_mat) 

print(Lr_report)

print("Accuracy:",Lr_score*100) 
Nn = MLPClassifier()

Nn.fit(X_train,y_train)

Nn_predict = Nn.predict(X_test)

Nn_score = Nn.score(X_test,y_test)

Nn_conf_mat = confusion_matrix(y_test,Nn_predict)

Nn_report = classification_report(y_test,Nn_predict)

print(Nn_conf_mat)

print(Nn_report)

print("Accuracy:",Nn_score*100)
Rf = RandomForestClassifier()

Rf.fit(X_train,y_train)

Rf_predict = Nn.predict(X_test)

Rf_conf_mat = confusion_matrix(y_test,Rf_predict)

Rf_score = Rf.score(X_test,y_test)

Rf_report = classification_report(y_test,Rf_predict)

print(Rf_conf_mat)

print(Rf_report)

print("Accuracy:",Rf_score*100)