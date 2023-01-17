# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")
data.head()
sns.pairplot(data,hue='target')
data.columns
sns.pairplot(data[['thalach','oldpeak','slope','target']],hue='target')
sns.heatmap(data.corr(),cmap='coolwarm')
ax = sns.countplot(x='sex',hue='sex',data=data)

ax.set_xticklabels(["Female","Male"])

ax.legend(['Female','Male'])
count = data.groupby('sex').count()['age']

count_male = data['sex'].value_counts()[1]

count_female = data['sex'].value_counts()[0]

from matplotlib.pyplot import pie

fig1, ax1 = plt.subplots()

ax1.pie([count_female/(count_female+count_male),count_male/(count_female+count_male)], labels=['Female','Male'],autopct='%1.1f%%')
hrt_prob_data = data[data['target']==1]

not_hrt_prob = data[data['target']==0]

sns.distplot(hrt_prob_data['age'],bins=10)
sns.distplot(hrt_prob_data['thalach'],hist=False, rug=True, label='Heart Problem')

sns.distplot(not_hrt_prob['thalach'],hist=False, rug=True, label='No Heart Problem')
ax1 = sns.distplot(hrt_prob_data['oldpeak'],hist=False, rug=True, label='Heart Problem',bins=10)

ax2 = sns.distplot(not_hrt_prob['oldpeak'],hist=False, rug=True, label='No Heart Problem',bins=10)

ax1.set_xlim(-1,4)

ax2.set_xlim(-1,4)
print('mean of ST depression when no heart problem:',not_hrt_prob['oldpeak'].mean())

print('mean of ST depression when there is heart problem:',hrt_prob_data['oldpeak'].mean())
data.info()
data['cp'] = data['cp'].astype('object')

data['fbs'] = data['fbs'].astype('object')

data['restecg'] = data['restecg'].astype('object')

data['exang'] = data['exang'].astype('object')

data['slope'] = data['slope'].astype('object')

data['thal'] = data['thal'].astype('object')
data_categorical = pd.get_dummies(data, drop_first=True)

data_categorical.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(data_categorical.drop('target',axis=1))
scaled_features = scaler.transform(data_categorical.drop('target',axis=1))
feature_data = pd.DataFrame(scaled_features,columns=(data_categorical.drop('target',axis=1)).columns[:])
feature_data.head()
from sklearn.model_selection import train_test_split
X = feature_data

y = data_categorical['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,marker='o')
knn_12 = KNeighborsClassifier(n_neighbors=12)

knn_12.fit(X_train,y_train)

pred_12 = knn_12.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred_12))

print('\n')

print(classification_report(y_test,pred_12))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
fig = plt.figure(figsize=(10,10))

important_feat = pd.Series(rfc.feature_importances_,index=X_train.columns)

sns.set_style('whitegrid')

important_feat.sort_values().plot.barh()

plt.title('Important Features')