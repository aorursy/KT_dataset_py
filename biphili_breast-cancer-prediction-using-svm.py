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



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
df.shape
df.columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df.head()
df.shape
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder() 

df['diagnosis'] = lb.fit_transform(df['diagnosis'])

df
df.describe().T
X=df.iloc[:,1:32].values

#X

y=df.iloc[:,0].values

#y
sns.pairplot(df,vars=['radius_mean','texture_mean','area_mean','smoothness_mean'],hue='diagnosis')

plt.ioff()
sns.countplot(df['diagnosis'])

plt.ioff()
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)

plt.ioff()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)

plt.ioff()
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape
X_test.shape
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svc_model=SVC()
svc_model.fit(X_train,y_train)
y_predict=svc_model.predict(X_test)
y_test
y_predict
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)

plt.ioff()
print(classification_report(y_test,y_predict))
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0,1))

X_train_scaled=sc.fit_transform(X_train)

X_test_scaled=sc.fit_transform(X_test)
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)

plt.ioff()
#sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=y_train)
svc_model.fit(X_train_scaled,y_train)
y_predict=svc_model.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)

plt.ioff()
print(classification_report(y_test,y_predict))
param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid_predictions=grid.predict(X_test_scaled)
cm=confusion_matrix(y_test,grid_predictions)
sns.heatmap(cm,annot=True)

plt.ioff()
print(classification_report(y_test,grid_predictions))
svc_model=SVC(C=10,gamma=1,kernel='rbf')

svc_model.fit(X_train_scaled,y_train)
y_predict=svc_model.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_predict)

cm
df.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(df)
scaled_data=scaler.transform(df)
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6));

plt.scatter(x_pca[:,0],x_pca[:,1],c=df['diagnosis'],cmap='plasma')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component');
pca.components_
df_comp=pd.DataFrame(pca.components_,)

df_comp
plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='plasma');