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
df=pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')
df

df.quality.value_counts()
df.quality.unique()
df.isna().sum()
df['volatile acidity'].fillna(value=0.339691,inplace=True)
df['pH'].fillna(value=3.218395,inplace=True)
df['fixed acidity'].fillna(value=7.000000,inplace=True)
df['sulphates'].fillna(value=0.510000,inplace=True)
df.drop(['citric acid','residual sugar','chlorides'],axis=1,inplace=True)
a=[2,6,9]
label=['bad','good']
df['quality']=pd.cut(df['quality'],bins=a,labels=label)
df.head()
df.quality.value_counts()
from sklearn.preprocessing import LabelEncoder
labelencoder_target=LabelEncoder()
df['quality']=labelencoder_target.fit_transform(df['quality'])
df.quality.head(15)
df.describe()
corr=df.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=[23,23])
sns.heatmap(df.corr(),annot=True)
sns.countplot('quality',data=df)
sns.barplot(x='quality',y='alcohol',data=df)
f,(axes,sa)= plt.subplots(1,2,figsize=(16,5))
sns.distplot(df['fixed acidity'],ax=sa)
sns.violinplot(x='quality',y='fixed acidity',data=df,hue=df['quality'],ax=axes)
f,axes=plt.subplots(1,2,figsize=(16,6))
sns.distplot(df['volatile acidity'],ax=axes[0])
sns.violinplot(x='quality',y='volatile acidity',data=df,hue=df['quality'],ax=axes[1])
f,axes=plt.subplots(2,2,figsize=(16,5))
sns.distplot(df['alcohol'],ax=axes[0,0])
sns.violinplot(x='quality',y='alcohol',data=df,ax=axes[0,1])
sns.distplot(df['sulphates'],ax=axes[1,0])
sns.violinplot(x='quality',y='sulphates',data=df,ax=axes[1,1])
df.drop(['density','volatile acidity'],axis=1,inplace=True)
df=pd.get_dummies(df,columns=['type'],drop_first=True)
df
y=df.quality
y=y.values.reshape(-1,1)
y

X=df.drop('quality',axis=1).values
X
X,y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(X,y)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train.ravel())
from sklearn.model_selection import cross_val_score
cross_val_score(estimator=clf,X=X_train,y=y_train.ravel(),cv=7)
from sklearn.metrics import accuracy_score
y_pred_train=clf.predict(X_train)
accuracy_score(y_train,y_pred_train)
from sklearn.metrics import accuracy_score
y_pred_test=clf.predict(X_test)
accuracy_score(y_test,y_pred_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_test)
from sklearn.ensemble import RandomForestClassifier
clf_rand=RandomForestClassifier(n_estimators=500,criterion='entropy',max_features=6,max_depth=10,random_state=42)
clf_rand.fit(X_train,y_train.ravel())
cross_val_score(estimator=clf_rand,X=X_train,y=y_train.ravel(),cv=7).mean()
y_pred_train=clf_rand.predict(X_train)
accuracy_score(y_train,y_pred_train)

y_pred_test=clf_rand.predict(X_test)
accuracy_score(y_test,y_pred_test)
confusion_matrix(y_test,y_pred_test)
from sklearn.model_selection import GridSearchCV
random_classifier = RandomForestClassifier()
parameters = { 'max_features':np.arange(4,7),'n_estimators':[58],'min_samples_leaf': [5,10,15],'criterion':['entropy']}
random_grid = GridSearchCV(random_classifier, parameters, cv = 7)
random_grid.fit(X_train,y_train.ravel())
random_grid.best_params_
cross_val_score(estimator=random_classifier,X=X_train,y=y_train.ravel(),cv=7).mean()
random_grid.score(X_train,y_train)
y_pred_train=random_grid.predict(X_train)
accuracy_score(y_train,y_pred_train)
y_pred_test=random_grid.predict(X_test)
accuracy_score(y_test,y_pred_test)
confusion_matrix(y_test,y_pred_test)
from sklearn.naive_bayes import GaussianNB
cls_nb=GaussianNB()
cls_nb.fit(X_train,y_train.ravel())
cross_val_score(estimator=cls_nb,X=X_train,y=y_train.ravel(),cv=7).mean()
y_prd=cls_nb.predict(X_train)
accuracy_score(y_train,y_prd)
y_prd_ttest=cls_nb.predict(X_test)
accuracy_score(y_test,y_prd_ttest)
confusion_matrix(y_test,y_prd_ttest)