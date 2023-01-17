# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
brain_df=pd.read_csv('../input/emotions.csv')

brain_df.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

sns.countplot(x=brain_df.label,color='mediumseagreen')

plt.title('Emotional sentiment class,fontsize=20')

plt.ylabel('class counts',fontsize=18)

plt.xlabel('class counts',fontsize=18)

plt.xticks(rotation='vertical')
brain_df.count
label_df=brain_df['label']

brain_df.drop('label',axis=1,inplace=True)

brain_df.head()
%%time

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score,train_test_split

pl_random_forest=Pipeline(steps=[('random_forest',RandomForestClassifier())])

scores=cross_val_score(pl_random_forest,brain_df,label_df,cv=10,scoring='accuracy')

print('Accuracy for RandomForest:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score,train_test_split

pl_log_reg=Pipeline(steps=[('scaler',StandardScaler()),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])

scores=cross_val_score(pl_log_reg,brain_df,label_df,cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression:',scores.mean())
%%time

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scale=scaler.fit_transform(brain_df)

pca=PCA(n_components=20)

pca_vectors=pca.fit_transform(scale)

for index,var in enumerate(pca.explained_variance_ratio_):

    print('variances by PCA',(index+1),":",var)

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(25,8))

sns.scatterplot(x=pca_vectors[:,0],y=pca_vectors[:,1],hue=label_df)

plt.title('PCA distribution ,fontsize=20')

plt.ylabel('PCA 1',fontsize=18)

plt.xlabel('PCA 2',fontsize=18)

plt.xticks(rotation='vertical')
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score,train_test_split

pl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=2)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])

scores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression 2 PCA component:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score,train_test_split

pl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=10)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])

scores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression 2 PCA component:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score,train_test_split

pl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=20)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])

scores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression 2 PCA component:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

pl_mlp=Pipeline(steps=[('scaler',StandardScaler()),('mil_ann',MLPClassifier(hidden_layer_sizes=(1275,637)))])

scores=cross_val_score(pl_mlp,brain_df,label_df,cv=10,scoring='accuracy')

print('ANN:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

pl_svm=Pipeline(steps=[('scaler',StandardScaler()),('svm',LinearSVC())])

scores=cross_val_score(pl_svm,brain_df,label_df,cv=10,scoring='accuracy')

print('SVM:',scores.mean())
%%time

from sklearn.pipeline import Pipeline

import xgboost as xgb

pl_xgb=Pipeline(steps=[('svm',xgb.XGBClassifier(objective='multi:softmax'))])

scores=cross_val_score(pl_xgb,brain_df,label_df,cv=10,scoring='accuracy')

print('XGBoost:',scores.mean())