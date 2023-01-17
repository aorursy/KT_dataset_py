# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections



# Deep Learning Libraries

from keras.models import Sequential

from keras.layers import Dense,Dropout



# Misc. Libraries

from imblearn.datasets import fetch_datasets

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score

import warnings

warnings.filterwarnings("ignore")



import seaborn as sns

import matplotlib.pyplot as plt
data=pd.read_csv("../input/heart.csv")
print('Heart Disease', round(data['target'].value_counts()[1]/len(data) * 100,2), '% of the target ')

print('No heart Disease', round(data['target'].value_counts()[0]/len(data) * 100,2), '% of the target')
sns.countplot('target', data=data, palette="winter")

plt.title('Class Distributions \n 0: No Disease || 1: Disease', fontsize=14)

data.head()
sns.set(style="white", palette="PuBuGn_d", color_codes=True)
fig, ax = plt.subplots(1, 2, figsize=(18,4))

age = data['age'].values

sex = data['sex'].values

sns.distplot(age, ax=ax[0], color='purple')

ax[0].set_title('Distribution of age', fontsize=14)

ax[0].set_xlim([min(age), max(age)])

sns.distplot(sex, ax=ax[1], color='b')

ax[1].set_title('Distribution of sex', fontsize=14)

ax[1].set_xlim([min(sex), max(sex)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

cp = data['cp'].values

trestbps = data['trestbps'].values

sns.distplot(cp, ax=ax[0], color='green')

ax[0].set_title('Distribution of chest pain', fontsize=14)

ax[0].set_xlim([min(cp), max(cp)])

sns.distplot(trestbps, ax=ax[1], color='orange')

ax[1].set_title('Distribution of trestbps', fontsize=14)

ax[1].set_xlim([min(trestbps), max(trestbps)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

chol = data['chol'].values

fbs = data['fbs'].values

sns.distplot(chol, ax=ax[0], color='brown')

ax[0].set_title('Distribution of cholestrol', fontsize=14)

ax[0].set_xlim([min(chol), max(chol)])

sns.distplot(fbs, ax=ax[1], color='blue')

ax[1].set_title('Distribution of fasting blood sugar', fontsize=14)

ax[1].set_xlim([min(fbs), max(fbs)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

restecg = data['restecg'].values

thalach = data['thalach'].values

sns.distplot(restecg,ax=ax[0], color='r')

ax[0].set_title('Distribution of ecg resting electrode', fontsize=14)

ax[0].set_xlim([min(restecg), max(restecg)])

sns.distplot(thalach, ax=ax[1], color='b')

ax[1].set_title('Distribution of thalach', fontsize=14)

ax[1].set_xlim([min(thalach), max(thalach)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

exang = data['exang'].values

oldpeak = data['oldpeak'].values

sns.distplot(exang,ax=ax[0], color='yellow')

ax[0].set_title('Distribution of exang', fontsize=14)

ax[0].set_xlim([min(exang), max(exang)])

sns.distplot(oldpeak, ax=ax[1], color='b')

ax[1].set_title('Distribution of oldpeak', fontsize=14)

ax[1].set_xlim([min(oldpeak), max(oldpeak)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

slope = data['slope'].values

ca = data['ca'].values

sns.distplot(slope,ax=ax[0], color='red')

ax[0].set_title('Distribution of slope', fontsize=14)

ax[0].set_xlim([min(slope), max(slope)])

sns.distplot(ca, ax=ax[1], color='green')

ax[1].set_title('Distribution of ca', fontsize=14)

ax[1].set_xlim([min(ca), max(ca)])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

thal = data['thal'].values

target = data['target'].values

sns.distplot(thal,ax=ax[0], color='blue')

ax[0].set_title('Distribution of thal', fontsize=14)

ax[0].set_xlim([min(thal), max(thal)])

sns.distplot(target, ax=ax[1], color='green')

ax[1].set_title('Distribution of target', fontsize=14)

ax[1].set_xlim([min(target), max(target)])

plt.show()

data.describe()
data.isnull().any()
data.target.value_counts()
data[data.duplicated()==True]
data.drop_duplicates(inplace=True)

data[data.duplicated()==True]
cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark")
data = data.sample(frac=1)



# total heart disease data classes 164 rows.

non_hd_data = data.loc[data['target'] == 0]

hd_data = data.loc[data['target'] == 1][:138]



b_data = pd.concat([non_hd_data,hd_data])



# Shuffle dataframe rows

b_data = b_data.sample(frac=1, random_state=7)



b_data.head()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



# Entire DataFrame

corr = data.corr()

sns.heatmap(corr, cmap=cmap, annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)





sub_sample_corr = b_data.corr()

sns.heatmap(sub_sample_corr, cmap=cmap, annot_kws={'size':20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.show()
print('Distribution of the Classes in the new balanced dataset')

print(b_data['target'].value_counts()/len(b_data))

sns.countplot('target', data=b_data, palette='winter')

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
fig, axes = plt.subplots(ncols=6,figsize=(20,4))

sns.boxplot(x='target',y='age',data=b_data, palette='Oranges', ax=axes[0])

axes[0].set_title('Age vs Target distribution')



sns.boxplot(x='target',y='sex' ,data=b_data, palette='Oranges', ax=axes[1])

axes[1].set_title("Sex vs Target distribution")



sns.boxplot(x='target',y='exang' ,data=b_data, palette='Oranges', ax=axes[2])

axes[2].set_title("exang vs Target distribution")



sns.boxplot(x='target',y='oldpeak' ,data=b_data, palette='Oranges', ax=axes[3])

axes[3].set_title("Oldpeak vs Target distribution")



sns.boxplot(x='target',y='ca' ,data=b_data, palette='Oranges', ax=axes[4])

axes[4].set_title("CA vs Target distribution")



sns.boxplot(x='target',y='trestbps',data=b_data, palette='Oranges',ax=axes[5])

axes[5].set_title("trestbps vs Target distribution")

fig, axes = plt.subplots(ncols=7,figsize=(20,4))

sns.boxplot(x='target',y='cp',data=b_data, palette='winter', ax=axes[0])

axes[0].set_title('Chest Pain vs Target distribution')



sns.boxplot(x='target',y='fbs' ,data=b_data, palette='winter', ax=axes[1])

axes[1].set_title("fbs vs Target ")



sns.boxplot(x='target',y='restecg' ,data=b_data, palette='winter', ax=axes[2])

axes[2].set_title("restecg vs Target distribution")



sns.boxplot(x='target',y='thalach' ,data=b_data, palette='winter', ax=axes[3])

axes[3].set_title("thalach vs Target ")



sns.boxplot(x='target',y='slope' ,data=b_data, palette='winter', ax=axes[4])

axes[4].set_title("slope vs Target distribution")



sns.boxplot(x='target',y='chol' ,data=b_data, palette='winter', ax=axes[5])

axes[5].set_title("chol vs Target ")



sns.boxplot(x='target',y='thal' ,data=b_data, palette='winter', ax=axes[6])

axes[6].set_title("thal vs Target ")

from scipy.stats import norm



f,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(20,4))

age_d=b_data['age'].loc[b_data['target']==0].values

sns.distplot(age_d,ax=ax1,fit=norm,color='g')

ax1.set_title("Age Distirbution \n (Non Heart Disease)", fontsize='14')



sex_d=b_data['sex'].loc[b_data['target']==0].values

sns.distplot(sex_d,ax=ax2,fit=norm,color='red')

ax2.set_title("Sex Distribution \n (Non Heart Disease)", fontsize='14')



exang_d=b_data['exang'].loc[b_data['target']==1].values

sns.distplot(sex_d,ax=ax3,fit=norm,color='blue')

ax3.set_title("exang Distribution \n (Heart Disease)", fontsize='14')



'''oldpeak_d=b_data['oldpeak'].loc[b_data['target']==1].values

sns.distplot(oldpeak_d,ax=ax4,fit=norm,color='blue')

ax4.set_title("oldpeak Distribution \n (Heart Disease)", fontsize='14')'''



oldpeak_d=b_data['oldpeak'].values

sns.distplot(oldpeak_d,ax=ax4,fit=norm,color='blue')

ax4.set_title("oldpeak Distribution \n (Non Heart Disease)", fontsize='14')





f,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(1,5,figsize=(20,4))

ca_d=b_data['ca'].loc[b_data['target']==1].values

sns.distplot(ca_d,ax=ax1,fit=norm,color='y')

ax1.set_title("CA Distirbution \n (Heart Disease)", fontsize='14')



trestbps_d=b_data['trestbps'].loc[b_data['target']==0].values

sns.distplot(trestbps_d,ax=ax2,fit=norm,color='red')

ax2.set_title("trestbps Distribution \n (Non Heart Disease)", fontsize='14')



cp_d=b_data['cp'].loc[b_data['target']==1].values

sns.distplot(cp_d,ax=ax3,fit=norm,color='orange')

ax3.set_title("cp Distribution \n (Heart Disease)", fontsize='14')



fbs_d=b_data['oldpeak'].values

sns.distplot(fbs_d,ax=ax4,fit=norm,color='purple')

ax4.set_title("fbs Distribution \n (Heart Disease)", fontsize='14')



thalach_d=b_data['thalach'].loc[b_data['target']==1].values

sns.distplot(thalach_d,ax=ax5,fit=norm,color='cyan')

ax5.set_title("thalach Distribution \n (Non Heart Disease)", fontsize='14')





X=b_data.iloc[:,:-1]

y=b_data.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=20)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,max_depth=5)

model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)

accuracy
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[1,500], 'max_depth':[1, 15]}

clf = GridSearchCV(model, parameters, cv=5)

clf.fit(X_train,Y_train)

print(clf.best_params_)
model = RandomForestClassifier(n_estimators=500,max_depth=1)

model.fit(X_train, Y_train)

Y_pred=model.predict(X_test)

cm=confusion_matrix(Y_test.tolist(),Y_pred.tolist())

print("Accuracy:",accuracy_score(Y_test,Y_pred))

print("Precision Score :",precision_score(Y_test,Y_pred))

print("f1 Score :",f1_score(Y_test,Y_pred))

print("Confusion Matrix: \n",cm)

total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])

print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])

print('Specificity : ', specificity)
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():

    classifier.fit(X_train, Y_train)

    training_score = cross_val_score(classifier, X_train, Y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, Y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, Y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, Y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, Y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
log_reg_score = cross_val_score(log_reg, X_train, Y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, X_train, Y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, Y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, Y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
X_train.shape
model=Sequential()

model.add(Dense(128, init="uniform", input_dim=13, activation='relu'))

model.add(Dense(64, init ="uniform", activation="relu"))

model.add(Dense(1, init="uniform", activation="sigmoid"))

model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

model.summary()

history=model.fit(X_train,Y_train, epochs=100, batch_size=100)
plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

model.evaluate(X_test,Y_test)