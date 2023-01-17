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
df=pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/train.csv')
df
df.info()
#lets check for the null values in the dataset

df.isnull().sum()
#lets first delete the columns which are not much necessary 

df=df.drop(['id','Region_Code','Vintage'],axis=1)
#now lets visualize the dataset to get some insight information from the dataset

import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(df[df['Response']==1]['Gender'])
#it is quite clear from the above figure that there are high chances of getting insurance to male as compared to female
sns.countplot(df[df['Response']==0]['Gender'])
#here also mostly males dont get the insurance
sns.countplot(df[df['Driving_License']==1]['Gender'])
# few females have driving_license as compare to mens
sns.countplot(df[df['Response']==1]['Previously_Insured'])
#people who have not insured previously have high chances of buying insurance
sns.countplot(df[df['Response']==1]['Vehicle_Age'])
#vehicles which are 1-2 year older gets insurance easily
sns.countplot(df[df['Response']==1]['Vehicle_Damage'])
#its quite obvious that damaged vehicle can claim insurance easily
sns.distplot(df['Annual_Premium'])
sns.countplot(df['Response'])
#well there is  a huge imbalance in the classes which may lead the classifier to predict the class with higher count
#lets prepare the dataset for model building

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
y=df['Response']

x=df.drop(['Response'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

lr=LogisticRegression(max_iter=100000)

list_models=[]

list_scores=[]

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)

list_models.append('logistic regression')
cm_1=confusion_matrix(y_test,pred_1)

labels = ['true_neg','false_pos','false_neg','true_pos']

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm_1, annot=labels, fmt='')

#lets analyse the results more precisely on the basis of these four parametrs 

TP = cm_1[0][0]

FN = cm_1[0][1]

FP = cm_1[1][0]

TN = cm_1[1][1]
#we have created the list for each 4 parameters to analyse the results more accurately and precisely

list_tp=[]

list_fn=[]

list_fp=[]

list_tn=[]
list_tp.append(TP)

list_fn.append(FN)

list_fp.append(FP)

list_tn.append(TN)
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
plt.figure(figsize=(12,5))

plt.bar(range(1,21),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy score')

plt.show()

        
#k=19 gives the best accuracy score

knn=KNeighborsClassifier(n_neighbors=19)

knn.fit(x_train,y_train)

pred_2=knn.predict(x_test)

score_2=accuracy_score(y_test,pred_2)

list_models.append('kneighbors')

list_scores.append(score_2)
cm_2=confusion_matrix(y_test,pred_2)

labels = ['true_neg','false_pos','false_neg','true_pos']

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm_2, annot=labels, fmt='')
TP = cm_2[0][0]

FN = cm_2[0][1]

FP = cm_2[1][0]

TN = cm_2[1][1]
list_tp.append(TP)

list_fn.append(FN)

list_fp.append(FP)

list_tn.append(TN)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_3=rfc.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('random forest')
cm_3=confusion_matrix(y_test,pred_3)

labels = ['true_neg','false_pos','false_neg','true_pos']

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm_3, annot=labels, fmt='')
TP = cm_3[0][0]

FN = cm_3[0][1]

FP = cm_3[1][0]

TN = cm_3[1][1]
list_tp.append(TP)

list_fn.append(FN)

list_fp.append(FP)

list_tn.append(TN)
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(x_test)

score_4=accuracy_score(y_test,pred_4)

list_scores.append(score_4)

list_models.append('xgboost')

cm_4=confusion_matrix(y_test,pred_4)

labels = ['true_neg','false_pos','false_neg','true_pos']

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm_4, annot=labels, fmt='')
TP = cm_4[0][0]

FN = cm_4[0][1]

FP = cm_4[1][0]

TN = cm_4[1][1]
list_tp.append(TP)

list_fn.append(FN)

list_fp.append(FP)

list_tn.append(TN)
plt.figure(figsize=(12,5))

plt.bar(list_models,list_scores)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')

plt.show()
#lets compare the true positive values among classifiers

plt.figure(figsize=(12,5))

sns.barplot(x=list_models,y=list_tp)
#comparison of false positive values among classifiers

plt.figure(figsize=(12,5))

sns.barplot(x=list_models,y=list_fp)
#compariosn of false negative values among classifiers

plt.figure(figsize=(12,5))

sns.barplot(x=list_models,y=list_fn)
#comparison of true negative values among classifiers

plt.figure(figsize=(12,5))

sns.barplot(x=list_models,y=list_tn)
#the conclusion we can draw is that random forest classifier performs slightly poor as comparison to others