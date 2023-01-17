import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os



from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn import ensemble

from sklearn import model_selection





from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



from sklearn.metrics import roc_curve, auc #for model evaluation

df = pd.read_csv('../input/heart.csv')
df.head()
#See the columns

df.columns
df.shape
#what is the target variable

df['target'].unique()
#lets see how many nulls do we have

print(df.isnull().sum())
#Data types of features

df.dtypes
df.describe()
#count of target variables

print(df.groupby('target')['target'].count())

sns.countplot("target",data = df)
# How age and heart disease relates

plt.figure(figsize = (19, 8))

sns.countplot('age', data=df,hue='target');

plt.title('Age v/s Cancer');
#Lets see the % of different age groups that have heart disease to understand better

age_target = df[df['target'] ==1].groupby('age')['target'].sum()

bins = [10, 20, 30, 40,50,60,70,80,90]

binned_age = pd.cut(age_target.values, bins=bins).value_counts()

plt.figure(figsize=(16, 8))

plt.hist(df.age,bins = 10)

plt.xlabel("Age")

plt.ylabel("Count")
age_target.values
#lets see how heart disease is related with Sex

sns.countplot('sex',data = df, hue='target')
#Count of each sex

print(df.groupby('sex')['sex'].count())

sns.countplot("sex",data = df)
#  % of each sex having heart disease

df[df['target'] ==1].sex.value_counts()/df.sex.value_counts()
df[df['target'] ==1].sex.value_counts()
features = ['trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','cp']

f, axes = plt.subplots(3, 5, figsize=(20, 20))

plt.suptitle('Distribution plots')

i=0

for feature in features:

        sns.distplot(df[feature],hist=False,label=feature,ax=axes[i // 5][i % 5]);

        i+=1
#lets see if there is any correaltion amoungst them

plt.figure(figsize=(15, 8))

sns.heatmap(df[features].corr(),annot = True)
#trestbps

a, b = 1, 10

m, n = df.trestbps.min(), df.trestbps.max()

df['trestbps'] = (df.trestbps - m) / (n - m) * (b - a) + a
#trestbps

a, b = 1, 10

m, n = df.chol.min(), df.chol.max()

df['chol'] = (df.chol - m) / (n - m) * (b - a) + a
#trestbps

a, b = 1, 10

m, n = df.thalach.min(), df.thalach.max()

df['thalach'] = (df.thalach - m) / (n - m) * (b - a) + a
X = df.loc[:, df.columns != 'target'].values

y = df['target'].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 
model = ensemble.RandomForestClassifier(n_estimators=800)



# create the ensemble model

seed = 450

kfold = model_selection.StratifiedKFold(random_state=seed)

results = model_selection.cross_val_score(model, X_train, 

                    y_train, cv=10)

for i in range(len(results)):

    print("Fold", i+1, "score: ", results[i])

print("Cross-validation score average on original data: ", results.mean())
df = df.loc[:, df.columns != 'target']
model.fit(X, y)

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(25, 12))                                                        



feature_importances = pd.DataFrame(importances,index = df.columns, columns = ['importance'])

feature_importances.sort_values('importance', ascending = False).plot(kind = 'bar',

                        figsize = (35,8), color = 'r', yerr=std[indices], align = 'center')

plt.xticks(rotation=90)

plt.xlabel("features",fontsize=16)

plt.show()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
ac=accuracy_score(y_test,y_pred)

print("Accuracy ", ac)
rc=recall_score(y_test, y_pred, average='macro')

print("recall score ",rc)



pc=precision_score(y_test, y_pred, average='macro') 



print("precision score ",pc)
cm = confusion_matrix(y_test, y_pred)

print(cm)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
#pruned=[]

#for i in range(len(feature_importances)):

 #   if(feature_importances.importance[i] < 0.04):

  #      pruned.append(feature_importances.index[i])



#len(pruned)