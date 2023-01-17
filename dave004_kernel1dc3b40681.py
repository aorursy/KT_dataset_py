import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn import preprocessing
pd.set_option('display.max_columns',500)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data1 = pd.read_csv("../input/adult-census-income/adult.csv")

x=data1.drop('income',axis=1)

y=data1['income']
x,y = train_test_split(data1,random_state=7)
x.head()
x.groupby('income').size()
x.info()
x.describe()
x.shape
(x=='?').sum()
((x=='?').sum()*100/32561).round(2)
((y=='?').sum()*100/32561).round(2)
#data[data[::] != '?']

x = x[(x['workclass']!='?')& (x['occupation']!='?') & (x['native.country']!='?')]
#data[data[::] != '?']

y = y[(y['workclass']!='?')& (y['occupation']!='?') & (y['native.country']!='?')]
(x=='?').sum()
(y=='?').sum()
x.info()
sns.pairplot(x)
correlation = x.corr()

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlation, vmin=-1, vmax=1)

fig.colorbar(cax)

#sns.heatmap(x.select_dtypes([object]), annot=True, annot_kws={"size": 7})


name = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']



for c in name:

    sns.boxplot(x=x[c],data=x)



    plt.show()

x.select_dtypes(['object']).head()
x['income'].unique()
x['workclass'].unique()
x['education'].unique()
x['occupation'].unique()
x['sex'].unique()
x['workclass'].unique()
x['native.country'].unique()
y['native.country'].unique()
y.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)
x.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)
x['native.country'].unique()
x['net_capital']=x['capital.gain']-x['capital.loss']

x.drop(['capital.gain','capital.loss'],1,inplace=True)
y['net_capital']=y['capital.gain']-y['capital.loss']

y.drop(['capital.gain','capital.loss'],1,inplace=True)
y.head()
x.head()


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:

    sns.distplot(x[c], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

    plt.show()


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:

    sns.distplot(y[c], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

    plt.show()
d = x.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]
d1 = y.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]
d.head()
d1.head()
from sklearn.preprocessing import Normalizer
pt = preprocessing.QuantileTransformer(output_distribution='normal')

d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])

pt = preprocessing.QuantileTransformer(output_distribution='normal')

d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])



d=pd.DataFrame(Normalizer().fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])



d1=pd.DataFrame(Normalizer().fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])

pt = preprocessing.QuantileTransformer(output_distribution='normal')

d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])

pt = preprocessing.QuantileTransformer(output_distribution='normal')

d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])

name = ['age','fnlwgt','education.num','net_capital','hours.per.week']



for c in name:

    sns.distplot(d[c], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

    plt.show()
name = ['age','fnlwgt','education.num','net_capital','hours.per.week']



for c in name:

    sns.distplot(d1[c], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

    plt.show()
sns.heatmap(x.corr(),annot = True)
sns.heatmap(y.corr(),annot = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for c in x.select_dtypes(['object']).columns:

    

        x[c]=le.fit_transform(x[c])

        
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for c in y.select_dtypes(['object']).columns:

    

        y[c]=le.fit_transform(y[c])

        
d.head()
d1.head()
x.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)
y.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)
x=pd.merge(x,d,left_index=True,right_index=True)
y=pd.merge(y,d,left_index=True,right_index=True)
x.head()
x.shape
y.head()
#pca

#treebaseapproach

#rfe
plt.figure(figsize=(20,10))

sns.heatmap(x.corr(),annot = True)
plt.figure(figsize=(20,10))

sns.heatmap(y.corr(),annot = True)
x_train = x.drop('income',1)

y_train = x['income']
x_test = x.drop('income',1)

y_test = x['income']
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb
from sklearn.metrics import confusion_matrix,accuracy_score
import warnings

warnings.filterwarnings('ignore')
rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))





plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
from sklearn.ensemble import RandomForestClassifier

 

# Feature importance values from Random Forests

rf = RandomForestClassifier(n_jobs=-1, random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = RandomForestClassifier(random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))




rf = AdaBoostClassifier( random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = AdaBoostClassifier( random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))
rf = GradientBoostingClassifier( random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = GradientBoostingClassifier( random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))
rf = xgb.XGBClassifier(random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = xgb.XGBClassifier(random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))
x['income']=le.fit_transform(x['income'])
x.head()
for c in x.select_dtypes(['object']).columns:

    cont = pd.get_dummies(x[c],prefix='Contract')

    x = pd.concat([x,cont],axis=1)

    x.drop(c,1,inplace=True)

    
for c in y.select_dtypes(['object']).columns:

    cont = pd.get_dummies(y[c],prefix='Contract')

    y = pd.concat([y,cont],axis=1)

    y.drop(c,1,inplace=True)

    
x.head()
x.shape
x_train = x.drop('income',1)

y_train = x['income']
x_test = x.drop('income',1)

y_test = x['income']
rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))





plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')

rfe = rfe.fit(x_train,y_train)



col = x_train.columns[rfe.support_]



acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))



print('Number of features selected: {}'.format(rfe.n_features_))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))



plt.figure()

plt.xlabel('k')

plt.ylabel('CV accuracy')

plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

plt.show()
from sklearn.ensemble import RandomForestClassifier

 

# Feature importance values from Random Forests

rf = RandomForestClassifier(n_jobs=-1, random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = RandomForestClassifier(random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))




rf = AdaBoostClassifier( random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = AdaBoostClassifier( random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))
rf = GradientBoostingClassifier( random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = GradientBoostingClassifier( random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))
rf = xgb.XGBClassifier(random_state=1)

rf.fit(x_train, y_train)

feat_imp = rf.feature_importances_



cols = x_train.columns[feat_imp >= 0.01]

est_imp = xgb.XGBClassifier(random_state=1)

est_imp.fit(x_train[cols], y_train)

 

# Test accuracy

acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))

print('Number of features selected: {}'.format(len(cols)))

print('Test Accuracy {}'.format(acc))

print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))