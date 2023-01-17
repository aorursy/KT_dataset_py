import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.plotly as py
import warnings
warnings.filterwarnings('ignore')
import os
os.getcwd()
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.describe()
train_df.info()
missing_vals=train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Title']=0
for i in train_df:
    train_df['Title']=train_df.Name.str.extract('([A-Za-z]+)\.')

test_df['Title']=0
for i in test_df:
    test_df['Title']=test_df.Name.str.extract('([A-Za-z]+)\.')
    
pd.crosstab(train_df.Title, train_df.Sex).style.background_gradient(cmap='Oranges')
def ageClass (df, var1, var2):
    df[var2]= np.where(df[var1]<18, 'child', 'adult')
    return df
## on training dataframe:
train_df = ageClass(train_df, 'Age','AgeClass')
train_df['AgeClass']= np.where(np.isnan(train_df['Age']), 'unk', train_df['AgeClass'])

## on testing dataframe:
test_df = ageClass(test_df, 'Age','AgeClass')
test_df['AgeClass'] = np.where(np.isnan(test_df['Age']), 'unk', test_df['AgeClass'])

pd.crosstab(train_df.Title, train_df.AgeClass).style.background_gradient(cmap='Oranges')
avgage_df = train_df.groupby(['Title','Sex','AgeClass'])['Age'].mean().reset_index().rename(index=str, columns={'Age': 'AvgAge'})
avgage_df = avgage_df[avgage_df['AgeClass']!='unk']
## training:  
conditions = [
    (train_df['Parch'] > 2),  ## assume there are children therefore passenger is adult
    (train_df['Title'] == 'Miss') & (train_df['SibSp'] > 0),  ## assume there are siblings aboard (given 'Miss' indicated no spouse) therefore passenger is more likely to be of child age
    (train_df['Title'] == 'Mr') & (train_df['SibSp'] > 1),    ## assume 'Mr.' is traveling with siblings, and therefore more likely to be of child age
    (train_df['Title'] == 'Mr'),
    (train_df['Title'] == 'Miss'),
    (train_df['Title'] == 'Dr'),
    (train_df['Title'] == 'Master'),
    (train_df['Title'] == 'Mrs')
    ]
choices = ['adult', 'child', 'child', 'adult', 'adult', 'adult', 'child', 'adult']
train_df['AgeClass']= np.select(conditions, choices, default=train_df['AgeClass'])

## testing:
conditions = [
    (test_df['Parch'] > 2),  ## assume there are children therefore passenger is adult
    (test_df['Title'] == 'Miss') & (test_df['SibSp'] > 0),  ## assume there are siblings aboard (given 'Miss' indicated no spouse) therefore passenger is more likely to be of child age
    (test_df['Title'] == 'Mr') & (test_df['SibSp'] > 1),    ## assume 'Mr.' is traveling with siblings, and therefore more likely to be of child age
    (test_df['Title'] == 'Mr'),
    (test_df['Title'] == 'Miss'),
    (test_df['Title'] == 'Dr'),
    (test_df['Title'] == 'Master'),
    (test_df['Title'] == 'Mrs')
    ]
choices = ['adult', 'child', 'child', 'adult', 'adult', 'adult', 'child', 'adult']
test_df['AgeClass']= np.select(conditions, choices, default='adult')
test_df.head()
train_df = train_df.merge(avgage_df, how='left', on=['Title','Sex','AgeClass'], copy=True)
train_df['Age'] = np.where(train_df['Age']>0, train_df['Age'], train_df['AvgAge'])
train_df = train_df.round({'Age': 0, 'AvgAge': 1, 'Fare': 0})

test_df = test_df.merge(avgage_df, how='left', on=['Title','Sex','AgeClass'], copy=True)
test_df['Age'] = np.where(test_df['Age']>0, test_df['Age'], test_df['AvgAge'])
test_df = test_df.round({'Age': 0, 'AvgAge': 1, 'Fare': 0})
print(train_df.Age.isnull().any())
print(test_df.Age.isnull().any())
train_df['AgeBin'] = pd.cut(train_df.Age, 5)
train_df.groupby(['AgeBin'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
conditions = [
    (train_df['Age'] <= 16),
    (train_df['Age'] > 16) & (train_df['Age'] <= 32),
    (train_df['Age'] > 32) & (train_df['Age'] <= 48),
    (train_df['Age'] > 48) & (train_df['Age'] <= 64),
    (train_df['Age'] > 64)
    ]
choices = [0,1,2,3,4]
train_df['AgeBin']= np.select(conditions, choices, default=train_df['Age'])

conditions = [
    (test_df['Age'] <= 16),
    (test_df['Age'] > 16) & (test_df['Age'] <= 32),
    (test_df['Age'] > 32) & (test_df['Age'] <= 48),
    (test_df['Age'] > 48) & (test_df['Age'] <= 64),
    (test_df['Age'] > 64)
    ]
choices = [0,1,2,3,4]
test_df['AgeBin']= np.select(conditions, choices, default=test_df['Age'])

f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked', data=train_df, ax=ax[0,0])
ax[0,0].set_title('Count of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=train_df, ax=ax[0,1])
ax[0,1].set_title('Embarked by Sex')
sns.countplot('Embarked', hue='Survived', data=train_df, ax=ax[1,0])
ax[1,0].set_title('Survival Rate by Port of Embarkment')
sns.countplot('Embarked', hue='Pclass', data=train_df, ax=ax[1,1])
ax[1,1].set_title('Class by Port of Embarkment')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
train_df['Embarked'].fillna('S', inplace=True)
test_df['Embarked'].fillna('S', inplace=True)
print(train_df.Embarked.isnull().any())
print(test_df.Embarked.isnull().any())
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
train_df['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%.2f', ax=ax[0], shadow=True, colors=['blue','orange'])
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train_df, ax=ax[1], palette=['blue','orange'])
ax[1].set_title('Survived')
plt.show()
train_df.info()
feature_count = [0,1,2,3]
features = ['Sex','Pclass','Embarked','AgeClass']
f,ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
survival_rate = train_df['Survived'].mean()
for f in feature_count:
    sns.barplot(features[f],'Survived', data=train_df, ax=ax[f])
    ax[f].set_ylabel('Survival Rate')
    ax[f].axhline(survival_rate, ls='--', color='grey')
plt.show()
# train_df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color='orange', ax=ax[0])
# ax[0].set_title('Rate of Survival by Sex')
# sns.countplot('Sex', hue='Survived', data=train_df, ax=ax[1], palette=['blue','orange'])
# ax[1].set_title('Survival Status by Sex')
feature_count = [0,1,2,3]
features = ['Sex','Pclass','Embarked','AgeClass']
f,ax = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
for f in feature_count:
    sns.violinplot(features[f],'Age', hue='Survived', data=train_df, split=True, ax=ax[f], palette=['blue','orange'])
    ax[f].set_yticks(range(0,100,10))
plt.show()
pd.crosstab(train_df.Sex, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.Pclass, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.Embarked, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.Embarked, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.AgeClass, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.AgeBin, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.Parch, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
pd.crosstab(train_df.SibSp, train_df.Survived, margins=True).style.background_gradient(cmap='winter')
feature_count = [0,1,2,3,4]
features = ['AgeClass','AgeBin','Pclass','Embarked','SibSp','Parch']
for f in feature_count:
    sns.factorplot(features[f],'Survived', hue='Sex', data=train_df, palette=['purple','teal'])
    plt.show()
f,ax = plt.subplots(1,2,figsize=(18,8))
survival = [0, 1]
title = ['DID NOT Survive', 'DID Survive']
colors = ['blue','orange']
for s in survival:
    train_df[train_df['Survived']==s].Age.plot.hist(ax=ax[s], bins=20, edgecolor='black', color=colors[s])
    x=list(range(0,85,5))
    ax[s].set_xticks(x)
    ax[s].set_title(title[s])

plt.show()

f,ax = plt.subplots(1,2,figsize=(18,8))
survival = [0, 1]
title = ['DID NOT Survive', 'DID Survive']
colors = ['teal','purple']
for s in survival:
    train_df[train_df['Survived']==s].Fare.plot.hist(ax=ax[s], bins=20, edgecolor='black', color=colors[s])
    x=list(range(0,550,50))
    ax[s].set_xticks(x)
    ax[s].set_title(title[s])

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=train_df, ax=ax[0])
ax[0].set_title('Survival rate by number of Siblings/Spouses')
sns.barplot('Parch','Survived',data=train_df, ax=ax[1])
ax[1].set_title('Survival rate by number of Parents/Children on board')
plt.close(2)
plt.show()
f,ax=plt.subplots(1,3,figsize=(20,8))
pclass = [1,2,3]
for p in pclass:
    i = p-1
    sns.distplot(train_df[train_df['Pclass']==p].Fare,ax=ax[i])
    ax[i].set_title('Fares in Class '+str(p))
train_df['Fare_Range']=pd.qcut(train_df['Fare'],4)
train_df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
avgfare = train_df.groupby(['Pclass'])['Fare'].mean().reset_index().rename(index=str, columns={'Fare': 'AvgFare'})
test_df = test_df.merge(avgfare, how='left', on=['Pclass'], copy=True)
test_df['Fare'] = np.where(test_df['Fare']>0, test_df['Fare'], test_df['AvgFare'])
test_df = test_df.round({'Fare': 0})
conditions = [
    (train_df['Fare'] <= 8),
    (train_df['Fare'] > 8) & (train_df['Fare'] <= 14),
    (train_df['Fare'] > 14) & (train_df['Fare'] <= 31),
    (train_df['Fare'] > 31),
    ]
choices = [0,1,2,3]
train_df['FareBin']= np.select(conditions, choices, default=train_df['Fare'])

conditions = [
    (test_df['Fare'] <= 8),
    (test_df['Fare'] > 8) & (test_df['Fare'] <= 14),
    (test_df['Fare'] > 14) & (test_df['Fare'] <= 31),
    (test_df['Fare'] > 31),
    ]
choices = [0,1,2,3]
test_df['FareBin']= np.select(conditions, choices, default=test_df['Fare'])
sns.heatmap(train_df.corr(), annot=True, cmap='summer', linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
train_df.groupby(['Title'])['Survived'].agg(['sum','mean']).style.background_gradient(cmap='summer_r')
train_df['Female'] = np.where(train_df.Sex == 'female', 1, 0)
train_df['Embarked_S'] = np.where(train_df.Embarked == 'S', 1, 0)
train_df['Embarked_C'] = np.where(train_df.Embarked == 'C', 1, 0)
train_df['Embarked_Q'] = np.where(train_df.Embarked == 'Q', 1, 0)
train_df['FamilySize'] = train_df.SibSp + train_df.Parch


test_df['Female'] = np.where(test_df.Sex == 'female', 1, 0)
test_df['Embarked_S'] = np.where(test_df.Embarked == 'S', 1, 0)
test_df['Embarked_C'] = np.where(test_df.Embarked == 'C', 1, 0)
test_df['Embarked_Q'] = np.where(test_df.Embarked == 'Q', 1, 0)
test_df['FamilySize'] = test_df.SibSp + test_df.Parch
train_df = train_df.drop(['Name','Age','Sex','SibSp','Parch','Ticket','Fare','Embarked','Title','AgeClass','AvgAge','Fare_Range'], axis=1)

test_df = test_df.drop(['Name','Age','Sex','SibSp','Parch','Ticket','Fare','Embarked','Title','AgeClass','AvgAge', 'AvgFare'], axis=1)
train_df = train_df.set_index('PassengerId', drop=True)

test_df = test_df.set_index('PassengerId', drop=True)
# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(train_df)
# df_normalized = pd.DataFrame(np_scaled)
# list(train_df.columns.values)
# df_normalized = df_normalized.rename(columns={0: 'Survived', 1: 'Pclass', 2: 'AgeBin', 3: 'FareBin', 4: 'Female', 5: 'Embarked_S', 6: 'Embarked_C', 7: 'Embarked_Q', 8: 'FamilySize'})
sns.heatmap(train_df.corr(), annot=True, cmap='summer', linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
RANDOM = 8675309
tr,ts = train_test_split(train_df, test_size=0.4, random_state=RANDOM, stratify=train_df['Survived'])
tr_x = tr[tr.columns[1:]]
tr_y = tr[tr.columns[:1]]
ts_x = ts[ts.columns[1:]]
ts_y = ts[ts.columns[:1]]
X = train_df[train_df.columns[1:]]
y = train_df['Survived']
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(tr_x, tr_y)
pred1 = model.predict(ts_x)
print('Accuracy for rbf SVM: ', metrics.accuracy_score(pred1, ts_y))
model = LogisticRegression()
model.fit(tr_x, tr_y)
pred2 = model.predict(ts_x)
print('Accuracy for Logistic Regression: ', metrics.accuracy_score(pred2, ts_y))
model = DecisionTreeClassifier()
model.fit(tr_x, tr_y)
pred3 = model.predict(ts_x)
print('Accuracy for Decision Tree: ', metrics.accuracy_score(pred3, ts_y))
model = svm.SVC(kernel='linear', C=.1, gamma=0.1)
model.fit(tr_x, tr_y)
pred4 = model.predict(ts_x)
print('Accuracy for linear SVM: ', metrics.accuracy_score(pred4, ts_y))
model = KNeighborsClassifier(n_neighbors=10)
model.fit(tr_x, tr_y)
pred5 = model.predict(ts_x)
print('Accuracy for KNN: ', metrics.accuracy_score(pred5, ts_y))
a = pd.Series()
ix = list(range(1,15))
x = list(range(0,15))
for i in ix:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(tr_x, tr_y)
    pred = model.predict(ts_x)
    a = a.append(pd.Series(metrics.accuracy_score(pred, ts_y)))
plt.plot(ix, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()
max_acc = a.values.max()
k_opts = np.where(a.values == max_acc)
print('Accuracy for K neighbors: ', a.values)
print('Highest accuracy: ', max_acc)    
print('Selection for K: ', ([x[0] for x in k_opts][0])+1)
model = GaussianNB()
model.fit(tr_x, tr_y)
pred6 = model.predict(ts_x)
print('Accuracy for Naive Bayes: ', metrics.accuracy_score(pred6, ts_y))
model = RandomForestClassifier(n_estimators=150)
model.fit(tr_x, tr_y)
pred7 = model.predict(ts_x)
print('Accuracy for Random Forest: ', metrics.accuracy_score(pred7, ts_y))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=RANDOM)

mean_array = []
accuracy_array = []
std_array = []

classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes', 'Random Forest']
models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=150)]
for i in models:
    cv_result = cross_val_score(i, X, y, cv=kfold, scoring = 'accuracy')
    mean_array.append(cv_result.mean())
    std_array.append(cv_result.std())
    accuracy_array.append(cv_result)
models2_df = pd.DataFrame({'CV Mean': mean_array, 'CV Standard Deviation': std_array}, index=classifiers)
models2_df
plt.subplots(figsize=(16,6))
box = pd.DataFrame(accuracy_array, index=[classifiers])
box.T.boxplot()
models2_df['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()
f,ax = plt.subplots(2,4,figsize=(16,6))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[0,0], annot=True, fmt='2.0f')
ax[0,0].set_title('Radial SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[0,1], annot=True, fmt='2.0f')
ax[0,1].set_title('Linear SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=10), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[0,2], annot=True, fmt='2.0f')
ax[0,2].set_title('KNN (k=10)')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=150), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[0,3], annot=True, fmt='2.0f')
ax[0,3].set_title('Random Forest')

y_pred = cross_val_predict(LogisticRegression(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[1,0], annot=True, fmt='2.0f')
ax[1,0].set_title('Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[1,1], annot=True, fmt='2.0f')
ax[1,1].set_title('Decision Tree')

y_pred = cross_val_predict(GaussianNB(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[1,2], annot=True, fmt='2.0f')
ax[1,2].set_title('Naive Bayes')

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
from sklearn.model_selection import GridSearchCV
C = [0.05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1]
gamma = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
kernel = ['rbf', 'linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
gd = GridSearchCV(estimator = svm.SVC(), param_grid = hyper, verbose = True)
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
n_estimators = range(100, 1000, 50)
hyper = {'n_estimators': n_estimators}
gd = GridSearchCV(estimator = RandomForestClassifier(random_state=RANDOM), param_grid=hyper, verbose=True)
gd.fit(X,y)
print(gd.best_score_)
print(gd.best_estimator_)
from sklearn.ensemble import VotingClassifier
ensemble1 = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                        ('RBF', svm.SVC(probability=True, kernel='rbf', C=0.75, gamma=0.2)),
                                        ('RF', RandomForestClassifier(n_estimators=650, random_state=RANDOM)),
                                        ('LR', LogisticRegression(C=0.05)),
                                        ('DT', DecisionTreeClassifier(random_state=RANDOM)),
                                        ('NB', GaussianNB()),
                                        ('SVM', svm.SVC(kernel='linear', probability=True))],
                            voting='soft').fit(tr_x, tr_y)
acc = ensemble1.score(ts_x, ts_y)
print('Accuracy: ', acc)
cross = cross_val_score(ensemble1, X, y, cv = 10, scoring='accuracy')
print('Cross Validation Score: ', cross.mean())
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=3), random_state=RANDOM, n_estimators=500)
model.fit(tr_x, tr_y)
pred=model.predict(ts_x)
acc = metrics.accuracy_score(pred, ts_y)
print('Accuracy for bagged KNN: ', acc)
result = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print('Cross Validation for bagged KNN: ', result.mean())
model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state=RANDOM, n_estimators=100)
model.fit(tr_x, tr_y)
pred=model.predict(ts_x)
acc = metrics.accuracy_score(pred, ts_y)
print('Accuracy for bagged decision tree: ', acc)
result = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print('Cross Validation for bagged decision tree: ', result.mean())
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 200, random_state = RANDOM, learning_rate = 0.1)
result = cross_val_score(ada, X, y, cv=10, scoring='accuracy')
print('Cross Validation for AdaBoost: ', result.mean())
n_estimators = list(range(100, 1000, 100))
learn_rate = [.05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1]
hyper = {'n_estimators': n_estimators, 'learning_rate': learn_rate}
gd = GridSearchCV(estimator = AdaBoostClassifier(), param_grid=hyper, verbose=True)
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
ada = AdaBoostClassifier(n_estimators=100, random_state=RANDOM, learning_rate=.05)
result = cross_val_predict(ada, X, y, cv=10)
sns.heatmap(confusion_matrix(y, result), cmap='winter', annot=True, fmt='2.0f')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(n_estimators=500, random_state=RANDOM, learning_rate=0.1)
result = cross_val_score(grad, X, y, cv=10, scoring='accuracy')
print('Cross Validation for SGB: ', result.mean())
import xgboost as xg
xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result = cross_val_score(xgboost, X, y, cv=10, scoring='accuracy')
print('Cross Validation for XGB: ', result.mean())
f, ax = plt.subplots(2,2, figsize=(16,8))
model = RandomForestClassifier(n_estimators=650, random_state=RANDOM)
model.fit(X, y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.75, ax=ax[0,0])
ax[0,0].set_title('Random Forest')

model = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM)
model.fit(X, y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0,1])
ax[0,1].set_title('AdaBoost')

model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=RANDOM)
model.fit(X, y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1,0])
ax[1,0].set_title('SGB')

model = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
model.fit(X, y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1,1])
ax[1,1].set_title('XGB')

plt.show()
print(test_df.info())
print(train_df.info())
model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state=RANDOM, n_estimators=100)
model.fit(X, y)
pred=model.predict(test_df)
PassengerId = pd.DataFrame(test_df.axes[0].tolist())
Survived = pd.DataFrame(pred)
# pd.DataFrame(pred, index=, columns='Survived')
submission = pd.DataFrame({
        "PassengerId": PassengerId[0],
        "Survived": Survived[0]
    })