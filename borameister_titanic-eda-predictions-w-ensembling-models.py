# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.graph_objects as go


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test  = pd.read_csv('/kaggle/input/titanic/test.csv')
train.sample(10)
print('Train Data Info\n')
train.info()
print('\n\n\nTest Data Info\n')
test.info()
print('Train Data\n')
print(train.isnull().sum().sort_values(ascending=False))
print('\nTest Data\n')
print(test.isnull().sum().sort_values(ascending=False))
fig, ax  = plt.subplots(1,2, figsize=(25,5))

ax[0].set_title('Train Nulls')
sns.barplot(x=train.columns, y=train.isnull().sum(), ax=ax[0])
ax[1].set_title('Test Nulls')
sns.barplot(x=test.columns, y=test.isnull().sum(), ax=ax[1])
plt.show()
train.describe()
cat1 = ['Survived(obj)', 'Pclass', 'Sex', 'Embarked']

train['Survived(obj)'] = ['Yes' if i == 1 else 'No' for i in train.Survived]

fig, axs = plt.subplots(2,2, figsize=(10,7))

c = 0
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        sns.barplot(train[cat1[c]].value_counts().index, train[cat1[c]].value_counts().values, ax=axs[i][j])
        axs[i][j].set_title(cat1[c])
        axs[i][j].set_ylabel('Frequency')
        c += 1
train.drop(['Survived(obj)'], axis=1, inplace=True)
# cat1 = ['Survived(obj)', 'Pclass', 'Sex', 'Embarked']

# train['Survived(obj)'] = ['Yes' if i == 1 else 'No' for i in train.Survived]

# fig, axs = plt.subplots(2,2, figsize=(15,10))

# c = 0
# for i in range(axs.shape[0]):
#     for j in range(axs.shape[1]):
#         sns.countplot(train[cat1[c]], ax=axs[i][j], hue=train[cat1[c]])
#         axs[i][j].set_title(cat1[c])
#         c += 1
# train.drop(['Survived(obj)'], axis=1, inplace=True)
cat2 = ['SibSp','Parch']

fig, axs = plt.subplots(1,2, figsize=(15,10))

for i in range(axs.shape[0]):
    axs[i].pie(train[cat2[i]].value_counts().values, labels=train[cat2[i]].value_counts().index, autopct='%1.1f%%')
    axs[i].set_title(cat2[i])
cat2 = ['Name', 'Ticket', 'Cabin']

for i in cat2:
    print(train[i].value_counts(),'\n\n')
plt.figure(figsize=(12,7))
train[train.Fare > 200].groupby('Name')['Fare'].max().plot(kind='barh', color='black')
plt.title('People Paid Fares More Than 200£')
plt.xlabel('Fare')
plt.show()
num = ['Age','Fare']

fig, axs = plt.subplots(1,2, figsize=(14,6))

for i in range(axs.shape[0]):
    sns.distplot(train[num[i]], ax=axs[i], color='blue')
    axs[i].set_title(num[i])
    
f, axs2 = plt.subplots(1,2, figsize=(14,6))
for i in range(axs.shape[0]):
    sns.boxplot(y = num[i], ax=axs2[i], data=train)
#     sns.swarmplot(y = num[i], ax=axs2[i], data=train, x='Pclass', color='.25')
    axs2[i].set_title(num[i])
train.groupby('Sex')[['Survived']].mean()
pvt = train.groupby(['Pclass','Sex'], as_index=0)['Survived'].mean()

plt.figure(figsize=(10,7))
sns.heatmap(pvt.pivot('Pclass','Sex','Survived'), cmap='Blues', annot=True)

train.groupby(['Pclass','Sex'])[['Survived']].mean()#.plot(kind='barh')
from collections import Counter as count

def detect_outliers(data, features):
    indices = []
    for f in features:
        q1 = np.quantile(data[f], .25) #np.nanpercentile(data[f], .25)
        q3 = np.quantile(data[f], .75) #np.nanpercentile(data[f], .75)
        iqr = np.abs(q3 - q1)
        
        min_val = q1 - 1.5 * iqr
        max_val = q3 + 1.5 * iqr
        
#         indices.extend([idx for idx, i in enumerate(data[f]) if i < min_val or i > max_val])
        indices.extend(data[(data[f] < min_val) | (data[f] > max_val)].index)
    
    outliers_morethan2 = [k for k,v in count(indices).items() if v > 2]
    
    return outliers_morethan2
outlier_indices = detect_outliers(train, ['SibSp','Parch','Fare'])
print(len(outlier_indices))

train = train.iloc[~train.index.isin(outlier_indices)].reset_index()
# or
# traindata = train.drop(outlier_indices, axis=0).reset_index()
data = pd.concat([train, test], axis=0).reset_index()

data.isnull().sum().sort_values(ascending=False)[data.isnull().sum() > 0]
data.groupby(['Pclass','Embarked'])[['Fare']].agg(['mean','median'])
data[data.Embarked.isnull()]
data[data.Fare.isnull()]
plt.figure(figsize=(10,7))
sns.boxplot(y='Fare', x='Embarked', data=data)
plt.show()
data['Embarked'] = data.Embarked.fillna('C')
print('Missing values in "Embarked" column: ',data.Embarked.isnull().sum())
data['Fare'] = data.Fare.fillna(data[(data.Pclass == 3) & (data.Embarked == 'S')]['Fare'].median())

print('Missing values in "Fare" column: ',data.Fare.isnull().sum())
print('Null Age values: ',data.Age.isnull().sum())
plt.figure(figsize=(8,5))
sns.boxplot(y=data.Age, x=data.Sex, hue=data.Pclass)
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(y=data.Age, x=data.SibSp)

plt.figure(figsize=(10,5))
sns.boxplot(y=data.Age, x=data.Parch)
plt.show()
data['SexLabeled'] = [0 if i =='female' else 1 for i in data.Sex]

plt.figure(figsize=(10,7))
sns.heatmap(data[['Age','Parch','Pclass','SibSp','SexLabeled']].corr(), cmap='Blues', annot=True)
plt.show()
# indices of missing values
null_indices = [i for i in data[data.Age.isnull()].index]


for idx in null_indices:
    age_med = data.Age[(data.Pclass == data.Pclass.iloc[idx]) & (data.SibSp == data.SibSp.iloc[idx]) & (data.Parch == data.Parch.iloc[idx])].median()
    
    if not np.isnan(age_med):
        data.Age.iloc[idx] = age_med
    else:
        data.Age.iloc[idx] = data.Age.median()
print('Missing values in "Age" column: ',data.Age.isnull().sum())
# def age_outliers(data, features):
#     indices = []
#     for f in features:
#         q1 = np.quantile(data[f], .25) #np.nanpercentile(data[f], .25)
#         q3 = np.quantile(data[f], .75) #np.nanpercentile(data[f], .75)
#         iqr = np.abs(q3 - q1)
        
#         min_val = q1 - 1.5 * iqr
#         max_val = q3 + 1.5 * iqr
        
# #         indices.extend([idx for idx, i in enumerate(data[f]) if i < min_val or i > max_val])
#         indices.extend(data[(data[f] < min_val) | (data[f] > max_val)].index)
    
#     outliers_morethan2 = [k for k,v in collections.Counter(indices).items()]
    
#     return outliers_morethan2


# data = data.drop(age_outliers(data, ['Age']), axis= 0).reset_index(drop=True)
data['Cabin'] = data.Cabin.fillna('U')
data['Cabin'] = [i[0][0] for i in data.Cabin]

data.Cabin.value_counts(dropna=False)
plt.figure(figsize=(10,5))
sns.barplot(data.Cabin, data.Survived)

data.groupby('Cabin').Survived.agg({'mean','count'})

data.sample(5)
plt.figure(figsize=(9,6))
sns.barplot(x=data.SibSp, y=data.Survived)
plt.title('# of Siblings vs Survival Rate')

data.groupby('SibSp').Survived.agg(['count','mean'])
plt.figure(figsize=(9,6))
sns.barplot(x=data.Parch, y=data.Survived)
plt.title('# of Family Members vs Survival Rate')

data.groupby('Parch').Survived.agg(['count','mean'])
plt.figure(figsize=(9,6))
sns.barplot(x=data.Pclass, y=data.Survived)
plt.title('# of Ticket Class vs Survival Rate')

data.groupby('Pclass').Survived.mean()
g = sns.FacetGrid(data=data, col='Survived', size=4)
g.map(sns.distplot, 'Age')

data.groupby('Age').Survived.mean().nlargest(15)
g = sns.FacetGrid(data= data, col='Survived', row='Pclass',size=3.5)
g.map(plt.hist, 'Age', bins=25)
plt.show()
g = sns.FacetGrid(data= data, col='Embarked', size=4)
g = g.map(sns.barplot, 'Pclass', 'Survived','Sex', palette='Set2')
g.add_legend()
g.set_axis_labels('Pclass', 'Survival Rate')

data.groupby(['Sex','Pclass','Embarked']).Survived.mean()
g = sns.FacetGrid(data = data, row='Embarked', col='Survived')
g = g.map(sns.barplot, 'Sex', 'Fare', palette='Set2')

data.groupby(['Sex','Embarked','Survived']).Fare.mean()
data['Title'] = [i.split('.')[0].split(',')[1].strip() for i in data.Name]
plt.figure(figsize=(9,6))
sns.countplot(data.Title)
plt.xticks(rotation=45)

data.Title.value_counts()
data['Title'] = data['Title'].replace([data.Title.value_counts().index[idx] for idx,title in enumerate(data.Title.value_counts()) if data['Title'].value_counts()[idx] < 50], 'other')
data['Title'] = [0 if title=='Mr' else 1 if title=='Mrs' or title=='Miss' else 2 if title=='Master' else 3 for title in data['Title']]
plt.figure(figsize=(9,6))
sns.barplot(x=data['Title'], y=data.Survived)
plt.xticks(np.arange(data.Title.nunique()),['Mr','Mrs','Master','Other'])
plt.show()
data['Family Size'] = data['Parch'] + data['SibSp'] + 1
data.head(15)
plt.figure(figsize=(9,6))
sns.barplot(x=data['Family Size'], y=data.Survived)

data.groupby('Family Size').Survived.mean()
data['Family Size'] = [0 if i==6 or i==5 else 1 if i==1 or i==7 else 2 for i in data['Family Size']]
sns.barplot(x=data['Family Size'], y=data.Survived)


data.groupby('Family Size').Survived.agg(['count','mean'])
import string 

# string.punctuation : '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

new_tickets = []
for i in data.Ticket:
    if len(i.split()) > 1:  # if not only digits
        for j in string.punctuation:
            i = i.replace(j,'')
        new_tickets.append(i.split()[0])
        
    else: # if only digits
        new_tickets.append('NC')
data.Ticket = new_tickets
data[['Family Size', 'Parch', 'SibSp']].corr()
data.head(10)
data.drop(['level_0','Name','index','SexLabeled'], axis=1, inplace=True)
# data.drop(['level_0','Cabin','Name','PassengerId','index','SexLabeled'], axis=1, inplace=True)

data.head()
one_hot_list = ['Embarked','Pclass','Sex','Title','Family Size','Cabin','Ticket']

data = pd.get_dummies(data, columns= one_hot_list, drop_first= True)

data.head()
train = data[~data.Survived.isna()].reset_index(drop=True)
test  = data[data.Survived.isna()].drop(['Survived'], axis=1).reset_index(drop=True)
test_ID = test['PassengerId']
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

x_train = train.drop(['Survived'], axis=1)
y_train = train.Survived.values.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_trainn, x_val, y_trainn, y_val = train_test_split(x_train, y_train, test_size= 0.25, random_state=3)

sns.barplot(x=['x_train','x_val','y_train','y_val','test'], y=[len(x_train),len(x_val),len(y_train),len(y_val),len(test)])
plt.title('# of Samples')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

clf_list = [KNeighborsClassifier(),
          LogisticRegression(),
          LinearDiscriminantAnalysis(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          SVC()]

knn_params = {'n_neighbors':np.linspace(1,50,5, dtype=int),
             'weights':['uniform','distance'], 
             'metric':['euclidean','minkowski']}

lr_params = {'penalty':['l1','l2'],
            'C':np.logspace(-1,0,6)}

lda_prarams = {'solver':['svd','eigen']}


dtc_params = {'criterion':['gini','entropy'],
             'min_samples_leaf':range(1,65,10), # samples required to be a leaf(not an internal node)
             'max_depth':range(1,20,3)} 

rfc_params = {'n_estimators':range(100,301,100),
             'bootstrap':[False]}

svc_params = {'C':np.logspace(0,2,4),
             'gamma':np.logspace(-1,0,4),
             'probability':[True]}


grid_params = [knn_params, lr_params, lda_prarams, dtc_params, rfc_params, svc_params]
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

cv_results = []
best_estimators = []

for clf, param in zip(clf_list, grid_params):
    
    gs  = GridSearchCV(estimator= clf, param_grid= param, cv= StratifiedKFold(5), scoring='accuracy')
    gs.fit(x_trainn, y_trainn)
    
    cv_results.append(gs.best_score_)
    best_estimators.append(gs.best_estimator_)
    print(clf,'\n\naccuracy: ',gs.score(x_val, y_val),'\n\n')
plt.figure(figsize=(8,5))
sns.pointplot(y=cv_results,x=['K-nearest Neighbor', 'Logistic Regression','LDA','Decision Trees','Random Forest','SVC'])
plt.title('Accuracies')
plt.xticks(rotation=30, color='black', fontweight='bold')
plt.yticks(np.arange(0.5,0.9,0.05))
plt.show()
from sklearn.ensemble import VotingClassifier


for voting in ('hard', 'soft'):
    globals()['votingclf_' + voting] = VotingClassifier(estimators=[('knn', best_estimators[0]),
                                             ('logreg', best_estimators[1]), 
                                             ('lda', best_estimators[2]), 
                                             ('dt', best_estimators[3]), 
                                             ('rf', best_estimators[4]),
                                             ('svc', best_estimators[5])],
                                             voting=voting)
    
    globals()['votingclf_' + voting] = globals()['votingclf_' + voting].fit(x_trainn, y_trainn)
    print('accuracy for voting = {}: {}'.format(voting, globals()['votingclf_' + voting].score(x_val, y_val)))
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


xgb  = XGBClassifier(n_estimators=1000, learning_rate=0.01, reg_lambda=2)
gbc  = GradientBoostingClassifier(n_estimators=500)
lgbm = LGBMClassifier(n_estimators=100)

for i in [xgb, gbc, lgbm]:
    print(i)
    print('score: ',i.fit(x_trainn, y_trainn).score(x_val, y_val))
    print('accuracy: ',sum((i.predict(x_val) == y_val)[0])/len(y_val))
    print('auc: ',roc_auc_score(i.predict(x_val), y_val))
    print('confusion matrix\n',confusion_matrix(y_val, i.predict(x_val), labels=[0,1]), '\n\n')
params = {'loss_function':'Logloss',
         'eval_metric':'AUC',
         'verbose':200,
         'iterations':1000}

cbc  = CatBoostClassifier(**params)
cbc.fit(x_trainn, y_trainn,
       eval_set=[(x_trainn, y_trainn), (x_val, y_val)],
       plot=True)
final_estimators = [SVC(), LogisticRegression(), LinearDiscriminantAnalysis()]

estimators = [('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.01, reg_lambda=2)),
              ('gbc', GradientBoostingClassifier(n_estimators=500)),
              ('lgb', LGBMClassifier(n_estimators=100))]
#               ('catb', CatBoostClassifier(**params))]

for i in final_estimators:
    sc = StackingClassifier(estimators = estimators,
                           final_estimator= i,
                           cv = StratifiedKFold(5, shuffle=True, random_state=3))
    
    sc.fit(x_trainn, y_trainn)
    print(i, '\nScore: ' ,sc.score(x_val, y_val), '\n\n')
    print('Accuracy: ' ,sum((sc.predict(x_val) == y_val)[0])/len(y_val),'\n')
    print('Confusion Matrix\n', confusion_matrix(y_val, sc.predict(x_val)),'\n\n\n')

test_predictions = pd.Series(sc.predict(test), name = "Survived").astype(int)

results = pd.concat([test_ID, test_predictions],axis = 1)
results.to_csv("titanic.csv", index = False)