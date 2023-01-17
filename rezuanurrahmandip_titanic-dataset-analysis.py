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
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from matplotlib import cm

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection  import train_test_split, KFold, StratifiedKFold  

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score



from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")



whole_data = train_data.append(test_data)

whole_data.reset_index(inplace=True)

whole_data.drop('index',inplace=True,axis=1)

print("Dimension of whole dataset: {}".format(whole_data.shape)+"\n")

print("Dimension of train dataset: {}".format(train_data.shape)+"\n")

print("Dimension of test dataset: {}".format(test_data.shape)+"\n")



train_data = train_data.drop('PassengerId', axis = 1)

train_data.head()
train_data.info()
train_data.apply(lambda x: sum(x.isnull()))
test_data.apply(lambda x: sum(x.isnull()))
whole_data.loc[:,['Age', 'SibSp', 'Parch', 'Fare']].describe()
cat_var=['Survived','Pclass', 'Sex', 'Embarked']



for i in cat_var:

     print (train_data[i].value_counts()) 

plt.figure(figsize=(5,5))

labels = ('Not survived', 'Survived')

x_pos = np.arange(len(labels))

plt.bar(x_pos, train_data['Survived'].value_counts(), align='center', alpha=0.5, width=0.25)

plt.xticks(x_pos, labels)

plt.ylabel('Frequency')

plt.title('Survival Count')
plt.figure(figsize=(5,5))

labels = ('Not survived', 'Survived')

x_pos = np.arange(len(labels))

df_temp = pd.crosstab(train_data["Survived"], train_data['Pclass']).plot(kind = 'bar', stacked='True')

plt.xticks(x_pos, labels)

plt.ylabel('Frequency')

plt.title('Survival Count')

plt.title('Survived'+" vs "+'Pclass')

ax = sns.FacetGrid(train_data,  margin_titles=True, size = 4)

ax = ax.map(sns.barplot, 'Pclass', 'Survived') 

ax.add_legend()
plt.figure(figsize=(5,5))

labels = ('Not survived', 'Survived')

x_pos = np.arange(len(labels))

df_temp = pd.crosstab(train_data["Survived"], train_data['Sex']).plot(kind = 'bar', stacked='True')

plt.xticks(x_pos, labels)

plt.ylabel('Frequency')

plt.title('Survival Count')

plt.title('Survived'+" vs "+'Sex')
ax = sns.FacetGrid(train_data,  margin_titles=True, size = 4)

ax = ax.map(sns.barplot, 'Sex', 'Survived') 

ax.add_legend()
ax = sns.FacetGrid(train_data,  margin_titles=True, size = 4)

ax = ax.map(sns.barplot, 'Embarked', 'Survived') 

ax.add_legend()
plt.figure(figsize=(12,10))

plt.hist([train_data[train_data['Survived']==1]['Age'].dropna(), train_data[train_data['Survived']==0]['Age'].dropna()], stacked=True, color = ['g','b'],

bins = 20,label = ['Survived','Not survived'])

plt.xlabel('Age')

plt.ylabel('Count of passengers')

plt.legend()
whole_data.boxplot(column ='Age', by = ['Sex'])
whole_data.boxplot(column ='Age', by = ['Sex','Pclass'])
corr = whole_data.corr()

ax = plt.subplots(figsize =(7, 6))

sns.heatmap(corr,annot = True)
whole_data.drop(['PassengerId', 'Survived'], axis =1).apply(lambda x: sum(x.isnull()))
whole_data['Fare'].fillna(whole_data.Fare.median(), inplace = True)

whole_data['Embarked'].fillna(whole_data.Embarked.mode()[0], inplace = True)

whole_data['Cabin'].fillna("N", inplace = True) 

whole_data['Cabin']=whole_data['Cabin'].dropna().map(lambda x: x[0])
pd.crosstab(whole_data['Pclass'], whole_data['Cabin'])
for i, row in whole_data.iterrows():

    whole_data.loc[(whole_data['Pclass']==1) & (whole_data['Cabin']=="N"), 'Cabin'] = 'C'

    whole_data.loc[(whole_data['Pclass']==2) & (whole_data['Cabin']=="N"), 'Cabin'] = 'F'

    whole_data.loc[(whole_data['Pclass']==3) & (whole_data['Cabin']=="N"), 'Cabin'] = 'F'
whole_data['Age'] = whole_data.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
whole_data= whole_data.drop(['Name', 'Parch', 'SibSp', 'Ticket'], axis=1)
var_mod=whole_data.dtypes[whole_data.dtypes == "object"].index.tolist()

var_mod.append('Pclass')

le = LabelEncoder()

for i in var_mod:

    whole_data[i] = le.fit_transform(whole_data[i])
predictor = whole_data.drop(['PassengerId', 'Survived'], axis = 1).columns.tolist()

dataset = whole_data[:891].drop('PassengerId', axis=1)

outcome = 'Survived'



X_train, X_test, y_train, y_test = train_test_split(dataset[predictor], dataset[outcome], test_size = 0.30, random_state = 0)

    

def _modelfit(xgb_model, dataset, outcome, **kwargs):  

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)

    cvresult = xgb.cv(xgb_model.get_xgb_params(), dtrain, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=5, metrics='error', early_stopping_rounds=50, seed = 0)  

    n_boosting=cvresult.index[cvresult['test-error-mean'] == cvresult['test-error-mean'].min()].tolist()[0]+1

    print(cvresult)

    xgb_model.set_params(n_estimators=n_boosting)

    xgb_model.fit(X_train, y_train, eval_metric='error') #auc

    dtrain_predictions = xgb_model.predict(X_test)

    dtrain_predprob = xgb_model.predict_proba(X_test)[:,1]

    print ("Accuracy score (test): {}".format(accuracy_score(y_test, dtrain_predictions)))

    

    

    plot_importance(xgb_model, color = 'green')

    plt.show()

    

clf = XGBClassifier(max_depth=6, n_estimators=2000, objective='binary:logistic', 

                     subsample=1.0, colsample_bytree=1.0, random_state = 0)



_modelfit(clf, dataset, outcome, predictor = predictor)
def _grid_depth_weight(dataset, predictor, outcome):

       

    param_test = {

     'max_depth':[i for i in range(3,11,1)],

     'min_child_weight':[i for i in range(1,11,1)]    

    }

    

    clf = XGBClassifier(n_estimators = 23, objective='binary:logistic', subsample=1.0, colsample_bytree=1.0, random_state=0)

     

    _grid = GridSearchCV(clf, param_grid=param_test, scoring='accuracy', 

                         n_jobs=4, cv=StratifiedKFold(n_splits=5)).fit(X_train,y_train)

    

    clf=_grid.best_estimator_.fit(X_train,y_train)

    

    return _grid.best_params_, _grid.best_score_



_grid_depth_weight(dataset, predictor, outcome)
def _grid_gamma(dataset, predictor, outcome):

    

    param_test = {

      'gamma':[i/10.0 for i in range(0,20)]   

    }

    

    clf = XGBClassifier(n_estimators =23, objective='binary:logistic', max_depth=6, min_child_weight=5, 

                        subsample=1.0, colsample_bytree=1.0, random_state=0)

    

    _grid = GridSearchCV(clf, param_grid = param_test, scoring='accuracy',n_jobs=4, 

                         cv=StratifiedKFold(n_splits=5)).fit(X_train,y_train)

    

    clf=_grid.best_estimator_.fit(X_train, y_train)

    

    return _grid.best_params_, _grid.best_score_



_grid_gamma(dataset, predictor, outcome)
def _grid_sample(dataset, predictor, outcome):

    

    param_test = {

    'subsample':[i/10.0 for i in range(1,11)],

    'colsample_bytree':[i/10.0 for i in range(1,11)]  

    }

    

    clf = XGBClassifier(n_estimators=23, objective='binary:logistic', max_depth=6, min_child_weight=5, 

                                                     gamma = 0.0, seed =0)

    

    _grid = GridSearchCV(clf, param_grid = param_test, scoring='accuracy',n_jobs=4, 

                         cv=StratifiedKFold(n_splits=5)).fit(X_train,y_train)

    

    clf=_grid.best_estimator_.fit(X_train,y_train)

    

    return _grid.best_params_, _grid.best_score_



_grid_sample(dataset, predictor, outcome)
def _grid_reg(dataset, predictor, outcome):

    

    param_test = {

    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 1, 100]  

    }

    

    clf = XGBClassifier(n_estimators=23, objective='binary:logistic', max_depth=6, min_child_weight=5, 

                                                     gamma = 0.0, subsample = 0.9, colsample_bytree = 0.7, seed = 0)   

                         

    _grid = GridSearchCV(clf, param_grid = param_test, scoring='accuracy',n_jobs=4, 

                         cv=StratifiedKFold(n_splits=5)).fit(X_train,y_train)

    

    clf=_grid.best_estimator_.fit(X_train,y_train)

    

    return _grid.best_params_, _grid.best_score_



_grid_reg(dataset, predictor, outcome)
clf = XGBClassifier(

 learning_rate =0.05,

 n_estimators=2000,

 max_depth=8,

 min_child_weight=3,

 gamma=0.3,

 subsample=1,

 colsample_bytree=0.9,

 reg_alpha = 0.005,

 objective='binary:logistic',

 random_state=0)



_modelfit(clf, dataset, outcome)
Survived = clf.predict(whole_data[891:].drop(['Survived', 'PassengerId'], axis=1)).astype(int)

PassengerId = whole_data[891:].PassengerId

xgb = pd.DataFrame({'PassengerId': PassengerId ,'Survived': Survived})

print("Dimension of final set: {}".format(xgb.shape)+"\n")

print(xgb.head())

xgb.to_csv('final_2.csv', index = False)