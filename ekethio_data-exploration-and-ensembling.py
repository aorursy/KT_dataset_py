# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn import ensemble, linear_model, svm

from scipy.stats import boxcox

import warnings

warnings.filterwarnings('ignore')

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ntrain = train.shape[0]

ids = test['PassengerId']

target = train.pop('Survived')



data = pd.concat([train, test], ignore_index = True)



data.Embarked.fillna('S', inplace = True)

data.Age = data.Age.fillna(train.Age.median())

data.Fare = data.Fare.fillna(data.Fare.median())

print(data.isnull().sum())

data.Cabin.fillna('NA', inplace = True)

data.Cabin[data.Cabin != np.nan] = data.Cabin[data.Cabin !=np.nan].apply(lambda x: str(x)[0])





train = data.iloc[:ntrain]

train['Survived'] = target



sns.factorplot('Survived', hue = 'Sex', data = train[train.Cabin != 'N'], col = 'Cabin', kind = 'count', col_wrap = 4)

plt.show()

sns.factorplot('Survived', hue = 'Sex', data = train, col = 'SibSp', kind = 'count', col_wrap = 4,palette = 'Set2')

plt.show()



data = data.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis = 1)

cat = data.select_dtypes(include = ['object']).dtypes.index

num = data.select_dtypes(include =['float64', 'int64'] ).dtypes.index

for col in cat:

    le = LabelEncoder()

    le.fit(data[col])

    data[col] = le.transform(data[col])

 



cat = pd.get_dummies(data[cat])

skewed_feats =  (data[num].skew() > 0.5).index

data[skewed_feats] = data[skewed_feats] + 1

for feat in skewed_feats:

    

    data[feat], lam = boxcox(data[feat])

all_data = pd.concat([cat, data[num]], axis = 1)



train = all_data.iloc[:ntrain]



test = all_data.iloc[ntrain:]









def train_models(estimators, data, label): 

    result = {}

    for estimator in estimators:

        

        score = cross_val_score(estimators[estimator], data, label, scoring = "accuracy").mean()

        result[""+ str(estimator)] = score 

    return pd.Series(result)



estimators = {}

#estimators['Linear Regression'] = linear_model.LinearRegression()

estimators['Random Forest '] = ensemble.RandomForestClassifier()

estimators['LogisticRegression'] = linear_model.LogisticRegression()

estimators['SVM'] = svm.SVC()

x = train_models(estimators, train.values, target.values)

print (x)

x.plot(kind = 'bar')





clf = ensemble.RandomForestClassifier(n_estimators = 100, min_samples_leaf = 10)

params = {'max_features' : ['sqrt', 'log2', 'auto'], 'max_depth': [9, 10, 11]}

gridsearch = GridSearchCV(clf, param_grid = params, cv = 5 )

gridsearch.fit(train.values, target.values)

print (gridsearch.best_params_)



clf = ensemble.RandomForestClassifier(n_estimators = 59, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 11)

clf.fit(train.values, target.values)

preds = clf.predict(test)

sub = pd.DataFrame()

sub['PassengerID'] = ids

sub['Survived'] = preds



sub.to_csv('subm.csv', index = False)







# Any results you write to the current directory are saved as output.