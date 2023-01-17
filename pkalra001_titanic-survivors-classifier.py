# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import ensemble, svm, linear_model, tree

from sklearn import model_selection

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df3 = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df2 = pd.read_csv('/kaggle/input/titanic/test.csv')

df1 = pd.read_csv('/kaggle/input/titanic/train.csv')

train = df1.copy(deep = True)

test = df2.copy(deep = True)

data_chng = [train, test]

for idx, data in enumerate(data_chng):

    data.drop(['Cabin', 'PassengerId', 'Ticket'], axis = 1,inplace = True)

    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    title_names = data['Title'].value_counts() < 10

    data['Title'] = data['Title'].apply(lambda x : 'Misc' if title_names.loc[x] == True else x)

    data['Age'].fillna(data['Age'].median(), inplace = True)

    data['Fare'].fillna(data['Fare'].median(), inplace = True)

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    data['familySize'] = data['SibSp'] + data['Parch'] + 1

    data['Alone'] = 1

    data['Alone'].loc[data['familySize'] > 1] = 0

    data['FareBin'] = pd.qcut(data['Fare'], 4)

    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

    data['Sex_en'] = LabelEncoder().fit_transform(data['Sex'])

    data['Embarked_en'] = LabelEncoder().fit_transform(data['Embarked'])

    data['FareBin_en'] = LabelEncoder().fit_transform(data['FareBin'])

    data['AgeBin_en'] = LabelEncoder().fit_transform(data['AgeBin'])

    data['Title_en'] = LabelEncoder().fit_transform(data['Title'])

col_en = ['Pclass', 'Embarked_en', 'FareBin_en', 'AgeBin_en', 'Sex_en', 'SibSp', 'Parch', 'familySize', 'Alone']

MLA = [

    ensemble.GradientBoostingClassifier(), 

    ensemble.RandomForestClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    svm.SVC(),

    tree.DecisionTreeClassifier(),

    XGBClassifier()

]

#Algorith comparison

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.3, train_size = 0.6, random_state = 0)

MLA_col = ['MLA_name', 'MLA train accuracy mean', 'MLA test accuracy mean', 'MLA time']

MLA_compare = pd.DataFrame(columns = MLA_col)

row = 0

for alg in MLA:

    MLA_compare.loc[row, 'MLA_name'] = alg.__class__.__name__

    cross_val = model_selection.cross_validate(alg,train[col_en], train['Survived'], cv = cv_split, return_train_score = True)

    MLA_compare.loc[row, 'MLA train accuracy mean'] = cross_val['train_score'].mean()

    MLA_compare.loc[row, 'MLA test accuracy mean'] = cross_val['test_score'].mean()

    MLA_compare.loc[row, 'MLA time'] = cross_val['fit_time'].mean()

    row+=1

MLA_compare.sort_values(by = ['MLA test accuracy mean'], ascending = False, inplace = True)

print(MLA_compare)

'''

#finding best parameters using grid search

grid_param = [{'learning_rate' : [0.01,0.03,0.05,0.1,0.25], 'max_depth' : [2,3,4,6,8,10,None], 'n_estimators' : [50, 100, 300], 'random_state' : [0]}]

tuned_model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid = grid_param, scoring = 'roc_auc', cv = cv_split)

tuned_gbc = tuned_model.fit(train[col_en], train['Survived']) '''

gbc = ensemble.GradientBoostingClassifier(learning_rate = 0.05, max_depth = 2, n_estimators = 300, random_state = 0).fit(train[col_en], train['Survived'])

df2['Survived'] = gbc.predict(test[col_en])

submit = df2[['PassengerId', 'Survived']]

submit.to_csv('../working/submit05.csv', index = False)
