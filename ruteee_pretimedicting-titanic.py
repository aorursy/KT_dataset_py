# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import BaggingClassifier



from sklearn.decomposition import PCA





from sklearn.pipeline import Pipeline



import seaborn as sns

import matplotlib.pyplot as plt



import missingno as msno
path ="/kaggle/input/titanic/"

gender_sub = pd.read_csv(path + "gender_submission.csv", index_col=0)

train = pd.read_csv(path + "train.csv", index_col=0)

#plot of missing values

msno.matrix(train)
#counting null values

train.isnull().sum()
#Drop of columns Cabin

#Since we have 76% of the Cabin data missing, we will drop this column.

train.drop(columns=['Cabin'], axis=1, inplace=True)
# Analysing age by gender

train.groupby([train['Sex']]).mean()
age_fe = train['Age'][train['Sex'] == 'female'].mean()

age_ma = train['Age'][train['Sex'] == 'male'].mean()


ag_f = lambda a : age_fe if a == 'female' else age_ma



idx = train['Age'][train['Age'].notna() == False].index

train[['Age','Sex']].loc[idx]
#Filling age according to gender

train['Age'] = train['Age'].fillna(pd.Series(map(ag_f, train['Sex'].loc[idx]), index=idx))
train[['Age','Sex']].loc[idx]
train
#Treating embarked with most frequently strategy

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

#Because Sex, Embarked and Pclass are categorical features that can be used as numeri

train_processed = pd.get_dummies(train, columns = ['Sex', 'Embarked', 'Pclass',])

train_processed_sel = train_processed.drop(columns=['Name', 'Ticket', 'Fare','Parch','Pclass_1','Embarked_C','Sex_male',])



train_processed_sel.isnull().sum()

train_processed_sel
plt.figure(figsize=(12,10))

sns.set()

sns.heatmap(train_processed_sel.corr(), cmap='BrBG', annot=True)
#With column age





X_train, X_test, y_train, y_test = train_test_split(train_processed_sel.drop(columns="Survived"), train_processed_sel['Survived'],

                                                    test_size=0.3,

                                                    random_state=123,

                                                    shuffle=True,

                                                    stratify=train_processed_sel["Survived"])
model = XGBClassifier()



pipe = Pipeline(steps = [("scale", MinMaxScaler()),

                         ('fs', SelectKBest()),

                         ("clf",BaggingClassifier())])

                

search_space = [{"clf": [BaggingClassifier()],

                "clf__base_estimator": [XGBClassifier()],

                 "clf__base_estimator__objective": ['binary:logistic'],

                 "clf__base_estimator__n_estimators": [600],

                 "clf__base_estimator__max_depth": [16],

                 "clf__base_estimator__learning_rate": [0.008,0.005,0.01],

                 "clf__base_estimator__random_state": [123],

                 "clf__base_estimator__subsample": [0.3,0.51],

                 "clf__base_estimator__colsample_bytree": [0.3,0.51,],

                 "clf__base_estimator__base_score": [0.2,0.8],

                 "clf__base_estimator__reg_alpha": [1,0],

                 "clf__base_estimator__reg_lambda": [0],

                 "fs__score_func":[chi2],

 

                 "fs__k":['all']},

                ]



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}



grid = GridSearchCV(estimator=pipe, 

                    param_grid=search_space,

                    cv=10,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy",

                    verbose=1)



#model without age

grid.fit(X_train, y_train)



print("score ", grid.best_score_, "Best Estimator", grid.best_estimator_)
result = pd.DataFrame(grid.cv_results_)

# result_b = pd.DataFrame(grid_b.cv_results_)

result[['rank_test_Accuracy','mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy']].sort_values(by='rank_test_Accuracy')
#with age

pred = grid.predict(X_test)

print(accuracy_score(y_test, pred))





print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
test = pd.read_csv(path + "test.csv", index_col=0)
test.isnull().sum()
test.drop(columns=['Cabin'], axis=1, inplace=True)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())



test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
age_fe_t = test['Age'][test['Sex'] == 'female'].mean()

age_ma_t = test['Age'][test['Sex'] == 'male'].mean()



ag_f_t = lambda i : age_fe_t if i == 'female' else age_ma_t



idx_t = test['Age'][test['Age'].notna() == False].index



test['Age'] = test['Age'].fillna(pd.Series(map(ag_f_t, test['Sex'].loc[idx_t]), index=idx_t))



test.isnull().sum()
test_processed = pd.get_dummies(test, columns = ['Sex', 'Embarked', 'Pclass',])

test_processed_sel = test_processed.drop(columns=['Name', 'Ticket', 'Fare','Parch','Pclass_1','Embarked_C','Sex_male', ])

test_processed_sel.isnull().sum()
index = test_processed_sel.index

# index.values


pred= grid.best_estimator_.predict(test_processed_sel)



sub = pd.DataFrame(pred, columns=['Survived'])

sub['PassengerId'] = index.values

sub.set_index(['PassengerId'], inplace=True)

sub



sub.to_csv("sub.csv")
sub
!pwd
sub