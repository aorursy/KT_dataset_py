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

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



y = df_train['Survived']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']



X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])



X.fillna(X.mean())

X_test.fillna(X_test.mean())
from xgboost import XGBClassifier



model = XGBClassifier(learning_rate = 0.05,

                     n_estimators=300,

                     max_depth = 4)

model.fit(X, y)
predictions = model.predict(X_test)

print(predictions)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X)

imputed_X_test = my_imputer.transform(X_test)
predictions = model.predict(X_test)

print(predictions)

import matplotlib.pyplot as plt

from xgboost import plot_importance



plot_importance(model)

plt.show()
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
import pandas as pd

import numpy as np

import hyperopt

!pip install catboost==0.23.2

from catboost import Pool, CatBoostClassifier, cv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.fillna(-999, inplace=True)

test.fillna(-999, inplace=True)
y = train.Survived

X = train.drop(['Survived'], axis =1)



X_test = test





cate_features_index = np.where(X.dtypes != float)[0]

cate_features_index
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y, train_size=0.20, random_state=15)
model = CatBoostClassifier(iterations=2000, learning_rate=0.01, l2_leaf_reg=4, depth=6, rsm=1, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)
model.fit(Xtrain,ytrain, cat_features=cate_features_index, eval_set=(Xtest,ytest))

print(model.get_best_score())
pred = model.predict(X_test)

print(pred)
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': pred})

output.to_csv('cat_boost_submission.csv', index=False)

print("Your submission was successfully saved!")
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
df_train.drop(['Ticket','PassengerId'], axis=1, inplace=True)



gender_mapper = {'male': 0, 'female': 1}

df_train['Sex'].replace(gender_mapper, inplace=True)



df_train['Title'] = df_train['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])

df_train['Title'] = [0 if x in ['Mr.', 'Miss.', 'Mrs.'] else 1 for x in df_train['Title']]

df_train = df_train.rename(columns={'Title': 'Title_Unusual'})

df_train.drop('Name', axis = 1, inplace=True)



df_train['Cabin_Known'] = [0 if str(x) == 'nan' else 1 for x in df_train['Cabin']]

df_train.drop('Cabin', axis=1, inplace=True)



emb_dummies = pd.get_dummies(df_train['Embarked'], drop_first=True, prefix='Embarked')

df_train = pd.concat([df_train, emb_dummies], axis = 1)

df_train.drop('Embarked', axis=1, inplace=True)



df_train['Age'] = df_train['Age'].fillna(int(df_train['Age'].mean()))

                     
df_train.head()
X = df_train.drop('Survived', axis=1)  #features or predictors excluding the target value

y = df_train['Survived']   #target value aka the one we are trying to predict



X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.8)
ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)

X_test_scaled = ss.transform(X_test)
from tpot import TPOTClassifier
tpot = TPOTClassifier(verbosity=2, max_time_mins=10)

tpot.fit(X_train_scaled, y_train)
tpot.fitted_pipeline_
tpot.score(X_test_scaled, y_test)