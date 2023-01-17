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
train_dir = "/kaggle/input/titanic/train.csv"

test_dir = "/kaggle/input/titanic/test.csv"
train_data = pd.read_csv(train_dir)

test_data = pd.read_csv(test_dir)
train_data.head()
test_data.head()
w_sur = train_data.loc[(train_data.Sex == "female")]["Survived"]

sum(w_sur)/len(w_sur)
m_sur = train_data.loc[(train_data.Sex == "male")]["Survived"]

sum(m_sur)/len(m_sur)
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train_data.info()
train_data.describe()
age_mean = train_data["Age"].mean()

age_std = train_data["Age"].std()
train_dir = "/kaggle/input/titanic/train.csv"

test_dir = "/kaggle/input/titanic/test.csv"
data = [train_data, test_data]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'

    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

    dataset["Fare_pp"] = dataset["Fare"] / (dataset["Parch"] + dataset["SibSp"] + 1)

    

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    #dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna("NA")

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
train_data.head()
num_features = ["Fare_pp", "Age_Class", "relatives"]

cat_features = ["Pclass","Sex", "travelled_alone", "Embarked", "Title"]
age_mean = train_data["Age"].mean()

age_std = train_data["Age"].std()
num_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="constant", fill_value=np.random.randint(age_mean - age_std, age_mean+age_std))),

    ("scaler", StandardScaler())



])



cat_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("encoder", OneHotEncoder())

])

   

full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_features),

    ("cat", cat_pipeline, cat_features),

])
y = train_data["Survived"]
train_data = train_data.drop(['Survived'], axis=1)

X = full_pipeline.fit_transform(train_data)
train_data.head()
X[0]
# for f in num_features + cat_features+ rem_features :

#     sns.factorplot(x=y, y=train_data[f])

#     plt.legend()

#     plt.show()
tx, vx, ty, vy = train_test_split(X, y, test_size=0.1, random_state=42)
svc = SVC()
from sklearn.model_selection import GridSearchCV

from scipy.stats import expon, reciprocal



param_distribs = {

        'C': [1, 5, 10], 'kernel': ('linear', 'rbf'),

        'gamma': ["scale", "auto"] 

    }

grid_search = GridSearchCV(svc, param_distribs,

                                cv=5, scoring='neg_mean_squared_error',

                                verbose=2)



grid_search.fit(X, y)
grid_search.best_params_

best_svc = grid_search.best_estimator_

best_svc.fit(tx, ty)
best_svc.score(vx, vy)

acc_svc = round(best_svc.score(vx, vy) * 100, 2)

acc_svc
# from sklearn.model_selection import RandomizedSearchCV

# from scipy.stats import expon, reciprocal



# param_distribs = {

#         'kernel': ['linear', 'rbf'],

#         'C': reciprocal(20, 200000),

#         'gamma': expon(scale=1.0),

#     }



# rnd_search = RandomizedSearchCV(svc, param_distributions=param_distribs,

#                                 cv=5, scoring='neg_mean_squared_error',

#                                 verbose=2, random_state=42)



# rnd_search.fit(X, y)
svc.fit(tx,ty)
svc.score(vx, vy)

acc_svc2 = round(svc.score(vx, vy) * 100, 2)

acc_svc2
X_test = full_pipeline.transform(test_data)

predictions_1 = best_svc.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_1})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")