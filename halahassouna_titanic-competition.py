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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.head()
print(train_data.shape)

print(test_data.shape)
from matplotlib import pyplot as plt

import seaborn as sns
sex_pivot = train_data.pivot_table(index='Sex', values="Survived")

sex_pivot.plot.bar()

plt.show()
class_pivot = train_data.pivot_table(index='Pclass', values="Survived")

class_pivot.plot.bar()

plt.show()
train_data['Age'].describe()
survived = train_data[train_data["Survived"]==1]

survived.head()
died = train_data[train_data["Survived"]==0]

died.head()
survived["Age"].plot.hist(alpha=0.75, color='green', bins=50)

died["Age"].plot.hist(alpha=0.25, color='red', bins=50)

plt.legend(['Survived','Died'])

plt.show()
sns.scatterplot(x=train_data['Survived'], y=train_data['Embarked'])
sns.scatterplot(x=train_data['Age'], y=train_data['Embarked'])
sns.scatterplot(x=train_data['Pclass'], y=train_data['Embarked'])
sns.barplot(x=train_data['Age'], y=train_data['Sex'])
train_data = train_data.drop(['PassengerId'], axis = 1)
train_data = train_data.drop(['Name', 'Ticket'], axis = 1)
train_data.head()
train_data.shape
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_data.head()
test_data.shape
train_data.isnull().sum()
train_data = train_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)
train_data = train_data.drop(['Embarked'], axis = 1)

train_data.shape
y = train_data['Survived']

y
X = train_data.drop(['Survived'], axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

X_train.head()
X_valid.head()
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])



label_X_train.head()
X_train.isnull().sum()

X_valid.isnull().sum()
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

imputed_X_train.head()
imputed_X_train.isnull().sum()
from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.metrics import mean_absolute_error #evaluate the erroe





model = LogisticRegression()

model.fit(imputed_X_train,y_train)

prediction_lr=model.predict(imputed_X_valid)

print(accuracy_score(y_valid, prediction_lr))

print(mean_absolute_error(y_valid, prediction_lr))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000)

model.fit(imputed_X_train,y_train)

prediction_rf=model.predict(imputed_X_valid)



print(accuracy_score(y_valid, prediction_rf))

print(mean_absolute_error(y_valid, prediction_rf))
from sklearn.svm import SVC, LinearSVC



model = SVC()

model.fit(imputed_X_train,y_train)

prediction_svm=model.predict(imputed_X_valid)

print(accuracy_score(y_valid, prediction_svm))

print(mean_absolute_error(y_valid, prediction_svm))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model= LinearDiscriminantAnalysis()

model.fit(imputed_X_train,y_train)

prediction_lda=model.predict(imputed_X_valid)

print(accuracy_score(y_valid, prediction_lda))

print(mean_absolute_error(y_valid, prediction_lda))
categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 

                        X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train[my_cols].copy()

X_valid = X_valid[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer()



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer()),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
model = LogisticRegression()

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(imputed_X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(imputed_X_valid)



# Evaluate the model

score = accuracy_score(y_valid, preds)

print('MAE:', score)

print(mean_absolute_error(y_valid, prediction_lda))
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(imputed_X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(imputed_X_valid, y_valid)], 

             verbose=False)
# predictions = my_model.predict(imputed_X_valid)

# print(accuracy_score(predictions, y_valid))
train_data.shape
test_data.head()
# test_data_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# output = pd.DataFrame({'Id': test_data_sub.index, 'Survived': prediction_rf})

# output.to_csv('submission.csv', index=False)

# print('done!')