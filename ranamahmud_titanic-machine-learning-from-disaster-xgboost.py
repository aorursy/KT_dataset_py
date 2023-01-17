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
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
train.shape
train.describe()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
df_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df_sub.head()


# Create arrays for the features and the target: X, y

X, y = train.drop(['PassengerId','Name','Survived'], axis = 1), train[['Survived']]

X.isnull().sum()
# fill missing values with median age

X.Age = X.Age.fillna(X.Age.median())

X.Cabin = X.Cabin.fillna(X.Cabin.mode())

X.Embarked = X.Embarked.fillna(X.Embarked.mode())

X.isnull().sum()
X.dtypes
test.shape
X_test = test.drop(['PassengerId', 'Name'], axis = 1)
X_test.isnull().sum()
X_test.shape
X_test.Cabin = X_test.Cabin.astype('str')

X_test.Fare = X_test.Fare.astype('float')
X_test.isnull().sum()
X_test.Age.fillna(X_test.Age.median(), inplace = True)

X_test.Cabin.fillna(X_test.Cabin.mode(), inplace = True)

X_test.Fare.fillna(X_test.Fare.median(), inplace = True)

X_test.isnull().sum()
X['type'] = 'train'
X_test['type'] = 'test'
df = pd.concat([X, X_test])
# Get list of categorical column names

cat_columns = df.columns[df.dtypes == object].tolist()

print(df[cat_columns].head())

df.type.value_counts()
le = LabelEncoder()
df[['Sex','Ticket','Cabin','Embarked']] = df[['Sex','Ticket','Cabin','Embarked']].apply(lambda x: le.fit_transform(x.astype('str')))
df.head()
X = df[df.type == "train"].drop("type", axis = 1)

X.head()
X_test_orig = df[df.type == "test"].drop("type", axis = 1)

X_test_orig.head()
X_test.shape


# Create the training and test sets

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)



# Instantiate the XGBClassifier: xg_cl

xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=42)



# Fit the classifier to the training set

xg_cl.fit(X_train,y_train)



# Predict the labels of the test set: preds

preds = xg_cl.predict(X_test)



# Compute the accuracy: accuracy

accuracy = float(np.sum(preds==y_test[['Survived']].values.ravel()))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
gbm_param_grid = {

    'num_rounds': [1,5, 10, 15, 20],

    'eta_vals' : [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9],

        'colsample_bytree': [0.3,0.4, 0.5, 0.7, 0.9],

    'n_estimators': [5,10,15,20,25],

    'max_depth': range(2, 20)

}



# Instantiate the classifier: gbm

gbm = xgb.XGBClassifier(objective='binary:logistic', seed=42)



# Perform random search: grid_mse

randomized_accuracy = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator  = gbm, scoring = "accuracy", n_iter = 5, cv = 4, verbose = 1)





# Fit randomized_accuracy to the data

randomized_accuracy.fit(X,y)



# Print the best parameters and best accuracy

print("Best parameters found: ", randomized_accuracy.best_params_)

print("Best Accuracy Score: ", np.sqrt(np.abs(randomized_accuracy.best_score_)))

X_test.shape
y_predicted = randomized_accuracy.best_estimator_.predict(X_test_orig)
sub = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":y_predicted})
sub.to_csv('submission.csv', index = False)

sub.head()