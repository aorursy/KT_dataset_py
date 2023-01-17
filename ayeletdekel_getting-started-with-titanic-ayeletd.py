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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men=train_data.loc[train_data.Sex=='male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# Let's get down to business (To defeat the Huns, Hah!)



from sklearn.model_selection import train_test_split



#remove data with no classification, if any

# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0, subset=['Survived'], inplace=True)

y = train_data.Survived

X_full=train_data.drop(['Survived'], axis=1)



# Name, Tickect and Cabin identify single users and hence are risky and should not be used

X_2=X_full.drop(['Name', 'Ticket', 'Cabin'], axis=1)

X_test2=test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)



# one-hot encoding for the non-numerical feature values

X = pd.get_dummies(X_2)

X_test = pd.get_dummies(X_test2)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state=0)



# fill in missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

imputed_X_test = pd.DataFrame(imputer.transform(X_test))



# Fit a xgb classifier for the data

from xgboost import XGBClassifier

my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

my_model.fit(imputed_X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(imputed_X_valid, y_valid)], 

             verbose=False)



#predict for the cross validation set

predict_cv = my_model.predict(imputed_X_valid)



from sklearn.metrics import accuracy_score

cv_score=accuracy_score(y_valid,predict_cv)

print ("CV Score: ",cv_score)



# That actually doesn't look bad. let's predict for the test set and save

test_predictions = my_model.predict(imputed_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})

output.to_csv('my_submission.csv', index=False)

print("Submission was successfully saved!")
