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
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
train.Parch.value_counts()
train['Sex'] = train['Sex'].apply(lambda x : 1 if x=='male' else 0)

test['Sex'] = test['Sex'].apply(lambda x : 1 if x=='male' else 0)
train['Embarked'] = train['Embarked'].map({'S':1, 'C':2, 'Q':3})

test['Embarked'] = test['Embarked'].map({'S':1, 'C':2, 'Q':3})
my_imputer = SimpleImputer()



features = ['Pclass','Sex','Age','Parch', 'Fare', 'Embarked']



imputed_train = pd.DataFrame(my_imputer.fit_transform(train[features]))

imputed_test = pd.DataFrame(my_imputer.fit_transform(test[features]))



y = train.Survived



X = imputed_train.copy()

X_test = imputed_test.copy()
train_X, val_X, train_y, val_y = train_test_split(imputed_train, y, train_size=0.8, test_size=0.2, random_state=0)
#Define the models



model_1 = RandomForestRegressor(n_estimators = 50, random_state = 0)

model_2 = RandomForestRegressor(n_estimators = 100,criterion = 'mae', random_state = 0)

model_3 = RandomForestRegressor(n_estimators = 100, min_samples_split = 20, random_state = 0)

model_4 = RandomForestRegressor(n_estimators = 200, min_samples_split = 20, random_state = 0)

model_5 = RandomForestRegressor(n_estimators = 100, max_depth = 7, random_state = 0)
#Function comparing different models

def score_model(model, X_train = train_X, X_val = val_X, y_train = train_y, y_val = val_y):

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return(mean_absolute_error(y_val, predictions))



models = [model_1,model_2,model_3,model_4,model_5]



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("model %d MAE : %d"%(i+1, mae))
# I don't know why the error in each of the above cases is 0, anyways, so we can use any of the aforementioned models

# because the error in each case is 0.



best_model = model_2
# Define a model



my_model = best_model
# Fit the model to the training data

my_model.fit(imputed_train, y)



# Generate predictions on test data

predict = my_model.predict(imputed_test)



submission['Survived'] = predict
submission.to_csv('submission.csv', index = False)