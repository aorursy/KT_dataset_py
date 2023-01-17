# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the Data

X_full = pd.read_csv('/kaggle/input/titanic/train.csv')

X_test_full = pd.read_csv ('/kaggle/input/titanic/test.csv')



X_full.info()
X_full.isnull().sum()
# Separate target from predictors

y = X_full.Survived



features = ["Pclass", "Sex", "SibSp", "Parch",'Embarked','Fare','Age']

X = pd.get_dummies(X_full[features])

X_test = pd.get_dummies(X_test_full[features])



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
X_train.head()
sns.heatmap(X_train.isnull() ,yticklabels=False ,cbar=False , cmap='viridis')
# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.impute import SimpleImputer



# Fill in the lines below: imputation

my_imputer = SimpleImputer() # Your code here

final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Fill in the lines below: imputation removed column names; put them back

final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns
final_X_train.isnull().sum()
final_X_valid.isnull().sum()
#def cabin(cabin_n,letter):

 # if  (str(cabin_n).find(letter) != -1):

  #  return 1

  #else :

   # return 0
#train_data['cabin_a'] = train_data['Cabin'].apply(lambda x: cabin(x,'A'))

#train_data['cabin_b'] = train_data['Cabin'].apply(lambda x: cabin(x,'B'))

#train_data['cabin_c'] = train_data['Cabin'].apply(lambda x: cabin(x,'C'))
#cabin_a = train_data.loc[train_data.cabin_a == 1 ]

#cabin_a_s = train_data.loc[train_data.cabin_a == 1 ]["Survived"]

#rate_cabin_a = sum(cabin_a_s)/len(cabin_a)



#print("% of cabin_a who survived:", rate_cabin_a)
sns.barplot(x='Sex',y='Survived',hue='Pclass' , data=train_data )
X_test.isnull().sum()
X_test.shape
final_X_test = pd.DataFrame(my_imputer.transform(X_test))



# Fill in the lines below: imputation removed column names; put them back

final_X_test.columns = X_test.columns
#test_data['Age'].fillna(test_data['Age'].median(), inplace = True)

#test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)

#test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)

#test_data['Cabin'].fillna(0, inplace = True)
final_X_test.isnull().sum()
final_X_test.shape
final_X_test.head()
# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=10, random_state=0)

model_5 = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0)

model_6 = DecisionTreeRegressor(random_state=0) 

model_7 = RandomForestClassifier(random_state=0)

model_8 = LinearRegression()

model_9 = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

model_10 = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5, model_6,model_7, model_8, model_9, model_10]
from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    accuracy = model.score(X_v,y_v)

    return accuracy



for i in range(0, len(models)):

    accuracy = score_model(models[i],final_X_train,final_X_valid,y_train, y_valid)

    print("Model %d score: {}".format(accuracy))
model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=0)

model.fit(final_X_train, y_train)

predictions = model.predict(final_X_test)



output = pd.DataFrame({'PassengerId': X_test_full.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_006.csv', index=False)

print("Your submission was successfully saved!")