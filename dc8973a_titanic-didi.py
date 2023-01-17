# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

main_file_path = '../input/train.csv'
data_train_raw = pd.read_csv(main_file_path)
data_train_raw.sample(10)
data_train_raw.describe()
#check number of null value out of 891 records for each variable
data_train_raw.isnull().sum()
#data vitualization: Pclass, Sex
import seaborn as sns
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train_raw,
              palette={"male": "blue", "female": "orange"},
              markers=["*", "o"], linestyles=["-", "--"]);
#sex is a highly correlated with survived based on the plot below. Only male in PClass 1 has significant higher survival rate than male in 2 or 3. For female, only PClass 3 has significant low survival rate.
data_train_raw.plot.scatter(x='Fare', y='Survived')
data_train_raw['Fare'].corr(data_train_raw['Pclass'])
#It doesn't make any common sense that Pclass is negatively correlated with Fare.So I will drop Fare in this case.
#data visualization: embarked
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train_raw)
#passengers boarded from C tend to have higher survival rate than ones boarded from S or Q, but the confidence levels are overlapping. So no significant difference.
#data visualization: Parch
sns.barplot(x="Parch", y="Survived",hue='Sex', data=data_train_raw)
#no significant trend/patterns
#data visualization: SibSp
sns.barplot(x="SibSp", y="Survived",hue='Sex', data=data_train_raw)
#no significant trend/patterns
#variables to drop: Cabin (high null count), Ticket (no visible pattern), Name(no visible pattern), Fare(no visible pattern; negative relationship to Pclass seems odd), Embarked(no visible pattern), Parch, SibSp
variables_dropped = ['Cabin','Ticket','Name','Fare','Embarked','Parch', 'SibSp']
data_train = data_train_raw.drop(variables_dropped, axis = 1)
#variables to fix null values: Age
#simplify age and fill null first before visulization
def simplify_ages(df):
    df.Age = df.Age.fillna(30)
    bins = (0, 2, 10, 18, 50, 120)
    group_names = ['Baby', 'Child', 'Teen', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
data_train = simplify_ages(data_train)
data_train.sample(10)

#data visulization: Age
sns.barplot(x="Age", y="Survived",hue="Sex", data=data_train);
sns.barplot(x="Age", y="Survived", data=data_train);
#Babies mostly survied regardless of gender
#load and clean test data
data_test_raw=pd.read_csv('../input/test.csv')
data_test = data_test_raw.drop(variables_dropped, axis = 1)
data_test = simplify_ages(data_test)
data_test.head()
data_train.head()
#instead of normalizing using labelencoder(commented below), use one-hot encoding
# from sklearn import preprocessing
# def encode_features(df_train, df_test):
#     features = ['Age', 'Sex']
#     df_combined = pd.concat([df_train[features], df_test[features]])
    
#     for feature in features:
#         le = preprocessing.LabelEncoder()
#         le = le.fit(df_combined[feature])
#         df_train[feature] = le.transform(df_train[feature])
#         df_test[feature] = le.transform(df_test[feature])
#     return df_train, df_test
# data_train, data_test = encode_features(data_train, data_test)

predictors = ['Pclass','Age', 'Sex']
train_predictors = data_train[predictors]
test_predictors = data_test[predictors]

encoded_data_train =pd.get_dummies(data_train)
encoded_train_predictors = pd.get_dummies(train_predictors)

encoded_train_predictors.head()
from sklearn.model_selection import train_test_split

X = encoded_train_predictors
y = data_train['Survived']

# for model validation, split data_train into training and validation data, for both predictors and target
train_X, val_X, train_y, val_y = train_test_split(X.as_matrix(), y.as_matrix(),random_state = 0)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


#get MAE between actual survived and predicted survived from train data
# forest_model = RandomForestClassifier()
# forest_model.fit(train_X, train_y)
# survived_preds = forest_model.predict(val_X)
# print(accuracy_score(val_y, survived_preds))
#75%+ accurate score is not bad!

from xgboost import XGBRegressor
xgboost_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgboost_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(val_X, val_y)], verbose=False)
predictions = xgboost_model.predict(val_X)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))
encoded_data_test = pd.get_dummies(data_test)
encoded_data_test.head()
ids = encoded_data_test['PassengerId']
test_predictors = encoded_data_test.drop('PassengerId', axis=1)
test_predictors.head()
predictions = xgboost_model.predict(test_predictors.as_matrix())
print(predictions)
output2 = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
print(output2)
output2.to_csv('submission3.csv', index=False)