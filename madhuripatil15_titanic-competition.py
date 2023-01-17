# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
print('Test data shape:',test_data.shape)

print('Train data shape:',train_data.shape)
train_data.info()
cols_with_missing_data= [cols for cols in train_data.columns if train_data[cols].isnull().any()]

percentage_of_missing_data = [((train_data[col].isnull().sum()/ len(train_data[col]))*100).round(2) for col in cols_with_missing_data]

print(cols_with_missing_data,percentage_of_missing_data)
test_data.head()
test_data.info()
# Let's copy the dataset

train = train_data.copy()

test = test_data.copy()
#drop columns with high missing values

# let's also drop the columns which seems to be not required for predictive analysis.[Name,PassengerId,Ticket]

cols_to_drop = ['Name', 'PassengerId', 'Ticket', 'Cabin']

train.drop(cols_to_drop,axis = 1,inplace = True)

test.drop(cols_to_drop,axis = 1,inplace = True)



# lets check the shape of datasets

print('Train shape: ', train.shape)

print('Test shape: ', test.shape)
train.head()
test.head()
train.Embarked.value_counts()
test.Fare.value_counts()
train['Age'] = train['Age'].fillna(0.5)

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts()[0])
test['Age'] = test['Age'].fillna(0.5)

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
gender = pd.get_dummies(train['Sex'],columns = ['Male','Female'],drop_first = True)

gender_test = pd.get_dummies(test['Sex'],columns = ['Male','Female'],drop_first = True)

gender.head(5)


embarked  = pd.get_dummies(train['Embarked'],prefix = 'Embarked',drop_first = True)

embarked_test  = pd.get_dummies(test['Embarked'],prefix = 'Embarked')

embarked.head()

embarked_test.head()
Pclass = pd.get_dummies(train['Pclass'], prefix = 'Pclass', drop_first = True)

Pclass_test = pd.get_dummies(test['Pclass'], prefix = 'Pclass', drop_first = True)
Pclass.head()

Pclass_test.head()
test_d = pd.concat([embarked_test, gender_test,Pclass_test,test], axis = 1)
test_d.head()
df  = pd.concat([embarked, gender, Pclass, train], axis = 1)
df.head()
duplicate_col = ['Sex','Embarked','Pclass']

df.drop(duplicate_col,axis = 1,inplace = True)

test_d.drop(duplicate_col, axis = 1, inplace = True)
df.columns

f = ['Embarked_C', 'Embarked_Q', 'Embarked_S', 'male', 'Pclass_2','Pclass_3', 'Age', 'SibSp', 'Parch', 'Fare']
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize = (12,10))

sns.heatmap(df[f].corr(), annot = True)
X = df.drop(['Survived'], axis = 1)

y = df['Survived']
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error,accuracy_score, mean_absolute_error # accuracy_score is for classification and r2_score for regression



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBRegressor



from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, test_size = 0.3,random_state = 0)
#Cross_val_score

def get_score(model):

    scores = cross_val_score(model, X_train, y_train, cv = 5)

    return scores.mean().round(3), scores.std().round(2)

    
#LogisticRegression

model1 = LogisticRegression(random_state = 0)



model1.fit(X_train, y_train)

predict = model1.predict(X_valid)



mae = mean_squared_error(y_valid,predict).round(3)

acc = accuracy_score(y_valid,predict).round(3)

print('Accuracy using accuracy score:',acc)

print('MAE:',mae)

print('Acurracy and standard deviation :',get_score(model1))



#RandomForestClassifier

model2 = RandomForestClassifier(n_estimators = 100, random_state = 1)



model2.fit(X_train, y_train)



predict = model2.predict(X_valid)



mae = mean_squared_error(y_valid,predict).round(3)

print('MAE:',mae)



acc = (accuracy_score(y_valid,predict) * 100).round(2)

print('Accuracy using accuracy score:',acc)



print('Acurracy and standard deviation :',get_score(model2))

#DecisionTreeClassifier

model3 = DecisionTreeClassifier(random_state = 1)



model3.fit(X_train, y_train)



predict = model3.predict(X_valid)



mae = mean_squared_error(y_valid,predict).round(3)

print('MAE:',mae)



acc = accuracy_score(y_valid,predict).round(3)

print('Accuracy using accuracy score:',acc)



print('Acurracy and standard deviation :',get_score(model3))



from sklearn.naive_bayes import BernoulliNB



model4 = BernoulliNB(alpha = 0.05)

model4.fit(X_train, y_train)





predict = model4.predict(X_valid)



mae = mean_squared_error(y_valid,predict).round(3)

print('MAE:',mae)



acc = accuracy_score(y_valid,predict).round(3)

print('Accuracy using accuracy score:',acc)



print('Acurracy and standard deviation :',get_score(model4))

# GradientBoost Classifier



from sklearn.ensemble import GradientBoostingClassifier



GB_clf = GradientBoostingClassifier(n_estimators = 250, learning_rate = 0.09, random_state = 100)

GB_clf.fit(X_train, y_train)

preds = GB_clf.predict(X_valid)



mae = mean_squared_error(y_valid,preds).round(3)

print('MAE:',mae)



acc = accuracy_score(y_valid,preds).round(3)

print('Accuracy using accuracy score:',acc)



print('Acurracy and standard deviation :',get_score(GB_clf))
from lightgbm import LGBMClassifier



lg_clf = LGBMClassifier(n_estimators=100,

                        num_leaves=100,

                        verbose=-1,

                        random_state=1)





lg_clf.fit(X_train, y_train)

prediction = lg_clf.predict(X_valid)



mae = mean_absolute_error(y_valid, prediction).round(3)

print("Mean Absolute Error:", mae)



acc = accuracy_score(y_valid, prediction) * 100

print('Accuracy: {0:.2f}%'.format(acc))
from sklearn.ensemble import AdaBoostClassifier

ab_clf = AdaBoostClassifier(n_estimators=100,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=1),

                            random_state=1)

ab_clf.fit(X_train, y_train)

prediction1 = ab_clf.predict(X_valid)



mae = mean_absolute_error(y_valid, prediction1).round(3)

print("MAE:", mae)



acc = accuracy_score(y_valid, prediction1) * 100

print('Accuracy: {0:.2f}%'.format(acc))
test_d.shape
predict_test = GB_clf.predict(test_d)

predict_test.shape
output = {'Survived':predict_test}

final_output = pd.DataFrame(output, index = test_data['PassengerId'])

print(final_output.head())

final_output = final_output.to_csv('Prediction.csv')