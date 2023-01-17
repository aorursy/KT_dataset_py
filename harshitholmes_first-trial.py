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
#The Trained File is load
Load = pd.read_csv('../input/train.csv')

print(Load.describe())
print(Load.head(), sep = '\n')
#The Prediction from the reports
Refer = pd.read_csv('../input/test.csv').dropna()

print(Refer.describe())
print(Refer.head(), sep = '\n')

#This carifying the features on the survived once
Gender_sub = pd.read_csv('../input/gender_submission.csv')

print(Gender_sub.describe())
print(Gender_sub.head(), sep = '\n')
#Here I am combining the train and test file to predict
Full_Data = [Load, Refer, Gender_sub]
print([Load, Refer, Gender_sub])
#First Phase
#Now onward the data will never check the Sex, PassengerId & Passenger_Name.
#It will pass the data which are exactly same at the test side.
for Each_Passenger in Full_Data :
    
    #Converterd to the integer dtype
    #Here the Gender taken and it reverify by the data
    #This steps is for inter check with train and test file
    Each_Passenger['passenger'] = Each_Passenger.Sex.str.extract('([A-Za-z]+)\. ', expand = False).apply(lambda x : x == Refer['PassengerId'] if type(x) == str else 1)
    Each_Passenger['passenger'] = pd.factorize(Each_Passenger['passenger'])[0]

    #And here it combine and coverted to one by overwrite
    #The person are unique and Sex can be same but not all of them are same
    Each_Passenger['passenger'] = Each_Passenger.Name.str.extract('([A-Za-z]+)\. ', expand = False)
    Each_Passenger['passenger'] = pd.factorize(Each_Passenger['passenger'])[0]
    
    train = Load.drop([ 'Ticket', 'Cabin'], axis = 1)
    test = Refer.drop([ 'Ticket', 'Cabin'], axis = 1)
    collect = [train, test]
    
    print(Each_Passenger['passenger'])
#Second Phase
#Cleaning data and Gather match data 
#Mean Data Preprocessing
#This will gather a big catelog for the features
#The connection in the Fare with Age because Fare Age defur but
#In the category age and fare will have some similarity
for Each_Passenger in Full_Data :
    
    Each_Passenger.loc[(Each_Passenger['Age'] <= 0) & (Each_Passenger['Fare'] <= 0) , 'Age']  =  0
    Each_Passenger.loc[(Each_Passenger['Age'] > 0) & (Each_Passenger['Age'] <= 6 ) | (Each_Passenger['Fare'] > 0) & (Each_Passenger['Fare'] <= 52) , 'Age'] = 1
    Each_Passenger.loc[(Each_Passenger['Age'] > 6) & (Each_Passenger['Age'] <= 13 ) | (Each_Passenger['Fare'] > 52) & (Each_Passenger['Fare'] <= 126) , 'Age'] = 2
    Each_Passenger.loc[(Each_Passenger['Age'] > 13) & (Each_Passenger['Age'] <= 22 ) | (Each_Passenger['Fare'] > 126) & (Each_Passenger['Fare'] <= 187) , 'Age'] = 3
    Each_Passenger.loc[(Each_Passenger['Age'] > 22) & (Each_Passenger['Age'] <= 50 ) | (Each_Passenger['Fare'] > 187) & (Each_Passenger['Fare'] <= 279) , 'Age'] = 4
    Each_Passenger.loc[(Each_Passenger['Age'] > 80) & (Each_Passenger['Age'] <= 80 ) | (Each_Passenger['Fare'] > 279) & (Each_Passenger['Fare'] <= 513) , 'Age'] = 5
    Each_Passenger.loc[(Each_Passenger['Age'] > 80) & (Each_Passenger['Fare'] > 513), 'Age']; 
    
    print(Each_Passenger.describe())
    print(Each_Passenger['Age'].tail(20))
#Third Phase
#The children with sibling are gather on there values with parents on the base of catalog
#The two features are created from three data list
for Each_Passenger in Full_Data :
    
    Each_Passenger.loc[(Each_Passenger['Parch'] <= 0) | (Each_Passenger['Parch'] <= 0) , 'Parch']  =  0
    Each_Passenger.loc[(Each_Passenger['Parch'] > 0) & (Each_Passenger['Parch'] <= 1 ) | (Each_Passenger['SibSp'] > 0) & (Each_Passenger['SibSp'] <= 1) , 'Parch'] = 1
    Each_Passenger.loc[(Each_Passenger['Parch'] > 1) & (Each_Passenger['Parch'] <= 2 ) | (Each_Passenger['SibSp'] > 1) & (Each_Passenger['SibSp'] <= 2) , 'Parch'] = 2
    Each_Passenger.loc[(Each_Passenger['Parch'] > 2) & (Each_Passenger['Parch'] <= 3 ) | (Each_Passenger['SibSp'] > 2) & (Each_Passenger['SibSp'] <= 3) , 'Parch'] = 3
    Each_Passenger.loc[(Each_Passenger['Parch'] > 3) & (Each_Passenger['Parch'] <= 4 ) | (Each_Passenger['SibSp'] > 3) & (Each_Passenger['SibSp'] <= 4) , 'Parch'] = 4
    Each_Passenger.loc[(Each_Passenger['Parch'] > 4) & (Each_Passenger['Parch'] > 4), 'Parch'];
    
    Each_Passenger.loc[(Each_Passenger['Pclass'] <= 0) & (Each_Passenger['Parch'] <= 0) , 'Pclass']  =  0
    Each_Passenger.loc[(Each_Passenger['Pclass'] > 0) & (Each_Passenger['Pclass'] <= 1 ) | (Each_Passenger['Parch'] > 0) & (Each_Passenger['Parch'] <= 1) , 'Pclass'] = 1
    Each_Passenger.loc[(Each_Passenger['Pclass'] > 1) & (Each_Passenger['Pclass'] <= 2 ) | (Each_Passenger['Parch'] > 1) & (Each_Passenger['Parch'] <= 2) , 'Pclass'] = 2
    Each_Passenger.loc[(Each_Passenger['Pclass'] > 2) & (Each_Passenger['Pclass'] <= 3 ) | (Each_Passenger['Parch'] > 2) & (Each_Passenger['Parch'] <= 3) , 'Pclass'] = 3
    Each_Passenger.loc[(Each_Passenger['Pclass'] > 4) & (Each_Passenger['Parch'] > 4), 'Pclass'];
    
    print(Each_Passenger[['Parch', 'Pclass']])
def Uni(Each_Passenger) :
    
    return Each_Passenger
tr = Uni(train) 
tr[['Age', 'Parch', 'Pclass']] = tr[['Age', 'Parch', 'Pclass']]
tr['Name'] = tr['Name'].apply(lambda x : x == Name if type(x) == float else 1)
tr['Embarked'] = tr['Embarked'].map({"S" : 1, "C" : 2, "Q" : 3}).astype(float)

X = tr['passenger'].fillna(0)
y = tr[['Embarked', 'Age',  'Parch', 'Pclass', 'Name',  'Survived']].fillna(0)

#Select  model
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as aus
from sklearn.model_selection import cross_val_score as cs
from sklearn.metrics import explained_variance_score as ax

#To avoid unrelvent errors
import warnings
warnings.filterwarnings('ignore')

#Split the your data as trainning and test sets
train_X, test_X, train_y, test_y = tts(y, X, train_size = 0.33, test_size = 0.33, random_state = 42)

print(len(train_X))
print(len(train_y))
print(len(test_X))
print(len(test_y))
#Classifying the splited data and check accuracy
model = dtr()
model.fit(train_X, train_y)

a = model.score(test_X, test_y)
print('Score with model', a)
z = cs(model, test_X, test_y)

print('This is error in list', z)
#Predict your data
prediction = model.predict(test_X)

ans = aus(test_y, prediction)

Final_score1 = round(model.score(train_X, train_y) * 100, 6)

print('Error', ans)
print('In Percentange : ',Final_score1)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer as im 

my_imputer = im()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

print(train_X)
print(train_y)

def get_mae(X, y):
    
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = prediction

mae_without_categoricals = get_mae(predictors_without_categoricals, test_y)


print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
from xgboost import XGBRegressor as xgr

nxt_model_checker = xgr(n_estimators = 296, learning_rate = 0.08)

nxt_model_checker.fit(train_X, train_y, verbose=False)
prediction_next_one = nxt_model_checker.predict(test_X)

Final_score2 = round(nxt_model_checker.score(train_X, train_y) * 100, 6)

print("Mean Absolute Error : " + str(aus(prediction_next_one, test_y)))
print('In Percentange : ',Final_score2)
print('Average Percentage in the model is :', (Final_score2 + Final_score1) / 2)