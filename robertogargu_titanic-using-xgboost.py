#import all necessary libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import data sets
train_data = pd.read_csv("../input/train.csv")
val_data = pd.read_csv("../input/test.csv")

#check data for missing values 
print(train_data.isnull().sum())
print(val_data.isnull().sum())
#create a data set containing all data for completing and cleaning
all_data = [train_data,val_data]

#fill in missing values
for feature in all_data:    
    feature['Age'].fillna(feature['Age'].median(), inplace = True)
    feature['Embarked'].fillna(feature['Embarked'].mode()[0], inplace = True)
    feature['Fare'].fillna(feature['Fare'].median(), inplace = True)
    
Target = train_data.Survived

#drop columns from train data set; 'Ticket' and 'PassangerId' will not be used, 'Cabin' has too many missing values    
drop_columns_train = ['Name','PassengerId','Cabin', 'Ticket','Survived']
train_data.drop(drop_columns_train, axis=1, inplace = True)

Test_Ids = val_data.PassengerId

drop_columns_test = ['Name','PassengerId','Cabin', 'Ticket']
val_data.drop(drop_columns_test, axis=1, inplace = True)
one_hot_train = pd.get_dummies(data=train_data)
one_hot_test = pd.get_dummies(data=val_data)

print(one_hot_train.isnull().sum())
print(one_hot_test.isnull().sum())
from xgboost import XGBClassifier

X_train = one_hot_train
y_train = Target

xgb_model = XGBClassifier()
xgb_model.fit(X_train,y_train,verbose=False)
predictions = xgb_model.predict(one_hot_test)
my_submission = pd.DataFrame({'PassengerId': Test_Ids, 'Survived': predictions})

my_submission.to_csv('submission.csv', index=False)