#Author: Ana Laura García

#The mistake is that the data is not splitted, so the model doesn´t have a parameter that it can compare the predictions with.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

test_data_path = '../input/test.csv'

train_data_path = '../input/train.csv'

test_data = pd.read_csv(test_data_path)

train_data = pd.read_csv(train_data_path)

features =['Pclass','Sex','Age']

train_y = train_data.Survived

train_x = train_data[features]

test_x =test_data[features]

#print(train_x.head())

#print(test_x.head())





#Before the model is built, imputation is needed in order for the data to be read correctly:

#We make a copy so we keep the original files

new_train_x = train_x.copy()

new_test_x = test_x.copy()



#The following solves the problem of missing data for the features but this only handles variables type 

#float or int

col_missing_data = (col for col in new_train_x if new_train_x[col].isnull().any)

for col in col_missing_data:

     new_train_x[col+'was_missing']= new_train_x[col].isnull()

     new_test_x[col+'was_missing']= new_test_x[col].isnull()

#we still need to use hot-encoding for the sex category (before or after??) solvinng the missing data problem

#print (new_test_x.describe())
#T know the number of colums that have string and numerical values:

low_cardinality_cols = [cname for cname in new_train_x.columns if 

                               new_train_x[cname].nunique() < 3 and

                               new_train_x[cname].dtype == "object"]

numeric_cols = [cname for cname in new_train_x.columns if 

                                new_train_x[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = new_train_x[my_cols]

test_predictors = new_test_x[my_cols]

#Lines that hot-encode the columns of object type

hotencoded_train_features = pd.get_dummies(train_predictors)

hotencoded_test_features = pd.get_dummies(test_predictors)

print(hotencoded_test_features.head())

#Starts imputator process:

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

# Uses the method fit_transform, it fits and then fills the missing data using the mean

#When running this code, an exception is thrown, probably because the sex category is a no-value category,

#The problem mencioned above was solved through hot-ecoding the sex category as I thought.

final_train_x= my_imputer.fit_transform(hotencoded_train_features)

final_test_x = my_imputer.transform(hotencoded_test_features)

#Buidling the model

#Now that we have the sex-category hot encoded, and the age and Pclass columns imputed, we use the

#variables final_train_x and final_test_x

from sklearn.tree import DecisionTreeRegressor

train_model = DecisionTreeRegressor(random_state = 0)

##from xgboost import XGBRegressor

##second_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

train_model.fit(final_train_x, train_y)

#Train the model with the train data

##second_model.fit(final_train_x, train_y, verbose=False)

#Finally making the predictions using the test data which was carfully taken care of through hot-encoding it

#and imputating it.

predictions = train_model.predict(final_test_x)

#print(predictions)







#Predictions were given as a float number, Idk if that's an error in my model, I ended up rounding:

rounded_preds =[]

for x in predictions:

    result = round(x,0)

    rounded_preds = np.append(rounded_preds, [result], axis=0)   

print(rounded_preds)

columnPass =test_data['PassengerId']

#Creation of the dataFrame

mysubmission = pd.DataFrame({'PassengerId': columnPass, 'Survived': rounded_preds})

#The following line doesn't run correctly, I wanted to put a title 'Survivors' to the predictions column

mysubmission.to_csv('submission.csv', index=False)

    






