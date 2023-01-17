# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
from sklearn.ensemble import RandomForestClassifier #random forest classifier
from sklearn import preprocessing #preprocessing
import os #I/O help
from sklearn.impute import SimpleImputer #Imputer helps fill missing values in dataset


# Any results you write to the current directory are saved as output.
#Read in the training data
df_train = pd.read_csv("../input/train.csv")
#Read in the test data
df_test = pd.read_csv("../input/test.csv")

#Create a model to predict survival

#Function creates the variables dataframe for prediction and testing	
def create_vars_df(complete_df):
	#Preprocess data
	#(i.e. sklearn decision trees can not automatically process categorical data. must convert to an integer)
	le = preprocessing.LabelEncoder()
	le.fit(["male", "female"])
	transformed_sex_column = le.transform(complete_df['Sex'])
	
	#Build new dictionary with passenger sex data
	new_dict = {'Sex' : pd.Series(transformed_sex_column)}
	
	#Choose other non-categorical variables to base the prediction on
	prediction_vars = ['Age', 'Pclass']
	
	#For each of the non-categorical vairbales, add their column to the input variables dictionary
	for var in prediction_vars:
		new_dict.update({var: pd.Series(complete_df[var])})
		
	#Build a pandas df with the input variables dict
	vars_df = pd.DataFrame(new_dict)
	
	#For null values, insert the mean for that column (making that data point essentially a non-value)
	myImputer  = SimpleImputer()
	vars_df = myImputer.fit_transform(vars_df)
	#vars_df.fillna(vars_df.mean(), inplace=True)
	
	return vars_df

#Build the training input and ground truth df's
x_train = create_vars_df(df_train)
y_train = df_train['Survived']
#Build the test input df
x_test = create_vars_df(df_test)


#Make a random forest model, fit the model to the training data
model = RandomForestClassifier()
model.fit(x_train, y_train)
#Use random forest model to build labeled predicitons for test data
y_pred = model.predict(x_test)

#Build the output array of id:prediciton
pred_array = np.empty([df_test.shape[0], 2], dtype=int)
pred_array = [df_test['PassengerId'], y_pred]

#Build the prediction df from the array
prediction = pd.DataFrame({'PassengerId' : pd.Series(df_test['PassengerId']), 'Survived' : pd.Series(y_pred)})

#Write the prediction to file for submission
prediction.to_csv('submission.csv', index=False)
