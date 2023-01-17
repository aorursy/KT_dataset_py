#Import necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Import the dataset
train = pd.read_csv("../input/dsn-ai-oau-july-challenge/train.csv")
test = pd.read_csv('../input/dsn-ai-oau-july-challenge/test.csv')
sub = pd.read_csv('../input/dsn-ai-oau-july-challenge/sample_submission.csv')
train.head()
#Check for missing values
train.isna().sum()
#General Info about thye data
train.info()

train.hist();
plt.tight_layout()
#sns.pairplot(train)
train['Product_Fat_Content'].value_counts()
train['Product_Type'].value_counts()
train['Supermarket _Size'].value_counts()
train['Supermarket_Location_Type'].value_counts()
train['Supermarket_Type'].value_counts()
sub.head()
df_train = train.copy() #make a copy of the training data
df_test = test.copy() #make a copy of the test data
#drop all id columns
df_train1= df_train.drop(columns =['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier'])
df_test1 =df_test.drop(columns =['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier'])
#info about the training data
df_train1.info()
#Return the column namesn with object datatype
obj_dtype= df_train1.select_dtypes(include=['object']).columns
#Encode all categorical variables
new_train =pd.get_dummies(df_train1, columns=obj_dtype)
new_test = pd.get_dummies(df_test1, columns=obj_dtype)
new_train.head()
#new_train.info()
#Check for missing values
new_train.isna().sum()
new_test.isna().sum()
#mean of Product weight for training data
new_train.Product_Weight.mean()
#Statistical info of the data
new_train.describe()
#fill missing values with the mediann
new_train.Product_Weight.fillna(12.60, inplace=True)
new_test.describe()
#fill missing values with the median
new_test.Product_Weight.fillna(12.85, inplace=True)
#Label
y = new_train['Product_Supermarket_Sales']
#Features
X = new_train.drop('Product_Supermarket_Sales', axis=1)
X.shape, new_test.shape
from sklearn.model_selection import train_test_split
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape
from sklearn.linear_model import LinearRegression
#Initialize the model
lr = LinearRegression()
#train the model
lr.fit(X_train, y_train)
#Make predictions on the test set
pred = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
#Calculate the RMSE
mean_squared_error(y_test,pred)**(0.5)
sub.head()
sub_pred = pd.DataFrame(pred)
sub['Product_Supermarket_Sales']= sub_pred
sub.to_csv('Submission.csv', index=False)
