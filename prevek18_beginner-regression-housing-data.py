# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))
#Create dataframe
df= pd.read_csv("../input/AmesHousing.csv")
#Check dataframe load
df.head()


#Get number of rows and columns
df.shape
#Since Regression needs numerical features,convert categorical columns into dummy variables
df1= pd.get_dummies(df)
df1.head()
#Look for columns with any NaN(missing) values
df1.columns[df1.isna().any()].tolist()
#Number of NaN values columnwise
df1.isna().sum()
#Define function to impute series with it's median
def impute_median(series):
    return series.fillna(series.median())
df1['Lot Frontage']= df1['Lot Frontage'].transform(impute_median)
df1['Mas Vnr Area']=df1['Mas Vnr Area'].transform(impute_median)
df1['BsmtFin SF 1']=df1['BsmtFin SF 1'].transform(impute_median)
df1['BsmtFin SF 2']=df1['BsmtFin SF 2'].transform(impute_median)
df1['Bsmt Unf SF']=df1['Bsmt Unf SF'].transform(impute_median)
df1['Total Bsmt SF']=df1['Total Bsmt SF'].transform(impute_median)
df1['Bsmt Full Bath']=df1['Bsmt Full Bath'].transform(impute_median)
df1['Bsmt Half Bath']=df1['Bsmt Half Bath'].transform(impute_median)
df1['Garage Cars']=df1['Garage Cars'].transform(impute_median)
df1['Garage Area']=df1['Garage Area'].transform(impute_median)
#Check remaining columns with NaN values
df1.columns[df1.isna().any()].tolist()
#Drop this column
df2=df1.drop('Garage Yr Blt',axis=1)
#Define target array y
y= df2['SalePrice'].values
y
#Create feature array X
X= df2.drop('SalePrice',axis=1).values
X
#Check X's shape
X.shape
#Check Y's shape
y.shape
#Reshape y to have 1 column
y=y.reshape(-1,1)
y.shape
#Split the arrays into training and testing data sets
X_train, X_test,y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=42)
#Create a regressor object
reg= LinearRegression()

#Fit training set to the regressor
reg.fit(X_train,y_train)

#Make predictions with the regressor
y_pred = reg.predict(X_test)

#Calculate accuracy
R2= reg.score(X_test,y_test)
print(R2)