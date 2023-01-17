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
real_estate=pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")
real_estate.head(10)
#summary
real_estate.describe()
#Removing NA values in the dataset
real_estate.dropna()
#mean
house_age_mean=real_estate["X2 house age"].mean()
print("house_age_mean=", house_age_mean)

mean_convenience_stores=real_estate["X4 number of convenience stores"].mean()
mean_convenience_stores
print("mean_convenience_stores=",mean_convenience_stores)
import seaborn as sns
#regression plot
sns.regplot(x="X2 house age", y="Y house price of unit area",data=real_estate)

from scipy import stats
#correlation
pearson_coef,p_value=stats.pearsonr(real_estate["X2 house age"],real_estate["Y house price of unit area"])
print("The correlation between house age and house price of unit area is", pearson_coef, "with p value of", p_value)
#regression plot
sns.regplot(x="X3 distance to the nearest MRT station", y="Y house price of unit area",data=real_estate)

#correlation
pearson_coef1,p_value1=stats.pearsonr(real_estate["X3 distance to the nearest MRT station"],real_estate["Y house price of unit area"])
print("The correlation between distance to the nearest MRT station and house price of unit area is", pearson_coef1, "with p value of", p_value1)
#regression plot
sns.regplot(x="X4 number of convenience stores", y="Y house price of unit area",data=real_estate)

#correlation
pearson_coef2,p_value2=stats.pearsonr(real_estate["X4 number of convenience stores"],real_estate["Y house price of unit area"])
print("The correlation between the number of convenience stores and house price of unit area is", pearson_coef2, "with p value of", p_value2)
#regression plot
sns.regplot(x="X5 latitude", y="Y house price of unit area",data=real_estate)

#correlation
pearson_coef3,p_value3=stats.pearsonr(real_estate["X5 latitude"],real_estate["Y house price of unit area"])
print("The correlation between the  latitude and house price of unit area is", pearson_coef3, "with p value of", p_value3)
#regression plot
sns.regplot(x="X6 longitude", y="Y house price of unit area",data=real_estate)

#correlation
pearson_coef4,p_value4=stats.pearsonr(real_estate["X6 longitude"],real_estate["Y house price of unit area"])
print("The correlation between longitude and house price of unit area is", pearson_coef4, "with p value of", p_value4)
#Linear Regression

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#Simple Linear Regression

X2=real_estate[["X2 house age"]]
Y=real_estate[["Y house price of unit area"]]
lm.fit(X2,Y)
Yhat=lm.predict(X2)
Yhat
print("Intercept=",lm.intercept_)
print("Coefficient=",lm.coef_)
print("R squared=", lm.score(X2,Y))
#residual plot

sns.residplot(x="X2 house age", y="Y house price of unit area", data=real_estate)
#Actual values vs Predicted values of house price per unit area

ax1=sns.distplot(real_estate["Y house price of unit area"], hist=False,color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
#Multiple linear regression

Z=real_estate[["X1 transaction date","X2 house age", "X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude"]]
lm.fit(Z,Y)
Yhat1=lm.predict(Z)
Yhat1
#Actual vs Predicted plot of house price of unit area

ax2=sns.distplot(real_estate["Y house price of unit area"], hist=False,color="r", label="Actual Value")
sns.distplot(Yhat1, hist=False, color="b", label="Fitted Values", ax=ax2)
from sklearn.model_selection import train_test_split

#We split the data such that 70% is for training while 30% is for testing

x_train,x_test,y_train,y_test=train_test_split(Z,Y, test_size=0.3, random_state=0)
y_test
#Actual vs Predicted plot after splitting the data 
ax3=sns.distplot(real_estate["Y house price of unit area"], hist=False,color="r", label="Actual Value")
sns.distplot(y_test, hist=False, color="b", label="Fitted Values", ax=ax3)