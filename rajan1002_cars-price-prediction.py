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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import  LinearRegression

from sklearn.metrics import  mean_squared_error

from sklearn.preprocessing import LabelEncoder

from statsmodels.formula.api import ols

car=pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")

print(car)

car.describe()

car.columns

car.dtypes

car.isnull().sum()

car.head()
car.drop("Unnamed: 0",axis=1,inplace=True)





car["vin"].value_counts()

car.drop("vin",axis=1,inplace=True)
car["lot"].value_counts()

car.drop("lot",axis=1,inplace=True)
car["country"].value_counts()

car.drop("country",axis=1,inplace=True)
sns.countplot("title_status",data=car)

dic={"clean vehicle":0,"salvage insurance":1}

car["title_status"]=car["title_status"].map(dic)
sns.boxplot(x="price",data=car)

print(car["price"].describe())

des_price=car["price"].describe()

Q1=des_price[4]

Q3=des_price[6]

IQR_price=Q3-Q1

max_price=IQR_price+1.5*Q3

min_price=IQR_price-1.5*Q1

print(Q1)

print(Q3)

print(IQR_price)



car.drop(car[car["price"]>max_price].index,axis=0,inplace=True)

car.drop(car[car["price"]<min_price].index,axis=0,inplace=True)
sns.boxplot(x="mileage",data=car)

print(car["mileage"].describe())

des_mileage=car["mileage"].describe()



Q1=des_mileage[4]

Q3=des_mileage[6]

IQR_mileage=Q3-Q1

max_mileage=IQR_mileage+1.5*Q3

min_mileage=IQR_mileage-1.5*Q1

print(Q1)

print(Q3)

print(IQR_mileage)



car.drop(car[car["mileage"]>max_mileage].index,axis=0,inplace=True)

car.drop(car[car["mileage"]<min_mileage].index,axis=0,inplace=True)
sns.countplot(x="brand",data=car)

plt.xticks(rotation=90)

plt.show()
sns.countplot(x="model",data=car)

plt.xticks(rotation=90)

plt.show()
sns.countplot("year",data=car)

plt.xticks(rotation=90)

plt.show()
sns.countplot("color",data=car)

plt.xticks(rotation=90)

plt.show()
sns.countplot("state",data=car)

plt.xticks(rotation=90)

plt.show()
sns.countplot("condition",data=car)

plt.xticks(rotation=90)

plt.show()
cars=car.copy(deep=True)

le_brand=LabelEncoder()

le_model=LabelEncoder()

le_color=LabelEncoder()

le_state=LabelEncoder()

le_condition=LabelEncoder()

cars["brand"]=le_brand.fit_transform(cars["brand"])

cars["model"]=le_model.fit_transform(cars["model"])

cars["color"]=le_color.fit_transform(cars["color"])

cars["state"]=le_state.fit_transform(cars["state"])

cars["condition"]=le_condition.fit_transform(cars["condition"])
model_fit=ols("price~brand+year+title_status+mileage",data=cars).fit()

print(model_fit.summary())
X=cars[["brand","model","year","title_status","mileage"]]

x=X.values

Y=cars["price"]

y=Y.values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=940)

lgr=LinearRegression(fit_intercept=True)

model=lgr.fit(x_train,y_train)

prediction=lgr.predict(x_test)

print("COD=",model.score(x_test,y_test))

rms=print("rms error=",np.sqrt(mean_squared_error(y_test,prediction)))
residual=y_test-prediction

sns.regplot(prediction,residual,fit_reg=False)





test=pd.DataFrame({"true value":(y_test),"predicted value":(prediction)})



print(test.head(10))