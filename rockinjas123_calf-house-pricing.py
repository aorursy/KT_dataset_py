import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb # plotting purpose

import matplotlib.pyplot as plt # plotting graphs

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/housing/housing.csv") # reading the data
df
df.hist(bins=50,figsize=(15,10))
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

scatter_matrix(df[attributes],figsize=(15,12),color="purple")
df.plot.scatter("longitude","latitude",alpha=0.1)  #this shows the area with more population density
df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=df["population"]/100,label="population",c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,figsize=(10,6))

plt.legend()
df.dtypes   # checking the dtatypes
ocean=pd.get_dummies(df["ocean_proximity"])

ocean
df[ocean.columns]=ocean # combining the dummy variables to our dataframe

df
df=df.drop(["ocean_proximity"],axis=1)
df.corr()   # checking which variables are coreelated with each other
plt.figure(figsize=(15,12))   # plotting a heatmap to get a clear visual of the corellation the skinny color shows the one with high correlation

sb.heatmap(df.corr(), annot=True)
df.isnull().sum()  # checking for the missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].mean(),inplace=True)
df.isnull().sum()  # checking for the missing values
x=df.drop(["median_house_value"],axis=1)

y=df["median_house_value"]
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
x_scaled = mms.fit_transform(x.values)

y_scaled = mms.fit_transform(y.values.reshape(-1,1)) 

x_scaled
y_scaled
# now splitting the training and tesing data

from sklearn.model_selection import train_test_split as tts

train_x,test_x,train_y,test_y=tts(x_scaled,y_scaled,random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score
lr=LinearRegression()

model1=cross_val_score(lr,test_x,test_y, cv=10)

model1.mean(),model1.std()
dtr=DecisionTreeRegressor(max_depth=4)

model1=cross_val_score(dtr,test_x,test_y, cv=10)

model1.mean(),model3.std()
rfr=RandomForestRegressor(max_depth=8)

model3=cross_val_score(rfr,test_x,test_y, cv=10)

model3.mean(),model3.std()
knr=KNeighborsRegressor(n_neighbors=14)

model4=cross_val_score(knr,test_x,test_y, cv=10)

model4.mean(),model4.std()
rnr=RANSACRegressor()

model5=cross_val_score(rnr,test_x,test_y, cv=10)

model5.mean(),model5.std()
rr=Ridge(random_state=19)

model6=cross_val_score(rr,test_x,test_y, cv=10)

model6.mean(),model6.std()
rfr=RandomForestRegressor(max_depth=8)

temp=rfr.fit(train_x,train_y)
train_pred=temp.score(train_x,train_y)

train_pred
test_pred=temp.score(test_x,test_y)

test_pred