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

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm
delhi=pd.read_csv("/kaggle/input/housing-prices-in-metropolitan-areas-of-india/Delhi.csv")
print(delhi)
delhi.columns
delhi.rename(columns={"No. of Bedrooms":"Bedrooms", "24X7Security":"Security","Children'splayarea":"Playarea"}, inplace=True)
y1=np.log(delhi["Price"])
sns.regplot(x="Area",y=y1,data=delhi,fit_reg=False)
delhi.drop(delhi[delhi["Area"]>=8000].index,inplace=True)

y1=np.log(delhi["Price"])

sns.regplot(x="Area",y=y1,data=delhi,fit_reg=False)
sns.countplot(x=delhi["IndoorGames"],data=delhi)
sns.countplot(x=delhi["Gymnasium"],data=delhi)
sns.countplot(x="SwimmingPool",data=delhi)
sns.countplot(x=delhi["IndoorGames"],data=delhi)
delhi.replace(9,np.nan,inplace=True)

delhi.dropna(axis=0,how="any",inplace=True)
le_location=LabelEncoder()

delhi["Location"]=le_location.fit_transform(delhi["Location"])





x1=delhi.drop(["Price","MaintenanceStaff","SwimmingPool","LandscapedGardens","ShoppingMall","SportsFacility","ATM","ClubHouse","StaffQuarter","Cafeteria","MultipurposeRoom","WashingMachine","Wifi","BED","Microwave","DiningTable","Wardrobe","Sofa","Refrigerator","GolfCourse","TV"],axis=1,inplace=False)

x1=sm.add_constant(x1)

y1=np.log(delhi["Price"])

model=sm.OLS(y1,x1).fit()

print(model.summary())
delhi.drop(["MaintenanceStaff","SwimmingPool","LandscapedGardens","ShoppingMall","SportsFacility","ATM","ClubHouse","StaffQuarter","Cafeteria","MultipurposeRoom","WashingMachine","Wifi","BED","Microwave","DiningTable","Wardrobe","Sofa","Refrigerator","GolfCourse","TV"],axis=1,inplace=True)



x=delhi.drop("Price",axis=1,inplace=False)

x=x.values

y=np.log(delhi["Price"]).values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lgr=LinearRegression(fit_intercept=True)

model_1=lgr.fit(x_train,y_train)

prediction=lgr.predict(x_test)

residual=y_test-prediction

sns.regplot(prediction,residual,fit_reg=False)
print(model_1.score(x_test,y_test))
print(r2_score(y_test,prediction))
print(np.sqrt(mean_squared_error(y_test,prediction)))