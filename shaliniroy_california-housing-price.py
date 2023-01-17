# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/california-housing-prices/housing.csv")

df.info()

df["ocean_proximity"].value_counts()
from sklearn.preprocessing import LabelBinarizer



lb=LabelBinarizer()

prox=lb.fit_transform(df["ocean_proximity"])

df=df.drop("ocean_proximity",axis=1)
print(prox)

prox_df=pd.DataFrame(prox,columns=["<1H Ocean","Inland","Near Ocean","Near Bay","Island"])

import matplotlib.pyplot as plt

%matplotlib inline

df.hist(bins=50,figsize=(20,15))

plt.show()
#fill n/a 

df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].median())
import seaborn as sns



corr_matrix=df.corr()

sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')

plt.title("Correlation map")

plt.show()
from pandas.plotting import scatter_matrix



corr1=["median_house_value","median_income","total_rooms","housing_median_age","latitude"]

scatter_matrix(df[corr1],figsize=(20,15))
df.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

plt.show()

#outliers detected
#remove outliers

df=df[df["median_house_value"]<500000]
df["bedrooms_per_household"]=df["total_bedrooms"]/df["households"]

df["rooms_per_household"]=df["total_rooms"]/df["households"]



corr_matrix=df.corr()

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")

plt.show()

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)
df=df.drop(["total_rooms","total_bedrooms"],axis=1)



df["<1H Ocean"]=prox_df["<1H Ocean"]

df["Inland"]=prox_df["Inland"]

df["Near Ocean"]=prox_df["Near Ocean"]

df["Near Bay"]=prox_df["Near Bay"]

df["Island"]=prox_df["Island"]

df
#train test split

from sklearn.model_selection import train_test_split



y=df["median_house_value"]

X=df.drop(["median_house_value"],axis=1)

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler



ss=StandardScaler()

cols=["longitude","latitude","housing_median_age","population","households","median_income","bedrooms_per_household","rooms_per_household"]

X_train[cols]=ss.fit_transform(X_train[cols])

X_test[cols]=ss.transform(X_test[cols])
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



lr=LinearRegression()

lr.fit(X_train,y_train)

pred=lr.predict(X_test.iloc[:786])

rms=mean_squared_error(pred,y_test.iloc[:786])

print(rms)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train, y_train)

pred=tree_reg.predict(X_test.iloc[:786])

rms=mean_squared_error(pred,y_test.iloc[:786])

print(rms)

tree_reg.score(X_train,y_train)
output_csv = pd.DataFrame({'Label':pred})



output_csv.to_csv('output.csv',index=False)