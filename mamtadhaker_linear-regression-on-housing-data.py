# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataframe =  pd.read_csv("../input/MELBOURNE_HOUSE_PRICES_LESS.csv")
dataframe.dtypes
dataframe.head()
dataframe['Type'].unique()
dataframe["Date"] = pd.to_datetime(dataframe["Date"],dayfirst=True)
len(dataframe["Date"].unique())/4
var = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").std()

count = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").count()

mean = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()
mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))
means = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").mean()

errors = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").std()
dataframe.columns
means.columns
means.head()
errors.head()
means.drop(["Price",

            "Postcode",

           "Distance", "Propertycount"],axis=1).plot(yerr=errors)
dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()
pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)

dataframe[(dataframe["Type"]=="h") & 

          (dataframe["Distance"]<14) &

          (dataframe["Distance"]>13.7) 

          #&(dataframe["Suburb"] =="Northcote")

         ].sort_values("Date", ascending=False).dropna().groupby(["Suburb","SellerG"]).mean()
sns.kdeplot(dataframe[(dataframe["Suburb"]=="Northcote")

         & (dataframe["Type"]=="u")

         & (dataframe["Rooms"] == 2)]["Price"])
plt.figure(figsize=(20,15))

my_axis = sns.kdeplot(dataframe["Price"][((dataframe["Type"]=="u") &

                                (dataframe["Distance"]>8) &

                                (dataframe["Distance"]<10) &

                                (dataframe["Rooms"] > 2)#&

                                #(dataframe["Price"] < 1000000)

                               )])

my_axis.axis(xmin=0, xmax=2000000)
sns.lmplot("Distance","Price",dataframe[(dataframe["Rooms"]<=4) & 

                                         (dataframe["Rooms"]> 2) & 

                                        (dataframe["Type"]=="h") &

                                        (dataframe["Price"]< 1000000)

                                       ].dropna(),hue="Rooms", size=10)
dataframe.columns
dataframe[(dataframe["Rooms"]>2) & (dataframe["Type"] == "h")].dropna().groupby("Distance").mean().plot()
sns.pairplot(dataframe.dropna())
fig, ax = plt.subplots(figsize=(15,15)) 

sns.heatmap(dataframe[dataframe["Type"] == "h"].corr(), annot=True)
from sklearn.model_selection import train_test_split
dataframe_dr = dataframe.dropna().sort_values("Date")
from datetime import date
all_Data = []
###########

##Find out days since start

days_since_start = [(x - dataframe_dr["Date"].min()).days for x in dataframe_dr["Date"]]
dataframe_dr["Days"] = days_since_start
#suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Type", "Method"]])

suburb_dummies = pd.get_dummies(dataframe_dr[["Type", "Method"]])

#suburb_dummies = pd.get_dummies(dataframe_dr[[ "Type"]])

#suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Method"]])
all_Data = dataframe_dr.drop(["Address","Price","Date", "SellerG","Suburb","Type","Method","CouncilArea","Regionname"],axis=1).join(suburb_dummies)
X = all_Data
y = dataframe_dr["Price"]
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

X.columns
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)

ranked_suburbs
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)

plt.ylim([200000,1000000])

plt.xlim([200000,1000000])
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))