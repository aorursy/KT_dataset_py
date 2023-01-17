import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt


dataframe =  pd.read_csv("../input/Melbourne_housing_extra_data.csv")
dataframe.head()
dataframe["Date"] = pd.to_datetime(dataframe["Date"],dayfirst=True)
len(dataframe["Date"].unique())/4
var = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").std()

count = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").count()

mean = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()
mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))
means = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].dropna().sort_values("Date", ascending=False).groupby("Date").mean()

errors = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].dropna().sort_values("Date", ascending=False).groupby("Date").std()
#fig, ax = plt.subplots()

means.plot(yerr=errors)
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
sns.kdeplot(dataframe["Price"][((dataframe["Type"]=="u") &

                                (dataframe["Distance"]>3) &

                                (dataframe["Distance"]<10) &

                                (dataframe["Rooms"] > 2)#&

                                #(dataframe["Price"] < 1000000)

                               )])
sns.lmplot("Distance","Price",dataframe[(dataframe["Rooms"]<=4) & 

                                         (dataframe["Rooms"]> 2) & 

                                        (dataframe["Type"]=="h") &

                                        (dataframe["Price"]< 1000000)

                                       ].dropna(),hue="Rooms", size=10)
dataframe[dataframe["Rooms"]<4].dropna().groupby(["Distance","Rooms"]).mean()
dataframe.columns
sns.pairplot(dataframe.dropna())
sns.heatmap(dataframe[dataframe["Type"] == "h"].corr(), annot=True)
from sklearn.cross_validation import train_test_split
dataframe_dr = dataframe.dropna().sort_values("Date")
dataframe_dr = dataframe_dr[dataframe_dr["Type"]=="u"]
dataframe_dr = dataframe_dr
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
all_Data = dataframe_dr.drop(["Address","Price","Date", "SellerG","Suburb","Type","Method"],axis=1).join(suburb_dummies)
all_Data.head().to_csv("tony.csv")
X = all_Data
y = dataframe_dr["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LinearRegression
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