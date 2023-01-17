import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing, svm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
file1= '../input/dsn-ai-oau-july-challenge/train.csv'

file2= '../input/dsn-ai-oau-july-challenge/test.csv'

file3='../input/dsn-ai-oau-july-challenge/sample_submission.csv'
train_data= pd.read_csv(file1,error_bad_lines= False)

train_data
test_data= pd.read_csv(file2,error_bad_lines= False)

test_data
sample= pd.read_csv(file3, error_bad_lines= False)

sample
train_data.info()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.describe()
test_data.describe()
train_data["Supermarket _Size"].value_counts()
train_data["Supermarket _Size"]= train_data["Supermarket _Size"].fillna("Medium", axis= 0)
train_data.Product_Weight.value_counts()
train_data["Product_Fat_Content"].value_counts()
LF= train_data["Product_Fat_Content"] == "Low Fat"

NF= train_data["Product_Fat_Content"] == "Normal Fat"

ULF= train_data["Product_Fat_Content"] ==  "Ultra Low fat"
train_data.loc[LF, "Product_Weight"] = train_data.loc[LF, "Product_Weight"].fillna(train_data.loc[LF, "Product_Weight"].mean())

train_data.loc[NF, "Product_Weight"] = train_data.loc[NF, "Product_Weight"].fillna(train_data.loc[NF, "Product_Weight"].mean())

train_data.loc[ULF, "Product_Weight"] = train_data.loc[ULF, "Product_Weight"].fillna(train_data.loc[ULF, "Product_Weight"].mean())
train_data.isnull().sum()
test_data["Supermarket _Size"].value_counts()
test_data["Supermarket _Size"]= test_data["Supermarket _Size"].fillna("Medium", axis= 0)
L= test_data["Product_Fat_Content"]== "Low Fat"

N= test_data["Product_Fat_Content"]== "Normal Fat"

UL= test_data["Product_Fat_Content"] == "Ultra Low fat"
test_data.loc[L, "Product_Weight"]= test_data.loc[L, "Product_Weight"].fillna(test_data.loc[L, "Product_Weight"].mean())

test_data.loc[N, "Product_Weight"] = test_data.loc[N, "Product_Weight"].fillna(test_data.loc[N, "Product_Weight"].mean())

test_data.loc[UL, "Product_Weight"]= test_data.loc[UL, "Product_Weight"].fillna(test_data.loc[UL, "Product_Weight"].mean())
test_data.isnull().sum()
train_data.corr()
sns.scatterplot(x= train_data["Product_Weight"], y= train_data["Product_Supermarket_Sales"])

plt.show()
sns.scatterplot(x= train_data["Product_Shelf_Visibility"] , y= train_data["Product_Supermarket_Sales"])

plt.show()
sns.scatterplot(x= train_data["Product_Price"], y= train_data["Product_Supermarket_Sales"])

plt.figure(figsize=(10,6))

plt.show()
sns.set(style= "darkgrid")

sns.countplot(y= "Product_Type", hue= "Supermarket_Type",data= train_data)

plt.title("Product Type to Supermarket Type")

plt.show()

#This chart illustrates Supermarket Type 1 as the most sort after. it encompasses most of the products in large amount
#Same illustration showing how much product can be found in Supermarket Type 1

train_data.Supermarket_Type.value_counts().plot(kind= 'pie')

plt.show()
sns.barplot(y= train_data["Supermarket_Type"], x= train_data["Product_Supermarket_Sales"])

plt.figure(figsize=(40,15))

plt.show()

#In respect to Product Supermarket Sales, Supermarket Type 3 makes the highest Sales
sns.set(style= "darkgrid")

sns.countplot(y= "Product_Type", hue= "Supermarket_Location_Type", data= train_data)

plt.title("Product Type with Clusters")

plt.show()

#Cluster 3 is seen to encompass more products
sns.barplot(y=train_data["Supermarket_Location_Type"], x= train_data["Product_Supermarket_Sales"])

plt.title("Clusters with Sales")

plt.show()

#But this shows Cluster 2 as the best sales making cluster
PW= train_data.groupby("Product_Fat_Content").sum().reset_index()

PW

#Normal fat products are seen to have the best product weight but did poorly in making more sales
plt.figure(figsize= (15,10))

sns.set(style= "darkgrid")

g= sns.barplot(PW["Product_Fat_Content"], PW["Product_Supermarket_Sales"])

for index, row in PW.iterrows():

    g.text(row.name, row.Product_Supermarket_Sales, round(row.Product_Supermarket_Sales,2), color= "black", ha= "center")

    g.set_xticklabels(g.get_xticklabels(), fontsize= 18)

    g.set_xlabel("Product_Fat_Content", fontsize= 18)

plt.title("Fat Content to Product Sales")

plt.show()

#Product with Low fat seems to make the best sales
sns.set(style="darkgrid")

sns.countplot( y= "Product_Type", hue= "Supermarket _Size", data= train_data)

plt.title("Product Type to Market Size")

plt.show()

#Medium sized supermarket has most products
sns.barplot(train_data["Supermarket _Size"], train_data["Product_Supermarket_Sales"])

plt.title("Supermarket sales per size")

plt.show()

#This still ascertain medium sized supermarket as the best sales making market
sns.set(style= "darkgrid")

sns.countplot(y= "Supermarket_Type", hue= "Supermarket _Size", data= train_data)

plt.show()
PV= train_data.groupby("Product_Type").sum().reset_index()

PV
plt.figure(figsize= (15,10))

sns.set(style= "darkgrid")

g= sns.barplot(PV["Product_Type"],PV["Product_Supermarket_Sales"])

for index, row in PV.iterrows():

    g.text(row.name, row.Product_Price, round(row.Product_Price,2), color= 'black',ha= 'center')

    g.set_xticklabels(g.get_xticklabels(), rotation= 90, fontsize= 18)

    g.set_xlabel("Product_Type", fontsize= 18)

plt.title("Product Type to Supermarket Sales")

plt.show()

#We clearly see Sanck Foods, followed by Fruits and vegetables, Household, Frozen Foods to Canned as the Best Five Sales Making Products
PT= train_data[train_data["Product_Type"] == "Snack Foods"]["Supermarket_Type"]

print(PT.value_counts())

sns.countplot(PT)

plt.title("Best Market with Snack Food")

#Supermarket Type 1 is seen to have the highest number of Snack Food but this doesnt clarify it to be making the best sales on Snack Foods

#Type 1 isn't but Type 3
columns= ("Product_Identifier","Supermarket_Identifier","Product_Supermarket_Identifier","Product_Fat_Content","Product_Type","Supermarket _Size","Supermarket_Location_Type","Supermarket_Type")

for x in columns:

    Le= LabelEncoder()

    train_data[x]= Le.fit_transform(train_data[x].values)

train_data.dtypes
col= ("Product_Identifier","Supermarket_Identifier","Product_Supermarket_Identifier","Product_Fat_Content","Product_Type","Supermarket _Size","Supermarket_Location_Type","Supermarket_Type")

for y in col:

    Le= LabelEncoder()

    test_data[y]= Le.fit_transform(test_data[y].values)

test_data.dtypes
train_new= train_data.drop(["Product_Identifier","Supermarket_Identifier"], axis=1)

test_new= test_data.drop(["Product_Identifier","Supermarket_Identifier"], axis=1)
X= train_new.drop("Product_Supermarket_Sales", axis= 1)

y= train_new["Product_Supermarket_Sales"]

#X= preprocessing.scale(X)
train_x, val_x, train_y, val_y= train_test_split(X,y, test_size= 0.2, random_state= 60)
LR= LinearRegression()

val= LR.fit(train_x, train_y)

pred= val.predict(val_x)
print("mse:", mean_squared_error(pred,val_y))

print("r2_score:", r2_score(pred,val_y))

print("Rmse:", np.sqrt(mean_squared_error(pred,val_y)))
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.preprocessing import MinMaxScaler


abr= AdaBoostRegressor(n_estimators= 50, learning_rate= 0.1, random_state=0)    

rfr = RandomForestRegressor( n_estimators= 100, random_state=0, verbose=False)

vall= abr.fit(train_x,train_y)

predd= vall.predict(val_x)
print("mse:", mean_squared_error(predd, val_y))

print("Rmse:", np.sqrt(mean_squared_error(predd,val_y)))
abr.fit(X,y)

test_yhat= abr.predict(test_new)
submission= pd.read_csv(file3)

submission.head()
submission.Product_Supermarket_Sales = test_yhat

submission.head()
submission.to_csv("july.csv", index= False)