import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/bigmart-sales-data/Train.csv")

test = pd.read_csv("../input/bigmart-sales-data/Test.csv")
# Preview First 05 Rows of the Data

train.head()
train.info()
# Target Variable

train.columns
sns.distplot(train.Item_Outlet_Sales, color = "m")

plt.show()
train.Item_Outlet_Sales.describe()
sns.distplot(train.Item_Visibility, color = "red");
sns.distplot(train.Item_Weight.dropna(), color = "g");
sns.distplot(train.Item_MRP, color = "r");
train.head()
test.Item_Fat_Content.value_counts()
test.Item_Fat_Content.replace(to_replace = ["LF", "low fat"], 

                              value = ["Low Fat", "Low Fat"], inplace=True)

test.Item_Fat_Content.replace(to_replace = ["reg"], value = ["Regular"], 

                              inplace = True)
# Replacement of LF and low fat

train.Item_Fat_Content.replace(to_replace = ["LF", "low fat"], 

                              value = ["Low Fat", "Low Fat"], inplace=True)

# Replacing reg into Regular

train.Item_Fat_Content.replace(to_replace = ["reg"], value = ["Regular"], 

                              inplace = True)
# Item Fat Content

train.Item_Fat_Content.value_counts().plot(kind = "bar")
# Item Fat Content

train.Item_Type.value_counts().plot(kind = "bar")



# By Sns

sns.countplot(x = "Item_Type", data = train)

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x = "Item_Type", data = train)

plt.xticks(rotation = 90)

plt.show()
# Outlet _Identifier

train.Outlet_Identifier.value_counts().plot(kind = "bar")
# Outlet _Size

train.Outlet_Size.value_counts().plot(kind = "bar")
# Outlet_Type

train.Outlet_Type.value_counts().plot(kind = "bar");
# Num vs Num

train.head()
plt.scatter(train.Item_Weight, train.Item_Outlet_Sales, color = "magenta");
plt.figure(figsize = [10, 8])

plt.scatter(train.Item_Visibility, train.Item_Outlet_Sales, color = "red");
plt.scatter(train.Item_MRP, train.Item_Outlet_Sales, color = "hotpink")

# Price Per Unit
# Cat Vs Numerical

sns.boxplot(train.Item_Fat_Content, train.Item_Outlet_Sales)
train.groupby("Item_Fat_Content")["Item_Outlet_Sales"].describe().T

# Hint: Refer Empirical Rule and Contradictory Rule - Chebyshev Inequality
# Cat Vs Numerical

plt.figure(figsize = [13,6])

sns.boxplot(train.Item_Type, train.Item_Outlet_Sales)

plt.xticks(rotation = 90)

plt.title("Boxplot - Item Type Vs Sales")

plt.xlabel("Item Type")

plt.ylabel("Sales")

plt.show()
# Cat Vs Numerical

plt.figure(figsize = [13,6])

sns.boxplot(train.Outlet_Identifier, train.Item_Outlet_Sales)

plt.xticks(rotation = 90)

plt.title("Boxplot - Oultet ID Vs Sales")

plt.xlabel("Outlets")

plt.ylabel("Sales")

plt.show()
# Outlet Size

# Cat Vs Numerical

plt.figure(figsize = [13,6])

sns.boxplot(train.Outlet_Size, train.Item_Outlet_Sales)

plt.xticks(rotation = 90)

plt.title("Boxplot - Oultet Size Vs Sales")

plt.xlabel("Outlets")

plt.ylabel("Sales")

plt.show()
pd.DataFrame(train.groupby("Outlet_Size")["Outlet_Identifier"].value_counts()).T
# Missing Value

train.isnull().sum()[train.isnull().sum()!=0]
weightna = train[train.Item_Weight.isnull()]
weightna.head()
# Combining the Dataset

combined = pd.concat([train,test], ignore_index=True, sort = False)
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.Item_Fat_Content.value_counts()
# Pattern

train[train.Item_Identifier=="FDX07"]["Item_Visibility"].median()



# Missing value Imputation

train.loc[29, "Item_Weight"]= train[train.Item_Identifier=="FDC14"]["Item_Weight"].median()



# Finding ID | np.where(train.Item_Weight.isna())

ids = train[pd.isnull(train.Item_Weight)]["Item_Identifier"]

locs = ids.index # Finding Index of the Item Weight Missing Values



# Missing Value Final Code

for i in range(0, len(ids)):

    train.loc[locs[i],"Item_Weight"]=train[train.Item_Identifier==ids.values[i]]["Item_Weight"].median()
# Missing Value Imputation - Item Weight | Lambda

combined["Item_Weight"]=combined.groupby("Item_Identifier")["Item_Weight"].transform(lambda x:x.fillna(x.median()))
# Missing Values - Item Visibility

combined["Item_Visibility"] = combined.groupby("Item_Identifier")["Item_Visibility"].transform(lambda x:x.replace(to_replace = 0,value = x.median()))
plt.figure(figsize = [10,7])

plt.scatter(combined["Item_Visibility"], combined["Item_Outlet_Sales"], color = "red")
combined[combined["Item_Identifier"]=="FDY07"]
train[train.Item_Identifier=="FDY07"]["Item_Visibility"]
# Imputation of FDY 07

combined.loc[(combined.Item_Identifier=="FDY07") & (combined["Item_Visibility"]!=0), 

        "Item_Visibility"]=0.121848
combined.head()
# Lets Deal with Tier 2

train.loc[train["Outlet_Location_Type"]=='Tier 2',"Outlet_Size"]="Small"
#train.loc[train["Outlet_Location_Type"]=='Tier 1',"Outlet_Size"]
# Feature Engineering

train.head()
# Size

pd.DataFrame(combined.groupby(["Outlet_Type", "Outlet_Location_Type"])

             ["Outlet_Size"].value_counts())
# Imputting Rule 2 Tier 2 and S1 - Small

combined.loc[[(combined["Outlet_Location_Type"]=="Tier 2") & 

             (combined["Outlet_Type"]=="Supermarket Type1"),

            "Outlet_Size"]]=["Small"]
# Imputting Rule 1 Tier 3 and Grocery Store - Medium

combined.loc[[(combined["Outlet_Location_Type"]=="Tier 3") & 

             (combined["Outlet_Type"]=="Grocery Store"),

            "Outlet_Size"]]=["Medium"]
combined.isnull().sum()
combined.head()
# Price Per Unit

combined["Price_Per_Unit"] = combined["Item_MRP"]/combined["Item_Weight"]
# Outlet Age

combined["Outlet_Age"] = 2013 - combined.Outlet_Establishment_Year
combined.Item_Type.unique()
perishables = ['Dairy', 'Meat', 'Fruits and Vegetables','Breakfast',

              'Breads','Seafood']
# Function

def badalde(x):

    if(x in perishables):

        return("Perishables")

    else:

        return("Non Perishables")

    

combined.Item_Type.apply(badalde)
# np.where

np.isin(combined.Item_Type, perishables)
np.where(combined.Item_Type.isin(perishables), "Perishables", 

         "Non Perishables")
# Loop

badlale = []

for i in range(0, len(combined)):

    if(combined.Item_Type[i] in perishables):

        badlale.append("Perishables")

    else:

        badlale.append("Non Perishables")
combined["ItemType_Cat"]=pd.Series(badlale)
combined.head()
str(combined.Item_Identifier[0])[:2]
item_id =[]

for i in combined.Item_Identifier:

    item_id.append(str(i)[:2])
combined["Item_IDS"]=pd.Series(item_id)
combined.head()
plt.figure(figsize=[10,7])

plt.scatter(combined["Price_Per_Unit"], combined["Item_MRP"], color = "red")
# Dropping the Columns

combined.columns
newdata = combined.drop(['Item_Identifier','Item_MRP','Item_Type','Outlet_Identifier',

       'Outlet_Establishment_Year',], axis = 1)
print(newdata.shape)
# Applying OHE

dummydata = pd.get_dummies(newdata)
dummydata.head()
# Split the Data in Train and Test

newtrain = dummydata[0:train.shape[0]]
# Test

newtest = dummydata[8523:dummydata.shape[0]]
newtest.drop("Item_Outlet_Sales",axis = 1, inplace = True)
print(newtrain.shape)

print(newtest.shape)
newtrain.columns
newtest.columns
# Scaling the Dataset

from sklearn.preprocessing import StandardScaler

nayasc = StandardScaler()
newtrain.drop("Item_Outlet_Sales", axis = 1).shape
newtrain.columns[newtrain.columns!="Item_Outlet_Sales"]
# Standardized Train Set

scaledtrain = pd.DataFrame(nayasc.fit_transform(newtrain.drop("Item_Outlet_Sales", axis = 1)), 

             columns = newtrain.columns[newtrain.columns!="Item_Outlet_Sales"])
# Standardized Test Set

scaledtest = pd.DataFrame(nayasc.transform(newtest), columns=newtest.columns)
# Random Forest

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor()

gbm = GradientBoostingRegressor()
gbm.fit(scaledtrain, newtrain.Item_Outlet_Sales)
gbm_pred = gbm.predict(scaledtest)
# Submit on AV

solution = pd.DataFrame({"Item_Identifier":test["Item_Identifier"],

                        "Outlet_Identifier":test["Outlet_Identifier"],

                        "Item_Outlet_Sales":gbm_pred})
solution.to_csv("GBM Model.csv", index = False) # 1164.224735564618
rf.fit(scaledtrain, newtrain.Item_Outlet_Sales)
pred = rf.predict(scaledtest)
# Submit on AV

solution = pd.DataFrame({"Item_Identifier":test["Item_Identifier"],

                        "Outlet_Identifier":test["Outlet_Identifier"],

                        "Item_Outlet_Sales":pred})
solution.head()
solution.to_csv("RandomF Model.csv", index = False) # 1224.9984365775733.