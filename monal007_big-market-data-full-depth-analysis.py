import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats
df  =pd.read_csv("../input/datasetbig-market/train.csv")
df
import missingno as msno
msno.matrix(df)
df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
plt.show()
df["Item_Identifier"]
#make a new column for item code

for i in range(len(df)):
    df.loc[i , "Item_Code"] = df.loc[i,"Item_Identifier"][:2]
df["Item_Code"]
df.info()
# drop Item_Identifier
df  =df.drop(["Item_Identifier"] , axis =1)
df
## count the item_code and find unique values
df["Item_Code"].value_counts()
sb.countplot(df["Item_Code"])
plt.grid()
plt.show()
print(df["Item_Code"].value_counts())
plt.scatter(df["Item_Weight"], df["Item_Outlet_Sales"])
sb.boxplot(df["Item_Weight"])
plt.grid()
print(df["Item_Weight"].median())
sb.distplot(df["Item_Weight"].dropna())
print(df["Item_Weight"].kurt())
print(df["Item_Weight"].skew())

print(df["Item_Weight"].isna().sum())


df["Item_Weight"].fillna(df["Item_Weight"].mean() ,  inplace = True)

print(df["Item_Weight"].isna().sum())

sb.distplot(df["Item_Weight"].dropna())
print(df["Item_Weight"].kurt())
print(df["Item_Weight"].skew())

df["Item_Fat_Content"].value_counts()
plt.figure(figsize=(10,8))
sb.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
sb.countplot(df["Item_Fat_Content"] , hue = df["Outlet_Size"])
## mapping
fat = {
    "Low Fat":"Low Fat","Regular":"Regular","low fat":"Low Fat","LF":"Low Fat","reg":"Regular"
}

df.loc[: , "Item_Fat_Content"] = df.loc[: , "Item_Fat_Content"].map(fat)
sb.countplot(df["Item_Fat_Content"] , hue = df["Outlet_Size"])
print(df["Item_Fat_Content"].value_counts())
#### converting this category into numerical!
## mapping
cat = {
    "Low Fat": 1,"Regular":0
}

df.loc[: , "Item_Fat_Content"] = df.loc[: , "Item_Fat_Content"].map(cat)
### lets check the co-relation again
df.corr()
df["Item_Visibility"]
sb.distplot(df["Item_Visibility"])
plt.show()
print(df["Item_Visibility"].skew())
sb.boxplot(df["Item_Visibility"])
#### Lets remove the outliers
cols = "Item_Visibility"
high = 0.18
low = 0.0
df = df[(df[cols] > low) & (df[cols] < high)]
#### Lets check again

sb.boxplot(df["Item_Visibility"])
sb.distplot(df["Item_Visibility"])
plt.show()
print(df["Item_Visibility"].skew())
df["Item_Visibility"] = np.log10(df["Item_Visibility"])
df["Item_Visibility"]
sb.distplot(df["Item_Visibility"])
plt.show()
print(df["Item_Visibility"].skew())
df.corr()
df["Item_Type"]
print(df["Item_Type"].value_counts())
sb.countplot(df["Item_Type"])
plt.xticks(rotation = 90)
plt.show()
food = {
    "Household": "Others","Frozen Foods":"Snack Foods" ,"Dairy":"Snack Foods", "Canned" : "Snack Foods", "Soft Drinks":"Drinks",
  "Hard Drinks":"Drinks"  ,"Starchy Foods":"Baking Goods","Breads":"Baking Goods","Meat":"Seafood","Breakfast":"Fruits and Vegetables",
    "Starchy Foods":"Baking Goods","Fruits and Vegetables":"Fruits and Vegetables","Snack Foods":"Snack Foods","Baking Goods":"Baking Goods",
    "Health and Hygiene":"Health and Hygiene","Seafood":"Seafood","Others":"Others",
}

df.loc[: , "Item_Type"] = df.loc[: , "Item_Type"].map(food)
print(df["Item_Type"].value_counts())
sb.countplot(df["Item_Type"])
plt.xticks(rotation = 90)
plt.grid()
plt.show()
encode = {
    "Snack Foods":7 ,"Fruits and Vegetables":6,"Others":5,"Baking Goods":4,"Drinks":3,"Health and Hygiene":2,"Seafood":1
}

df.loc[: , "Item_Type"] = df.loc[: , "Item_Type"].map(encode)
print(df["Item_Type"].value_counts())
sb.countplot(df["Item_Type"])
plt.xticks(rotation = 90)
plt.grid()
plt.show()
sb.pairplot(df )
sb.boxplot(df["Item_Type"])
df
df["Item_MRP"].describe()
sb.distplot(df["Item_MRP"])
print("Kurt" , df["Item_MRP"].kurt())
print("Skew", df["Item_MRP"].skew())
plt.grid()
plt.show()

sb.boxplot(df["Item_MRP"])
print("Co-relation of sales and mrp is :",stats.pearsonr(df["Item_Outlet_Sales"] , df["Item_MRP"])[0])
plt.scatter(df["Item_MRP"],df["Item_Outlet_Sales"])
plt.show()
df["Outlet_Identifier"]
### No meaning here , lets drop this column
df =df.drop(["Outlet_Identifier"] , axis =1)
df["Outlet_Establishment_Year"]
plt.hist(df["Outlet_Establishment_Year"])
plt.grid()
plt.show()
print("Co-relation of sales and mrp is :",stats.pearsonr(df["Item_Outlet_Sales"] , df["Outlet_Establishment_Year"])[0])
sb.heatmap(df.corr() , cmap="RdYlGn" , annot =True)
sb.boxplot(df["Outlet_Establishment_Year"])
sb.countplot(df["Outlet_Establishment_Year"] , hue = df["Outlet_Size"])
df
df["Outlet_Size"].value_counts()
sb.countplot(df["Outlet_Size"])
df["Outlet_Size"].isna().sum()
df["Outlet_Size"].fillna(value = "High" , inplace = True)
sb.countplot(df["Outlet_Size"] , hue = df["Item_Type"])
plt.grid()
## we will encode it 
out = {
    "Medium":2 , "Small":1,"High":3
}

df.loc[: , "Outlet_Size"] = df.loc[: , "Outlet_Size"].map(out) 
df.corr()
sb.countplot(df["Outlet_Size"] , hue = df["Item_Fat_Content"])
plt.grid()
print(df["Outlet_Location_Type"].value_counts())
sb.countplot(df["Outlet_Location_Type"])
sb.countplot(df["Outlet_Location_Type"] , hue = df["Outlet_Size"] )
print("Outlet size == 1 is medium size , 2 is small ,  3 is high")
mapp = {
    "Tier 3" : 3 , "Tier 2":1 , "Tier 1":2
}
df.loc[: , "Outlet_Location_Type"] = df.loc[: , "Outlet_Location_Type"].map(mapp)
sb.countplot(df["Outlet_Location_Type"] , hue = df["Outlet_Size"] )
print("Outlet size == 1 is medium size , 2 is small ,  3 is high")
df.corr()
sb.countplot(df["Outlet_Location_Type"] , hue = df["Outlet_Establishment_Year"] )
df
print(df["Outlet_Type"].value_counts())
sb.countplot(df["Outlet_Type"])
plt.xticks(rotation = 90)
plt.grid()
plt.show()
sb.countplot(df["Outlet_Type"] , hue = df["Outlet_Location_Type"])
plt.xticks(rotation = 90)
plt.show()
print("3 is tier 3 , 2 is tier 1 and 1 is tier 2")
sb.countplot(df["Outlet_Type"] , hue = df["Outlet_Establishment_Year"])
plt.xticks(rotation = 90)
plt.show()
## map
types = {"Supermarket Type1":4 , "Supermarket Type2":3 , "Supermarket Type3":2 , "Grocery Store":1
    
}
df.loc[: , "Outlet_Type"] = df.loc[: , "Outlet_Type"].map(types)
df
df.corr()
df["Item_Code"]
sb.countplot(df["Item_Code"])

## map
types = {"FD":3 , "DR":1 , "NC":2    
}
df.loc[: , "Item_Code"] = df.loc[: , "Item_Code"].map(types)
sb.boxplot(df["Item_Code"])
df.corr()
###############################################################################################################################
