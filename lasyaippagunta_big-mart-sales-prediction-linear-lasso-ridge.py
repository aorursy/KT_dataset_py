# Importing Libraries

import pandas as pd

import numpy as np

from scipy.stats import mode

import seaborn as sns

from matplotlib import pyplot as plt
# Reading the Train and Test CSV files

train = pd.read_csv("../input/bigmart-dataset/Train.csv")

test = pd.read_csv("../input/bigmart-dataset/Test.csv")
train.head()
train.describe()
train.info()
# Join both the train and test dataset

train['source']='train'

test['source']='test'



dataset = pd.concat([train,test], ignore_index = True)

print("Train dataset shape:",train.shape)

print("Test dataset shape:",test.shape)

print("Concatenated dataset shape:",dataset.shape)
dataset.info()
dataset.head()
dataset.isnull().sum()
#To find percentage of test set in the dataset

print(dataset["Item_Outlet_Sales"].isnull().sum()/dataset.shape[0]*100,"%")
# pivot_table() allows us to create a table that contains the mean values of identifiers

avg = pd.pivot_table(dataset,values='Item_Weight', index='Item_Identifier',aggfunc='mean')

avg
# We find that all examples containing the same Item_Identifier value have same Item_Weight.

# This proves the fact that the mean is same as their value.

# For example the mean value for the value "DRA12" in the Item_Identifier is same as the Item_Weight for 

# individual examples in the Item_Weight column.



df=dataset[dataset['Item_Identifier'].str.contains("DRA12")]

df
dataset[:][dataset['Item_Identifier'] == 'DRI11']

def impute(cols):

    Weight = cols[1]

    Identifier = cols[0]

    

    if pd.isnull(Weight):

        return avg['Item_Weight'][avg.index == Identifier]

    else:

        return Weight

print ('Orignal Number of missing values in Item_Weight:',sum(dataset['Item_Weight'].isnull()))



# Applying the impute() function to impute null values of Item_Weight

dataset['Item_Weight'] = dataset[['Item_Identifier','Item_Weight']].apply(impute,axis=1).astype(float)



print ('Number of missing values in Item_Weight after imputation: ',sum(dataset['Item_Weight'].isnull()))
# Finding unique Outlet Types

dataset.Outlet_Type.unique()
# pivot_table() allows us to create a table that contains the mode of identifiers

mode = pd.pivot_table(dataset, values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())

mode
# Imputing Outlet_Size missing values with their mode

def impute_mode(cols):

    size = cols[1]

    Type = cols[0]

    

    if pd.isnull(size):

        return mode.loc['Outlet_Size'][mode.columns == Type][0]

    else:

        return size

print ('Orignal Number of missing values in Outlet_Size:',sum(dataset['Outlet_Size'].isnull()))



# Applying the impute() function to impute null values of Item_Weight

dataset['Outlet_Size'] = dataset[['Outlet_Type','Outlet_Size']].apply(impute_mode,axis=1)



print ('Number of missing values in Outlet_Size after imputation: ',sum(dataset['Outlet_Size'].isnull()))
dataset.isnull().sum()
dataset.head()
dataset.Item_Fat_Content.unique()
dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})

dataset.Item_Fat_Content.unique()
dataset['Outlet_Year'] = 2020 - dataset['Outlet_Establishment_Year']

dataset.drop(['Outlet_Establishment_Year'],axis=1,inplace=True)

dataset.Outlet_Year.unique()
dataset.to_csv('insight.csv')

dataset.head()
vmean = dataset.pivot_table(index = "Item_Identifier",  values = "Item_Visibility")
dataset.loc[(dataset["Item_Visibility"] == 0.0), "Item_Visibility"] = dataset.loc[(dataset["Item_Visibility"] == 0.0), "Item_Identifier"].apply(lambda x : vmean.at[x, "Item_Visibility"])
# Turning all categorical variables into numerical values can be done by mapping each categorical value with  

# respective FREQUENCY of the values in the column



cat_var = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type','Item_Identifier','Outlet_Identifier']

for i in cat_var:

    p  = dataset[i].value_counts().to_dict()

    dataset[i] = dataset[i].map(p)
dataset.head()
#Divide into test and train:

train = dataset.loc[dataset['source']=="train"]

test = dataset.loc[dataset['source']=="test"]

#Drop unnecessary columns:

test.drop(['source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)
train.head()
test.head()
corr_matrix=train.corr()

corr_matrix['Item_Outlet_Sales']
sns.pairplot(data=train,y_vars=['Item_Outlet_Sales'],x_vars=['Item_Identifier','Item_Weight', 'Item_Fat_Content',

       'Item_Visibility', 'Item_Type', 'Item_MRP','Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',

       'Outlet_Type', 'Outlet_Year'])

plt.figure(figsize = (10,5))

sns.heatmap(corr_matrix, cmap = "RdYlGn", annot = True)
train= train.drop(['Item_Identifier','Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type'],axis=1)
train.skew()
# Before Scaling

fig, ax = plt.subplots(2,3,figsize = (15,15))

sns.distplot(train["Item_Visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(train["Item_MRP"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(train["Outlet_Identifier"], kde =True, ax=ax[0,2], color = "orange")

sns.distplot(train["Outlet_Size"], kde =True, ax=ax[1,0], color = "magenta")

sns.distplot(train["Outlet_Type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(train["Outlet_Year"], kde =True, ax=ax[1,2])
for i in train.columns:

    train[i] =np.log(train[i])
train.head()
#After Scaling the variables

fig, ax = plt.subplots(2,3,figsize = (15,15))

sns.distplot(train["Item_Visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(train["Item_MRP"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(train["Outlet_Identifier"], kde =True, ax=ax[0,2], color = "orange")

sns.distplot(train["Outlet_Size"], kde =True, ax=ax[1,0], color = "magenta")

sns.distplot(train["Outlet_Type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(train["Outlet_Year"], kde =True, ax=ax[1,2])
y=train["Item_Outlet_Sales"]

x= train.drop(["Item_Outlet_Sales"],axis=1)

x.head()
from matplotlib import pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
reg = LinearRegression()

reg = reg.fit(x_train,y_train)
reg.coef_
reg.intercept_
y_pred = reg.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2_score = r2_score(y_test, y_pred)

print('Root Mean Squared Error:',rmse )

print('R2_Score:',r2_score*100,"%" )
residue_lreg = y_test - y_pred

#Plotting Residual Plot

plt.scatter(y_test,residue_lreg, c = "blue")

plt.xlabel("Residual Plot for Linear Regression")

plt.ylabel("y_test")

plt.axhline(y = 0)
from sklearn.linear_model import Lasso, Ridge

ls = Lasso(alpha = 0.009)

ls = ls.fit(x_train, y_train)
ls.coef_
ls.intercept_
ls_pred = ls.predict(x_test)
rmse_LS = np.sqrt(metrics.mean_squared_error(y_test, ls_pred))

print('Root Mean Squared Error:',rmse_LS )
from sklearn.metrics import r2_score

r2_score_LS = r2_score(y_test, ls_pred)

print('R2_Score:',r2_score_LS*100,"%" )
#RESIDUE VALUE AFTER LASSO REGRESSION

residue_lasso = y_test - ls_pred

#Plotting Residual Plot

plt.scatter(y_test,residue_lasso, c = "blue")

plt.xlabel("Residual Plot for Lasso Regression")

plt.ylabel("y_test")

plt.axhline(y = 0)
#Ridge Regression

rr = Ridge(alpha = 0.009)

rr.fit(x_train, y_train)
#Prediction AFTER Ridge regression

rr_pred = rr.predict(x_test)
#Accuracy score check

r2_score_RR = r2_score(y_test, y_pred)

print('R2_Score:',r2_score_RR*100,"%" )
#RMSE

rmse_ridge = np.sqrt(metrics.mean_squared_error(y_test, rr_pred))

rmse_ridge
#residue after ridge

residue_rr = y_test-rr_pred

#Plotting Residual Plot

plt.scatter(y_test,residue_rr, c = "blue")

plt.xlabel("Residual Plot for Ridgeo Regression")

plt.ylabel("y_test")

plt.axhline(y = 0)
#Linear Regression

Lreg_coef = pd.Series(reg.coef_,index =x.columns)

Lreg_coef.plot(kind="bar", title= "Linear")
# Lasso Regression

lasso_coef = pd.Series(ls.coef_,index =x.columns)

lasso_coef.plot(kind="bar", title= "Lasso")
# Ridge Regression

ridge_coef = pd.Series(rr.coef_,index =x.columns)

ridge_coef.plot(kind="bar", title= "Ridge")
df1 = pd.DataFrame(columns=["Linear Regression", "Ridge Regression","Lasso Regression"])

for i in range(len(rr.coef_)):

    df1=df1.append({"Linear Regression":reg.coef_[i],"Ridge Regression":rr.coef_[i], "Lasso Regression":ls.coef_[i]}, ignore_index = True)

df1
test.head()
test= test.drop(["Item_Outlet_Sales"],axis=1)

test_= test.drop(['Item_Identifier','Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type'],axis=1)
test_.head()
test_.skew()
for i in test_.columns:

    test_[i] =np.log(test_[i])
fig, ax = plt.subplots(2,3,figsize = (15,15))

sns.distplot(test_["Item_Visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(test_["Item_MRP"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(test_["Outlet_Identifier"], kde =True, ax=ax[0,2], color = "orange")

sns.distplot(test_["Outlet_Size"], kde =True, ax=ax[1,0], color = "magenta")

sns.distplot(test_["Outlet_Type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(test_["Outlet_Year"], kde =True, ax=ax[1,2])
test_.head()
item_outsale_pred = reg.predict(test_)
item_outsale_pred
#Performing inverse transformation

actual_item_outsale = np.exp(item_outsale_pred+1)
actual_item_outsale
#ADDING THE PREDICTED ITEM_OUTLET_SALE COLUMNS TO TEST DATA

test = pd.read_csv("../input/bigmart-dataset/Test.csv")

test["Item_Outlet_Sales"] = actual_item_outsale
test
test.to_csv('testLR.csv')
vis = pd.read_csv("insight.csv")

vis.head()
#FINDING FREQUENCY COUNT OF OUTLET TYPE

sns.countplot(data = vis, x = "Outlet_Type",hue = "Outlet_Size")

plt.xticks(rotation =90)
# Relation between Outlet_Identifier and Item_Outlet_Sales

import matplotlib.pyplot as plt

plt.figure(figsize = (10,5))

sns.barplot(data = vis, x = "Outlet_Identifier", y= "Item_Outlet_Sales")
plt.figure(figsize = (10,5))

sns.barplot(data = vis, x = "Outlet_Size", y= "Item_Outlet_Sales")
import matplotlib.pyplot as plt

plt.figure(figsize = (10,5))

sns.barplot(data = vis, x = "Outlet_Type", y= "Item_Outlet_Sales")
plt.figure(figsize = (10,5))

sns.barplot(data = vis, x = "Outlet_Location_Type", y= "Item_Outlet_Sales")
vis.groupby("Outlet_Year")["Item_Outlet_Sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
#Understanding to item_type per year with respective to mean of each respective year item outlet sales

vis.groupby("Item_Type")["Item_Outlet_Sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
#Understanding to outlet_type per year with respective to mean of each respective year item outlet sales

vis.groupby("Outlet_Type")["Item_Outlet_Sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
sns.countplot(vis["Item_Fat_Content"]).set_title="Item_Fat_Content"

print(vis["Item_Fat_Content"].value_counts(normalize=True))

plt.xticks(rotation=90)

plt.show()
sns.countplot(vis["Outlet_Size"]).set_title="Outlet_Size"

print(vis["Outlet_Size"].value_counts(normalize=True))

plt.xticks(rotation=90)

plt.show()
sns.countplot(vis["Outlet_Location_Type"]).set_title="Outlet_Location_Type"

print(vis["Outlet_Location_Type"].value_counts(normalize=True))

plt.xticks(rotation=90)

plt.show()
sns.countplot(vis["Item_Type"]).set_title="Item_Type"

print(vis["Item_Type"].value_counts(normalize=True))

plt.xticks(rotation=90)

plt.show()
sns.jointplot(train["Item_Outlet_Sales"],train["Item_MRP"],kind="reg")
sns.jointplot(train["Item_Outlet_Sales"],train["Item_Visibility"],kind="reg")