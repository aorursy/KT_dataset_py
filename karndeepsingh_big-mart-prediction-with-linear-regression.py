#IMPORTING LIBRARIES:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats
#Importing Dataset:

data_shopping= pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
data_shopping.head()
#Making a Copy of Original Data

data =data_shopping.copy()
#Lowering the down column names

data.columns = data.columns.str.lower()
#Calculating Missing Values

(data.isnull().sum()/len(data))*100
group_mean_weight = data.pivot_table(index = ["item_type"], values = "item_weight", aggfunc = [np.mean])
group_mean_weight
mean_weight = group_mean_weight.iloc[:,[0][0]]
mean_weight
# Function to impute Missing Value in item_weight column:



def missing_value(cols):

    item_type = cols[0]

    item_weight =cols[1]

    if pd.isnull(item_weight):

        if item_type == "Baking Goods":

            return 12.277

        elif item_type == "Breads":

            return 11.347

        elif item_type == "Breakfast":

            return 12.768

        elif item_type == "Canned":

            return 12.30

        elif item_type == "Dairy":

            return 13.42

        elif item_type == "Frozen Foods":

            return  12.867061

        elif item_type == "Fruits and Vegetables":

            return 13.224769

        elif item_type == "Hard Drinks":

            return 11.400328

        elif item_type == "Health and Hygiene":

            return 13.142314

        elif item_type == "Household":

            return 13.384736

        elif item_type == "Meat":

            return 12.817344

        elif item_type == "Others":

            return 13.853285

        elif item_type == "Seafood":

            return 12.552843

        elif item_type == "Snack Foods":

            return 12.987880

        elif item_type == "Soft Drinks":

            return 11.847460

        elif item_type == "Starchy Foods":

            return 13.690731

    return item_weight   

        

       

            

            

            

            
#Imputing the missing value by using defined function

data["item_weight"] = data[["item_type","item_weight"]].apply(missing_value, axis = 1)
data.head()
#FINDING FREQUENCY COUNT OF OUTLET TYPE

sns.countplot(data = data, x = "outlet_type",hue = "outlet_size")

plt.xticks(rotation =90)
# Function for Imputing Missing value in Outlet_Size column:



def impute_size(cols):

    size = cols[0]

    ot_type = cols[1]

    if pd.isnull(size):

        if ot_type == "Supermarket Type1":

            return "Small"

        elif ot_type == "Supermarket Type2":

            return "Medium"

        elif ot_type == "Grocery Store":

            return "Small"

        elif ot_type == "Supermarket Type3":

            return "Medium"

    return size    
#USING ABOVE DEFINED FUNCTION IMPUTE MISSING VALUES IN OUTLET SIZE COLUMNS

data["outlet_size"] = data[["outlet_size","outlet_type"]].apply(impute_size, axis = 1)
data["item_fat_content"].unique()
data["item_fat_content"] = data["item_fat_content"].str.replace("LF", "low fat").str.replace("reg", "regular").str.lower()
data["item_fat_content"].unique()
data.head()
mean_visibility = data.pivot_table(index = "item_identifier",  values = "item_visibility")
data.loc[(data["item_visibility"] == 0.0), "item_visibility"] = data.loc[(data["item_visibility"] == 0.0), "item_identifier"].apply(lambda x : mean_visibility.at[x, "item_visibility"])

                                                                                        
#understanding outlet_identifier depending on item_outlet_sales

import matplotlib.pyplot as plt

plt.figure(figsize = (10,5))

sns.barplot(data = data, x = "outlet_identifier", y= "item_outlet_sales")
#Understanding to ultet_establishment per year with respective to mean of each respective year item outlet sales

data.groupby("outlet_establishment_year")["item_outlet_sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
#Understanding to item_type per year with respective to mean of each respective year item outlet sales

data.groupby("item_type")["item_outlet_sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
#Understanding to outlet_type per year with respective to mean of each respective year item outlet sales

data.groupby("outlet_type")["item_outlet_sales"].mean().plot.bar()

plt.ylabel("Mean of Item outlet sales")
data.head()
cols = ['item_identifier', 'item_fat_content',

       'item_type', 'outlet_identifier',

       'outlet_establishment_year', 'outlet_size', 'outlet_location_type',

       'outlet_type']
#MAPPING EACH CATEGORICAL COLUMN WITH RESPECTIVE FREQUENCY OF THE VALUES IN THE COLUMNS

for i in cols:

    x  = data[i].value_counts().to_dict()

    data[i] = data[i].map(x)
#RESULTING DATASET AFTER CATEGORICAL VALUES CONVERTED TO NUMERICAL COLUMN

data.head()
#COPYING DATA 

new_data= data.copy()
#FINDING CORRELATION BETWEEN EACH COLUMNS BY USING HEATMAP

plt.figure(figsize = (10,5))

sns.heatmap(new_data.corr(), cmap = "RdYlGn", annot = True)
#REMOVING LESS CORRELATED COLUMNS 

new_data =new_data.drop(["item_weight","item_identifier", "item_type", "item_fat_content","outlet_location_type"], axis = 1)
new_data.head()
#CALCULATING THE SKEWNESS OF THE DATA

new_data.skew()
# Before Transformation

fig, ax = plt.subplots(4,2,figsize = (15,15))

sns.distplot(new_data["item_visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(new_data["item_mrp"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(new_data["outlet_identifier"], kde =True, ax=ax[1,0], color = "orange")

sns.distplot(new_data["outlet_type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(new_data["outlet_size"], kde =True, ax=ax[2,0], color = "magenta")

sns.distplot(new_data["outlet_establishment_year"], kde =True, ax=ax[3,0])

sns.distplot(new_data["item_outlet_sales"], kde =True, ax=ax[3,1])



new_data.columns
for i in new_data.columns:

    new_data[i] =np.log(new_data[i])
new_data.head()
# After Transformation

fig, ax = plt.subplots(4,2,figsize = (15,15))

sns.distplot(new_data["item_visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(new_data["item_mrp"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(new_data["outlet_identifier"], kde =True, ax=ax[1,0], color = "orange")

sns.distplot(new_data["outlet_type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(new_data["outlet_size"], kde =True, ax=ax[2,0], color = "magenta")

sns.distplot(new_data["outlet_establishment_year"], kde =True, ax=ax[3,0])

sns.distplot(new_data["item_outlet_sales"], kde =True, ax=ax[3,1])
new_data.skew()
#Independent Variables:

x = new_data.drop("item_outlet_sales", axis = 1) 



#Depenedent Variables 

y = new_data["item_outlet_sales"].values.reshape(-1,1)

#Splitting The data  into Train and Test Dataset:

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size =0.20, random_state = 3)
#Applying Linear Regression Model

from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_train, y_train)
#Prediction

y_pred = regressor.predict(x_test)
#Accuracy of Model (Apply R2_score)

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test, y_pred)
#Checking Root Mean Square error

from math import sqrt

rmse = sqrt(mean_squared_error(y_test,  y_pred))

rmse
#Residue of the Linear Regression Model 

residue_lr = y_test -y_pred
#Plotting Residual Plot

plt.scatter(y_test,residue_lr, c = "red")

plt.xlabel("residual")

plt.ylabel("y_test")

plt.axhline(y = 0)
#Importing LASSO AND RIDGE from sklearn library:

#Apply Lasso Regularization Technique

from sklearn.linear_model import Lasso, Ridge

ls = Lasso(alpha = 0.009)

ls.fit(x_train, y_train)

#prediction by LASSO model

ls_pred = ls.predict(x_test)
#Accuracy After Lasso(by R2_score)

r2_score(y_test,ls_pred)
#Root Mean Square Error

rmse_lasso = sqrt(mean_squared_error(y_test, ls_pred))

rmse_lasso
#Getting Lasso Coefficent

lasso_coeff = pd.Series(ls.coef_, index =x.columns) 
#Visualization of Coefficent after LASSO 

lasso_coeff.plot(kind = "bar")
#RESHAPING THE PREDICTED VALUES

ls_pred= ls_pred.reshape(-1,1)



ls_pred
#RESIDUE VALUE AFTER LASSO REGRESSION

residue = y_test - ls_pred
plt.scatter(y_test.reshape(-1,1),residue, c = "red")

plt.xlabel("residual")

plt.ylabel("y_test")

plt.axhline(y = 0)
#Ridge Regression

rr = Ridge(alpha = 0.009)

rr.fit(x_train, y_train)
#Prediction AFTER Ridge regression

rr_pred = rr.predict(x_test)
#Accuracy score check

r2_score(y_test, y_pred)
#RMSE

rmse_ridge = sqrt(mean_squared_error(y_test, rr_pred))

rmse_ridge
#residue after ridge

residue_rr = y_test-rr_pred
#plotting of residual graph after RIDGE REGRESSION

plt.scatter(y_test,residue_rr, c = "red")

plt.xlabel("residual")

plt.ylabel("y_test")

plt.axhline(y = 0)
test_data = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")
test_data.head()
test = test_data.copy()
#LOWERING THE COLUMNS NAMES 

test.columns = test.columns.str.lower()
#Calculating Missing Values

(test.isnull().sum()/len(test))*100
#calculating the mean of item_weight with respective to item_type

group_mean = test.pivot_table(index = ["item_type"], values = "item_weight", aggfunc = [np.mean])
group_mean
mean_weigh = group_mean.iloc[:,[0][0]]
mean_weigh
# Function to impute Missing Value in item_weight column:



def missing_value1(cols):

    item_type = cols[0]

    item_weight =cols[1]

    if pd.isnull(item_weight):

        if item_type == "Baking Goods":

            return 12.277

        elif item_type == "Breads":

            return 10.86

        elif item_type == "Breakfast":

            return  13.759603

        elif item_type == "Canned":

            return 12.393565

        elif item_type == "Dairy":

            return 12.955040

        elif item_type == "Frozen Foods":

            return  12.101543

        elif item_type == "Fruits and Vegetables":

            return 13.146659

        elif item_type == "Hard Drinks":

            return 11.844417

        elif item_type == "Health and Hygiene":

            return 13.216929

        elif item_type == "Household":

            return 13.270504

        elif item_type == "Meat":

            return 12.702148

        elif item_type == "Others":

            return 14.009725

        elif item_type == "Seafood":

            return 13.241136

        elif item_type == "Snack Foods":

            return 12.684256

        elif item_type == "Soft Drinks":

            return 11.691965

        elif item_type == "Starchy Foods":

            return 13.618247

    return item_weight  
#applying the above function to the dataset

test["item_weight"] = test[["item_type","item_weight"]].apply(missing_value1, axis = 1)
#Frequency count of the outlet_type 

sns.countplot(data = test, x = "outlet_type",hue = "outlet_size")

plt.xticks(rotation =90)
# Function for Imputing Missing value in Outlet_Size column:



def impute_size1(cols):

    size = cols[0]

    ot_type = cols[1]

    if pd.isnull(size):

        if ot_type == "Supermarket Type1":

            return "Small"

        elif ot_type == "Supermarket Type2":

            return "Medium"

        elif ot_type == "Grocery Store":

            return "Small"

        elif ot_type == "Supermarket Type3":

            return "Medium"

    return size    
test["outlet_size"] = test[["outlet_size","outlet_type"]].apply(impute_size1, axis = 1)
test["item_fat_content"].unique()
test["item_fat_content"] = test["item_fat_content"].str.replace("LF", "low fat").str.replace("reg", "regular").str.lower()
test["item_fat_content"].unique()
mean_item_visibility = test.pivot_table(index = "item_identifier",  values = "item_visibility")
mean_item_visibility.head()
test_d = test.copy()
columns = ['item_identifier', 'item_fat_content',

       'item_type', 'outlet_identifier',

       'outlet_establishment_year', 'outlet_size', 'outlet_location_type',

       'outlet_type']

for i in columns:

    x  = test_d[i].value_counts().to_dict()

    test_d[i] = test_d[i].map(x)
test_d.head()
new_test_data = test_d.copy()
new_test_data =new_test_data.drop(["item_weight","item_identifier", "item_type", "item_fat_content","outlet_location_type"], axis = 1)
new_test_data.head()
#CHECKING THE SKEWNESS OF THE TEST DATA

new_test_data.skew()
#VISUALIZING THE SKEWNESS OF THE DATASET

# Before Transformation

fig, ax = plt.subplots(3,2,figsize = (15,15))

sns.distplot(new_test_data["item_visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(new_test_data["item_mrp"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(new_test_data["outlet_identifier"], kde =True, ax=ax[1,0], color = "orange")

sns.distplot(new_test_data["outlet_type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(new_test_data["outlet_size"], kde =True, ax=ax[2,0], color = "magenta")

sns.distplot(new_test_data["outlet_establishment_year"], kde =True, ax=ax[2,1])



new_test_data.columns
#Applying the log transformation

for i in new_test_data.columns:

    new_test_data[i] =np.log(new_test_data[i]+1)
#After Transformation 

fig, ax = plt.subplots(3,2,figsize = (15,15))

sns.distplot(new_test_data["item_visibility"], kde =True, ax=ax[0,0], color = "red")

sns.distplot(new_test_data["item_mrp"], kde =True, ax=ax[0,1], color = "blue")

sns.distplot(new_test_data["outlet_identifier"], kde =True, ax=ax[1,0], color = "orange")

sns.distplot(new_test_data["outlet_type"], kde =True, ax=ax[1,1], color = "black")

sns.distplot(new_test_data["outlet_size"], kde =True, ax=ax[2,0], color = "magenta")

sns.distplot(new_test_data["outlet_establishment_year"], kde =True, ax=ax[2,1])

new_test_data.head()
#Skewness After Transformation:

new_test_data.skew()
item_outsale_pred = regressor.predict(new_test_data)
item_outsale_pred
#Performing inverse transformation

actual_item_outsale = np.exp(item_outsale_pred+1)
actual_item_outsale
#ADDING THE PREDICTED ITEM_OUTLET_SALE COLUMNS TO TEST DATA

test["item_outlet_sale"] = actual_item_outsale
test