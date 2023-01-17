# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split  

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
train = pd.read_csv("../input/black-friday/train.csv")

train.head(10)
train["Product_Category_1"].isna().sum(), train["Product_Category_2"].isna().sum(), train["Product_Category_3"].isna().sum()
train["Product_Category_3"].isna().sum()/550068*100, train["Product_Category_2"].isna().sum()/550068*100, train["Product_Category_1"].isna().sum()/550068*100
### it looks like product category 3 has more null values which is close to 70 percent of the data, so we delete the feature.

### keep product category 2 and 1.
train.drop(["Product_Category_3"],  axis=1, inplace=True)
train.head(10)
train["Product_Category_2"].fillna(train["Product_Category_2"].median(), inplace = True)
train["Product_Category_2"].isna().sum(), train["Product_Category_1"].isna().sum()
train.Age.unique()
Label_Encoder = LabelEncoder()



train["Gender"] = Label_Encoder.fit_transform(train["Gender"])

train["City_Category"] = Label_Encoder.fit_transform(train["City_Category"])

train["Age_Category"] = Label_Encoder.fit_transform(train["Age"]) #then we drop the Age variable. lets delete after visualization
# # we can also use fn to create a category of the above variables 



# def Age_Category(Age):

#     if Age == '0-17':

#         return 1

#     elif Age == '18-25':

#         return 2

#     elif Age == '26-35':

#         return 3

#     elif Age == '36-45':

#         return 4

#     elif Age == '46-50':

#         return 5

#     elif Age == '51-55':

#         return 6

#     else :

#         return 7

# train["Age_Category"] = train["Age"].apply(Age_Category)        # now we can delete Age .. but let us do that after visualization
train.head()
train.Gender.unique(), train.Age.unique(),   train.City_Category.unique() 



    # make sure no missing values    

    # also make sure 0 is for gender and 1 is form menA=0, B = 1, C = 2 , 
train.head(5)     
sns.boxplot(x =train["Gender"], y = train["Purchase"])
sns.boxplot(x =train["City_Category"], y = train["Purchase"], hue=train["Gender"])
fig, ax = plt.subplots(figsize = (12,8))

    

ax = sns.boxplot(x =train["Age"], y = train["Purchase"], hue= train["Gender"], palette='bright')
train.head(10)
sns.violinplot("Marital_Status", y="Purchase", data = train)
fig = plt.subplots(figsize = (12, 8))

sns.boxplot(x="Occupation", y = "Purchase", hue="Gender", data = train)
train["Marital_Status"].value_counts()
fig = plt.subplots(figsize= (12,8))

train["Purchase"].value_counts().hist()
# no significnt change in purchasing behavior between married and non-married.
train.Purchase[1:100].sort_values(ascending = False)
train.head(2)
forget = train[["User_ID", "Product_ID", "Age"]]



train.drop(forget, axis =1, inplace = True )
train.head()
train.Stay_In_Current_City_Years.unique()
def stay(Stay_In_Current_City_Years):

        if Stay_In_Current_City_Years == '4+':

            return 4

        else:

            return Stay_In_Current_City_Years

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].apply(stay).astype(int)    
train.dtypes
fig, ax = plt.subplots(figsize = (15,8))

ax = sns.heatmap(train.corr(), annot=True, cmap="YlGnBu" )
X = train.drop("Purchase", axis=1)

y = train["Purchase"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
train.dtypes
#model

%time

#cutting down on the max number of samples each estimator can see improves training time



# rf_regressor = RandomForestRegressor(n_estimators=100)





rf_regressor = RandomForestRegressor(n_jobs=-1, 

                              random_state=42)



rf_regressor.fit(X_train, y_train)
rf_regressor.score(X_test, y_test)
y_pred = rf_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)

r2
MAE = mean_absolute_error(y_test, y_pred)

MAE
MSE = mean_squared_error(y_test, y_pred)

MSE
fig, ax = plt.subplots(figsize=(12,5))

ax = plt.scatter(y_test, y_pred, c="blue")
# 1. RandomSearchCV 



grid =  {"n_estimators": [10,50,100],

       "max_depth": [None,10,20,30,40,50,],

       "max_features": ["auto", "sqrt"],

       "min_samples_leaf": [2,10,15],

       "min_samples_split": [2,5,20]}

randomsearchCV = RandomizedSearchCV(rf_regressor, param_distributions = grid, n_iter = 5, cv=5,  verbose = True, n_jobs=2 )

     

         #"Verbose is a general programming term for produce lots of logging output. You can think of it as asking the program to "tell me everything about what you are doing all the time". 

          #Just set it to true and see what happens."

            

            

        # if you specify n_jobs to -1, it will use all cores in the CPU (100% CPU). 

          #If it is set to 1 or 2, it will use one or two cores only 
%time



randomsearchCV.fit(X_train, y_train)

randomsearchCV.best_params_
rf_regressor_tune = RandomForestRegressor(n_estimators=100, max_depth = 40, max_features = 'auto', min_samples_leaf =10,

                                     min_samples_split=2 )
rf_regressor_tune.fit(X_train, y_train) 
y_pred_tune = rf_regressor_tune.predict(X_test)

y_pred_tune
# rf_regressor.score(X_test, y_test)



r2_tune = r2_score(y_test, y_pred_tune)  

r2_tune
MAE_tune = mean_absolute_error(y_test, y_pred_tune)

MAE_tune
MSE_tune = mean_squared_error(y_test, y_pred_tune)

MSE_tune
# compare prediction before and after Tunning



compare = {"R^2_score":[r2_tune, r2],

            "Mean Squared Error": [MSE_tune, MSE],

            "Mean Absolute Error": [MAE_tune, MAE]}



compare
Compare = pd.DataFrame(compare, index=[["After_tune", "Before Tune"]])

Compare
fig, ax = plt.subplots(figsize=(12,5))

ax = plt.scatter(y_test, y_pred_tune, c="blue")
test = pd.read_csv("../input/black-friday/test.csv")
test.head()
test["Product_Category_1"].isna().sum(), test["Product_Category_2"].isna().sum(), test["Product_Category_3"].isna().sum()
### Lets do same data preprocessing as we do in the train data

test["Product_Category_1"].isna().sum()/550068*100, test["Product_Category_2"].isna().sum()/550068*100, test["Product_Category_3"].isna().sum()/550068*100
test["Product_Category_2"].fillna(test["Product_Category_2"].median(), inplace = True)

test["Product_Category_3"].fillna(test["Product_Category_3"].median(), inplace = True)
test.head()
Label_Encoder = LabelEncoder()





test["Gender"] = Label_Encoder.fit_transform(test["Gender"])

test["City_Category"] = Label_Encoder.fit_transform(test["City_Category"])

test["Age_Category"] = Label_Encoder.fit_transform(test["Age"])
test["Stay_In_Current_City_Years"].unique()
def stay(Stay_In_Current_City_Years):

        if Stay_In_Current_City_Years == '4+':

            return 4

        else:

            return Stay_In_Current_City_Years

test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].apply(stay).astype(int) 
forget = test[["User_ID", "Product_ID", "Age"]]



test.drop(forget, axis =1, inplace = True )
test.head()
test.drop("Product_Category_3", axis = 1, inplace = True)
test_pred = rf_regressor_tune.predict(test)

test_pred
test.columns
len(y_pred)
# test = pd.read_csv("../input/black-friday/test.csv")
Prediction = pd.DataFrame()

Prediction["User_ID"] = test["User_ID"]

Prediction["Purchase Prediction"] = test_pred
Prediction
print(rf_regressor_tune.feature_importances_)
test.head(1)
# test.drop(["User_ID", "Product_ID", "Age"], axis = 1, inplace=True)
columns = pd.DataFrame({"Features": test.columns, 

                        "Feature Importance" :rf_regressor_tune.feature_importances_})
columns.sort_values("Feature Importance", ascending = False).reset_index(drop=True)
sns.barplot(y="Features", x = "Feature Importance", data = columns)