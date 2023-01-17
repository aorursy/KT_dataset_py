import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
filename = "../input/black-friday/train.csv"

train=  pd.read_csv(filename)

train.head()
test= pd.read_csv("../input/black-friday/test.csv")
test.head()
train.describe()
# Checking for null values
train['Product_Category_1'].isna().mean()*100, train['Product_Category_2'].isna().mean()*100, train['Product_Category_3'].isna().mean()*100
# droping Product_Category_3 column
train.drop(["Product_Category_3"],  axis=1, inplace=True)

train.columns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Purchase, bins = 25)
plt.xlabel("Amount spent in Purchase")
plt.ylabel("Number of Buyers")
plt.title("Purchase amount Distribution")
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
sns.countplot(train.Marital_Status)
sns.countplot(train.Product_Category_1)
plt.xticks()
sns.countplot(train.Product_Category_2)
plt.xticks(rotation=90)
corr = numeric_features.corr()

#correlation matrix
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr,  annot=True,annot_kws={'size': 15})
sns.countplot(train.Gender)
sns.countplot(train.Age)
sns.countplot(train.City_Category)
sns.countplot(train.Stay_In_Current_City_Years)
marital_status_pivot= train.pivot_table(index='Marital_Status',values='Purchase', aggfunc=np.mean)
marital_status_pivot
marital_status_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Marital_Status")
plt.ylabel("Purchase")
plt.title("Marital_Status and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()
Product_category_1_pivot = train.pivot_table(index='Product_Category_1', values="Purchase", aggfunc=np.mean)
Product_category_1_pivot
Product_category_1_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Marital_Status")
plt.ylabel("Purchase")
plt.title("Marital_Status and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()
Product_category_2_pivot = train.pivot_table(index='Product_Category_2', values="Purchase", aggfunc=np.mean)

Product_category_2_pivot.plot(kind='bar', color='brown',figsize=(12,7))
plt.xlabel("Product_Category_2")
plt.ylabel("Purchase")
plt.title("Product_Category_2 and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()
gender_pivot = train.pivot_table(index='Gender', values="Purchase", aggfunc=np.mean)
gender_pivot
gender_pivot.plot(kind='bar', color='orange',figsize=(12,7))
plt.xlabel("Gender")
plt.ylabel("Purchase")
plt.title("Gender and Purchase Analysis " "AVERAGE")
plt.xticks(rotation=0)
plt.show()
age_pivot = train.pivot_table(index='Age', values="Purchase", aggfunc=np.sum)
age_pivot
age_pivot.plot(kind='bar', color='pink',figsize=(12,7))
plt.xlabel("Age")
plt.ylabel("Purchase")
plt.title("Age and Purchase Analysis " "AVERAGE")
plt.xticks(rotation=0)
plt.show()
city_pivot = train.pivot_table(index='City_Category', values="Purchase", aggfunc=np.mean)
city_pivot
city_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("City_Category")
plt.ylabel("Purchase")
plt.title("City_Category and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()

Stay_In_Current_City_Years_pivot = train.pivot_table(index='Stay_In_Current_City_Years', values="Purchase", aggfunc=np.mean)
Stay_In_Current_City_Years_pivot
Stay_In_Current_City_Years_pivot.plot(kind='bar', color='red',figsize=(12,7))
plt.xlabel("Stay_in_Current_City_Years")
plt.ylabel("Purchase")
plt.title("Stay_in_Current_City_Years and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()
test.head()
# Join Train and Test Dataset
train['source']='train'
test['source']='test'

df = pd.concat([train,test], ignore_index = True, sort = False)

print(train.shape, test.shape, df.shape)
test.drop(["Product_Category_3"],  axis=1, inplace=True)
df.drop(["Product_Category_3"],  axis=1, inplace=True)
print(train.shape, test.shape, df.shape)
#Check the percentage of null values per variable
df.isnull().mean()*100
# Replacing Null Values in Product_Category_2 with the median of the column
df["Product_Category_2"].fillna(train["Product_Category_2"].median(), inplace = True)

#Get index of all columns with product_category_1 equal 19 or 20 from train

ind = df.index[(df.Product_Category_1.isin([19,20])) & (df.source == "train")]
df = df.drop(ind)
df.shape
df.dtypes
#Filter categorical variables and get dataframe will all strings columns names except Item_identfier and outlet_identifier
category_cols = df.select_dtypes(include=['object']).columns.drop(["source"])
#Print frequency of categories
for col in category_cols:
    #Number of times each value appears in the column
    frequency = df[col].value_counts()
    print("\nThis is the frequency distribution for " + col + ":")
    print(frequency)
gender_dict = {'F':0, 'M':1}
df["Gender"] = df["Gender"].apply(lambda x: gender_dict[x])

df["Gender"].value_counts()
age_dict={'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
df['Age']=df['Age'].apply(lambda x:age_dict[x])
df['Age'].value_counts()
city={'A':0,'B':1,'C':2}
df['City_Category']=df['City_Category'].apply(lambda x: city[x])
df['City_Category'].value_counts()
def stay(Stay_In_Current_City_Years):
        if Stay_In_Current_City_Years == '4+':
            return 4
        else:
            return Stay_In_Current_City_Years
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].apply(stay).astype(int) 
#Divide into test and train:
train = df.loc[df['source']=="train"]
test = df.loc[df['source']=="test"]

#Drop unnecessary columns:
test.drop(['source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_clean.csv",index=False)
test.to_csv("test_clean.csv",index=False)
train= pd.read_csv('train_clean.csv')
train.head()
test= pd.read_csv('test_clean.csv')
test.head()
X = train.drop(['Product_ID','User_ID','Purchase'], axis=1)
y = train["Purchase"]
# splitting train and test set
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)
#model
%time



rf_regressor = RandomForestRegressor(n_jobs=-1, 
                              random_state=42)

rf_regressor.fit(X_train, y_train)

rf_regressor.score(X_test, y_test)
y_pred = rf_regressor.predict(X_test)
R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
print("R2_Score_tune: {}\n Mean_absolute_error_: {}\n Mean_Square_error_tune: {}".format(R2, MAE, MSE))
fig, ax = plt.subplots(figsize=(12,5))
ax = plt.scatter(y_test, y_pred, c="brown")
# 1.RandomSearchCV 

grid =  {"n_estimators": [10,50,100],
       "max_depth": [None,10,20,30,40,50,],
       "max_features": ["auto", "sqrt"],
       "min_samples_leaf": [2,10,15],
       "min_samples_split": [2,5,20]}
randomsearchCV = RandomizedSearchCV(rf_regressor, param_distributions = grid, n_iter = 5, cv=5,  verbose = True, n_jobs=-1)
%time

randomsearchCV.fit(X_train, y_train)
randomsearchCV.best_params_
rf_regressor_tune = RandomForestRegressor(n_estimators=100, max_depth = 40, max_features = 'auto', min_samples_leaf =15,
                                     min_samples_split=5 )
rf_regressor_tune.fit(X_train, y_train) 
y_pred_tune = rf_regressor_tune.predict(X_test)

R2_tune= r2_score(y_test, y_pred_tune)
MAE_tune= mean_absolute_error(y_test, y_pred_tune)
MSE_tune= mean_squared_error(y_test, y_pred_tune)
print("R2_Score_tune: {}\n Mean_absolute_error_: {}\n Mean_Square_error_tune: {}".format(R2_tune, MAE_tune, MSE_tune))
# compare prediction before and after Tunning

compare = {"R^2_score":[R2_tune, R2],
            "Mean Squared Error": [MSE_tune, MSE],
            "Mean Absolute Error": [MAE_tune, MAE]}


Compare = pd.DataFrame(compare, index=[["After_tune", "Before Tune"]])
Compare

fig, ax = plt.subplots(figsize=(12,5))
ax = plt.scatter(y_test, y_pred_tune, c="brown")
test.head()
# No. of features used to train the model must match with the input
# Droppping User_id , Product_id and purchase columns
predicted= test[['User_ID','Product_ID']]
test =test.drop(['User_ID','Product_ID','Purchase'],axis=1)
test.head()
test_pred = rf_regressor_tune.predict(test)
test_pred
predicted['Predicted_Purchase']=test_pred
predicted.head()
#saving calculated purchase in a csv file
predicted.to_csv("predict.csv",index=False)
