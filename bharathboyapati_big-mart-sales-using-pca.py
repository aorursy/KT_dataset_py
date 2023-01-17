import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Train = pd.read_csv("../input/Train.csv")

Test =pd.read_csv("../input/Test.csv")
Train.head()
Train.info()
#Missing values imputation
Train.isnull().sum().sort_values(ascending = False)
Train.Outlet_Size.value_counts()

plt.subplots(figsize=(10,4))

sns.countplot(Train['Outlet_Size'])
#So filling the missing values with medium

Train["Outlet_Size"] = Train["Outlet_Size"].fillna('Medium')
#Item weight should be filled with mean since it a continuous value

Train['Item_Weight']=Train['Item_Weight'].fillna(Train['Item_Weight'].mean())
#make sure that all the missing values are imputed

Train.isnull().sum().any()
num_col = Train._get_numeric_data()

num_col.head()
#so here 'Outlet_Establishment_Year' is basically not an integer so convert it back to categorical

Train.Outlet_Establishment_Year = pd.Categorical(Train.Outlet_Establishment_Year)
num_cols=Train._get_numeric_data()

num_cols.columns


# Writing a function to find percentiles, min and max values of the attributes

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_cols.apply(lambda x: var_summary(x)).T
sns.boxplot(num_cols['Item_Outlet_Sales'])
#so There are outliers lets remove them

num_cols['Item_Outlet_Sales']= num_cols['Item_Outlet_Sales'].clip_upper(num_cols['Item_Outlet_Sales'].quantile(0.95))
sns.boxplot(num_cols['Item_Visibility'])
#finding correlation for numaric columns with respect to Item_Outlet_Sales

corr = num_col.corr()["Item_Outlet_Sales"]

corr[np.argsort(corr,axis=1)].sort_values(ascending = False)
#using heat ma

cm =num_col.corr()

sns.set(font_scale=1.35)

f, ax = plt.subplots(figsize=(10,10))

hm=sns.heatmap(cm, annot = True, vmax =.8)
#Damn thre is one attribute which is negatively correlated so we need to kick that out.

num_cols =num_col.drop(["Item_Visibility"], axis =1)

#lets check multi correlation using VIF(Varience Inflation Factor)

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
# Performing OLS to know the significant attributes

lm = smf.ols('Item_Outlet_Sales ~ Item_MRP+Item_Weight', num_cols).fit()

lm.summary()
num_cols['intercept'] = lm.params[0]
for i in range(3):

    print (vif(num_cols[['Item_MRP','Item_Weight','intercept']].as_matrix(), i))
np.diag(np.linalg.inv(num_cols[['Item_MRP','Item_Weight','intercept']].corr().as_matrix()),0)
Final_Num_Cols = num_cols.drop(["Item_Weight"],axis =1)
Final_Num_Cols.columns
cat_col = Train.drop(['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales'],axis = 1)

cat_col.columns
#Item_Identifier is just like seriel number so just ignore it, i mean drop it.

cat_cols =cat_col.drop(['Item_Identifier'], axis =1)
plt.subplots(figsize=(10,4))

sns.countplot(cat_cols['Item_Fat_Content'])
plt.subplots(figsize=(23,4))

sns.countplot(cat_cols.Item_Type)
plt.subplots(figsize=(10,4))

sns.countplot(cat_cols.Outlet_Identifier)
plt.subplots(figsize=(10,4))

sns.countplot(cat_cols.Outlet_Size)
plt.subplots(figsize=(10,4))

sns.countplot(Train.Outlet_Establishment_Year)
plt.subplots(figsize=(10,4))

sns.countplot(cat_cols.Outlet_Type)
plt.subplots(figsize=(10,4))

sns.countplot(cat_cols.Outlet_Location_Type)
plt.subplots(figsize = (10,4))

sns.barplot(x = cat_cols['Item_Fat_Content'], y= num_cols['Item_Outlet_Sales'])
plt.subplots(figsize = (25,4))

sns.barplot(x = cat_cols['Item_Type'], y= num_cols['Item_Outlet_Sales'])
plt.subplots(figsize = (15,4))

sns.barplot(y = cat_cols['Outlet_Identifier'], x= num_cols['Item_Outlet_Sales'])


plt.subplots(figsize = (15,4))

sns.barplot(y = cat_cols['Outlet_Establishment_Year'], x= num_cols['Item_Outlet_Sales'])
plt.subplots(figsize = (15,4))

sns.barplot(y = cat_cols['Outlet_Size'], x= num_cols['Item_Outlet_Sales'])
plt.subplots(figsize = (15,4))

sns.barplot(y = cat_cols['Outlet_Location_Type'], x= num_cols['Item_Outlet_Sales'])
plt.subplots(figsize = (15,4))

sns.barplot(y = cat_cols['Outlet_Type'], x= num_cols['Item_Outlet_Sales'])
cat_cols.columns
# we should add Item_Outlet_Sales as categorical attributes does not have target.

categorical_col =pd.concat([cat_cols,Final_Num_Cols.Item_Outlet_Sales],axis=1)

categorical_col.columns
##Now we need to do stats model.api:



import statsmodels.api as sm

import statsmodels.formula.api as smf
lm1 = smf.ols('Item_Outlet_Sales ~Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Size+Outlet_Location_Type+Outlet_Type', categorical_col).fit()

lm1.summary()
import scipy.stats as stats
cat_cols.Item_Fat_Content.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Fat_Content=="Low Fat"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Fat_Content=="Regular"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Fat_Content=="LF"]

s4 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Fat_Content=="reg"]

s5 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Fat_Content=="low fat"]
stats.f_oneway(s1, s2, s3, s4,s5)
cat_cols.Item_Type.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Fruits and Vegetables"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Snack Foods"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Household"]

s4 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Frozen Foods"]

s5 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Dairy"]

s6 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Canned"]

s7 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Baking Goods"]

s8 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Health and Hygiene"]

s9 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Soft Drinks"]

s10 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Meat"]

s11 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Breads"]

s12 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Hard Drinks"]

s13 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Others"]

s14 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Starchy Foods"]

s15 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Breakfast"]

s16 = categorical_col.Item_Outlet_Sales[categorical_col.Item_Type=="Seafood"]
stats.f_oneway(s1, s2, s3, s4,s5,s6,s7,s8,s9,s9,s10,s11,s12,s13,s14,s15,s16)
cat_cols.Outlet_Identifier.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT027"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT013"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT046"]

s4 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT035"]

s5 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT049"]

s6 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT045"]

s7 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT018"]

s8 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT017"]

s9 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT010"]

s10 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Identifier=="OUT019"]
stats.f_oneway(s1, s2, s3, s4,s5,s6,s7,s8,s9,s9,s10)
cat_cols.Outlet_Size.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Size=="Medium"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Size=="Small"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Size=="High"]

stats.f_oneway(s1, s2, s3)
cat_cols.Outlet_Location_Type.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Location_Type=="Tier 3"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Location_Type=="Tier 2"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Location_Type=="Tier 1"]

stats.f_oneway(s1, s2, s3)
cat_cols.Outlet_Type.value_counts()
s1 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Type=="Supermarket Type1"]

s2 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Type=="Grocery Store"]

s3 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Type=="Supermarket Type3"]

s4 = categorical_col.Item_Outlet_Sales[categorical_col.Outlet_Type=="Supermarket Type2"]

stats.f_oneway(s1, s2, s3, s4)
Final_Cat_Cols = cat_cols.drop(['Item_Fat_Content', 'Outlet_Identifier'

       ],axis =1)
Final_Cat_Cols.columns
#now lets do dummies

dummies_concat =  pd.get_dummies(Final_Cat_Cols, columns =['Outlet_Size', 'Outlet_Location_Type',

       'Outlet_Type','Item_Type','Outlet_Establishment_Year'],drop_first =True)
#As we completed preprocessing for train now lets concat numaric and categorical attrebutes

final = pd.concat([Final_Num_Cols,dummies_concat], axis =1)
final.isnull().sum().sort_values(ascending = False)
final.columns
Final = final.sample(n = 4260, random_state = 123)

Final.head(4)
Final1x = Final.drop(['Item_Outlet_Sales'], axis= 1)

Final1y = Final.Item_Outlet_Sales
Final2 = final.drop(Final.index)

Final2.info()
Final2x = Final2.drop(['Item_Outlet_Sales'], axis= 1)

Final2y = Final2.Item_Outlet_Sales
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(

        Final1x,

        Final1y,

        test_size=0.20,

        random_state=123)
print (len(X_train), len(X_test))
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        Final2x,

        Final2y,

        test_size=0.20,

        random_state=123)
y_pred = linreg.predict(X_test)

print(y_pred.mean())
from sklearn import metrics

metrics.r2_score(Y_test, y_pred)
Residuals = Y_test - y_pred

sns.distplot(Residuals)
#there is an assumption that residuals must be normally distributed and we successfully achieved that
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))

rmse
Test.head()
Test.isnull().sum().sort_values(ascending= False)
Test.info()
Test.Outlet_Size.value_counts()
#So filling the missing values with medium

Test["Outlet_Size"] = Test["Outlet_Size"].fillna('Medium')
#Item weight should be filled with mean since it a continuous value

Test['Item_Weight']=Test['Item_Weight'].fillna(Train['Item_Weight'].mean())
Test.isnull().sum().any()
num_data = Test._get_numeric_data()

num_data.head()
Test.Outlet_Establishment_Year = pd.Categorical(Test.Outlet_Establishment_Year)
# Writing a function to find percentiles, min and max values of the attributes

def var_summary(x):

    return pd.Series([round(x.mean(),2), round(x.median(),2), round(x.min(),2), round(x.quantile(0.01),2), round(x.quantile(0.05),2), round(x.quantile(0.10),2),round(x.quantile(0.25),2),round(x.quantile(0.50),2),round(x.quantile(0.75),2), round(x.quantile(0.90),2),round(x.quantile(0.95),2),round(x.quantile(0.99),2),round(x.max(),2)], 

                  index=['MEAN','MEDIAN', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



#applying the above method to all numeric data

num_data.apply(lambda x: var_summary(x)).T
num_colss = num_data.drop(['Item_Weight',"Item_Visibility",'Outlet_Establishment_Year'],axis=1)

num_colss.columns
test_cat= num_colss
cat_data = Test.drop(['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',

       'Item_MRP', 'Outlet_Identifier'],axis =1)

cat_data.columns
cat_data.info()
all_dummies=pd.concat([Final_Cat_Cols,cat_data],axis=0)
#now lets do dummies

dummies_concat =  pd.get_dummies(all_dummies, columns =['Outlet_Size', 'Outlet_Location_Type',

       'Outlet_Type','Item_Type','Outlet_Establishment_Year'],drop_first =True)
tt = dummies_concat[0:8523:]

test = dummies_concat[8524::]
test.info()
test.isnull().sum().sort_values(ascending= False)
#As we completed preprocessing for train now lets concat numaric and categorical attrebutes

final1 = pd.concat([test_cat,test], axis =1)

final1.columns