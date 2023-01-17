import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns', 22)
file = '/kaggle/input/Sample - Superstore.xls'

sales = pd.read_excel(file)

sales
sales.columns = sales.columns.str.replace(' ', '_')
sales.dtypes
sales["Order_Date"] = sales["Order_Date"].astype("O")

sales["Ship_Date"] = sales["Ship_Date"].astype("O")

sales.isnull().sum()
if not None in sales["Country"]:

    print(True)

else:

    print(False)
sales["Product_Name"] = sales["Product_Name"].fillna(sales["Product_Name"].mode)
print(sales["Country"].value_counts())

print(sales["Product_Name"].value_counts())
plt.hist(sales["Category"])
sns.catplot(x="Category",y="Profit",data = sales)
sns.boxenplot(x="Category",y="Profit",data = sales)
sns.boxenplot(x="Region",y="Profit",data = sales)
sns.catplot(x="Region",y="Profit",data = sales)
# plt.figure(figsize=(16,14))

# sns.catplot(x="Sub-Category",y="Profit",data = sales)

fig_dims = (18, 16)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = "Sub_Category", y = "Profit", ax=ax, data=sales)
# fig, ax = plt.subplots(figsize=(18,15))

# sns.catplot(x='Sub-Category', y='Profit', data=sales, kind='swarm', ax=ax)

sns.catplot(x='Sub_Category', y='Profit', data=sales, height=7, aspect=2)
#One-way ANOVA fails with dataframe having completely unique values 

import statsmodels.formula.api as smf

model = smf.ols(formula='Profit~C(Sub_Category)',data=sales) 

results = model.fit()

print(results.summary())

#Ship_mode = 0.9..

#customer_name = .00467

#Customer_ID = .00467

#Segment = .407

#Country is unique

#City = .258

#State = 3.21e-60

#Region = 0.0489

#Product_id = 0.00

#Category = 3.47e-24

#Sub_category = 2.81e-181

#Product_name = LinAlgError: SVD did not converge

#ChiSquaretest

from scipy import stats

crosstab = pd.crosstab(sales["Discount"],sales["Profit"])

stats.chi2_contingency(crosstab)

#Postal_Code = 2.4057439656749844e-26

#Sales = 0.0

#Quantity = 0.0

#Discount = 0.0
plt.figure(figsize=(12,10))

plt.scatter(sales["Sales"],sales["Profit"]) 
fig_dims = (20, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxenplot(x='Quantity', y='Profit', data=sales)
fig_dims = (20, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxenplot(x='Discount', y='Profit', data=sales)
sales_df = sales.groupby("Customer_Name")

f=sales_df["Profit"].sum()

f.sort_values()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



for col in sales.columns.values:



       if sales[col].dtypes=='object':

            data=sales[col].append(sales[col])

            le.fit(data.values)

            sales[col]=le.transform(sales[col])

sales
sales.drop(columns=["Ship_Mode","Segment","Country","City"],inplace=True)

sales
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(sales))

print(z)

sales_o = sales[(z < 3).all(axis=1)]

sales_o.shape
from sklearn.model_selection import train_test_split

y = sales['Profit']

sales.drop(columns="Profit",inplace=True)





X_train,X_test,y_train,y_test = train_test_split(sales,y,test_size=0.3,random_state=24)

X_train
from sklearn.preprocessing import MinMaxScaler

minmaxscaling = MinMaxScaler()

X_train_scaled = minmaxscaling.fit_transform(X_train)

X_test_scaled = minmaxscaling.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=24)

rf.fit(X_train_scaled,y_train)

predicted = rf.predict(X_test_scaled)

predicted

comparision_df = pd.DataFrame({'actual':y_test,'predicted':predicted})

comparision_df
from sklearn.metrics import r2_score
rf.score(X_train_scaled,y_train)
rf.score(X_test_scaled,y_test)
r2_score(y_test,predicted)

#without outliers treatment
from sklearn.model_selection import train_test_split

y = sales_o['Profit']

sales_o.drop(columns="Profit",inplace=True)





X_train_o,X_test_o,y_train_o,y_test_o = train_test_split(sales_o,y,test_size=0.3,random_state=24)

X_train_o
from sklearn.preprocessing import MinMaxScaler

minmaxscaling = MinMaxScaler()

X_train_scaled_o = minmaxscaling.fit_transform(X_train_o)

X_test_scaled_o = minmaxscaling.transform(X_test_o)
from sklearn.ensemble import RandomForestRegressor

rf_o = RandomForestRegressor(random_state=24)

rf_o.fit(X_train_scaled_o,y_train_o)

predicted_o = rf_o.predict(X_test_scaled_o)

predicted_o

comparision_df_o = pd.DataFrame({'actual':y_test_o,'predicted':predicted_o})

comparision_df_o
r2_score(y_test_o,predicted_o)

#with outliers treatment
rf_o.score(X_train_scaled_o,y_train_o)
rf_o.score(X_test_scaled_o,y_test_o)