# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read the dataset

df = pd.read_csv('../input/superstore-data/superstore_dataset2011-2015.csv')
# without encoding error

df = pd.read_csv('../input/superstore-data/superstore_dataset2011-2015.csv',encoding = "ISO-8859-1")
df.columns
# Let's fetch the customer level details :



df_customer = df[['Customer ID','Customer Name', 'Segment']].drop_duplicates()
df_customer.head()
# Let's fetch details at Customer and Order Level : 



df_customer_order = df[['Customer ID','Order ID','Order Date', 'Ship Date', 'Ship Mode',]].drop_duplicates()
df_customer_order.head()
# Creating a new feature describing the time taken between placing of an order and 

# shipment of the same



df_customer_order['Order_to_Ship_Days'] = (pd.to_datetime(df_customer_order['Ship Date']) 

                                           - pd.to_datetime(df_customer_order['Order Date'])).dt.days
df_customer_order.head()
# Calculating average order-to-ship time(in days) at Customer level



df_customer_days = df_customer_order.groupby('Customer ID')['Order_to_Ship_Days'].mean()
# Saving this information as a Dataframe



df_customer_days = df_customer_days.to_frame()
# Set the column name correctly



df_customer_days.columns = ['Avg_Order_to_Ship_Days']
df_customer_days.head()
grp_customer = df.groupby(['Customer ID'])

df1 = grp_customer['Order ID','Sales'].agg({'Order ID':np.size,'Sales':np.sum})
# Creating a function which will do all the aggregation of metrics at Customer level : 



def agg_customer(x):

    d = []

    d.append(x['Order ID'].nunique())

    d.append(x['Sales'].sum())

    d.append(x['Shipping Cost'].sum())

    d.append(pd.to_datetime(x['Order Date']).min())

    d.append(pd.to_datetime(x['Order Date']).max())

    d.append(x['City'].nunique())

    return pd.Series(d, index=['#Purchases','Total_Sales','Total_Cost','First_Purchase_Date','Latest_Purchase_Date','Location_Count'])



df_agg = df.groupby('Customer ID').apply(agg_customer)
# Checking the names of the new aggregated dataframe



df_agg.columns
# Creating new features on top of the aggregated dataframe already created above : 



from datetime import datetime

df_agg['Duration'] = (df_agg['Latest_Purchase_Date'] - df_agg['First_Purchase_Date']).dt.days

df_agg['Frequency'] = df_agg['Duration']/df_agg['#Purchases']

df_agg['Days_Since_Last_Purchase'] = df_agg['Latest_Purchase_Date'].apply(lambda x: datetime.strptime('2016-01-01', "%Y-%m-%d") - x).dt.days

df_agg['Sales_Contribution'] = (df_agg['Total_Sales']/df_agg['Total_Sales'].sum())

df_agg['Average_Basket_Value'] = df_agg['Total_Sales']/df_agg['#Purchases']

df_agg['CLTV'] = df_agg['Total_Sales'] - df_agg['Total_Cost']

df_agg.sort_values(by="Latest_Purchase_Date",ascending = False).head()
df['Product ID'].nunique()
#  We would like to understand which products are high priced ones, medium and low priced ones 



df_prod = df.groupby(['Product ID'])['Quantity','Sales'].agg(np.sum)

df_prod['Average_Price_Point'] = df_prod['Sales']/df_prod['Quantity']

df_prod['Price_Point_Perc_Rank'] = df_prod['Average_Price_Point'].rank(pct=True)

df_prod['Ticket_Type'] = df_prod['Price_Point_Perc_Rank'].apply(lambda x: 'High' if x>0.7 else ('Medium' if (x>0.4 and x<=0.7) else 'Low'))

df_prod.sort_values(by='Price_Point_Perc_Rank',ascending=False).head()
# Check count of rows by Ticket Type :



df_prod['Ticket_Type'].value_counts()
# Here we join the base table with the Product level table to bring in

# the Ticket_Type column as a part of the base table



df_with_ticket = pd.merge(df,df_prod,left_on='Product ID',right_on=df_prod.index,how='inner')
df_with_ticket.shape
# Here we create a pivot table with Customer ID in rows

# and Ticket type in columns

# and count of rows as values

df_tickettype_pivot = pd.pivot_table(df_with_ticket[['Customer ID','Ticket_Type']],index=["Customer ID"],

               columns=["Ticket_Type"],aggfunc=[np.size])
# Fetch column names from level 2 of the Multi-Index



df_tickettype_pivot_columns = df_tickettype_pivot.columns.get_level_values(1).tolist()
df_tickettype_pivot.head()

# Merge customer aggregated data with Ticket type data 

df_customer_tickettype = pd.merge(df_agg,df_tickettype_pivot,on = 'Customer ID',how='inner')
df_agg.columns.tolist()
df_customer_tickettype.columns = df_agg.columns.tolist() + df_tickettype_pivot_columns
df_customer_tickettype.head()
# bringing in customer meta-data

df_total_customer = pd.merge(df_customer_tickettype,df_customer,on='Customer ID',how='inner')
df_total_customer.head()
# Add average Order-to-Ship days to the dataset



df_tot_cust_order_final = pd.merge(df_total_customer,df_customer_days,on='Customer ID',how='inner')
df_tot_cust_order_final.head()
# Read returns data

df_returns = pd.read_csv('../input/product-returns/Returned orders.csv')
df_returns.columns
# connect with product table to categorize the returned products : 

df_returns_tickettype = pd.merge(df_returns,df_prod,on='Product ID',how='inner')
df_returns_tickettype.columns
df_cust_prd_tktype = pd.pivot_table(df_returns_tickettype[['Customer ID','Ticket_Type']],index=["Customer ID"],

               columns=["Ticket_Type"],aggfunc=[np.size])
df_cust_prd_tktype.head()
df_cust_prd_tktype.columns = ['High_Returns','Medium_Returns','Low_Returns']
df_cust_prd_tktype.head()
df_cust_prd_tktype.fillna(0,inplace = True)
df_cust_prd_tktype.head()
# Merge with the final aggregated table created before to develop the final dataset to work on : 



data_final = pd.merge(df_tot_cust_order_final,df_cust_prd_tktype,on='Customer ID',how='left')
data_final.shape
data_final.head()
data_final = data_final.fillna({'High':0,

                                'Medium':0,

                                'Low':0,

                                'High_Returns':0,

                                'Medium_Returns':0,

                                'Low_Returns':0})
data_final.info()
# one hot encode the segment column



df_segment_ohe = pd.get_dummies(data_final['Segment'])
df_segment_ohe.head()
df_clean = pd.concat([data_final,df_segment_ohe],axis = 1)
df_clean.head()
# drop columns which are of text type or of no use



df_clean.drop(['Customer ID','Customer Name','First_Purchase_Date','Latest_Purchase_Date','Segment'

               ,'Total_Sales','Total_Cost']

                , axis=1,inplace = True)
df_clean.head()
df_clean.info()
# Let's split the feature and response variables : 

y = df_clean['CLTV']

X = df_clean[df_clean.columns.difference(['CLTV'])]
# Let's split the data now



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# Let's scale the data now



#Standardization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_std=sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)
corr = X.corr()

corr.style.background_gradient(cmap='coolwarm')
import matplotlib.pyplot as plt

import seaborn as sns



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation of features')

# Draw the heatmap using seaborn

#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)

sns.heatmap(X.corr(),linewidths=0.25,vmax=1.0, square=True, cmap="coolwarm", linecolor='k', annot=True)
# Wow! That was big! It is better to find the most correlated features

most_corr_features = corr.index[abs(corr["#Purchases"])>0.6]

plt.figure(figsize=(15,15))

sns.heatmap(X[most_corr_features].corr(),annot=True)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_std,y_train)
lr.coef_
from sklearn.metrics import r2_score,mean_squared_error
y_pred = lr.predict(X_test_std)
r2_score(y_pred,y_test)
mean_squared_error(y_pred,y_test)
np.sqrt(mean_squared_error(y_pred,y_test))
X_train.columns
from sklearn.linear_model import Lasso
# create a lasso regressor

lasso = Lasso(alpha=0.2, normalize=True)



# Fit the regressor to the data

lasso.fit(X_train_std,y_train)



# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)
df_coeffs = pd.DataFrame()

df_coeffs['feature_names'] = X_train.columns
df_coeffs['values'] = lasso_coef
df_coeffs
from sklearn.feature_selection import f_regression

ffs = f_regression(X_train,y_train)
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

lreg = LinearRegression()

rfe = RFE(lreg,5)

rfe = rfe.fit(X_train_std,y_train)
rfe.ranking_
df_coeffs['ranking'] = rfe.ranking_
df_coeffs
new_cols = df_coeffs[df_coeffs['ranking'] == 1]['feature_names']
X_new = X[new_cols]
# Let's split the data now

from sklearn.model_selection import train_test_split

X_new_train,X_new_test,y_new_train,y_new_test=train_test_split(X_new,y,test_size=0.2)
# Let's scale the data now



#Standardization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_new_train_std=sc.fit_transform(X_new_train)

X_new_test_std = sc.transform(X_new_test)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_new_train_std,y_new_train)
y_new_pred = lr.predict(X_new_test_std)
r2_score(y_new_pred,y_new_test)
mean_squared_error(y_new_pred,y_new_test)
np.sqrt(mean_squared_error(y_new_pred,y_new_test))