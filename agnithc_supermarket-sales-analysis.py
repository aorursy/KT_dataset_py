# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/supermarket-sales/supermarket_sales - Sheet1.csv",parse_dates=['Date'])
df.head()
#import dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#changing the date format
df['Date']=pd.to_datetime(df.Date)
df.head()
#extracting the date in the dataset
def extract_date(df,column):
    df[column+"_month"]=df[column].apply(lambda x:x.month)
extract_date(df,'Date')
df.head()
df['Date_month']= df[['Date_month']].astype("str")
df['Date_month'].replace({"1":"Jan","2":"Feb","3":"Mar"},inplace=True)
df
del df['Invoice ID']
del df['Time']
del df['gross margin percentage']
del df['Branch']
df.head(10)
#deleting unwanted columns as part of data cleaning
df.isnull().sum()
df.dtypes
#checking the data types of each of the columns
#Heatmap for Correlation between Sales Metrics
plt.figure(figsize=(12,8))
df_heatmap= sns.heatmap(df.corr(),annot=True,
                        linewidths=.5,cmap='RdBu')
plt.title('Heatmap for Correlation between Sales Metrics')
#analysing best performing city branch based on maximum revenue generated
df_test=df[['City','Total']]
df_grp=df_test.groupby(['City'],as_index=False).sum()
df_grp
#analysing best performing city branch based on gross income generated
df_pro=df[['City','gross income']]
df_grp1=df_pro.groupby(['City'],as_index=False).sum()
df_grp1
#analysing the sales revenue for each product line
df_sal=df[['Product line','Total']]
df_sal_grp= df_sal.groupby('Product line', as_index=False).sum()
df_sal_grp
#visualising sales revenue for each product line
plt.figure(figsize=(12,6))
sns.barplot(x='Total',y='Product line',data=df_sal_grp,palette='autumn')
plt.title('Sales revenue based on Product Line',fontsize=20)
plt.xlabel('Total Sales',fontsize=14)
plt.ylabel('Product Line',fontsize=14)
#Comparision of Product lines purchased based on Gender
plt.figure(figsize=(12,8))
sns.countplot(x='Product line', hue='Gender',data= df, 
              palette='spring')
plt.title('Comparision of Product lines purchased based on Gender',fontsize=20)
plt.xlabel('Product Line',fontsize=15)
plt.ylabel('Count',fontsize=15)
#analysing if the sales revenue generated is more through members or normal customers
df_cust=df[['Customer type','Total']]
df_grp_cust=df_cust.groupby('Customer type',as_index=False).sum()
df_grp_cust
#analysing the average spend of members vs normal customers
df_grp_cust_avg=df_cust.groupby('Customer type',as_index=False).mean()
df_grp_cust_avg
#analysing the most profitable product line in each of the city branches
df_prof_prod=df[['City','Product line','gross income']]
df_grp_prof_prod= df_prof_prod.groupby(['City','Product line'], as_index=False).sum()
df_grp_prof_prod_piv= df_grp_prof_prod.pivot(index='City',columns='Product line')
df_grp_prof_prod_piv

#visualzing the most profitable product line in each of the city branches
plt.figure(figsize=(14,12))
sns.barplot(x='City',y='gross income',hue='Product line',
           data= df_grp_prof_prod, palette='Set2')
plt.title('Most Profitable Product Lines in each City branch',fontsize=22)
plt.xlabel('City',fontsize=18)
plt.ylabel('Gross Income',fontsize=18)
plt.xticks(fontsize=12)
#average rating of each city branch based on their product line
df_rating= df[['City','Product line','Rating']]
df_rating_city=df_rating.groupby(['City','Product line'], as_index=False).mean()
df_rating_city_piv=df_rating_city.pivot(index='City',columns='Product line')
df_rating_city_piv
#visualising the best city branch based on their overall ratings
plt.figure(figsize=(12,8))
sns.boxplot(x='Rating',y='City',data= df)
plt.title('Analysing the City branch ratings by customers', fontsize=20)
plt.xlabel('Rating',fontsize=14)
plt.ylabel('City',fontsize=14)
#visualising the product line with the best rating per city branch
plt.figure(figsize=(20,12))
sns.barplot(x='City',y='Rating',hue='Product line',data= df)
plt.title('Analysing the Product line ratings of each city branch', fontsize=20)
plt.xlabel('City',fontsize=15)
plt.ylabel('Rating',fontsize=15)
#analysing quantity of products sold every month
df_mon=df[['Product line','Date_month','Quantity']]
df_mon_grp= df_mon.groupby(['Product line','Date_month'],as_index=False).sum()
df_mon_piv= df_mon_grp.pivot(index='Product line',columns='Date_month')
df_mon_piv
#visualizing the monthly product quantity sold
plt.figure(figsize=(12,8))
sns.lmplot(x='Product line',y='Quantity',hue='Date_month',fit_reg=False, 
           data=df_mon_grp)
plt.title('Quantity of Product line sold every month',fontsize=15)
plt.xlabel('Product Line',fontsize=12)
plt.ylabel('Product Quantity',fontsize=12)
plt.xticks(rotation=90)
#analysing income generated in each city branch every month
df_mon1=df[['City','Date_month','gross income']]
df_mon1_grp= df_mon1.groupby(['City','Date_month'],as_index=False).sum()
df_mon1_piv= df_mon1_grp.pivot(index='City',columns='Date_month')
df_mon1_piv
#Monthly Gross Income of every city branch
plt.figure(figsize=(15,8))
sns.barplot(x='Date_month',y='gross income',hue='City',
          data=df_mon1_grp)
plt.title('Monthly Gross Income of every city branch',fontsize=20)
plt.xlabel('Month',fontsize=14)
plt.ylabel('Gross Income',fontsize=14)
df.head(10)
#Payment methods preferred for each Product line
plt.figure(figsize=(14,8))
sns.countplot(x='Product line',hue='Payment',data=df)
plt.title('Payment methods preferred for each Product line',fontsize=18)
plt.xlabel('Product line',fontsize=12)
plt.ylabel('Count',fontsize=12)
#dataset for electronic accssories
df_elec= df.loc[df['Product line']=='Electronic accessories']
df_elec.head(20)
#Correlation between Unit Price and Gross Income for Electronic Accessories
plt.figure(figsize=(12,8))
sns.regplot(x='Unit price',y='gross income',data=df_elec)
plt.title('Correlation between Unit Price and Gross Income for Electronic Accessories',fontsize=18)
plt.xlabel('Unit Price',fontsize=12)
plt.ylabel('Gross Income',fontsize=12)
#training a regression model to predict the income for electronic accessories based on Unit price
x=df_elec[['Unit price']]
y=df_elec['gross income']
lm.fit(x,y)
Yhat=lm.predict(x)
lm.intercept_,lm.coef_
#prediction of the model
df_elec_profit= lm.intercept_+lm.coef_*100
df_elec_profit
df_yan_rat=df.loc[df['City']=='Yangon']
df_yan_rat.head()
plt.figure(figsize=(12,8))
sns.regplot(x='Unit price',y='Rating',data=df_yan_rat, color='Red')
plt.title('Correlation between Unit Price and Rating for Yangon',fontsize=18)
plt.xlabel('Unit Price',fontsize=12)
plt.ylabel('Rating',fontsize=12)
x1=df_yan_rat[['Unit price']]
y1=df_yan_rat['Rating']
lm.fit(x1,y1)
Yhat1=lm.predict(x1)
lm.intercept_,lm.coef_
Yhat1
