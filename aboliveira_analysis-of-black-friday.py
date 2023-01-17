# Warnings
import warnings
warnings.filterwarnings('ignore')

# Data and analysis
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

sns.set(style='darkgrid')
plt.rcParams["patch.force_edgecolor"] = True
df = pd.read_csv('../input/BlackFriday.csv')
# First 5 rows:
df.head(5)
print(df.info())
print('Shape: ',df.shape)
total_miss = df.isnull().sum()
perc_miss = total_miss/df.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head(3)
print('Unique Values for Each Feature: \n')
for i in df.columns:
    print(i, ':',df[i].nunique())
# Info about products
print('Number of products:',df['Product_ID'].nunique())
print('Number of categories:',df['Product_Category_1'].unique().max())
print('Highest and lowest purchase:',
      df['Purchase'].max(),',',df['Purchase'].min())
# Info about shoppers
print('Number of shoppers:',df['User_ID'].nunique())
print('Years in city:',df['Stay_In_Current_City_Years'].unique())
print('Age Groups:',df['Age'].unique())
count_m = df[df['Gender']=='M'].count()[0]
count_f = df[df['Gender']=='F'].count()[0]
print('Number of male clients:',count_m)
print('Number of female clients:',count_f)
print('Female Purchases:',round(df[df['Gender']=='F']['Purchase'].sum()/count_f,3))
print('Male Purchases:',round(df[df['Gender']=='M']['Purchase'].sum()/count_m,3))
plt.pie(df.groupby('Gender')['Product_ID'].nunique(),labels=['Male','Female'],
       shadow=True, autopct='%1.1f%%',colors=['steelblue','cornflowerblue'])
plt.title('Unique Item Purchases by Gender')
plt.show()
# Individual groupby dataframes for each gender
gb_gender_m = df[df['Gender']=='M'][['Product_Category_1','Gender']].groupby(by='Product_Category_1').count()
gb_gender_f = df[df['Gender']=='F'][['Product_Category_1','Gender']].groupby(by='Product_Category_1').count()

# Concatenate and change column names
cat_bygender = pd.concat([gb_gender_m,gb_gender_f],axis=1)
cat_bygender.columns = ['M ratio','F ratio']

# Adjust to reflect ratios
cat_bygender['M ratio'] = cat_bygender['M ratio']/df[df['Gender']=='M'].count()[0]
cat_bygender['F ratio'] = cat_bygender['F ratio']/df[df['Gender']=='F'].count()[0]

# Create likelihood of one gender to buy over the other
cat_bygender['Likelihood (M/F)'] = cat_bygender['M ratio']/cat_bygender['F ratio']

cat_bygender['Total Ratio'] = cat_bygender['M ratio']+cat_bygender['F ratio']
cat_bygender.sort_values(by='Likelihood (M/F)',ascending=False)
# Encoding the age groups
df['Age_Encoded'] = df['Age'].map({'0-17':0,'18-25':1,
                          '26-35':2,'36-45':3,
                          '46-50':4,'51-55':5,
                          '55+':6})

prod_byage = df.groupby('Age').nunique()['Product_ID']

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax = ax.ravel()

sns.countplot(df['Age'].sort_values(),ax=ax[0], palette="Blues_d")
ax[0].set_xlabel('Age Group')
ax[0].set_title('Age Group Distribution')
sns.barplot(x=prod_byage.index,y=prod_byage.values,ax=ax[1], palette="Blues_d")
ax[1].set_xlabel('Age Group')
ax[1].set_title('Unique Products by Age')

plt.show()
spent_byage = df.groupby(by='Age').sum()['Purchase']
plt.figure(figsize=(12,6))

sns.barplot(x=spent_byage.index,y=spent_byage.values, palette="Blues_d")
plt.title('Mean Purchases per Age Group')
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(df['Occupation'])
plt.title('Occupation Distribution')
plt.show()

plt.figure(figsize=(12,6))
prod_by_occ = df.groupby(by='Occupation').nunique()['Product_ID']

sns.barplot(x=prod_by_occ.index,y=prod_by_occ.values)
plt.title('Unique Products by Occupation')
plt.show()
spent_by_occ = df.groupby(by='Occupation').sum()['Purchase']
plt.figure(figsize=(12,6))

sns.barplot(x=spent_by_occ.index,y=spent_by_occ.values)
plt.title('Total Money Spent per Occupation')
plt.show()
plt.figure(figsize=(12,6))
prod_by_cat = df.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index,y=prod_by_cat.values, palette="Blues_d")
plt.title('Number of Unique Items per Category')
plt.show()
category = []
mean_purchase = []


for i in df['Product_Category_1'].unique():
    category.append(i)
category.sort()

for e in category:
    mean_purchase.append(df[df['Product_Category_1']==e]['Purchase'].mean())

plt.figure(figsize=(12,6))

sns.barplot(x=category,y=mean_purchase)
plt.title('Mean of the Purchases per Category')
plt.xlabel('Product Category')
plt.ylabel('Mean Purchase')
plt.show()
# Dictionary of product IDs with minimum purchase
prod_prices = df.groupby('Product_ID').min()['Purchase'].to_dict()
def find_price(row):
    prod = row['Product_ID']
    
    return prod_prices[prod]
df['Price'] = df.apply(find_price,axis=1)
df['Amount'] = round(df['Purchase']/df['Price']).astype(int)



