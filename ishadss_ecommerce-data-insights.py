#Import and read the csv

import pandas as pd

ecom = pd.read_csv('../input/ecommercepurchase-data/Ecommerce Purchases')
#View first 5 rows

ecom.head()
#Statistical descriptions

ecom.describe()
#View the size of the dataset

row, col = ecom.shape

print(f'There are {row} rows and {col} columns in this dataset')
#What is the average purchase price?

print('The average purchase price is {}'.format(ecom['Purchase Price'].mean()))
#How many people have the Language preference as 'en'

ecom[ecom['Language'] == 'en']['Language'].count()
#How many people have their Job as Lawyer

ecom[ecom['Job'] == 'Lawyer']['Job'].count()
ecom[(ecom['AM or PM'] == 'AM')].info()
ecom[(ecom['AM or PM'] == 'PM')].info()
#How many people purchased in AM and how many in PM

ecom['AM or PM'].value_counts()
#Top 5 most famous job titles

ecom['Job'].value_counts().head()
#what was the Purchase Price for this transaction that came from the lot 90WT

ecom[ecom['Lot']=='90 WT']['Purchase Price']
#What is the email of the person whose credit card number is 4926535242672853

ecom[ecom['Credit Card'] == 4926535242672853]['Email']
#How many people have American Express as their Credit Card Provider *and made a purchase above $95 ?

ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)]
 #Count of the people whose credit card expires in 2025

sum(ecom['CC Exp Date'].apply(lambda x: x[3:] == '25'))

 #top 5 most popular email providers/hosts 

ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head()
#drop ipAddress column

ecom = ecom.drop(['IP Address'], axis=1)

ecom.info()
#knowing the datatypes of columns

dt = ecom.dtypes

dt
#Dealing with duplicates

ecom[ecom.duplicated()]  #There are no duplicates in this dataset
#Missing values  #There are no missing_values in this dataset

missing_val = ecom.dropna()

missing_val.info()