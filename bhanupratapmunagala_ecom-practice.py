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
#Import libraries

import numpy as np
import pandas as pd
#Read data set 

ecom_df = pd.read_csv('../input/EcommercePurchases.csv')
#Verify dataset

ecom_df.head(5)
#Analyse data set

ecom_df.info()
#Average Purchase Price

ecom_df['Purchase Price'].mean()
#Highest & Lowest Purchases

ecom_df.loc[ecom_df['Purchase Price'].idxmax()]['Purchase Price']

#ecom_df['Purchase Price'].max()
ecom_df.iloc[ecom_df['Purchase Price'].idxmin()]['Purchase Price']

#ecom_df['Purchase Price'].min()
#How many people have English 'en' as their Language of choice on the website?

#ecom_df[ecom_df['Language'] == 'en'].count()
#ecom_df[ecom_df['Language'] == 'en']['Language'].count()
ecom_df[ecom_df['Language'] == 'en']['Language'].value_counts()
#How many people have the job title of "Lawyer" 

ecom_df[ecom_df['Job'] == 'Lawyer']['Job'].count()
#How many people made the purchase during the AM and how many people made the purchase during PM

ecom_df['AM or PM'].value_counts()
#What are the 5 most common Job Titles? 
ecom_df['Job'].value_counts().head(5)
#Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction?

ecom_df[ecom_df['Lot'] == '90 WT']['Purchase Price']
#What is the email of the person with the following Credit Card Number: 4926535242672853

ecom_df[ecom_df['Credit Card'] == 4926535242672853]['Email']
#How many people have American Express as their Credit Card Provider *and made a purchase above $95 ?

ecom_df[(ecom_df['CC Provider'] == 'American Express') & (ecom_df['Purchase Price'] > 95)].count()
#How many people have a credit card that expires in 2025? 
#Method 1 Using function

def year_check(Year):
    if '25' in Year.split('/')[1]:
        return True
    else:
        return False

    
sum(ecom_df['CC Exp Date'].apply(year_check))
#Method 2 using in-line lambda function

ecom_df[ecom_df['CC Exp Date'].apply(lambda Year: '25' in Year.split('/')[1])]['CC Exp Date'].count()
#Identify number of cards expiring month wise in 2025

ecom_df[ecom_df['CC Exp Date'].apply(lambda Year: '25' in Year.split('/')[1])]['CC Exp Date'].value_counts()
#What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...) 
#Using in-line lambda function

ecom_df['Email'].apply(lambda email: email.split('@')[1]).value_counts().head(5)