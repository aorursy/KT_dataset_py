# Run this code to get the file path of the dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd
commerce = pd.read_csv('/kaggle/input/ecommerce-purchases/Ecommerce Purchases')
commerce.info()
commerce.head()
commerce.shape
commerce['Purchase Price'].mean()
commerce['Purchase Price'].agg(['max','min'])
commerce[commerce['Language']=='en'].count()
commerce[commerce['Job']=='Lawyer'].count()
commerce['AM or PM'].value_counts()
commerce['Job'].value_counts()[:5]



person = commerce[commerce['Lot'] == '90 WT']

person['Purchase Price']
person2 = commerce[commerce['Credit Card'] == 4926535242672853]

person2['Email']
merican_xpress = commerce[(commerce['CC Provider'] == 'American Express') & (commerce['Purchase Price'] > 95)] 

merican_xpress.count()
trial = commerce['CC Exp Date'].str.split(pat = '/',expand=True)

trial[trial[1]=='25'].count()
trial2 = commerce['Email'].str.split(pat = '@', expand=True)

trial2[1].value_counts()[:5]