

import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





df = pd.read_csv('/kaggle/input/starbucks-customer-retention-malaysia-survey/Starbucks satisfactory survey encode cleaned.csv')

df.head()
for col in df.columns:

    print(col)
gender_df = df[['gender', 'timeSpend', 'spendPurchase']]

gender_pivot = pd.pivot_table(gender_df,index=['gender'],aggfunc=[np.mean,len])

gender_pivot #note we have an almost equal distribution of genders in the sample
print(df['timeSpend'].value_counts())

print(df['spendPurchase'].value_counts())
price_df = df[['spendPurchase', 'priceRate']]

price_pivot = pd.pivot_table(price_df,index=['spendPurchase'],aggfunc=[np.mean])

price_pivot
income_df = df[['income', 'spendPurchase', 'priceRate']]

income_pivot = pd.pivot_table(income_df,index=['income'],aggfunc=[np.mean])

income_pivot