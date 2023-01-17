# Please write me about all my shortcomings, this is my first attempt.

import numpy as np
import pandas as pd    #importing libraries
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../input/budget-2011-budget-ordinance-appropriations.csv')    
df.info()
df.head()

df['APPROPRIATION AUTHORITY DESCRIPTION'] = df['APPROPRIATION AUTHORITY DESCRIPTION'].apply(lambda x: x.replace(x[0:6], ''))
new_df = df[['DEPARTMENT','APPROPRIATION ACCOUNT DESCRIPTION','AMOUNT']] # new table 
sorted_df = new_df.sort_values('AMOUNT',ascending = False)     #sorting table 
sorted_df.head(30).plot(x ='DEPARTMENT', y = 'AMOUNT', kind='bar') #sorting  the top 30 by amount
sns.regplot(x = 'DEPARTMENT', y = 'AMOUNT', data = sorted_df.head(5), marker = "x", color = 'r', fit_reg = False)
df['DEPARTMENT'].unique()
AFD = []
i=0
a = 0
for dep in new_df['DEPARTMENT']:
    if dep == new_df['DEPARTMENT'].unique()[i]:
        a =  a + new_df['AMOUNT'][i]
    else:
        AFD.append(a)
        a = 0
        i+=1
AFD.append(a)
print(len(AFD))
print(len(new_df['DEPARTMENT'].unique()))
df2 = pd.DataFrame({
    'DEPARTMENT':new_df['DEPARTMENT'].unique(),
    'AMOUNT': AFD })
print(df2)
df2['AMOUNT'].describe()
df2['AMOUNT'].hist(bins = 20) # Distribution for Amount
df2.plot(x='DEPARTMENT', y='AMOUNT', kind = 'bar', figsize=(15,15), logy=True, legend=True, title="Amount for each department")

