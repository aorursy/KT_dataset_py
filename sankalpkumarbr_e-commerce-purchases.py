#import pandas module

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#read the csv file

pd.set_option('display.max.rows',None)

data=pd.read_csv('../input/ecommerce-purchases-csv/Ecommerce Purchases.csv')

#converting the data into pandas DataFrame

ecomm=pd.DataFrame(data)

#print DataFrame

ecomm
ecomm.head()    # Fetches the first five rows of the dataset.
ecomm.tail()   #Fetches the last five rows of the dataset
ecomm.shape  #shape of the dataframe
ecomm.info() # Give Summary
ecomm.ndim  #dimension of the dataframe
ecomm.size #size of the dataframe
ecomm.columns #columns name of the dataframe
ecomm.describe()  #statistical analysis on dataframe
ecomm.describe(include=['object', 'bool'])   # Calculation of Statistical Data wrt Objects and Boolean
language={'ru':'Russia','de':'German' ,'el':'Greece','pt':'Portuguese','en':'English','fr':'French','es':'Spanish','it':'Italian','zh':'Chinese'}

ecomm['Language']=ecomm['Language'].map(language)

ecomm

ecomm[ecomm['Lot'].duplicated()]
ecomm[ecomm['Lot'].duplicated()]['Lot'].count()    #total duplicate values in the Lot column
ecomm['Purchase Price'].max()
ecomm[ecomm['Purchase Price']==ecomm['Purchase Price'].max()]['Browser Info']
ecomm['Purchase Price'].min()
ecomm[ecomm['Purchase Price']==0.0]['Browser Info']
ecomm['AM or PM'].value_counts()
ecomm['AM or PM'].value_counts().plot(kind='bar',color=['r','b'])
ecomm['Company'].value_counts().head(5)
ecomm['Company'].value_counts().head(5).plot()
md={'Job':'Job Title'}

ecomm1=ecomm.rename(columns=md)

ecomm1
ecomm.groupby(by = 'Job')['Purchase Price'].max()
count = ecomm['CC Provider'].value_counts().sort_values()

count.head(1)
ecomm['Rank']=ecomm['Purchase Price'].rank(method='min')

ecomm
ecomm['Rank'].max()
ecomm['Purchase Price'].max()
common_jobs = ecomm['Job'].value_counts().sort_values(ascending = False).head()

common_jobs
sns.distplot(ecomm['Job'].value_counts(), bins=20, kde = False, color = 'red')
count_AE=ecomm[ecomm["CC Provider"]=="American Express"]

pur_95=count_AE['CC Provider'][count_AE["Purchase Price"]>95].count()

pur_95
# we can also write the above code as :
ecomm['CC Provider'][(ecomm["CC Provider"]=="American Express") & (ecomm["Purchase Price"]>95)].count()
pd.pivot_table(data=ecomm,index='Company',columns='AM or PM',aggfunc='mean',values='Purchase Price')

ecomm[ecomm['Credit Card']==3337758169645356]['Email']
ecomm[ecomm['CC Exp Date'].apply( lambda exp: exp[3:]=='22')]['CC Exp Date'].count()
ecomm[ecomm['Language']=='Greece']['Language'].count()
ecomm['Browser Info'].value_counts().head()
ecomm['Browser Info'].value_counts().head(5).plot(kind='bar')
ecomm['Language'].value_counts().head(2)
ecomm['CC Security Code'].value_counts()
ecomm['Email'].apply(lambda email:email.split('@')[1]).value_counts().head(5).plot.pie(autopct='%0.1f%%',shadow=True, radius =2, pctdistance=0.8)

plt.title("Browser Distribution")

plt.legend(loc='upper left')