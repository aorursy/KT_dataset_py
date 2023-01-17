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
# Load the dataset

df_2009_2010 = pd.read_excel("../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx", sheet_name = "Year 2009-2010")

df_2010_2011 = pd.read_excel("../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
# Concatenate the data frames of two years

df_concat = pd.concat([df_2009_2010,df_2010_2011],axis=0,ignore_index=True)

# Copy the dataset for pre-precessing/analytics

df=df_concat.copy()
# 1.Eyeball the data

df.head()
df.info()
# 2. Check if there is records with Null values

print(f"Fields with null values:\n{df.isnull().any()}")

# Null values in CustomerID: some orders may purchased by non-member-> no Customer No.?
# 3. For the purpose of RFM analysis, drop the order without customer ID (this customers are not trackable)

df.drop(index = df[df['Customer ID'].isnull()].index, inplace = True)

print(f"Fields with null values:\n{df.isnull().any()}")
# 4. Detact the abnormal records (Invoice start with 'C', but not a cancellation invocie, and vice verer)

print("Any invoice starts with 'C' but is not a cancelation order ?",((df[df['Invoice'].str.startswith("C",na=False)]['Quantity']) > 0).any())

print("Any invoice not starts with 'C' but is a cancelation order ?", ((df[df['Invoice'].str.startswith("C",na=False) == False]['Quantity']) < 0).any())
# 5. Calculate the sales amount of each order / Indicated whether the order is cancelled (chargeback) / turn Customer ID into INT

df['Sales'] = df['Quantity']*df['Price']

df['Chargeback'] = df['Invoice'].str.startswith("C",na=False) # show False (na=False) if element tested is not start with C (not a string).

df['Customer ID'] = df['Customer ID'].astype(int)
# 6. Present the cleaned table

print("Show cleaned data set")

df.head()

# df.info()
import matplotlib.pyplot as plt

import datetime

import calendar

%matplotlib inline
# Navigate the sales trend over months

df['Month']=df['InvoiceDate'].dt.month

df_plot=df.loc[:,['Sales','Month','Chargeback']].sort_values(by=['Month'])

# Prepare the data: Calculate the sales and chargeback amount

x=df_plot.Month.unique()

y1=df.groupby('Month')['Sales'].sum().values/1e6

y2=df[df.Chargeback==True].groupby('Month').Sales.sum().values/-1e6
# Sales trend Figure:

#1 Figure name and size

f1=plt.figure(figsize=(16,8))

plt.gcf()

#2 Grid

plt.grid(alpha=0.4)

plt.axis=([0,10,0,10])

# plt.xlim(0.5,12.5)

plt.ylim(-.2,3)

# 3 coordinate,tick scale/tick label/label rotation/font size

plt.xticks(x,rotation = 0.1,fontsize=16)

plt.xlabel('Months',fontsize=16)

plt.yticks(fontsize=16)

plt.ylabel(u'Total Sales (Minion \u00a3)',fontsize=16)

#4 plot y axis(#0022FF rgb(0,255))

plt.plot(x,y1,label='Sales Amount',color="blue",marker='o',markersize=10)

plt.plot(x,y2,label='Chargeback',color="red",marker = 'x',markersize=10)

#5 Legend#

plt.legend(loc='upper left',fontsize=20)

#6 Corrdinate showpoints

for a,b in zip(x,y1):

    plt.annotate('%.2f'%(b),xy=(a,b),xytext=(-15,15),textcoords='offset points',fontsize=12)

for a,b in zip(x,y2):

    plt.annotate('(%.2f)'%(b),xy=(a,b),xytext=(-15,15),textcoords='offset points',fontsize=12)    

plt.show()

plt.clf()
# 1.Only consider the non-chargeback orders when establishing the RFM models

df_C=df[df['Chargeback']==False]
# 2. An invoice is ordered at certain time, which can include several products -> Group the invoice and create a pivot table

table=df_C.pivot_table(index=['Invoice'],values=['InvoiceDate','Customer ID','Sales'],aggfunc={'InvoiceDate':'first','Customer ID':'first','Sales':'sum'})

table.head()
# 3. Group the customer ID to form a RFM table

table.rename(columns={'InvoiceDate':'R','Sales':'M'},inplace=True)

table['F'] = 1

table_RMF=table.pivot_table(index=['Customer ID'],values=['R','M','F'],aggfunc={'R':'max','M':'sum','F':'count'})

# 4. Normalised the recency, using the most recent day in the data set as the reference day

table_RMF['R']=(table_RMF['R'].max()-table_RMF['R']).dt.days

table_RMF.head()

# 5. Ignore the one off customers

table_RMF.drop(index=table_RMF[table_RMF['F']==1].index, inplace = True)
# 5. Simple Statistic analysis on the RFM table

table_RMF.describe()

# 4314 customers: F: Frequency (no. of orders), M: Monetary (UK Pound), Recency (days)
# End. Write the table into excel files for further analytics in Tableau

# df.to_csv('customer_segmentation.csv', index = True)

table_RMF.to_csv('RMF_table.csv',index = True)
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
#1 Standardise the data in RFM table

np.random.seed(42)

df_std=StandardScaler()

df_std=df_std.fit_transform(table_RMF)

x=df_std[:,0]

y=df_std[:,1]

z=df_std[:,2]
#2 K-means clustering

km=KMeans(n_clusters=6)

res=km.fit(df_std)

clusterNo=res.predict(df_std)
#3 Show the number of customers in each segment

# table_RMF['clusterNo']=clusterNo

seg,segNo=np.unique(clusterNo,return_counts=True)

print(seg,'\t',segNo) 
# Plot the 3D clustering results

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

f1=plt.figure(figsize=(16,8))

ax1 = f1.add_subplot(111, projection='3d')

ax1.set_xlabel('Recency',size=16)

ax1.set_ylabel('Frequency',size=16)

ax1.set_zlabel('Monetary',size=16)

ax1.scatter(x,y,z,c=clusterNo,cmap='plasma')