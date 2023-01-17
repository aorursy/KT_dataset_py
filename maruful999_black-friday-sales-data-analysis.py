#The warnings filter controls whether warnings are ignored, displayed, or turned into errors (raising an exception).

import warnings

warnings.filterwarnings('ignore')



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading the Data

data = pd.read_csv('../input/BlackFriday.csv')
#prints information about the DataFrame including the index dtype and column dtypes

data.info()
# Look at the DataFrame

data.head()
# Check which column has null/NaN values

data.isna().any()
# fill null values with 0.

data.fillna(value=0,inplace=True)



#conver data types into integer

data['Product_Category_2'] = data['Product_Category_2'].astype('int64')

data['Product_Category_3'] = data['Product_Category_3'].astype('int64')
actualGender = data[['User_ID','Gender']].drop_duplicates('User_ID')

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(actualGender['Gender'].value_counts(),labels=['Male','Female'],autopct='%1.2f%%',explode = (0.1,0))

plt.title("Actual ratio between Male & Female")

plt.axis('equal')

plt.legend()

plt.tight_layout()

plt.show()
# Let's check the ratio of purchase amount between male and female with duplicates User_id



fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(data[['Gender','Purchase']].groupby(by='Gender').sum(),labels=['Female','Male'],autopct='%1.2f%%',explode = (0.1,0))

plt.title("Ratio of purchase amount between male and female")

plt.axis('equal')

plt.legend()

plt.tight_layout()

plt.show()



print(data[['Gender','Purchase']].groupby(by='Gender').sum())
AgePurchase_DF = data[['Age','Purchase']].groupby('Age').sum().reset_index()

fig1,ax1 = plt.subplots(figsize=(12,6))

sns.barplot(x='Age',y='Purchase',data=AgePurchase_DF)

plt.title('Total purchase made by different Age')

plt.tight_layout()
AgeGenderMeritual_DF = data[['User_ID','Age','Gender','Marital_Status']].drop_duplicates()

AgeGenderMeritual_DF.sort_values(by='Age',inplace=True)

AgeGenderMeritual_DF['combinedG_M'] = AgeGenderMeritual_DF.apply(lambda x: '%s_%s' % (x['Gender'],x['Marital_Status']), axis=1)

fig1,ax1 = plt.subplots(figsize=(15,7))

sns.countplot(x= 'Age',hue='combinedG_M',data=AgeGenderMeritual_DF)
Occupation_DF = data[['User_ID','Occupation','Purchase']].drop_duplicates('User_ID')

Occupation_DF = Occupation_DF.drop('User_ID',1)

fig1, ax1 = plt.subplots(figsize = (12,7))

sns.countplot(x='Occupation',data=Occupation_DF)

plt.title('Top Occupations counts')

plt.tight_layout()
Occupation_DF1 = data[['Occupation','Purchase']].groupby('Occupation').sum().reset_index()

fig1,ax1=plt.subplots(figsize=(13,7))

sns.barplot(x='Occupation',y='Purchase',data=Occupation_DF1)

plt.title('Total purchasing By Occupation')

plt.tight_layout()
city_DF = data[['City_Category','Stay_In_Current_City_Years','Purchase']].groupby(['City_Category','Stay_In_Current_City_Years']).sum().reset_index()

fig1,ax1 = plt.subplots(figsize = (12,7))

sns.barplot(x='City_Category',y='Purchase',hue= 'Stay_In_Current_City_Years',data = city_DF)

plt.title("Top revenue from city & staying in city")

plt.tight_layout()
category1_DF = data[['Product_Category_1','Purchase']].groupby('Product_Category_1').sum().reset_index()

category2_DF = data[['Product_Category_2','Purchase']].groupby('Product_Category_2').sum().reset_index()

category3_DF = data[['Product_Category_3','Purchase']].groupby('Product_Category_3').sum().reset_index()



fig , (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(13,7))



sns.barplot(x='Product_Category_1',y='Purchase',data=category1_DF,ax=ax1)

sns.barplot(x='Product_Category_2',y='Purchase',data=category2_DF,ax=ax2)

sns.barplot(x='Product_Category_3',y='Purchase',data=category3_DF,ax=ax3)



plt.tight_layout()