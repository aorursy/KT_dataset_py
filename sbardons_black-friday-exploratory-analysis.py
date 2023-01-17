# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mlp

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
##reading the data 

df = pd.read_csv('../input/BlackFriday.csv')
df.shape
# Number of transactions for genre

df['Gender'].value_counts()  # 
#Pie chart for the transactions

explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(8,5))

plt.rcParams['font.size']=18

color_palette_list = ['#80bfff', '#ff99ff' ]

ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['Male','Female'],colors=color_palette_list[0:2], autopct='%1.1f%%',

        shadow=True, startangle=90)



ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle 

plt.tight_layout()

plt.legend()

plt.title('Percentage of Transactions by genre')

plt.show()
df2=df #backup original dataframe

df3 =df2.drop_duplicates(['User_ID'], keep='first') #removed duplicates

#df3.head()

df3['Gender'].value_counts()  # counts for females and males

# more elegant solution:    df.groupby(['Gender'])['User_ID'].nunique()



#Pie chart

explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(8,5))

plt.rcParams['font.size']=18

color_palette_list = ['#80bfff', '#ff99ff' ]

ax1.pie(df3['Gender'].value_counts(), explode=explode,labels=['Male','Female'],colors=color_palette_list[0:2], autopct='%1.1f%%',

        shadow=True, startangle=90)



ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle 

plt.tight_layout()

plt.title('Male and Female')

plt.legend()

plt.show()
label=df3['Age'].value_counts().index

fig2, ax2 = plt.subplots(figsize=(10,8))

ax2.pie(df3['Age'].value_counts(), labels=label,  

       autopct='%1.0f%%', 

       shadow=False, startangle=0,   

       pctdistance=0.6,labeldistance=1.1)

ax2.axis('equal')

ax2.set_title("Percentage split between different age groups in the dataset", fontsize=20)



ax2.legend(label,

          title="Age Groups",

          loc="center left",

          bbox_to_anchor=(1.2, 0.5, 0, 0.2))

plt.show()
n_bins=30

plt.subplots(figsize=(16,8))

plt.hist(df['Purchase'],bins=n_bins)

# Label axes

plt.xlabel('purchase')

plt.ylabel('count')

plt.title('Purchase Amount', fontsize=30)

plt.show()
df.groupby('Age')[['Purchase']].describe()
# plt.subplots(figsize=(16,8))

data=df.groupby('Age')[['Purchase']].mean()

data.plot(xticks=[1,2,3,4,5,6,7],figsize=(13,5))

plt.title('Mean Purchases by Age', fontsize=20)

plt.show()

derived=df.groupby('Product_ID')[['Purchase']].sum()/ sum(df['Purchase'])
df3['Gender'].value_counts()  # counts for females and males
(df.groupby(['Product_ID','Gender'])[['Gender']]).count().head()
##User by Occupation

plt.subplots(figsize=(12,7))

df.groupby(['Occupation'])['User_ID'].nunique().sort_values().plot('bar', color='m')

plt.xlabel('Occupation')

plt.ylabel('Count')

plt.title('User by Occupation')

plt.show()
##User by Location

plt.subplots(figsize=(10,5))

df.groupby(['City_Category'])['User_ID'].nunique().sort_values().plot('bar', color='g')

plt.xlabel('City_Category')

plt.ylabel('Count')

plt.title('User by City_Category')

plt.show()
##Total purchases by genre not normalized: the females are much less than males.

plt.subplots(figsize=(10,5))

df.groupby('Gender')['Purchase'].sum().sort_values().plot('bar',color=['#ff99ff','#80bfff'])  

plt.xlabel('Gender')

plt.ylabel('Total Purchases')

plt.title('Total spend Female-Males Not normalized')

plt.show()
# normalized

plt.subplots(figsize=(10,5))

(df.groupby('Gender')['Purchase'].sum()/df.groupby(['Gender'])['User_ID'].nunique()).sort_values().plot('bar',color=['#ff99ff','#80bfff'])

plt.xlabel('Gender')

plt.ylabel('Total Purchases')

plt.title('Total spend Female-Males Normalized')

plt.show()
plt.subplots(figsize=(10,5))

df.groupby('Age')['Purchase'].sum().plot('bar', color='b')  #add axes labels



plt.xlabel('Age')

plt.ylabel('Total Purchases')

plt.title('Age of customers')

plt.show()
plt.subplots(figsize=(10,5))

(df.groupby('Age')['Purchase'].sum()/df.groupby(['Age'])['User_ID'].nunique()).plot('bar', color='b')

plt.xlabel('Age')

plt.ylabel('Total Purchases')

plt.title('Age Customers Normalized')

plt.show()
###### See https://www.kaggle.com/arkhoshghalb/black-friday-analysis-regression-and-clustering/comments



ages_count = [df[df.Age== x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(df.Age.unique())]



stay_years = [df[df.Stay_In_Current_City_Years == x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(df.Stay_In_Current_City_Years.unique())]





f, (ax1, ax2) = plt.subplots(2,1, figsize=(16,12))



ages = sorted(df.Age.unique())

pd.DataFrame(ages_count, index=ages).sort_index(axis=1).T.plot.bar(stacked=True, width=0.3, ax=ax1, rot=0, fontsize=11)

ax1.set_xlabel('City Category', size=13)

ax1.set_ylabel('# Transactions', size=14)

ax1.set_title('# Transactions by city (separated by age)', size=14)

ax1.legend(loc="upper left",fontsize=10)



colors=['#FF0000','#FFFF00','#00FFFF','#C0C0C0','#800000',]

years = sorted(df.Stay_In_Current_City_Years.unique())

pd.DataFrame(stay_years, index=years).sort_index(axis=1).T.plot.bar(stacked=True, width=0.3, ax=ax2, rot=0,color=colors,fontsize=11)

ax2.set_xlabel('City Category', size=13)

ax2.set_ylabel('# Transactions', size=14)

ax2.set_title('# Transactions by city (separated by years in the City)', size=14)

ax2.legend(loc="upper left",fontsize=10)



plt.show()
##### test to see distribution for populations  ###

######https://www.kaggle.com/arkhoshghalb/black-friday-analysis-regression-and-clustering/comments



occupations_count = [df[df.Occupation== x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(df.Occupation.unique())]



stay_years = [df[df.Stay_In_Current_City_Years == x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(df.Stay_In_Current_City_Years.unique())]





f, (ax1, ax2) = plt.subplots(2,1, figsize=(16,12))



occupations = sorted(df.Occupation.unique())

pd.DataFrame(occupations_count, index=occupations).sort_index(axis=1).T.plot.bar(stacked=True, width=0.3, ax=ax1, rot=0, fontsize=11)

ax1.set_xlabel('City Category', size=13)

ax1.set_ylabel('# Transactions', size=14)

ax1.set_title('# Transactions by city (separated by age)', size=14)

ax1.legend(loc="upper left",fontsize=10)



colors=['#FF0000','#FFFF00','#00FFFF','#C0C0C0','#800000',]

years = sorted(df.Stay_In_Current_City_Years.unique())

pd.DataFrame(stay_years, index=years).sort_index(axis=1).T.plot.bar(stacked=True, width=0.3, ax=ax2, rot=0,color=colors,fontsize=11)

ax2.set_xlabel('City Category', size=13)

ax2.set_ylabel('# Transactions', size=14)

ax2.set_title('# Transactions by city (separated by years in the City)', size=14)

ax2.legend(loc="upper left",fontsize=10)



plt.show()
# (df.groupby('Marital_Status')['Purchase'].sum()/df.groupby(['Marital_Status'])['User_ID'].nunique()).sort_values().plot('bar',color=['#008000','#800080'])

df.groupby('Marital_Status')['Purchase'].sum().sort_values().plot('bar',color=['#008000','#800080'])  

plt.xlabel('Marital_Status')

plt.ylabel('Total Purchases')

plt.title('Normalized')

plt.show()
# Do not married spend more than married?

(df.groupby('Marital_Status')['Purchase'].sum()/df.groupby(['Marital_Status'])['User_ID'].nunique()).sort_values().plot('bar',color=['#008000','#800080'])

plt.xlabel('Marital_Status')

plt.ylabel('Total Purchases')

plt.title('Total spend by  marital status normalized')

plt.show()
df_0to17=df.loc[df['Age'] == '0-17']   ##create dataframe with age category
fig1, ax1 = plt.subplots(figsize=(12,7))

plt.title('Most sold product for Age group 0-17')

df_0to17.groupby(['Product_ID'])['Purchase'].count().nlargest(10).sort_values().plot('barh',color='r')

plt.show()
#filling missing data entries with zero

df=df.fillna(0)

df.head()
product='P00000142'  #change here accordingly

df_product=df.loc[df['Product_ID'] == product]

(df_product.groupby(['Product_ID','Gender','Age'])[['Age']]).count()