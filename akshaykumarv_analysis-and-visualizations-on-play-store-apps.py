import numpy as np,pandas as pd

import matplotlib.pyplot as plt,seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

ak0=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
ak0.shape

ak0.head()
ak0.columns
ak0.Rating.describe()
ak0.Rating.isnull().sum()
#a. Drop records where rating is missing since rating is our target/study variable

ak0['Rating']=ak0['Rating'].replace(np.NaN,ak0['Rating'].mean())
ak0.Rating.isnull().sum()
# b. Check the null values for the Android Ver column.

ak0['Android Ver'].isnull().sum()
# i. Are all 3 records having the same problem?

ak0[ak0['Android Ver'].isnull()]
# ii. Drop the 3 rd record i.e. record for “Life Made WIFI …”

ak1=ak0[~(ak0['App']=='Life Made WI-Fi Touchscreen Photo Frame')]
ak1
ak1[ak1['App']=='Life Made WI-Fi Touchscreen Photo Frame']
ak1.shape
ak1.isnull().sum()
ak1['Type']=ak1['Type'].replace(np.NaN,ak1['Type'].mode()[0])
# in android column need to replace the remaining missing value with mode

# iii. Replace remaining missing values with the mode

missing=ak1['Android Ver'].mode()[0]

ak1['Android Ver'].fillna(missing,inplace=True)
# c. Current ver – replace with most common value

missing2=ak1['Current Ver'].mode()[0]

missing2
# c. Current ver – replace with most common value

ak1['Current Ver'].fillna(missing2,inplace=True)
# to check

ak1.isnull()
#to check if there are any more missing values

ak1.isnull().sum()
#a. Which all variables need to be brought to numeric types?

ak1.info()
#b. Price variable – remove $ sign and convert to float 

ak1['Price']=ak1['Price'].replace('[/$,]','',regex=True).astype(float)
ak1['Price']
ak1['Installs'].apply(str)





ak1['Installs']=ak1['Installs'].str.replace('[/+]','',regex=True)

ak1['Installs']=ak1['Installs'].str.replace('[/,]','',regex=True)
ak1['Installs']=ak1['Installs'].astype(int)
ak1
ak1.shape
ak1['Installs']

#d. Convert all other identified columns to numeric

#we need to covert Reviews variable from object to numeric

ak1['Reviews']=ak1.Reviews.astype(float)
#to check

ak1['Reviews']
#a. Avg. rating should be between 1 and 5, as only these values are allowed on the play store. 

#i. Are there any such records? Drop if so.

ak1[(ak1['Rating']>5) |(ak1['Rating']<1)]





#there are no such records
#b. Reviews should not be more than installs as only those who installed can review the app. 

ak1[ak1['Reviews']>ak1['Installs']]
#i. Are there any such records? Drop if so.

ind=ak1[ak1['Reviews']>ak1['Installs']].index

ak1.drop(labels=ind,inplace=True)
#to chech 

ak1[ak1['Reviews']>ak1['Installs']]
#Identify and handle outliers –a. Price column

ak1['Price'].describe()
#i. Make suitable plot to identify outliers in price

plt.figure(figsize=(11,4))

plt.subplot(121)

sns.boxplot(y=ak1['Price'])

plt.subplot(122)

sns.distplot(ak1['Price'],kde=True)

sns.despine()
#ii. Do you expect apps on the play store to cost $200? Check out these cases

ak1[ak1['Price']>200]
ak1.drop(ak1[ak1['Price']>200].index,inplace=True)
#iii. After dropping the useless records, make the suitable plot again to identify outliers

plt.figure(figsize=(11,4))

plt.subplot(121)

sns.boxplot(y=ak1['Price'])

plt.subplot(122)

sns.distplot(ak1['Price'],kde=True)

sns.despine()
#iv. Limit data to records with price < $30

ak1[~(ak1['Price']<30)]
#iv. Limit data to records with price < $30

ak1.drop(ak1[~(ak1['Price']<30)].index,inplace=True)
#to check

ak1.shape
#b. Reviews column i. Make suitable plot

plt.figure(figsize=(11,4))

plt.subplot(121)

sns.boxplot(y=ak1['Reviews'])

plt.subplot(122)

sns.distplot(ak1['Reviews'],kde=True)

sns.despine()
#ii. Limit data to apps with < 1 Million reviews

#we are going to remove the data where reviews are more than 1m

print(ak1[ak1['Reviews']>1000000].shape[0])

index=ak1[ak1['Reviews']>1000000].index

ak1.drop(index,inplace=True)
#c. Installs i. What is the 95 th percentile of the installs?

np.percentile(ak1['Installs'],95)

#ii. Drop records having a value more than the 95 th percentile

df=ak1[ak1['Installs']>10000000.0]

ak1.drop(df.index,inplace=True)

ak1.shape
#5

sns.distplot(ak1['Rating'])

plt.show()

print('the skewness of the distribution is {}'.format(ak1['Rating'].skew()))

print('The median of the distribution is {} which is greater than mean of distribution {}'.format(ak1['Rating'].median(),ak1['Rating'].mean()))
#what are the top content rating values

ak1['Content Rating'].value_counts().max()
#a. Are there any values with very few records?

ak1['Content Rating'].value_counts()
#b. If yes, drop those as they won’t help in the analysis

drop=ak1[(ak1['Content Rating']=='Adults only 18+') | (ak1['Content Rating']=='Unrated') ]

ak1.drop(drop.index,inplace=True)
ak1['Content Rating'].value_counts()

#that particular record is dropped
#7

sns.jointplot(ak1['Size'],ak1['Rating'],data=ak1)

plt.show()

#8 Effect of price on rating a. Make a jointplot (with regression line)

sns.jointplot(x=ak1['Price'],y=ak1['Rating'],data=ak1,kind='reg')

plt.show()
#8-d

ak2=ak1[ak1['Price']>0]

ak2
#8-d

sns.jointplot(x='Price',y='Rating',data=ak2,kind='reg')



plt.show()
ak=ak1[['Reviews','Size','Rating','Price']]

sns.pairplot(ak)

plt.show()
#10-a

sns.barplot(ak1['Rating'],ak1['Content Rating'])

plt.show()
#10- b

ak1.groupby(['Content Rating'])['Rating'].mean()

#10-c

ak1.groupby(['Content Rating'])['Rating'].mean().plot.bar(color=['red','blue','green','yellow'])

plt.show()
type(ak1['Size'])