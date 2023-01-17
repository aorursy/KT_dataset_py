!pip install missingno 
import os 
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
sns.set(color_codes=True)
import missingno as msno

os.chdir('../input/')
melb=pd.read_csv('melb_data.csv')
melb.head()
#shape of the data
melb.shape
#summary statistics
melb.describe()
# information on dataframe
melb.info()
# Seperating numerical and categorical variables

num=melb.select_dtypes(exclude='object')
cat=melb.select_dtypes(exclude=['int64','float64'])
# List of numerical attributes
num.dtypes

# List of categorical attributes

cat.dtypes
# CHECKING FOR NaN VALUES IN THE VARIABLES

cat.isnull().sum()
plt.figure(figsize=(15,5))
sns.heatmap(cat.isnull(),yticklabels=0,cbar=True,cmap='magma')
num.isnull().sum()

plt.figure(figsize=(15,5))
sns.heatmap(num.isnull(),yticklabels=0,cbar=True,cmap='inferno')
plt.figure(figsize=(20,8))
sns.heatmap(num.corr(),annot=False,cmap='YlGnBu',square=True);
#Getting features that have a correlation value greater than 0.5 against sale price.

for val in range(len(num.corr()['Price'])):
    if abs(num.corr()['Price'].iloc[val]) > 0.5:
        print(num.corr()['Price'].iloc[val:val+1]) 
num_corr=num.corr()['Price'][:-1].sort_values(ascending=True)
num_corr
data = num[['Rooms','Bathroom','Bedroom2','Price']]
#correlation = data.corr(method='pearson')
plt.figure(figsize=(6,6))
sns.heatmap(data.corr(),annot=True,cmap='CMRmap_r');
# First lets consider the numeriicl variables.
#Car,BuildingArea and Yearbuilt has missing values.
sns.kdeplot(num.Car,Label='Car',color='b')

melb['Car'].describe()
melb['Car'].replace({np.nan:2},inplace=True)
melb['YearBuilt'].describe()
melb['YearBuilt'].replace({np.nan:'NA'},inplace=True)
sns.kdeplot(num.BuildingArea,Label='BuildingArea',color='b')
melb['BuildingArea'].describe()
melb['BuildingArea'].replace({np.nan:melb.BuildingArea.median()},inplace=True)
melb['CouncilArea'].describe()
sns.countplot(melb['CouncilArea'],palette='spring');
plt.xticks(rotation=90)

#Replacing the missing values with 'NA'
melb['CouncilArea'].replace({np.nan:'None'},inplace=True)
#Checking for replacements.
melb.isnull().sum()
#checking whether there are any duplicate rows

melb=melb.drop_duplicates(keep='first')


melb
num['Price'].describe()
num['Price'].median()
fig = plt.figure(figsize=(8,5))
print("Skew of SalePrice:", num.Price.skew())
plt.hist(num.Price,color='blue')
plt.show()
sns.distplot(melb['Price'],color='green');
fig = plt.figure(figsize=(8,5))
print("Skew of Log-Transformed Price:", np.log1p(num.Price).skew())
plt.hist(np.log1p(num.Price), color='black')
plt.show()


# Histogram of continuous numerical variables
num.hist(bins=10,figsize=(8,14),layout=(5,3),color='r');
# Scatterplots of YearBuilt,Lattitude,Distance and Longtitude Against Price

plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(4,4,1)
sns.scatterplot(data= num,y='Price',x='YearBuilt',color='green');
plt.subplot(4,4,2)
sns.scatterplot(data= num,y='Price',x='Lattitude',color='gold');
plt.subplot(4,4,3)
sns.scatterplot(data= num,y='Price',x='Distance',color='green');

plt.subplot(4,4,4)
sns.scatterplot(data= num,y='Price',x='Longtitude',color='gold');
# Scatterplots of Landsize,BuildingArea and Postcode Against Price
plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,3,1)
sns.scatterplot(data= num,y='Price',x='Landsize',color='green');
plt.subplot(3,3,2)
sns.scatterplot(data= num,y='Price',x='BuildingArea',color='gold');
plt.subplot(3,3,3)
sns.scatterplot(data= num,y='Price',x='Postcode',color='green');

#Scatterplots of Bedroom2,Bathroom , Rooms and Car Against Price
plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(4,4,1)
sns.scatterplot(data= num,y='Price',x='Bathroom',color='gold');
plt.subplot(4,4,2)
sns.scatterplot(data= num,y='Price',x='Bedroom2',color='green');
plt.subplot(4,4,3)
sns.scatterplot(data= num,y='Price',x='Rooms',color='gold');
plt.subplot(4,4,4)
sns.scatterplot(data= num,y='Price',x='Car',color='green');



plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)
plt.subplot(3,3,1)
sns.regplot(x='Landsize',y='Price',data=num,color='orange')
plt.subplot(3,3,2)
sns.regplot(x='BuildingArea',y='Price',data=num,color='blue')
plt.subplot(3,3,3)
sns.regplot(x='Postcode',y='Price',data=num,color='red')
plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)
plt.subplot(4,4,1)
sns.regplot(x='Bathroom',y='Price',data=num,color='orange')
plt.subplot(4,4,2)
sns.regplot(x='Bedroom2',y='Price',data=num,color='blue')
plt.subplot(4,4,3)
sns.regplot(x='Rooms',y='Price',data=num,color='orange')
plt.subplot(4,4,4)
sns.regplot(x='Car',y='Price',data=num,color='blue')

plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)
plt.subplot(4,4,1)
sns.regplot(x='YearBuilt',y='Price',data=num,color='green')
plt.subplot(4,4,2)
sns.regplot(x='Lattitude',y='Price',data=num,color='red')
plt.subplot(4,4,3)
sns.regplot(x='Longtitude',y='Price',data=num,color='green')
plt.subplot(4,4,4)
sns.regplot(x='Distance',y='Price',data=num,color='red')
# Relationship between Regionname and Price
plt.figure(figsize=(15,10))
plt.xticks(rotation=90)
sns.boxenplot(x= cat['Regionname'],y=num['Price'])
plt.show()


plt.figure(figsize=(15,10))

sns.violinplot(y=num['Price'],x=cat['Method'],palette='terrain_r')
plt.show()
#plt.subplot(2,2,2)



plt.figure(figsize=(26,20))
plt.xticks(rotation=90)
sns.boxplot(x=cat['Date'],y=num['Price'])
plt.show()

plt.figure(figsize=(15,25))
plt.xticks(rotation=90)
sns.boxplot(y=num['Price'],x=cat['CouncilArea'],palette='winter')
plt.show()




plt.figure(figsize=(8,10))
plt.xticks(rotation=90)
sns.boxenplot(y=num['Price'],x=cat['Type'],palette='spring')
plt.show()
