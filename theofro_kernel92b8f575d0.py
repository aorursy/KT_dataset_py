# First, we import the packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Data visualisation 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')
# Data visualisation 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline 
import sklearn
import warnings
warnings.filterwarnings('ignore')
# Let's import the data and have a quick look
dt = pd.read_csv('../input/BlackFriday.csv')
dt.head()
# Quick check for outliers in the Purchase variable 
dt_out = dt[np.abs(dt['Purchase']-dt['Purchase'].mean()) <= (3*dt['Purchase'].std())]
print(len(dt['Purchase']))
print(len(dt_out['Purchase']))
# Next a check for missing values 
dt.isnull().sum()
# Only two categories are missing data, and we can assume it is just that the product does not have subcategories
# Let's have a look at the values inside Product_Category_2 & Produc_Category_3 
print(dt['Product_Category_2'].unique())
print(dt['Product_Category_3'].unique())
# We therefore replace missing values with 0. Note that these are floats. 
# The float later becaming a nuisance in analysis so I have transformed the values to integers. 
dt.fillna(value=0,inplace=True)
dt['Product_Category_2'] = dt['Product_Category_2'].astype(int)
dt['Product_Category_3'] = dt['Product_Category_3'].astype(int)

dt.head()
# Since we are interested in customer behaviour, we aggregate Purchases to the customer level.
table = pd.pivot_table(dt, values='Purchase', index=['User_ID'], aggfunc=np.sum)
# Now we merge back client characteristics using User ID
pv = table.merge(dt, on='User_ID') 
pv.drop_duplicates(['User_ID'],keep='first',inplace=True) # Keep only one since you currently have tons of duplicates since User_ID can happen many times
pv.reset_index(drop=True,inplace=True)

# I drop categories I am not interested in at this point or that I have no way of interpreting. Ie: what are the product categories
pv.drop(['Product_ID','Product_Category_1','Product_Category_2','Product_Category_3','Purchase_y'],axis=1,inplace=True)
pv.rename(columns={'Purchase_x':'Purchase'},inplace=True)

# Let's have a look at our new database
pv.head()
# Now that we have a cleaner dataset to work with, lets look at some descriptives
f = plt.figure(figsize=(15,5))

ax = f.add_subplot(1,3,1)
sns.countplot(x='Age', data=pv, palette ='YlGnBu', order=dt['Age'].value_counts().index)
plt.title('Distribution of Age', fontsize='x-large')

ax = f.add_subplot(1,3,2)
sns.countplot(x='Gender', data=pv, palette='RdBu')
plt.title('Distribution of Gender', fontsize='x-large')

ax = f.add_subplot(1,3,3)
sns.countplot(x='Marital_Status', data=pv, palette='RdBu')
plt.title('Distribution of Marital Status', fontsize='x-large')
plt.xticks((0,1),('Not Married','Married'))
plt.xlabel('Marital Status')
# Note that Marital Status = 1 means married

plt.tight_layout()
f = plt.figure(figsize=(15,8))

ax = f.add_subplot(1,2,1)
sns.countplot(x='City_Category', data=pv.sort_values(by='City_Category'), palette ='YlGnBu')
ax.set_xlabel('City Category')

ax = f.add_subplot(1,2,2)
sns.countplot(x='Stay_In_Current_City_Years', data=pv, palette='YlGnBu', order=dt['Stay_In_Current_City_Years'].value_counts().index)
hide = ax.set_xlabel('Length of Stay in City in Years')
# Let's begin with a distribution of spending in our data
fig = plt.figure(figsize=(10,10))

pv['Purchase'].hist( rwidth = 0.8, bins=50,color ='g',alpha =.2)
plt.xlabel('Total spending per Client')
plt.ylabel('Number of Clients')

plt.title('Customer total spending Distribution')
x_tickers = plt.xticks(np.linspace(0,10000000,6),['0m','2m','4m','6m','8m','10m'])
# Gender
table = pd.pivot_table(pv, values='Purchase', index=['Gender'], aggfunc=np.mean)
table.round(2)
# Marital Status
table = pd.pivot_table(pv, values='Purchase', index=['Marital_Status'], aggfunc=np.mean)
table.round(2)
# Age
table = pd.pivot_table(pv, values='Purchase', index=['Age'], aggfunc=np.mean)
table.round(2)
# Occupation 
table = pd.pivot_table(pv, values='Purchase', index=['Occupation'], aggfunc=np.mean)
table.sort_values(by = 'Purchase', ascending=False).round(2)
# Stay in City
table = pd.pivot_table(pv, values='Purchase', index=['Stay_In_Current_City_Years'], aggfunc=np.mean)
table.sort_values(by = 'Purchase', ascending=False).round(2)
# City Category
table = pd.pivot_table(pv, values='Purchase', index=['City_Category'], aggfunc=np.mean)
table.round(2)
# Let's first look at our best products!
dt['Product_ID'].value_counts().head(10)
byProd = dt.groupby('Product_ID')

# Create a Dataframe with the number of purchases for a Product
prod = pd.DataFrame(byProd.count()['User_ID'])

# Create a Dataframe with the mean Purchase Price for a Product
mean_price = pd.DataFrame(byProd.mean()['Purchase']).round(2)

# Merge to create a database with columns: Product_ID, Nb_Sales, Mean_Price
prodprice = prod.merge(mean_price, on='Product_ID')
prodprice.rename(columns={'User_ID':'Nb_Sales','Purchase':'Mean_Price'},inplace=True)

prodprice.head()
# Have a quick look at the distribution of the mean price
fig = plt.figure(figsize=(10,10))

plot = plt.hist(prodprice['Mean_Price'], rwidth=0.95, color = 'g', alpha = .2)
x = plt.xlabel('Mean Product Price')
x = plt.title('Distribution of Mean Product Price')
# We plot this to see if there is an interesting observable relationship 
fig = plt.figure(figsize=(10,10))
x = sns.scatterplot(x='Mean_Price', y = 'Nb_Sales',data=prodprice)
f = plt.figure(figsize=(15,5))

ax = f.add_subplot(1,3,1)
sns.countplot(dt['Product_Category_1'],palette='Blues')

ax = f.add_subplot(1,3,2)
sns.countplot(dt[dt['Product_Category_2'] != 0]['Product_Category_2'],palette='Greens') 
# We remove the 0/NaN values else we get a huge left tail at 0 and can't really see the rest

ax = f.add_subplot(1,3,3)
sns.countplot(dt[dt['Product_Category_3'] != 0]['Product_Category_3'], palette ='Reds')

plt.tight_layout()
#we use only the main product category, but you could aggregate based on all three
byprodCat1 = dt.groupby('Product_Category_1').sum()['Purchase']
dtprod = pd.concat([pd.Series(range(1,19)),byprodCat1.reset_index(drop=True)],axis=1)
dtprod.rename(columns = {0:'ProdCat'}, inplace = True)

fig = plt.figure(figsize=(10,10))
sns.barplot(x='ProdCat',y='Purchase',data=dtprod,palette='Blues_r')
xlab = plt.xlabel('Product Category')
ylab = plt.ylabel('Total Purchases')
title = plt.title('Total Spending per Category', fontsize = 'x-large')
# I plot this here to see if there is a difference in product preference based on age
x= sns.catplot(x="Product_Category_1", hue="Age", kind="count", palette="pastel", edgecolor=".6",data=dt)
# Let's check if the Gender has an impact
x = sns.catplot(x="Product_Category_1", hue="Gender", kind="count", palette="pastel", edgecolor=".6",data=dt)
class_tests = pv.copy()
class_tests.head()
# Create a Dummy indicating a total purchase in the top 30 percentil
class_tests['top_30%'] = np.where(class_tests['Purchase'] >= class_tests['Purchase'].quantile(.7),1,0)
# The intervals for age need transformation
def map_age(age):
    if age == '0-17':
        return 8.5
    elif age == '18-25':
        return 21.5
    elif age == '26-35':
        return 30.5
    elif age == '36-45':
        return 40.5
    elif age == '46-50':
        return 48
    elif age == '51-55':
        return 53
    else:
        return 57
class_tests['Age_new'] = class_tests['Age'].apply(map_age)
#Create dummies for Gender and City Category 
dummies = pd.get_dummies(class_tests['Gender'], drop_first=True)#, drop_first=True
class_tests = pd.concat([class_tests,dummies],axis = 1)

dummies = pd.get_dummies(class_tests['City_Category'], drop_first=True) #,drop_first=True
class_tests = pd.concat([class_tests,dummies],axis = 1)

class_tests.head()
# Creating a variable which indicates the average spending of the occupation

code4 = (20,19,5,16,3)
code3 = (2,0,18,4,14)
code2 = (8,15,11,7,6)
code1 = (1,12,17,9,10,13)

def map_occu(occupation):
    if occupation in code1:
        return 1
    elif occupation in code2:
        return 2
    elif occupation in code3:
        return 3
    elif occupation in code4:
        return 4
    
class_tests['Occ_New'] = class_tests['Occupation'].apply(map_occu).astype(int)
X = class_tests.drop(['User_ID', 'Purchase', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'top_30%'],axis=1)
y = class_tests['top_30%']
X.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))

roc_auc_score(y_test,rfc_pred)
