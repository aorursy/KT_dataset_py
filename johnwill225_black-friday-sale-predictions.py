import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv("/kaggle/input/black-friday-predictions/train.csv")

test = pd.read_csv("/kaggle/input/black-friday-predictions/test.csv")
print(train.shape)

print(test.shape)

train.head()
sns.distplot(train['Purchase'])

print("Skewness : {}".format(train['Purchase'].skew()))

print("Kurtosis : {}".format(train.Purchase.kurt()))
# The distribution is moderately skewed
print(train['Purchase'].describe())

print(train[train['Purchase'] == train['Purchase'].min()].shape[0])

print(train[train['Purchase'] == train['Purchase'].max()].shape[0])
train.isnull().sum()
test.isnull().sum()
# Let's analyse the missing value

# Only this predictors Product_Category_2 & Product_Category_3 has missing values this might be due to that products did not fall under these two categories

train[train['Product_Category_2'].isnull()]['Product_ID'].value_counts()
# We analyse firt two top products

print(train[train['Product_ID']=='P00255842']['Product_Category_2'].value_counts(dropna=False))

print(train[train['Product_ID']=='P00278642']['Product_Category_2'].value_counts(dropna=False))
train[train['Product_Category_3'].isnull()]['Product_ID'].value_counts()
# We analyse firt two top products

print(train[train['Product_ID']=='P00265242']['Product_Category_3'].value_counts(dropna=False))

print(train[train['Product_ID']=='P00058042']['Product_Category_3'].value_counts(dropna=False))
# Our guess is correct that product doesn't fall under these categories, so it is safe to fill 0

train['Product_Category_2'].fillna(0,inplace=True)

test['Product_Category_2'].fillna(0,inplace=True)

train['Product_Category_3'].fillna(0,inplace=True)

test['Product_Category_3'].fillna(0,inplace=True)
# we remove '+' character

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].replace("4+","4")

test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].replace("4+","4")



train['Age'] = train['Age'].replace('55+','56-100')

test['Age'] = test['Age'].replace('55+','56-100')
# Product ID has so many unique values that won't help us but there is a pattern on product formation. We will split first 4 

# characters this might be some sellers name or for some identification they kept on it
train['Product_Name'] = train['Product_ID'].str.slice(0,4)

test['Product_Name'] = test['Product_ID'].str.slice(0,4)
sns.countplot(train['Product_Name'])
train.groupby('Product_Name')['Purchase'].describe().sort_values('count',ascending=False)
# Items which are only fall under Product_Category_1 list

pd_cat_1_purchase = train[(train['Product_Category_2'] == 0) & (train['Product_Category_3']==0)]['Purchase']

print("Total no. of Sold Items in Product_Category_1 {}".format(pd_cat_1_purchase.shape[0]))

print("Mean value {}".format(pd_cat_1_purchase.mean()))

print("Median value {}".format(pd_cat_1_purchase.median()))



# Items which are available in any two category

pd_cat_2_purchase = train[np.logical_xor(train['Product_Category_2'],train['Product_Category_3'])]['Purchase']

print("Total no. of Sold Items in Product_Category_1 & any one of the other two category {}".format(pd_cat_2_purchase.shape[0]))

print("Mean value is {}".format(pd_cat_2_purchase.mean()))

print("Median value is {}".format(pd_cat_2_purchase.median()))
# Items which are available in all category

pd_cat_all_purchase = train[(train['Product_Category_2'] != 0) & (train['Product_Category_3']!=0)]['Purchase']

print("Total no. of Sold Items in all Category {}".format(pd_cat_all_purchase.shape[0]))

print("Mean value is {}".format(pd_cat_all_purchase.mean()))

print("Median value is {}".format(pd_cat_all_purchase.median()))
train['Category_Weight'] = 0

train.loc[pd_cat_1_purchase.index,'Category_Weight'] = 1

train.loc[pd_cat_2_purchase.index,'Category_Weight'] = 2

train.loc[pd_cat_all_purchase.index,'Category_Weight'] = 3

# Each user has purchased atleast 6 items.

# Based on the count  we'll create a new variable called Frequent_Buyers which holds 1 for Users who purchased more than 100 items

# and 0 for less than 100 items
train['Frequent_Buyers'] = train.groupby('User_ID')['User_ID'].transform(lambda x : 1 if x.count() > 100 else 0)

test['Frequent_Buyers'] = test.groupby('User_ID')['User_ID'].transform(lambda x : 1 if x.count() > 100 else 0)
train.drop(['Product_ID','User_ID'],inplace=True,axis=1)

test.drop(['Product_ID','User_ID'],inplace=True,axis=1)
train['Age'].value_counts()
sns.barplot(train['Age'],train['Age'].value_counts().values)
# We'll create a new feature for Student

train['IsStudent'] = 1 * (train['Age']=='0-17')

test['IsStudent'] = 1 * (test['Age']=='0-17')
# Based on our income we spend more, so we'll order occupation by mean value of the purchase and we use the same order for test data also.

order_occupation_by_purchase = train.groupby('Occupation')['Purchase'].describe().sort_values('mean',ascending=False)['mean'].index
train['Occupation']
map_occupation = {k: v for v, k in enumerate(order_occupation_by_purchase)}

map_occupation
train['Occupation'] = train['Occupation'].apply(lambda x: map_occupation[x])

test['Occupation'] = test['Occupation'].apply(lambda x: map_occupation[x])
corrIndex = train.corr().nlargest(10,'Purchase')['Purchase'].index

corr = train[corrIndex].corr()
plt.figure(figsize=(16,8))

ax = sns.heatmap(corr,annot=True,cmap="YlGnBu")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
# There is no satisifactory correlation feature so we will avoid using Linear model.
f,ax = plt.subplots(1,2,figsize=(10,6))

sns.countplot(train['Gender'],ax=ax[0])

sns.barplot('Gender','Purchase',data=train,ax=ax[1])
f,ax = plt.subplots(1,2,figsize=(10,6))

sns.countplot(train['City_Category'],ax=ax[0])

sns.barplot('City_Category','Purchase',data=train,ax=ax[1])

# Customer from city B has purchased more items.

# Customer from city C has spent higher Amount Eventhough B has purchased more items.