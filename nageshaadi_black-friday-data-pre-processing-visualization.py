import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing data into python from the given csv file
dataset= pd.read_csv('../input/BlackFriday.csv')
print("****** Dataset - head *****", dataset.head(), sep="\n", end="\n\n\n\n")
print("****** Dataset - tail *****", dataset.tail(), sep="\n", end="\n\n")
dataset.dtypes
dataset.info()
dataset.columns
dataset.columns = ['UserID', 'ProductID', 'Gender', 'Age', 'Occupation', 'CityCategory',
       'StayYearsCity', 'MaritalStatus', 'ProdCat1',
       'ProdCat2', 'ProdCat3', 'Purchase']
dataset.columns
# Replacing 0s and 1s in the Marital status column with the appropriate strings
dataset['MaritalStatus'] = dataset['MaritalStatus'].replace(0, 'Unmarried')
dataset['MaritalStatus'] = dataset['MaritalStatus'].replace(1, 'Married')
dataset['MaritalStatus'].unique()
dataset_orig = dataset.copy()
# Importing required package
from sklearn.preprocessing import LabelEncoder
encode_x = LabelEncoder()
dataset.head()
# Encoding columns ProductID, 
dataset['ProductID'] = encode_x.fit_transform(dataset['ProductID'])
dataset['Gender'] = encode_x.fit_transform(dataset['Gender'])
dataset['Age'] = encode_x.fit_transform(dataset['Age'])
dataset['CityCategory'] = encode_x.fit_transform(dataset['CityCategory'])
dataset['MaritalStatus'] = encode_x.fit_transform(dataset['MaritalStatus'])
dataset.StayYearsCity.unique()
# Replacing '4+' years of with numerical number 4 
dataset['StayYearsCity'] = dataset['StayYearsCity'].replace('4+', 4)
dataset.info()
# Converting StayYearsCity from object to integer
dataset['StayYearsCity'] = dataset['StayYearsCity'].astype(str).astype(int)
dataset.info()
# Creating list of index 'without' null values on column ProdCat2
no_null_list1 = dataset[~dataset['ProdCat2'].isnull()].index.tolist()
dataset['ProductID'][no_null_list1].corr(dataset['ProdCat2'][no_null_list1])
# Creating list of index 'without' null values on column ProdCat2
no_null_list2 = dataset[~dataset['ProdCat3'].isnull()].index.tolist()
dataset['ProductID'][no_null_list2].corr(dataset['ProdCat3'][no_null_list2])
print("Missing Product Category2 values :", len(dataset)-len(no_null_list1))
print("Missing Product Category3 values :", len(dataset)-len(no_null_list2))
# Checking values contained in ProdCat2 and ProdCat3
dataset['ProdCat2'].unique()
dataset['ProdCat3'].unique()
dataset['ProdCat2'].fillna(value=0,inplace=True)
dataset['ProdCat3'].fillna(value=0,inplace=True)
# Recheck for missing values (NaN) in the dataset
dataset.isna().any()
# Obtaining categorical data in terms of Percentage for each column 
group_1 = dataset_orig.groupby(['Gender'])
group_2 = dataset_orig.groupby(["Age"])
group_3 = dataset_orig.groupby(["CityCategory"])
group_4 = dataset_orig.groupby(["Occupation"])

print (group_1[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_2[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_3[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_4[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")

plt.figure(figsize=(15,10))

# Pie chart for gender distribution
plt.subplot(2,2,1)
gender_count = [dataset_orig.Gender[dataset_orig['Gender']=='F'].count(),
                dataset_orig.Gender[dataset_orig['Gender']=='M'].count()]
gender_lab = dataset_orig.Gender.unique()
expl = (0.1,0)
plt.pie(gender_count, labels=gender_lab, explode=expl, shadow=True , autopct='%1.1f%%');

# Bar chart for Age
plt.subplot(2,2,2)
ordr =dataset_orig.groupby(["Age"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(dataset_orig['Age'], label=True, order=ordr)

# Bar chart for Occupation
plt.subplot(2,2,3)
ordr1 =dataset_orig.groupby(["Occupation"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(y=dataset_orig['Occupation'], label=True, order=ordr1)

# Donut chart for City Category
plt.subplot(2,2,4)
city_count = group_3[['Purchase']].count().values.tolist()
city_lab = dataset_orig.groupby(["CityCategory"]).count().index.values
my_circle = plt.Circle( (0,0), 0.4, color='white')
expl1 = (0,0.1,0)
plt.pie(city_count, labels=city_lab,explode=expl1, shadow=True, autopct='%1.1f%%')
plt.gcf().gca().add_artist(my_circle)


plt.show()
plt.figure(figsize=(8,6))
ordr2 =dataset_orig.groupby(["StayYearsCity"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(dataset_orig['StayYearsCity'], label=True, order=ordr2)
plt.show()
#Creating new column in the dataset 
dataset_orig['Gender_MaritalStatus'] = dataset_orig.apply(lambda x:'%s_%s' % (x['Gender'],x['MaritalStatus']),axis=1)
dataset_orig.Gender_MaritalStatus.unique()
group_5 = dataset_orig.groupby(["Gender_MaritalStatus"])
plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
count1 = group_5[['Purchase']].count().values.tolist()
lab1 = dataset_orig.groupby(["Gender_MaritalStatus"]).count().index.values
expl2 = (0,0,0.1,0.1)
plt.pie(count1, labels=lab1,explode=expl2, shadow=True, autopct='%1.1f%%')

plt.subplot(1,2,2)
sns.countplot(dataset_orig['Age'],hue=dataset_orig['Gender_MaritalStatus'])

plt.show()
# Bar chart for Age

sns.catplot(x='Gender_MaritalStatus', y='Purchase', data=dataset_orig, kind='boxen')

ordr_occ =dataset_orig.groupby(["Age"]).mean().sort_values(by='Purchase',ascending=False).index
sns.catplot(x='Age', y='Purchase', order=ordr_occ, data=dataset_orig, kind='bar')

ordr_occ =dataset_orig.groupby(["Occupation"]).mean().sort_values(by='Purchase',ascending=False).index
sns.catplot(x='Occupation', y='Purchase', order=ordr_occ, data=dataset_orig, kind='bar')

sns.catplot(x='CityCategory', y='Purchase', data=dataset_orig, kind='boxen')


plt.show()
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, cmap="YlGnBu", square=True,linewidths=.5, annot=True)
plt.show()
dataset[dataset.columns[0:]].corr()['Purchase'].sort_values(ascending=False)
# Obtaining top K columns which affects the Purchase the most
k= 8
corrmat.nlargest(k, 'Purchase')
# Replotting the heatmap with the above data
cols = corrmat.nlargest(k, 'Purchase')['Purchase'].index
cm = np.corrcoef(dataset[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
