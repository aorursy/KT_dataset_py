

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_palette('pastel')
data=pd.read_csv('../input/BlackFriday.csv')
data.head()
data.info()
data.describe()
#we can see that in the columns of Product_Category_2 and Product_Category_3 we have missing values,

# what you can notice is that they are a type of product,

#so these naN, are types of products that are not bought by the customer.
#Total Unique Clients

data.User_ID.nunique() 
#Total Unique Products

data.Product_ID.nunique()
Gender=data.loc[:,["User_ID","Gender"]]

Gender=Gender.drop_duplicates()
Gender.head()
Gender.Gender.value_counts()
#We can see that the majority of buyers are Men, a bit of surprise here since

#Women are the ones prone to go shopping. You can have this relate to the type of store that is.
DG=pd.DataFrame(Gender["Gender"].value_counts())

DG.plot(kind='pie', subplots=True, figsize=(5,5),autopct='%1.1f%%',explode=(0.05,0))
#We want to know which is the age range,that bought more this day and which less

#so we can target the age range for future projects

Age=data.loc[:,["User_ID","Age","Gender"]]

Age=Age.drop_duplicates()
Age.Age.value_counts()
sns.countplot(data=Age,x='Age', hue='Gender')
# We notice that the biggest buyers are Men, in any age range
Age_rang=pd.DataFrame(Age['Age'].value_counts())

Age_rang.plot(kind='pie', subplots=True, figsize=(10,7),autopct='%1.1f%%')
#Great, we can see that 34.8% are in the range between 26-35 years old, 

#followed by 19.8% between 35-45 and 18.1% between 18-25
Oc=data.loc[:,["User_ID","Occupation"]]

Oc=Oc.drop_duplicates()

Oc.Occupation.value_counts()
Oc_g=pd.DataFrame(Oc["Occupation"].value_counts())

sns.countplot(data=Oc, x = 'Occupation')
#this dataset has a categorical variable of occupation, we can see that those who occupy the 4 and the 0,

#are the biggest customers who buy something in the store
#Simply know how many products were sold in each category and a small data cleaning :p

data = data.fillna(0)
a=data.Product_Category_1.sum()

b=data.Product_Category_2.sum()

c=data.Product_Category_3.sum()

print(a,b,c)
#purchased of product units

data.Product_Category_1.sum()+data.Product_Category_2.sum()+data.Product_Category_3.sum()
#this is the totality of products purchased, this can be used for comparisons of previous or future years
#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------

#We want to know what category bought the most in value.

cons=data[['Occupation','Purchase']].groupby('Occupation').sum()
sort=cons.sort_values('Purchase', ascending=False)

print(sort)
city=data[['City_Category','Purchase']].groupby('City_Category').sum()

city
plt.figure(figsize =(10,3))

sns.set_style('whitegrid')

sns.set_palette('pastel')

data.groupby('City_Category').Purchase.sum().plot('bar')

plt.ylabel('City Category', fontsize=12)

plt.xlabel('Total Purchase Amounts', fontsize=12)

plt.title('Total Purchase Amounts of different City Categories', fontsize=12)

plt.show()



plt.pie(data["City_Category"].value_counts().values, labels=["B","C","A"], autopct="%1.0f%%", wedgeprops={"linewidth":1,"edgecolor":"white"})



plt.show()
#Well now we know the city that sold the most in total amount.