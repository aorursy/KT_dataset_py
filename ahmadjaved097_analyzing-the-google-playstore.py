import os

#print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/googleplaystore.csv')

df.head(3)
print('Different types of App Categories as present in the dataset are: ')

print('--------------------------------------------------------------------')



count = 1

for i in df['Category'].unique():

    print(count,': ',i)

    count = count + 1
df[df['Category'] == '1.9']
df.drop(df.index[[10472]],inplace = True)     #Removing the app on row 10472
sns.set_style('whitegrid')

plt.figure(figsize=(16,8))

plt.title('Number of apps on the basis of category')

sns.countplot(x='Category',data = df)

plt.xticks(rotation=90)

plt.show()
category = pd.DataFrame(df['Category'].value_counts())        #Dataframe of apps on the basis of category

category.rename(columns = {'Category':'Count'},inplace=True)
plt.figure(figsize=(15,6))

sns.barplot(x=category.index[:10], y ='Count',data = category[:10],palette='hls')

plt.title('Top 10 App categories')

plt.xticks(rotation=90)

plt.show()
family_category = len(df[df['Category'] == 'FAMILY'])/len(df)*100

games_category = len(df[df['Category'] == 'GAME'])/len(df)*100

beauty_category = len(df[df['Category'] == 'BEAUTY'])/len(df)*100

print('Percentage of Apps in the family category: {}%'.format(round(family_category,2)))

print('Percentage of Apps in the games category: {}%'.format(round(games_category,2)))

print('Percentage of Apps in the beauty category: {}%'.format(round(beauty_category,2)))
plt.figure(figsize=(15,8))

sns.countplot(x='Rating',data = df)

plt.xticks(rotation =90)

plt.title('Countplot for ratings')             

plt.show()
rating_greater_4 = len(df[df['Rating'] >= 4])/len(df)*100

print('Percentage of Apps having ratings of 4 or greater: {}%'.format(round(rating_greater_4,2)))
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M',''))

df['Size'] = df['Size'].apply(lambda x: str(x).replace('k','e-3'))
#Converting the data type of Size category to float wherever possible

def convert(val):

    try:

        return float(val)

    except:

        return val

df['Size'] = df['Size'].apply(lambda x: convert(x))
#Seperate the apps whose size is given from those whose size varies with the device.

sized = df[df['Size'] != 'Varies with device'].copy()
sized['Size'] = pd.to_numeric(sized['Size'])
plt.figure(figsize=(12,6))

plt.title('Distribution of App Sizes')

sns.distplot(sized['Size'],bins = 30,rug=True)

plt.show()
size_less_20 = len(sized[sized['Size'] <= 50 ])/len(sized)*100

print('Percentage of Apps in the beauty category: {}%'.format(round(size_less_20,2)))
order = ['0','0+','1+','5+','10+','50+','100+','500+','1,000+','5,000+','10,000+','50,000+','100,000+','500,000+','1,000,000+',

         '5,000,000+','10,000,000+',

         '50,000,000+','100,000,000+','500,000,000+','1,000,000,000+']

sns.set_style('whitegrid')

plt.figure(figsize=(22,8))

plt.title('Number of apps on the basis of Installs')

sns.countplot(x='Installs',data = df,palette='hls',order = order)

plt.xticks(rotation = 90)



plt.show()
print('{}% apps in the play store having more than 1,000,000 installs and {}% apps have more than 10,000,000+ downloads' .format(round(len(df[df['Installs'] == '1,000,000+'])/len(df)*100,2),round(len(df[df['Installs'] == '10,000,000+'])/len(df)*100,2)))
print('Apps on the basis of Type are classified as')

print('--------------------------------------------------------------------')



count = 1

for i in df['Type'].unique():

    print(count,': ',i)

    count = count + 1
plt.figure(figsize=(10,6))



# Data to plot

labels = ['Free','Paid']

sizes = [len(df[df['Type'] == 'Free']),len(df[df['Type'] == 'Paid'])]

colors = ['skyblue', 'yellowgreen','orange','gold']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.title('Percentage of Free and paid apps in playstore')

plt.pie(sizes, labels=labels,

autopct='%1.1f%%', startangle=380,colors=colors,explode=explode)



plt.axis('equal')

plt.show()
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$',''))

df['Price'] = pd.to_numeric(df['Price'])
paid_apps = df[df['Price'] != 0]
plt.figure(figsize=(8,6))

plt.title('Distribution of Paid App Prices')

sns.distplot(paid_apps['Price'],bins=50)

plt.show()
price_less_10 = len(paid_apps[paid_apps['Price'] <= 10])/len(paid_apps)*100

print('Percentage of Apps having price less than 10$: {}%'.format(round(price_less_10,2)))
paid_apps[paid_apps['Price'] >= 350]

print('Apps on the basis of Content Rating are classified as')

print('-------------------------------------------------------------------')



count = 1

for i in df['Content Rating'].unique():

    print(count,': ',i)

    count = count + 1
plt.figure(figsize=(12,6))

sns.countplot(x=df['Content Rating'],palette='hls')

plt.show()
print('Percentage of Apps having content rating as everyone: {}%'.format(round(len(df[df['Content Rating'] == 'Everyone'])/len(df)*100,2)))
plt.figure(figsize=(22,8))

plt.title('Number of Apps on the basis of Genre')

sns.countplot(x='Genres',data = df,palette='hls')

plt.xticks(rotation = 90)

plt.show()
print('Total Number of Genres: ',df['Genres'].nunique())
plt.figure(figsize=(22,8))

plt.title('Number of Apps on the basis of Android version required to run them')

sns.countplot(x='Android Ver',data = df.sort_values(by = 'Android Ver'),palette='hls')

plt.xticks(rotation = 90)



plt.show()
#function to convert columns to numeric data type from object data type

for i in df.columns:

    try:

        df[i] = pd.to_numeric(df[i])

    except:

        pass
plt.figure(figsize=(20,6))

sns.boxplot(x='Category',y='Rating',data = df)

plt.xticks(rotation=90)

plt.title('App ratings across different categories')

plt.show()
rating = pd.DataFrame(df['Rating'].describe()).T

rating
sns.set_style('whitegrid')

plt.figure(figsize=(15,8))

sns.scatterplot(y='Category',x='Reviews',data = df,hue='Category',legend=False)

plt.xticks(rotation=90)

plt.title('Number of reviews on the basis of Category')

plt.show()
#Number of apps having 0 reviews

len(df[df['Reviews'] == 0])

review_0_category = pd.DataFrame(df[df['Reviews'] == 0]['Category'].describe())

#App having maximum reviews.

max_review_app = df[df['Reviews'] == max(df['Reviews'])]