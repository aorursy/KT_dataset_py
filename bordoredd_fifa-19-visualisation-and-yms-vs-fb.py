#import dataset and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sbs
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#checking the first columns and rows
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.head()
data.info() #general information
#statistics about dataset
data.describe()
data.columns # Shows Columns
#Exploring Data
fig = plt.figure(figsize=(25, 10))
p = sbs.countplot(x='Nationality', data=data)
_ = plt.setp(p.get_xticklabels(), rotation=90)
sns.countplot(data['Age'])
plt.xticks(rotation=90)
plt.title('Age distribution')
plt.show()
#Turkish Footballers
def country(x):
    return data[data["Nationality"]==x][["Name","Age","Overall","Potential","Position"]]

country("Turkey")
#Search Club
def club(x):
    return data[data["Club"]==x][["Name","Age","Overall","Potential","Nationality"]]

club("Yeni Malatyaspor")
#Search Players
def Player_Search(x):
    return data[data["Name"]==x][["Name","Club","Age","Overall","Potential","Position"]]

Player_Search("Guilherme")
#Filtering Age
a=data["Age"]>41
data[a]
#Filtering Age with Nationality
data[np.logical_and(data['Age']>35, data['Nationality']=="Turkey" )]
#Preferred Foot
p = sbs.countplot(x='Preferred Foot', data=data)
# Top 10 left footed footballers

data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10).style.background_gradient('magma')
# Top 10 right footed footballers

data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10).style.background_gradient('magma')
yms=data[data["Club"]=='Yeni Malatyaspor']
print(yms.head())
fb=data[data["Club"]=="Fenerbahçe SK"]
print(fb.head())
print(fb['Position'].value_counts())

print(yms['Position'].value_counts())

sns.distplot(yms['Age'],color='black')
sns.distplot(fb['Age'],color='blue')
plt.title('Comparison of distribution of Age between Yeni Malatyaspor and Fenerbahçe players')
plt.tight_layout()
plt.show()
plt.subplot(1,2,1)
sns.countplot(fb['Nationality'],color='blue')
plt.xticks(rotation=90)
plt.title('Fenerbahçe SK')
plt.subplot(1,2,2)
sns.countplot(yms['Nationality'],color='black')
plt.xticks(rotation=90)
plt.title('Yeni Malatyaspor')
plt.tight_layout()
plt.show()
plt.subplot(1,2,1)
sbs.countplot(x='Preferred Foot', data=yms)
plt.title('Yeni Malatyaspor')
plt.subplot(1,2,2)
sbs.countplot(x='Preferred Foot', data=fb)
plt.title('Fenerbahçe')
plt.tight_layout()
plt.show()