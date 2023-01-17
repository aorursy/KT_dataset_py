# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')

data.info()


data.drop('Unnamed: 0',axis=1,inplace=True)

data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8, inplace = True)

data['Body Type'].fillna('Normal', inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(1, inplace = True)

data['Wage'].fillna('€200K', inplace = True)
data.head()
data.columns
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

sns.pairplot(data[['Age','Overall','Potential','Stamina','Strength']])
data.head()
sns.countplot(data['Age'])

plt.xticks(rotation=90)

plt.title('Age distribution')

plt.show()
sns.distplot(data['Age'])

plt.xticks(rotation=90)

plt.title('Age distribution')

plt.show()
countries=data['Nationality'].value_counts()

index=countries.index

con=pd.DataFrame({'Country':index,'Count':countries})

con['Percentage%']=(con['Count']/con['Count'].sum())*100

con
con=con['England':'Colombia']

plt.pie(con['Count'],labels=con['Country'],wedgeprops = {'linewidth': 3},autopct='%1.1f%%')

plt.title('Distribution of players from top 8 countries')

plt.tight_layout()

plt.show()
barca=data[data['Club']=='FC Barcelona']

madrid=data[data['Club']=='Real Madrid']

print(barca.head())

print(madrid.head())
print(barca['Position'].value_counts())

print(madrid['Position'].value_counts())
sns.distplot(barca['Age'],color='maroon')

sns.distplot(madrid['Age'],color='black')

plt.title('Comparison of distribution of Age between Barcelona and Real Madrid players')

plt.tight_layout()

plt.show()
sns.distplot(barca['Overall'],color='maroon')

sns.distplot(madrid['Overall'],color='black')

plt.title('Comparison of distribution of overall rating between Barcelona and Real Madrid players')

plt.tight_layout()

plt.show()
barca['Wage']=barca['Wage'].str[1:-1]

barca['Wage(in K)']=barca['Wage'].astype(int)

barca.drop('Wage',axis=1,inplace=True)



madrid['Wage']=madrid['Wage'].str[1:-1]

madrid['Wage(in K)']=madrid['Wage'].astype(int)

madrid.drop('Wage',axis=1,inplace=True)



print(barca['Wage(in K)'].sample(5))

print(madrid['Wage(in K)'].sample(5))
sns.distplot(barca['Wage(in K)'],color='maroon')

sns.distplot(madrid['Wage(in K)'],color='black')

plt.title('Wage comparison between Barca and Madrid players')

plt.tight_layout()

plt.show()
plt.subplot(1,2,1)

sns.countplot(barca['Nationality'],color='maroon')

plt.xticks(rotation=90)

plt.title('Barcelona')

plt.subplot(1,2,2)

sns.countplot(madrid['Nationality'],color='black')

plt.xticks(rotation=90)

plt.title('Real Madrid')

plt.tight_layout()

plt.show()
barca.head()
cols=['Preferred Foot','International Reputation','Weak Foot','Skill Moves']

for col in cols:

    plt.subplot(1,2,1)

    sns.countplot(barca[col])

    plt.title('Barcelona')

    plt.subplot(1,2,2)

    sns.countplot(madrid[col])

    plt.title('Real Madrid')

    plt.tight_layout()

    plt.show()
barca['Weight']=barca['Weight'].str[:-3]

barca['Weight']=barca['Weight'].astype(int)



madrid['Weight']=madrid['Weight'].str[:-3]

madrid['Weight']=madrid['Weight'].astype(int)
sns.distplot(barca['Weight'],color='maroon')

sns.distplot(madrid['Weight'],color='black')

plt.show()