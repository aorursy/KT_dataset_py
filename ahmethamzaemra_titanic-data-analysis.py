# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import unicodecsv

import seaborn as sns

import matplotlib.pyplot as plt

import os

path="../input"

os.chdir(path)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_df=pd.read_csv('train.csv')

titanic_df.head()
titanic_df.describe()
numeric_variables=list(titanic_df.dtypes[titanic_df.dtypes!='object'].index)

titanic_df[numeric_variables].head()
#Standarilizng the data Fare

def standardize_colum(column):

    return (column-column.mean())/column.std()
standardize_colum(titanic_df['Fare']).plot()

plt.title("Standardized Fare Chart")

plt.xlabel("Passenger Id")

plt.ylabel("standardized fare value")

plt.show()
average_age_titanic    =titanic_df['Age'].mean()

std_age_titanic        =titanic_df['Age'].std()

count_nan_age_titanic  =titanic_df['Age'].isnull().sum()



rand_1=np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic, size=count_nan_age_titanic)

# plot original Age values

# fill NaN values in Age column with random values generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

#convert them to int

titanic_df['Age']=titanic_df['Age'].astype(int)



titanic_df['Age'].hist(bins=70)

plt.title('Ages of peoples in Titanic')

plt.xlabel('Ages')

plt.ylabel('Number of people')

plt.show()
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, titanic_df['Age'].max()))

facet.add_legend()

fig, axis1 = plt.subplots(1,1,figsize=(18,6))

average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)

plt.show()
df1=(titanic_df.groupby(['Survived', 'Sex'])).count().unstack('Sex')['PassengerId']

df1[['male', 'female']].plot(kind='bar', stacked=True)

labels=['Died', 'Survived']



plt.title("Survived and Gender Relation")

plt.ylabel("number of people")

plt.show()
total_gender=titanic_df.groupby('Sex').size()

port_class_groups=titanic_df.groupby(['Sex'], as_index=False).get_group('female')

famele_survive=port_class_groups.groupby('Survived').count()*100/port_class_groups.count()
labels='famele died','famele survived'

values=famele_survive["Age"]

plt.pie(values, labels=labels,autopct='%1.1f%%', shadow=True)

plt.show()
total_gender=titanic_df.groupby('Sex').size()

port_class_groups=titanic_df.groupby(['Sex'], as_index=False).get_group('male')

famele_survive=port_class_groups.groupby('Survived').count()*100/port_class_groups.count()

labels='male died','male survived'

values=famele_survive["Age"]

plt.pie(values, labels=labels,autopct='%1.1f%%', shadow=True)

plt.show()
df2 =titanic_df.groupby(['Survived', 'Pclass'])['PassengerId'].count().unstack('Survived').fillna(0)

df2
df2[[0, 1]].plot(kind='bar', stacked=False)

plt.title('Embarked and Classes effect on surviving')

plt.ylabel('Number of People')

plt.xlabel("Passenger classes")

plt.legend(['Survived', 'Died'])

plt.show()
def correlation(x,y):

    std_x=(x-x.mean())/x.std(ddof=0)

    std_y=(y-y.mean())/y.std(ddof=0)

    return (std_x*std_y).mean()
tdf=titanic_df.dropna(subset=['Age'])
correlation(tdf['Age'],tdf['Fare'])