import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import missingno
import pandas as pd

titanic_df=pd.read_csv('/kaggle/input/titaniccsv/titanic.csv')

titanic_df
#Let's Explore the Data using pandas method ( 'shape','info','describe','dtype')



print(f"Data Frame shape: {titanic_df.shape}\n=================")

print(f"Data Frame columns: {titanic_df.columns}\n=================")

print(f"The type of each column : {titanic_df.dtypes}\n=================================")

print(f"How much missing value is there: {titanic_df.isnull().sum()}\n=====================")



print(f"Descb of Catg features: {titanic_df.describe(include=['O'])}\n=====================")
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")

plt.style.use("fivethirtyeight")
#lets have a qiuck look on the number of survived and those who didn't

# 0- means perished and 1- means survived 



titanic_survived=titanic_df['Survived'].value_counts()

titanic_survived
survived_percentage=titanic_survived[1]/titanic_survived.sum()

survived_percentage
#let's start doing some viz :)



titanic_survived.plot(kind='bar',color=("darkblue","red"))



plt.title("Survival Percentage -{:0.1%} of  Passangers".format(survived_percentage))

plt.xlabel("0 = Perished, 1 = Survived")

plt.ylabel('No of Passengers')



#Now lets change the yticks to percentage to easily visualize the percentage difference

titanic_df["Survived"].value_counts(normalize=True).plot(kind="bar",color=("darkblue","red"))



plt.title("Survival Histogram")

plt.ylabel("% of Passengers")

plt.xlabel("0 = Perished, 1 = Survived")
#first lets have a quick basic look 





titanic_df['Pclass'].value_counts()


sns.countplot(y='Pclass',data=titanic_df)



# As you can see more than 50% of the Passengers were in the third-Class.

#And Around 200 hundred passengers were both in the First Class and the Second Class
titanic_df['Pclass'].value_counts(normalize='True').plot(kind='bar')
n_male=len(titanic_df[titanic_df['Sex']=='male'])

n_Female=len(titanic_df[titanic_df['Sex']=='female'])

"Males: {:.1%}, Females: {:.1%}".format(n_male / len(titanic_df), n_Female / len(titanic_df))

sns.catplot('Embarked', data=titanic_df, hue='Pclass', kind='count', order=['C', 'Q', 'S'])

#Most of the Passengers came from S - Southampton
g = sns.catplot(x='Sex',y='Survived', data=titanic_df, kind="bar")

g.set_ylabels("Survival Probability")

plt.show()
missingno.matrix(titanic_df,figsize=(12,9))
titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].mean())
# We shall use the cut method  to convert continuous ages into categorical groups.

Category=pd.cut(titanic_df['Age'],bins=[0,2,17,65,99],labels=['Toddler/Baby','Child','Adult','Elderly'])

titanic_df.insert(5,"Age Group",Category)
titanic_df.head(10)
plt.figure(figsize=(12,7))

ax=sns.catplot(x='Age Group',y='Survived', data=titanic_df, kind="point")

ax.set_ylabels("Survival Probability")

#Let's see if the class of the passengers had an effect on their survival rate.

plt.figure(figsize=(12,7))

sns.catplot(x='Pclass',y='Survived',data=titanic_df, kind="point")
sns.catplot(x='Pclass',y='Survived',hue='Age Group',data=titanic_df, kind="point")
#Let's see if the Ticket fare  of the passengers had an effect on their survival rate.

sns.lmplot('Fare', 'Survived',hue='Pclass', data=titanic_df)


#first let's make a new column named 'Family_members '





titanic_df['Family_members']=titanic_df['Parch']+titanic_df['SibSp']

titanic_df.head()
# Look for > 0 or == 0 to set alone status

titanic_df.loc[titanic_df['Family_members'] != 0, 'Family_members'] = 'with Family'

titanic_df.loc[titanic_df['Family_members'] == 0, "Family_members"] = 'Alone'
titanic_df.head(10)
# Now let's get a simple visualization!

sns.catplot(x='Family_members',y='Survived',hue='Pclass' ,data=titanic_df, kind='point')