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

# we will add * in front and after the name philip.

name='Titanic'   #Name is defined as string  

#there are 6 letters in the name philip.We can use string center method to put star in front and behind the word philip 

name.center(8,'*')
name.center(10,'*')
name.center(8,'!')
name='Ship'

#The number of letter in name is calculated by using length function and 8 is added to add 4 stars before and end of the string

print(name.center(len(name)+8,'*'))
# We will be combaining the First and the Last Name 

first_name='Leonardo'

last_name='DiCaprio'

full_name=first_name+last_name

print(full_name)
# We will add space between the two names 

full_name=first_name+' ' +last_name

print(full_name)
print(first_name +'3')

print(first_name+str(3))

print(first_name*3)
# Importing the modules for which we want to find out the Version

import matplotlib

import sklearn

import scipy 

import seaborn as sns

import pandas as pd 

import numpy as np

import sys

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
print('matplotlib: {}'.format(matplotlib.__version__))

print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
# Link for getting the list of the Unicode emoji characters and sequences, with images from different vendors, CLDR name, date, source, and keywords - https://unicode.org/emoji/charts/full-emoji-list.html

# For the codes obtained from this website replace + with 000 and put \ in front of the code as dhown below 

#U+1F600 has to be represented as \U000F600 in the code 

print('\U0001F600')  #Grimming face emoji

print('\U0001F600','\U0001F600') # Printing the Emoji twice 

print('\U0001F600','\U0001F602','\U0001F643') # Printing three different emojis
A=np.array([[1,1],[1.5,4]])

b=np.array([2200,5050])

np.linalg.solve(A,b)
import matplotlib.pyplot as plt 

x=np.linspace(0,10,10)  # Equally spaced data with 10 points 

y=np.sin(x) # Generating the sine function 

plt.plot(x,y)

plt.xlabel('Time')     #Specifying the X axis label 

plt.ylabel('Speed')    #Specifying the Y axis label 

plt.title('My Cool Chart') #Specifying the title 

plt.show()
x=np.linspace(0,10,10)

y=np.array([1,5,6,3,7,9,13,50,23,56])

plt.scatter(x,y,color='r') # We can specify the color to the points

plt.xlabel('X-Value')     #Specifying the X axis label 

plt.ylabel('Y-Value')    #Specifying the Y axis label 

plt.title('My Cool Chart') #Specifying the title 

plt.show()
import pandas as pd

data=pd.read_csv('../input/train.csv')
data.head()
data.tail() 
Total=data.isnull().sum().sort_values(ascending=False)

Percent=round(Total/len(data)*100,2)

pd.concat([Total,Percent],axis=1,keys=['Total','Percent'])

pd.crosstab(data.Survived,data.Pclass,margins=True).style.background_gradient(cmap='gist_rainbow') 

#Margins=True gives us the All column values that is sum of values colums
percent = pd.DataFrame(round(data.Pclass.value_counts(dropna=False, normalize=True)*100,2))

## creating a df with the #

total = pd.DataFrame(data.Pclass.value_counts(dropna=False))

## concating percent and total dataframe



total.columns = ["Total"]

percent.columns = ['Percent']

pd.concat([total, percent], axis = 1)
data.describe()
data['Age'].describe()
data[data['Age']>70]
data['Pclass'].unique()
data[data['Pclass']==1].head()
data[data['Pclass']==1].count()
# Create a family size variable including the passenger themselve

data["FamilySize"] = data["SibSp"] + data["Parch"]+1

#titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

print(data["FamilySize"].value_counts())
data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
data.dtypes
data.sample(5)
import seaborn as sns #importing seaborn module 

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook

plt.style.use('fivethirtyeight')
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Survived'].value_counts().plot.pie(ax=ax[0],explode=[0,0.1],shadow=True,autopct='%1.1f%%')

ax[0].set_title('Survived',fontsize=30)

ax[0].set_ylabel('Count')

sns.set(font="Verdana")

sns.set_style("ticks")

sns.countplot('Survived',hue='Sex',linewidth=2.5,edgecolor=".2",data=data,ax=ax[1])

plt.ioff() # This removes the matplotlib notifications
data["FamilySize"] = data["SibSp"] + data["Parch"]+1

#titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

#data["FamilySize"].value_counts())

sns.countplot('FamilySize',data=data)
sns.barplot(x="Embarked", y="Survived",data=data);
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data);
import re

#GettingLooking the prefix of all Passengers

data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))



#defining the figure size of our graphic

plt.figure(figsize=(12,5))



#Plotting the result

sns.countplot(x='Title', data=data, palette="hls")

plt.xlabel("Title", fontsize=16) #seting the xtitle and size

plt.ylabel("Count", fontsize=16) # Seting the ytitle and size

plt.title("Title Name Count", fontsize=20) 

plt.xticks(rotation=45)

plt.show()
Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

                   }

data['Title']=data.Title.map(Title_Dictionary)
print('Chance of Survival based on Titles:')

print(data.groupby("Title")["Survived"].mean())

#plt.figure(figsize(12,5))

sns.countplot(x='Title',data=data,palette='hls',hue='Survived')

plt.xlabel('Title',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.title('Count by Title',fontsize=20)

plt.xticks(rotation=30)

plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',data=data)

plt.ioff() 
sns.factorplot('Embarked','Survived',hue='Sex',data=data)

plt.ioff() 
plt.figure(figsize=(12,6))

data[data['Age']<200000].Age.hist(bins=80,color='red')

plt.axvline(data[data['Age']<=100].Age.mean(),color='black',linestyle='dashed',linewidth=3)

plt.xlabel('Age of Passengers',fontsize=20)

plt.ylabel('Number of People',fontsize=20)

plt.title('Age Distribution of Passengers on Titanic',fontsize=25)
#print('Minumum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.min()),'$')

print('Mean age of Passenger on Titanic:',int(data[data['Age']<=100].Age.mean()),'Years')

print('Median age of Passenger on Titanic:',int(data[data['Age']<=100].Age.median()),'Years')

#print('Maximum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=9000000].PREVAILING_WAGE.max()),'$')
sns.stripplot(x="Survived", y="Age", data=data,jitter=True)
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=data);