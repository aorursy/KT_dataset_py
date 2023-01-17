import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as mpl
titanic = pd.read_csv("../input/titanic/train.csv")

titanic
df = titanic.copy()

df.head(10)
# it is important to know what kind of data we are working with



print(df.shape)

print(df.dtypes)
# it is also important to know the no. of NaN values and where they lie



df.isnull().sum()
# we drop the 'PassengerId' column right at the beginning



df.drop('PassengerId', axis = 1, inplace = True)

df.head(10)
# let us check the survival rate of the passengers



df['Survived'].mean()
# we can group data by class and view the averages for each column



df.groupby('Pclass').mean()
# let us see what these nos. actually mean



class_grouping = df.groupby('Pclass').mean()

class_grouping['Survived'].plot.bar(width = 0.2, color = 'red')

mpl.ylabel("Survival Chances")

mpl.show()

class_grouping['Age'].plot.bar(width = 0.2)

mpl.ylabel("Age")

mpl.show()
# we drop the 'Name' column



df.drop('Name', axis = 1, inplace = True)

df.head(10)
# now we group the data by sex



sex_grouping = df.groupby('Sex').mean()

sex_grouping
sex_grouping['Survived'].plot.bar(width = 0.5, color = 'green')

mpl.ylabel("Sex")

mpl.show()
# drawing a pie chart for number of males and females aboard



males = (df['Sex'] == "male").sum()

females = (df['Sex'] == "female").sum()



print(males)

print(females)

p = [males, females]

mpl.pie(p,    #giving array

       labels = ['Male', 'Female'], #Correspndingly giving labels

       colors = ['blue', 'red'],   # Corresponding colors

       explode = (0.10, 0),    #How much the gap should me there between the pies

       startangle = 0)  #what start angle should be given

mpl.axis('equal') 

mpl.show()
# let us see how the "female" across all "Pclass" fared



class_sex_grouping = df.groupby(['Pclass','Sex']).mean()

class_sex_grouping
class_sex_grouping['Survived'].plot.bar()

mpl.ylabel("Survival Chances")

mpl.show()
# changing Value for "male, female" string values to numeric values ; male=1 and female=2



def getNumber(str):

    if str=="male":

        return 1

    else:

        return 2

df["Gender"] = df["Sex"].apply(getNumber)



#We have created a new column called 'Gender' and 

#filling it with values 1 ,2 based on the values of 'Sex' column



df.head()
# we drop the 'Sex' column, since we have no use of it now



df.drop('Sex', axis = 1, inplace = True)

df.head(10)
df['Age'].describe()
group_age_split = pd.cut(df['Age'], np.arange(0, 80, 15))

age_grouping = df.groupby(group_age_split).mean()

age_grouping['Survived'].plot.bar(color = 'red')

mpl.ylabel("Survival Chances")

mpl.show()
#finding mean survived age

meanS= titanic[titanic.Survived==1].Age.mean()

print(meanS)



# finding the mean not survived age

meanNS=titanic[titanic.Survived==0].Age.mean()

print(meanNS)
df['age'] = np.where(pd.isnull(df['Age']) & df['Survived']==1  , meanS, df['Age'])

df.head(10)
df['age'].fillna(meanNS, inplace=True)

df
# checking if 'age' column is devoid of NaN values



df.isnull().sum()
# we can safely delete the 'Age' column now



df.drop('Age', axis = 1, inplace = True)

df.head(10)
# we drop the above columns

df.drop(['Ticket','Cabin'], axis = 1, inplace = True)

df.head(10)
# applying statistical approach on the above dataframe to analyse 

# which feature or column is affecting the survival rate and which is useless column



df.describe()
df.groupby('Embarked').mean()
embark_grouping = df.groupby('Embarked').mean()

embark_grouping['Survived'].plot.bar(width = 0.2, color = 'brown')

mpl.ylabel("Survival Chances")

mpl.show()
df.isnull().sum()
df.dropna(inplace = True)

df.isnull().sum()
df