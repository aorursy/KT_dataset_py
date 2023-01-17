#Using pandas and matplotlib packages of Python to explore titanic data



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots



#read csv train file and create pandas dataframe

df = pd.read_csv('../input/train.csv')



#get data summary

print("df shape {}".format(df.shape))



print("\ndf types")

print(df.dtypes)



print("\nfirst 5 lines")

print(df.head(5))



print("\ninfo")

df.info()



print("\ndisrptive statistics of numeric columns")

print(df.describe())



print("\ndescriptive summary of non-numeric data")

obj_cat = df.dtypes[df.dtypes == 'object'].index

print(obj_cat)

print(df[obj_cat].describe())



# with simple data description methods, I can sense that there are 

# 1: Useless variables: PassengerId, Embarked

# 2. Categorical variables: Survived, Pclass, Sex

# 3. Transformable variables: Age (can be grouped), Parch + SibSp = Family, 

# 4. Correlated/dependent: Name (can relate to Sex), Ticket, Cabin and Fare



# there are missing values in Age and Cabin columns. Check more on this from set-3 and 4.



# Let us explore the variables furthur by numerical and graphical representations:



#set-1

del df["PassengerId"]   # as this is just ID assigned to each passenger



#set-2

print("\nlet us make some pie plots to see the portion of passengers in each category:")

fig, (ax1,ax2,ax3) = plt.subplots(1,3)

df.Survived.value_counts().plot(kind='pie', ax = ax1, autopct='%1.1f%%')

df.Sex.value_counts().plot(kind='pie', ax = ax2, autopct='%1.1f%%')

df.Pclass.value_counts().plot(kind='pie', ax = ax3, autopct='%1.1f%%')

ax1.axis('equal')

ax2.axis('equal')

ax3.axis('equal')

plt.show()



#set-3



# missing values in categorical data (Cabin above) can be treated as another class of nan, 

# but in numeric variables, be careful



print("\ndescribe Age")

print(df['Age'].describe())



"""there are missing values. so, find how many. detect missing values with isnull()"""

missing = np.where(df['Age'].isnull())

print(len(missing[0]))

# fill missing values with either 0 (age = 0 no sense), 

#or mean/median (need to check), or impute or split data into age groups.

df.hist(column='Age',figsize=(6,6),bins=25)

plt.show()

# common ages between 20-35, take median of Age column to fill missing values

new_age = np.where(df['Age'].isnull(),28.,df['Age'])

df['Age'] = new_age





print("\n describe the age again to check the impact of imputation: \n", df['Age'].describe())



"""I can create new variable "Family":"""

df['Family'] = df['Parch'] + df['SibSp']



"""let us check the biggest family onboard"""

big_family = np.where(df['Family'] == max(df['Family']))

print(df.iloc[big_family])



"""since value of "Survived" is 0, so this big family didn't survive :( """



#set-4

print("Now let us explore Name, Ticket, Cabin and Fare features.")

sorted(df["Name"])[0:15] 

# this is just string of characters, we can relate the family names, title etc. 

# to pick families, but since there are unique 889 names and 

# we have SibSp+Parch information to count family members, 

# so leave it just to know the name of those who actually survived.



print("\nfirst 15 lines of Ticket feature", df["Ticket"][0:15])

print("\n describe Ticket: ",df["Ticket"].describe()) 

""" Ticket variable has 680 unique values, a lot to work on before using to categorize, 

and given no logical pattern in ticket numbers, let us remove it."""

del df["Ticket"]





print("\nfirst 15 lines of Cabin feature", df["Cabin"][0:15])

print("\n describe Cabin: ",df["Cabin"].describe())

""" 145 unigue values, names have structure capital letter + number, can be reduced to some levels to make categories.

let us reduce its number of levels"""

strcabin = df["Cabin"].astype(str) 

newcab = pd.Categorical(np.array([cabin[0] for cabin in strcabin]))

print("describe transformed Cabin feature: ",newcab.describe())

"""missing values in newcab variable now has new category 'nan' that can be used

in modelling."""

df['Cabin'] = newcab





print("\n check Fare variable, detect outliers with boxplots to check the spread in data)")

df['Fare'].plot(kind='box',figsize=(6,6))

plt.show()

#central 50 % data paid less than $50, interesting correlation as Class 3 passengers are far more than other two classes.



print("\n check the outliers in Fare")

outliers = np.where(df['Fare'] == max(df['Fare']))

print(df.loc[outliers])

print("\n all those who paid the maximum fare survived !!! \n")



"""this exercise has helped me to explore data from csv file, create datframe with pandas, 

explore data numerically and graphically using boxplots, pie-charts, histograms, 

transform features, impute null values and search for interesting instances. """ 