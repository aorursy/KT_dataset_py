# import libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn
# Display Raw Data

data = pd.read_csv('../input/titanic/train.csv')



data
## Displaying Null Values In Data

data.isnull().sum()
## Filling Null Values

data['Age'] = data['Age'].fillna(data['Age'].median())

# Filling Null Age Values with the Median

data['Embarked'] = data['Embarked'].fillna('S')

# Filling Null Embarked Values with Southampton, 2 values unlikely to make a huge difference



## Dropping Unnecessary Data

data = data.drop(columns = ['Cabin'])

data = data.drop(columns = ['PassengerId'])

data = data.drop(columns = ['Name'])

data = data.drop(columns = ['Ticket'])



data
# Post-Checking For Null Values

data.isnull().sum()
data
## Read Headers

data.columns
## Describing Numerical Data

data.describe()
data.groupby(['Survived'])['Survived'].count()
# Creating a correlation matrix of the data

corrMatrix = data.corr()['Survived'].sort_values(ascending = False)

print(corrMatrix)
# Creating a heatmap to display the correlation matrix

corr = data.corr()



# Display only the lower left of the heatmap

mask = np.triu(data.corr())



# Adjust size of the heatmap

f, ax = plt.subplots(figsize=(11, 9))



# Printing the heatmap

sn.heatmap(corr, annot = True, fmt='.1g', vmin = -1, vmax = 1, center = 0, cmap = 'bone', cbar_kws = {'orientation': 'horizontal'}, mask = mask, ax = ax)
# Filter Survived

survived_data = data[data['Survived'] == 1]



# Define Passenger Class Variables

first_class_survived = survived_data[survived_data['Pclass'] == 1]

second_class_survived = survived_data[survived_data['Pclass'] == 2]

third_class_survived = survived_data[survived_data['Pclass'] == 3]



[first_class_survived.Survived.count(), second_class_survived.Survived.count(), third_class_survived.Survived.count()]
# Define Passenger Class Variables

first_class = data[data['Pclass'] == 1]

second_class = data[data['Pclass'] == 2]

third_class = data[data['Pclass'] == 3]



[first_class.describe(), second_class.describe(), third_class.describe()]
# Defining Fare Distribution Intervals

Fares = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525]



# Filtering Class Data & Defining Variables

not_survived = data.loc[data['Survived'] == 0]



# Creating Histogram

data.Fare.plot(kind='hist', color='orange', edgecolor='black', figsize=(11,9), bins = Fares)

# Adding x Value Ticks Based On Fares

plt.xticks(Fares)



# Titles

plt.xlabel('Fare Costs', size = 18)

plt.ylabel('Frequency', size = 18)

plt.title('Distribution Of Passenger Fare Costs', size = 18)



plt.show()



# Finding Average Fare Cost

data['Fare'].mean()
# Countplot To Compare The Number Of People In Different Classes

plt.figure(figsize = (12,4))

sn.countplot(x = 'Pclass', data = data, color = 'orange', edgecolor = 'black') 

plt.title('Distribution of Passengers Across Classes', size = '24')

plt.ylabel('Frequency',size = 18)

plt.xlabel('Passenger Class',size = 18)

plt.show()



# Finding The Most Common Embarked Location Among Passengers

mode = data.Pclass.mode()

mode = data.Pclass.mode()

[mode, mode]
# Countplot To Compare The Number Of Survived In Different Classes

plt.figure(figsize = (12,4))

sn.countplot(x = 'Pclass', data = survived_data, color = 'orange', edgecolor = 'black') 

plt.title('Distribution of Survived Across Classes', size = '24')

plt.ylabel('Frequency',size = 18)

plt.xlabel('Survived Class',size = 18)

plt.show()



# Finding The Most Common Passenger Class Among Survived

mode1 = survived_data.Pclass.mode()

mode2 = not_survived.Pclass.mode()

[mode1, mode2]
sn.barplot(x="Pclass", y="Survived", data=data)
# Define Intervals

age_dist = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]



# Creating Histogram

data.Age.plot(kind='hist', color='orange', edgecolor='black', figsize=(11,9), bins = age_dist)

# Adding x Value Ticks Based On Fares

plt.xticks(age_dist)



# Titles

plt.xlabel('Age', size = 18)

plt.ylabel('Frequency', size = 18)

plt.title('Distribution Of Passenger Age', size = 18)



plt.show()
# Countplot To Compare Gender Of Passengers

plt.figure(figsize=(12,4))

sn.countplot(x='Sex', data=data, color='orange', edgecolor='black') 

plt.title('Distribution of Passenger Gender', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Gender',size=18)

plt.show()
# Countplot To Compare Number Of Siblings/Spouse

plt.figure(figsize=(12,4))

sn.countplot(x='SibSp', data=data, color='orange', edgecolor='black') 

plt.title('Distribution of Number of Passengers with Siblings/Spouse', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Siblings/Spouse',size=18)

plt.show()
# Countplot To Compare Number Of Parents/Children

plt.figure(figsize=(12,4))

sn.countplot(x='Parch', data=data, color='orange', edgecolor='black') 

plt.title('Distribution of Number of Passengers with Parents/Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Parents/Children',size=18)

plt.show()
# Countplot To Compare Number Of Parents/Children

plt.figure(figsize=(12,4))

sn.countplot(x='Embarked', data=data, color='orange', edgecolor='black') 

plt.title('Distribution of Passenger Embark Locations', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Embark Location',size=18)

plt.show()
# Plot For Survived Ages

plt.figure(figsize=(11,9))

plt.title("Distribution of Survived Ages")

plt.xlabel('Age', size = 18)

plt.ylabel('Survived', size = 18)

ax = sn.distplot(survived_data.Age, color = 'green')



# Calculate Average Age

average = survived_data.Age.mean()

print('Average Survived Age:', average)
# Creating Plot

plt.figure(figsize=(11,9))

ax = sn.distplot(first_class_survived.Age, color = 'gold')

sn.distplot(second_class_survived.Age, color = 'magenta')

sn.distplot(third_class_survived.Age, color = 'blue')

plt.legend(labels=['1st Class', '2nd Class', '3rd Class'])

plt.title('Distribution of Age Across Survived Classes', size=24)

plt.xlabel('Age', size=18)

plt.ylabel('Survived', size=18)



#Find Average Age Of Survived Passenger Classes

average_first = first_class_survived.Age.mean()

average_second = second_class_survived.Age.mean()

average_third = third_class_survived.Age.mean()

[average_first, average_second, average_third]
# Countplot To Compare Gender Of Survived

plt.figure(figsize=(12,4))

sn.countplot(x='Sex', data=survived_data, color='orange', edgecolor='black') 

plt.title('Distribution of Survived Gender', size='24')

plt.ylabel('Survived',size=18)

plt.xlabel('Gender',size=18)

plt.show()



# Find Number Of Survived For Each Gender

men = survived_data[survived_data['Sex'] == 'male']

women = survived_data[survived_data['Sex'] == 'female']



[women.Survived.count(), men.Survived.count()]
sn.barplot(x="Sex", y="Survived", data=data)
# Countplot To Compare Gender Of First Class Survived

plt.figure(figsize=(12,4))

sn.countplot(x='Sex', data=first_class_survived, color='gold', alpha=0.5, edgecolor='black') 

plt.title('Distribution of First Class Survived Gender', size='24')

plt.ylabel('Survived',size=18)

plt.xlabel('Gender',size=18)

plt.show()



# Find Number Of Survived For Each Gender

men1 = first_class_survived[first_class_survived['Sex'] == 'male']

women1 = first_class_survived[first_class_survived['Sex'] == 'female']



[women1.Survived.count(), men1.Survived.count()]
sn.barplot(x="Sex", y="Survived", data=first_class)
# Countplot To Compare Gender Of Second Class Survived

plt.figure(figsize=(12,4))

sn.countplot(x='Sex', data=second_class_survived, color='magenta', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Second Class Survived Gender', size='24')

plt.ylabel('Survived',size=18)

plt.xlabel('Gender',size=18)

plt.show()



# Find Number Of Survived For Each Gender

men2 = second_class_survived[second_class_survived['Sex'] == 'male']

women2 = second_class_survived[second_class_survived['Sex'] == 'female']



[women2.Survived.count(), men2.Survived.count()]
sn.barplot(x="Sex", y="Survived", data=second_class)
# Countplot To Compare Gender Of Third Class Survived

plt.figure(figsize=(12,4))

sn.countplot(x='Sex', data=third_class_survived, color='blue', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Third Class Survived Gender', size='24')

plt.ylabel('Survived',size=18)

plt.xlabel('Gender',size=18)

plt.show()



# Find Number Of Survived For Each Gender

men3 = third_class_survived[third_class_survived['Sex'] == 'male']

women3 = third_class_survived[third_class_survived['Sex'] == 'female']



[women3.Survived.count(), men3.Survived.count()]
sn.barplot(x="Sex", y="Survived", data=third_class)
# Countplot To Compare Number Of Parents/Children

plt.figure(figsize=(12,4))

sn.countplot(x='Parch', data=survived_data, color='orange', edgecolor='black') 

plt.title('Distribution of Number of Survived with Parents/Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Parents/Children',size=18)

plt.show()



# Find Number Survived For Each Number Of Children/Parents

count = survived_data[survived_data['Parch'] == 0].Survived.count()

count1 = survived_data[survived_data['Parch'] == 1].Survived.count()

count2 = survived_data[survived_data['Parch'] == 2].Survived.count()

count3 = survived_data[survived_data['Parch'] == 3].Survived.count()

count4 = survived_data[survived_data['Parch'] == 4].Survived.count()

count5 = survived_data[survived_data['Parch'] == 5].Survived.count()



[count, count1, count2, count3, count5]
sn.barplot(x="Parch", y="Survived", data=data)
# Countplot To Compare Number Of Parents/Children Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='Parch', data=first_class_survived, color='gold', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of First Class Survived with Parents/Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Parents/Children',size=18)

plt.show()
sn.barplot(x="Parch", y="Survived", data=first_class)
# Countplot To Compare Number Of Parents/Children Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='Parch', data=second_class_survived, color='magenta', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Second Class Survived with Parents/Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Parents/Children',size=18)

plt.show()
sn.barplot(x="Parch", y="Survived", data=second_class)
# Countplot To Compare Number Of Parents/Children Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='Parch', data=third_class_survived, color='blue', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Third Class Survived with Parents/Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Parents/Children',size=18)

plt.show()
sn.barplot(x="Parch", y="Survived", data=third_class)
# Countplot To Compare Number Of Siblings/Spouses

plt.figure(figsize=(12,4))

sn.countplot(x='SibSp', data=survived_data, color='orange', edgecolor='black') 

plt.title('Distribution of Number of Survived with Siblings/Spouses', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Siblings/Spouses',size=18)

plt.show()



# Find Number Survived For Each Number Of Children/Parents

count0 = survived_data[survived_data['SibSp'] == 0].Survived.count()

count01 = survived_data[survived_data['SibSp'] == 1].Survived.count()

count02 = survived_data[survived_data['SibSp'] == 2].Survived.count()

count03 = survived_data[survived_data['SibSp'] == 3].Survived.count()

count04 = survived_data[survived_data['SibSp'] == 4].Survived.count()



[count0, count01, count02, count03, count04]
sn.barplot(x="SibSp", y="Survived", data=data)
# Countplot To Compare Number Of Siblings/Spouses Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='SibSp', data=first_class_survived, color='gold', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of First Class Survived with Siblings/Spouses', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Siblings/Spouses',size=18)

plt.show()
sn.barplot(x="SibSp", y="Survived", data=first_class)
# Countplot To Compare Number Of Siblings/Spouses Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='SibSp', data=second_class_survived, color='magenta', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Second Class Survived with Siblings/Spouses', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Siblings/Spouses',size=18)

plt.show()
sn.barplot(x="SibSp", y="Survived", data=second_class)
# Countplot To Compare Number Of Siblings/Spouses Across Class

plt.figure(figsize=(12,4))

sn.countplot(x='SibSp', data=third_class_survived, color='magenta', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Third Class Survived with Siblings/Spouses', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Siblings/Spouses',size=18)

plt.show()
sn.barplot(x="SibSp", y="Survived", data=third_class)
# Countplot To Compare Embarked Locations

plt.figure(figsize=(12,4))

sn.countplot(x='Embarked', data=survived_data, color='orange', edgecolor='black') 

plt.title('Distribution of Number of Survived from Embarked Locations', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Embarked Location',size=18)

plt.show()



# Find Number Survived For Each Number Of Children/Parents

countc = survived_data[survived_data['Embarked'] == 'C'].Survived.count()

counts = survived_data[survived_data['Embarked'] == 'S'].Survived.count()

countq = survived_data[survived_data['Embarked'] == 'Q'].Survived.count()



[countc, counts, countq]
sn.barplot(x="Embarked", y="Survived", data=data)
# Countplot To Compare Number Of Survived Across Embarked Locations

plt.figure(figsize=(12,4))

sn.countplot(x='Embarked', data=first_class_survived, color='gold', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of First Class Survived Across Embarked Locations', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Embarked Location',size=18)

plt.show()
sn.barplot(x="Embarked", y="Survived", data=first_class)
# Countplot To Compare Number Of Survived Across Embarked Locations

plt.figure(figsize=(12,4))

sn.countplot(x='Embarked', data=second_class_survived, color='magenta', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Second Class Survived Across Embarked Locations', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Embarked Location',size=18)

plt.show()
sn.barplot(x="Embarked", y="Survived", data=second_class)
# Countplot To Compare Number Of Survived Across Embarked Locations

plt.figure(figsize=(12,4))

sn.countplot(x='Embarked', data=third_class_survived, color='blue', alpha=0.5, edgecolor='black') 

plt.title('Distribution of Number of Third Class Survived Across Embarked Locations', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Embarked Location',size=18)

plt.show()
sn.barplot(x="Embarked", y="Survived", data=third_class)
# Define Plot Intervals

Fares = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525]



# Plot For Survived Fares

plt.figure(figsize=(11,9))

plt.title("Distribution of Survived Fares")

plt.xlabel('Fare', size = 18)

plt.ylabel('Survived', size = 18)

ax = sn.distplot(survived_data.Fare, color = 'red', bins = Fares)

plt.xticks(Fares)



# Calculate Average Fare

average_fare = survived_data.Fare.mean()

print('Average Survived Fare:', average_fare)
# Setting A Maximum Fare Price For First Class For The Sake Of Reading Data

first_class_limit = first_class_survived[first_class_survived['Fare'] <= 175]



# Defining New Intervals For This Limited Dataset

limited_fares = [0, 25, 50, 75, 100, 125, 150, 175]



# Plot For Survived Fares

plt.figure(figsize=(11,9))

plt.title("Distribution of First Class Survived Fares")

plt.xlabel('Fare', size = 18)

plt.ylabel('Survived', size = 18)

sn.distplot(first_class_limit.Fare, color = 'gold', bins = limited_fares)

sn.distplot(second_class_survived.Fare, color= 'magenta', bins = limited_fares)

sn.distplot(third_class_survived.Fare, color= 'blue', bins = limited_fares)

plt.xticks(limited_fares)



# Calculate Average Fare

average_first_class_fare = first_class_survived.Fare.mean()

average_second_class_fare = second_class_survived.Fare.mean()

average_third_class_fare = third_class_survived.Fare.mean()

[average_first_class_fare, average_second_class_fare, average_third_class_fare]