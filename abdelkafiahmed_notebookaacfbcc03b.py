# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import packAges

import pandas as pd

import numpy as np

import brewer2mpl

import matplotlib.pyplot as plt

%matplotlib inline



# read in data

training_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)

# Set up some better defaults for matplotlib

from matplotlib import rcParams



#colorbrewer2 Dark2 qualitative color table

dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors



rcParams['figure.figsize'] = (10, 6)

rcParams['figure.dpi'] = 150

rcParams['axes.color_cycle'] = dark2_colors

rcParams['lines.linewidth'] = 2

rcParams['axes.facecolor'] = 'white'

rcParams['font.size'] = 14

rcParams['patch.edgecolor'] = 'white'

rcParams['patch.facecolor'] = dark2_colors[0]

rcParams['font.family'] = 'StixGeneral'
# view training data

training_df.head(10)
# check for data type and null values

training_df.info()
# view basic statistics 

training_df.describe()
### plot the Age distribution ###

plt.subplot(121)

plt.scatter(training_df.Age, training_df.Survived, color='k', alpha=0.05)

plt.ylim(-1, 2)

plt.yticks([0.0, 1.0], ['Died', 'Survived'], rotation='horizontal')

plt.xlim(0, training_df.Age.max())

plt.xlabel("Age")



plt.subplot(122)

plt.hist(training_df.Age.dropna(),bins=8)

plt.xlabel("Age")

plt.ylabel("Count")
### plot the class distribution ###

tclass = training_df.groupby(['Pclass', 'Survived']).size().unstack()

print(tclass)

plt.subplot(121)

plt.bar([0, 1, 2], tclass[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')

plt.ylabel("Count")

plt.xlabel("")

plt.legend(loc='upper left')

#normalize each row by transposing, normalizing each column, and un-transposing

tclass = (1. * tclass.T / tclass.T.sum()).T

plt.subplot(122)

plt.bar([0, 1, 2], tclass[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')

plt.ylabel("Fraction")

plt.xlabel("")
### plot the Gender distribution ###

#first convert Gender string to numeric value

training_df['Gender'] = 0 # create new column by giving a value

training_df.Gender = training_df.Sex.map({'female':0,'male':1}).astype(int) # map Sex to numeric value

test_df['Gender'] = 0 # create new column by giving a value

test_df.Gender = test_df.Sex.map({'female':0,'male':1}).astype(int) # map Sex to numeric value

Gender = training_df.groupby(['Gender', 'Survived']).size().unstack()

print(Gender)

plt.subplot(121)

plt.bar([0, 1], Gender[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1], Gender[1], bottom=Gender[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5], ['Female', 'Male'], rotation='horizontal')

plt.ylabel("Count")

plt.xlabel("")

plt.legend(loc='upper left')

#normalize each row by transposing, normalizing each column, and un-transposing

Gender = (1. * Gender.T / Gender.T.sum()).T

plt.subplot(122)

plt.bar([0, 1], Gender[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1], Gender[1], bottom=Gender[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5], ['Female', 'Male'], rotation='horizontal')

plt.ylabel("Fraction")

plt.xlabel("")
### fill in estimates for the missing Ages by Gender and class ###

# Inspection of the Name feature reveals that males <= 12 are usually called 'Master'

# and females are titled 'Miss' if they are not married. Female children generally have parent on board,

# but female single women don't. Misses with null Age and Parch >0 generally do not survive

# while those with Parch == 0 do survive.

# Mr. with null Ages generally do not survive

# Master. with null Age have variable survival

# Make a new 'Penalty' feature to include these observations about the null Ages

median_Ages = np.zeros((2,3))

for i in range(0,2):

        for j in range(0,3):

            median_Ages[i,j] = training_df[(training_df.Gender == i) & (training_df.Pclass == j+1)].Age.dropna().median()



training_df['AgeFill']=training_df.Age

training_df['Penalty']=0

# fill in null female Ages

for j in range(0,3):

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 0)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Mrs. ')),'AgeFill']=median_Ages[0,j]

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 0)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Ms. ')),'AgeFill']=median_Ages[0,j]

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 0)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Miss. '))&(training_df.Parch ==0),'AgeFill']=median_Ages[0,j]

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 0)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Miss. '))&(training_df.Parch >0),'AgeFill']=8

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 0)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Miss. '))&(training_df.Parch >0),'Penalty']=1

# fill in null male Ages

for j in range(0,3):

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 1)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Mr. ')),'AgeFill']=median_Ages[1,j]

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 1)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Mr. ')),'Penalty']=1

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 1)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Master. ')),'AgeFill']=8

    training_df.loc[(training_df.Age.isnull())&(training_df.Gender == 1)&(training_df.Pclass == j+1)&(training_df['Name'].str.contains('Dr. ')),'AgeFill']=median_Ages[1,j]

# check that it worked

print(training_df[training_df.Age.isnull() ][['Survived','Gender','Pclass','Age','AgeFill','Penalty']].head(10))

# do the same for the test data

# fill in estimates for the missing Ages by Gender and class

test_median_Ages = np.zeros((2,3))

for i in range(0,2):

        for j in range(0,3):

            test_median_Ages[i,j] = test_df[(test_df.Gender == i) & (test_df.Pclass == j+1)].Age.dropna().median()

test_df['AgeFill']=test_df.Age

test_df['Penalty']=0

# fill in null female Ages

for j in range(0,3):

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 0)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Mrs. ')),'AgeFill']=median_Ages[0,j]

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 0)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Ms. ')),'AgeFill']=median_Ages[0,j]

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 0)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Miss. '))&(test_df.Parch ==0),'AgeFill']=median_Ages[0,j]

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 0)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Miss. '))&(test_df.Parch >0),'AgeFill']=8

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 0)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Miss. '))&(test_df.Parch >0),'Penalty']=1

# fill in null male Ages

for j in range(0,3):

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 1)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Mr. ')),'AgeFill']=median_Ages[1,j]

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 1)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Mr. ')),'Penalty']=1

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 1)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Master. ')),'AgeFill']=8

    test_df.loc[(test_df.Age.isnull())&(test_df.Gender == 1)&(test_df.Pclass == j+1)&(test_df['Name'].str.contains('Dr. ')),'AgeFill']=median_Ages[1,j]



# fill in estimates for the missing Fares, by class, for training data

median_Fares = np.zeros(3)

for i in range(0,3):

    median_Fares[i] = training_df[(training_df.Pclass == i+1)].Fare.dropna().median()

training_df['FareFill']=training_df.Fare

for i in range(0,3):

    training_df.loc[(training_df.Fare.isnull())&(training_df.Pclass == i+1),'FareFill']=median_Fares[i]

# fill in estimates for the missing Fares, by class, for test data

test_median_Fares = np.zeros(3)

for i in range(0,3):

    test_median_Fares[i] = test_df[(test_df.Pclass == i+1)].Fare.dropna().median()

test_df['FareFill']=test_df.Fare

for j in range(0,3):

    test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass == i+1),'FareFill']=test_median_Fares[i]



# Normalize the AgeFill and FareFill features

training_df['AgeNorm'] = training_df.AgeFill

training_df.AgeNorm = (training_df.AgeNorm - training_df.AgeFill.mean()) / training_df.AgeFill.std()

test_df['AgeNorm'] = test_df.AgeFill

test_df.AgeNorm = (test_df.AgeNorm - test_df.AgeFill.mean()) / test_df.AgeFill.std()

training_df['FareNorm'] = training_df.FareFill

training_df.FareNorm = (training_df.FareNorm - training_df.FareFill.mean()) / training_df.FareFill.std()

test_df['FareNorm'] = test_df.FareFill

test_df.FareNorm = (test_df.AgeNorm - test_df.FareFill.mean()) / test_df.FareFill.std()

# check results

training_df.head(5)



# Group Ages to simplify machine learning algorithms.  0:0-5, 1:6-10, 2:11-15, 3:16-59, 4:>60

training_df['AgeGroup']=0

training_df.loc[(training_df.AgeFill<6),'AgeGroup']=0

training_df.loc[(training_df.AgeFill>=6) & (training_df.AgeFill <11),'AgeGroup']=1

training_df.loc[(training_df.AgeFill>=11) & (training_df.AgeFill <16),'AgeGroup']=2

training_df.loc[(training_df.AgeFill>=16) & (training_df.AgeFill <60),'AgeGroup']=3

training_df.loc[(training_df.AgeFill>=60),'AgeGroup']=4

test_df['AgeGroup']=0

test_df.loc[(training_df.AgeFill<6),'AgeGroup']=0

test_df.loc[(training_df.AgeFill>=6) & (test_df.AgeFill <11),'AgeGroup']=1

test_df.loc[(training_df.AgeFill>=11) & (test_df.AgeFill <16),'AgeGroup']=2

test_df.loc[(training_df.AgeFill>=16) & (test_df.AgeFill <60),'AgeGroup']=3

test_df.loc[(training_df.AgeFill>=60),'AgeGroup']=4

### plot Age groups ###

Age_group = training_df.groupby(['AgeGroup', 'Survived']).size().unstack()

print(Age_group)

plt.subplot(121)

plt.bar([0, 1, 2, 3, 4], Age_group[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1, 2, 3, 4], Age_group[1], bottom=Age_group[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0-5', '6-10', '11-15','16-59','>60'], rotation='horizontal')

plt.ylabel("Count")

plt.xlabel("")

plt.legend(loc='upper left')

# normalize each row by transposing, normalizing each column, and un-transposing

Age_group = (1. * Age_group.T / Age_group.T.sum()).T

plt.subplot(122)

plt.bar([0, 1, 2, 3, 4], Age_group[0], color=dark2_colors[2], label='Died')

plt.bar([0, 1, 2, 3, 4], Age_group[1], bottom=Age_group[0], color=dark2_colors[0], label='Survived')

plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0-5', '6-10', '11-15','16-59','>60'], rotation='horizontal')

plt.ylabel("Fraction")

plt.xlabel("")
# combine parent and sibling data into FamilySize

training_df['FamilySize']=training_df.SibSp+training_df.Parch

test_df['FamilySize']=test_df.SibSp+test_df.Parch

# Drop the non-numeric attributes, attributes with null values, and un-normalized attributes from the training data and test data

# so we can use the machine learning algorithms

training_df = training_df.drop(['Name','Cabin','Embarked','Age', 'AgeFill', 'Sex','Ticket','Fare', 'FareFill'],axis=1)

test_df = test_df.drop(['Name','Cabin','Embarked','Age', 'AgeFill', 'Sex','Ticket','Fare', 'FareFill'],axis=1)

# check results

training_df.head(5)
# convert dataframes to arrays for use with machine learning algorithms

training_data = training_df.values

test_data = test_df.values