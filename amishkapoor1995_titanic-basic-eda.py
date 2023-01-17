# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # importing numpy 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt # visualizations

import seaborn as sns # for more visualisations



# Any results you write to the current directory are saved as output.
# reading training and test data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_submission =  pd.read_csv('../input/gender_submission.csv')

train.head() #use this method to look at the different columns we have in the training dataset
test.head() #look at the test dataset
gender_submission.head() #taking a look at the 
train.shape #understanding shape of the dataset – there are 891 rows and 12 columns
train.info() #provides important information pertaining to the training data set



#Some of the important information is:

    #Age column has a lot of empty/NaN values, which we need to fill

    #Cabin colmn primarily contains Nan values and we will skip this column

    #There are 2 values in the Embarked column which are empty
train.isnull().sum() #this is telling us the exact number of values in each column which contain Nan values
#Computing the mean and median age 

mean_age = train['Age'].mean()



median_age = train['Age'].median()

print( ' mean age is =', mean_age, '\n', 'median age in =',median_age)
train.describe() 

#using this we get the summary statistics for training data set, such as mean, standard edviation, quartiles, minimum and maximum values
train.groupby(['Sex', 'Survived'])['Survived'].count()
sns.countplot('Survived', data = train)
#now we willexplore the training dataset and make some plots using seaborn and matplotlib libraries

# surviva by gender, by class, age

sns.countplot('Sex', hue = 'Survived', data = train)

plt.xlabel('Male or Female')

plt.ylabel('Number Survived')

plt.show()
train.groupby(['Pclass', 'Sex', 'Survived'])['Survived'].count()
plt.subplots(1,1)

p_class_1_train = train.loc[train['Pclass'] == 1]

sns.countplot('Sex', hue = 'Survived', data = p_class_1_train)

plt.title('P class 1')

plt.subplots(1,1)

p_class_2_train = train.loc[train['Pclass'] == 2]

sns.countplot('Sex', hue = 'Survived', data = p_class_2_train)

plt.title('P class 2')

plt.subplots(1,1)

p_class_3_train = train.loc[train['Pclass'] == 3]

sns.countplot('Sex', hue = 'Survived', data = p_class_3_train)

plt.title('P class 3')

plt.show()
sns.countplot(x= 'Pclass', hue = 'Survived', data = train)
#look at the age column

train['Age'].describe()
train['Age'].isna().sum()
train['Age'].median()
train['Age'] = train['Age'].fillna(train['Age'].median())

train['Age'].describe()
train.isna().sum()
train_f = train.drop('Cabin', 1)

train_f.describe()