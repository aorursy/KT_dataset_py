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
# load the training data using pandas

df = pd.read_csv('../input/train.csv', header=0)



# load the test data using pandas

tf = pd.read_csv('../input/test.csv', header=0)
# define a function for cleaning the data



def clean(df):

    """Cleans data and adds extra columns"""

    # fare

    df.loc[ (df.Fare.isnull(), 'Fare')] = df.Pclass*10

    

    # transform 'Sex' into 'Gender' column with binary integer

    # identifiers, female=0, male=1

    df['Gender'] = 4

    df['Gender'] = df['Sex'].map( {'female': 0, 'male':1} ).astype(int)



    # As Southampton has the highest number of Embarked. Assume S for and NaN

    # create new column

    df['Origin'] = df['Embarked']



    # replace any null valules with S

    df.loc[ (df.Embarked.isnull(), 'Origin')] = 'S'



    # replace letters with numbers

    df['Origin'] = df['Origin'].map( {'C': 0, 'S':1, 'Q':2} ).astype(int)



    # create feature which records whether the Embarked was originally missing

    df["EmbarkedIsNull"] = pd.isnull(df.Embarked).astype(int)

    

    # deal with missing values of Age

    # fill missing values with median ages fro passengers of that

    # gender and class



    # calculate median ages

    median_ages = np.zeros((2,3))



    for i in range(0,2):

        for j in range(0,3):

            median_ages[i,j] = df[(df['Gender'] == i) &\

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

     

    # make a new column where any null ages will be replaced by 

    # median age for that class



    # make a copy of age

    df['AgeFill'] = df['Age']



    # assign median ages

    for i in range(0,2):

        for j in range(0,3):

            df.loc[ (df.Age.isnull()) & (df.Gender == i) &\

                (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

    

    # create feature which records whether the Age wa originally missing

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    

    # Parch = number of parents or children on board

    # SibSp = number of siblings or spouses

    # create new features based on these

    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df.AgeFill * df.Pclass

    

    # (1) determine which columns are left which are not numeric

    # (2) send the pandas.DataFrame back to a numpy.array

    # (1):

    df.dtypes[df.dtypes.map(lambda x: x=='object')]

    # drop columns we will not use:

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # can also drop Age, as now have AgeFill

    df = df.drop(['Age'], axis=1)

    # can drop any row which still have missing values

    df = df.dropna()



    

    return df
# transfor 'Sex' into 'Gender' column with binary integer

# identifiers, female=0, male=1

#df['Gender'] = 4

#df['Gender'] = df['Sex'].map( {'female': 0, 'male':1} ).astype(int)



#df.head(3)
# Now follow a similar procedure for 'Embarked'

# Cherbour=0, Southampton=1, Queenstown=2



# first look at number of each in the data

cherbourg = 0

southampton = 0

queenstown = 0

    

for i in range(len(df['Embarked'])):

    if (df['Embarked'][i]=='C'):

        cherbourg += 1

    elif (df['Embarked'][i]=="S"):

        southampton += 1

    elif (df['Embarked'][i]=="Q"):

        queenstown += 1

        

print(cherbourg)

print(southampton)

print(queenstown)
# As Southampton has the highest number of Embarked. Assume S for and NaN

# create new column

#df['Origin'] = df['Embarked']



# replace any null valules with S

#df.loc[ (df.Embarked.isnull(), 'Origin')] = 'S'



# replace letters with numbers

#df['Origin'] = df['Origin'].map( {'C': 0, 'S':1, 'Q':2} ).astype(int)



# create feature which records whether the Embarked was originally missing

#df["EmbarkedIsNull"] = pd.isnull(df.Embarked).astype(int)



#df.head(3)
# deal with missing values of Age

# fill missing values with median ages fro passengers of that

# gender and class



# calculate median ages

#median_ages = np.zeros((2,3))



#for i in range(0,2):

#    for j in range(0,3):

#        median_ages[i,j] = df[(df['Gender'] == i) &\

#                              (df['Pclass'] == j+1)]['Age'].dropna().median()

        

#median_ages
# make a new column where any null ages will be replaced by 

# median age for that class



# make a copy of age

#df['AgeFill'] = df['Age']



# assign median ages

#for i in range(0,2):

#    for j in range(0,3):

#        df.loc[ (df.Age.isnull()) & (df.Gender == i) &\

#              (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]



#df.head(3)

        
# create feature which records whether the Age wa originally missing

#df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
# Parch = number of parents or children on board

# SibSp = number of siblings or spouses

#df['FamilySize'] = df['SibSp'] + df['Parch']



# create new feature

#df['Age*Class'] = df.AgeFill * df.Pclass
# (1) determine which columns are left which are not numeric

# (2) send the pandas.DataFrame back to a numpy.array



#df.dtypes[df.dtypes.map(lambda x: x=='object')]
# drop columns we will not use:

#df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)



# can also drop Age, as now have AgeFill

#df = df.drop(['Age'], axis=1)



# can drop any row which still have missing values

#df = df.dropna()



#df.head(3)
# clean training data

df = clean(df)



# convert to numpy array

train_data = df.values

train_data



# clean test data

tf = clean(tf)



# convert to numpy array

test_data = tf.values



# remove passengerId column

train_data = train_data[:,1:]

train_data



passenger_ids = test_data[:,0]



test_data = test_data[:,1:]
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier



# Create the random forest object which will include all the

# parameters for the fit

forest = RandomForestClassifier(n_estimators = 500)



# Fit the training data to Survived labels and create the 

# decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])



# Take the same decision trees and run it on the test data

# Test data should go through same cleaning process as Train data



output = forest.predict(test_data)

result = np.array([passenger_ids, output])

result.T

import csv as csv

predictions_file = open("results.csv", "wt")

p = csv.writer(predictions_file)

p.writerow(["PassengerId", "Survived"])

for row in result.T:

    p.writerow([int(row[0]), int(row[1])])



predictions_file.close()