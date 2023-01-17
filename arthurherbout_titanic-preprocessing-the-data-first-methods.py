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
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataframe = pd.read_csv('../input/train.csv')
print(dataframe)
# I need to convert all strings to integers

# I need to fill in the missing values of the data and make it complete

# female = 0, Male = 1

dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Here we have added a colomn "Gender" at the end of our dataframe

#print(X)



# Embarked from 'C', 'Q', 'S'

# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.



# All missing Embarked -> just make them embark from most common place

if len(dataframe.Embarked[ dataframe.Embarked.isnull() ]) > 0:

    dataframe.Embarked[ dataframe.Embarked.isnull() ] = dataframe.Embarked.dropna().mode().values



Ports = list(enumerate(np.unique(dataframe['Embarked'])))    # determine all values of Embarked,

print(Ports)

Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

print(Ports_dict)

dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# Here we have converted strings values in Embarked to integers 1, 2 or 3

#print(X)



#All the ages with no data : I will complete the empty entries with the median Age

median_age = dataframe['Age'].dropna().median()

if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:

    dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age

#print(X)





# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

#print(y)
print(dataframe)
y=dataframe['Survived']

#print(y)
# I get rid of the label column

dataframe = dataframe.drop(['Survived'], axis=1)
print(dataframe)
train_data = dataframe.values

labels = y.values
# TEST DATA

testdataframe = pd.read_csv('../input/test.csv',header = 0)

# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

testdataframe['Gender'] = testdataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Embarked from 'C', 'Q', 'S'

# All missing Embarked -> just make them embark from most common place

if len(testdataframe.Embarked[ testdataframe.Embarked.isnull() ]) > 0:

    testdataframe.Embarked[ testdataframe.Embarked.isnull() ] = testdataframe.Embarked.dropna().mode().values

# Again convert all Embarked strings to int

testdataframe.Embarked = testdataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)





# All the ages with no data -> make the median of all Ages

median_age = testdataframe['Age'].dropna().median()

if len(testdataframe.Age[ testdataframe.Age.isnull() ]) > 0:

    testdataframe.loc[ (testdataframe.Age.isnull()), 'Age'] = median_age



# All the missing Fares -> assume median of their respective class

if len(testdataframe.Fare[ testdataframe.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = testdataframe[ testdataframe.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        testdataframe.loc[ (testdataframe.Fare.isnull()) & (testdataframe.Pclass == f+1 ), 'Fare'] = median_fare[f]



# Collect the test data's PassengerIds before dropping it

ids = testdataframe['PassengerId'].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

testdataframe = testdataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
print(testdataframe)

print(dataframe)
test_data = testdataframe.values
#Standardization

from sklearn.preprocessing import StandardScaler

train_data_std = StandardScaler().fit_transform(train_data)
# Visualizing variance explained

from sklearn.decomposition import PCA

pca = PCA().fit(train_data_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

# 7 is the number of features of our dataset 

plt.xlim(0,6,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.ensemble import RandomForestClassifier