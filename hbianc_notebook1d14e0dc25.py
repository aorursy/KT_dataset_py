# This Python 3 environment comes with many helpful analytics libraries installed



# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Compute basic statistics 

data = pd.read_csv('../input/train.csv')

data.describe()
#missing age in the dataset

print(data['Age'].mean(), data['Age'].median())  
#Median and mean are alike. We can use median to fill in NaN in the 'Age' column

data['Age'].fillna(data['Age'].median(), inplace=True);
#Dealing with qualitative variables: Sex, Embarking city, Ticket, Cabin

#Sex: no need to have one column for female and one for male as data is complementary.

try:

    data['SexQ'] = data['Sex'].apply(lambda x: 1 if x=='female' else 0)

    del data['Sex']

except KeyError:

    col = 'Sex'

    print('{} column already deleted'.format(col))

#Embarking port

try:

    data['C'] = data['Embarked'].apply(lambda x: 1 if x=='C' else 0)

    data['S'] = data['Embarked'].apply(lambda x: 1 if x=='S' else 0)

    data['Q'] = data['Embarked'].apply(lambda x: 1 if x=='Q' else 0)

    del data['Embarked']

except KeyError:

    col = 'Embarked'

    print('{} column already deleted'.format(col))
#Support vector machine

from sklearn.svm import SVC



X = data.drop(['Survived', 'Ticket', 'Cabin', 'Name'], axis=1).values

y = data['Survived'].values



classifier = SVC(kernel='linear')

classifier.fit(X,y)