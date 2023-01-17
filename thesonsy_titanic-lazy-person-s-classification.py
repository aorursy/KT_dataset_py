# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
 # This should be a fairly application of random forest.



# A nice warm-up if you will.



train.drop("Name", inplace = True, axis = 1)

train.columns
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
for column in train.columns:

    if column != "Survived":

        

        try:

            sns.jointplot(x = "Survived", y = column, data = train)

            plt.show() # I ran into a problem because I'm lazy. But that's okay.

        except ValueError:

            pass

#Oh hey look what I done did. I get some usefu info so long as I'm not lazy.
train.drop("Ticket", axis = 1, inplace = True)
# Getting Dummy variables because sci-kit learn

embarked_dummies = pd.get_dummies(train['Embarked']) # Hates factor variables # This doth not work?
train.drop('Embarked', axis = 1, inplace = True)



new_train = pd.concat([train, embarked_dummies], axis = 1)





train['Cabin'] = train['Cabin'].str[0]
new_train['Cabin'] = new_train['Cabin'].str[0]
cabin_dummies = pd.get_dummies(new_train['Cabin'])

new_train = pd.concat([new_train.drop('Cabin', axis =1), cabin_dummies], axis = 1)
new_train.dropna(inplace = True)
correlationseries = new_train.corr()['Survived']
correlationseries.drop(['Survived', 'PassengerId'], inplace = True, axis = 0)

corrplot = correlationseries.plot.bar()

corrplot.set_title("Correlations")
farecorrplot = new_train.corr()['Fare'].drop('Fare',axis = 0).plot.bar()

farecorrplot.set_title("Fare Correlations")

plt.show()# Note P-class has negative order
#Forgot to apply all transformations on the test set. I don't like how they organize it like this

#Mais c'est la vie

test.drop("Name", inplace = True, axis = 1)

test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

test.drop("Ticket", axis = 1, inplace = True)

test_embarked_dummies = pd.get_dummies(test['Embarked'])



new_test = pd.concat([test, test_embarked_dummies], axis = 1)

new_test['Cabin'] = new_test['Cabin'].str[0]

test_cabin_dummies = pd.get_dummies(new_test['Cabin'])

new_test = pd.concat([new_test.drop('Cabin', axis =1), test_cabin_dummies], axis = 1)

new_test.drop('Embarked', axis = 1, inplace = True)

new_test.dropna(inplace = True)
data = [new_train, new_test]



for d in data:

    d.drop('PassengerId', axis = 1, inplace = True)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
train_data = new_train.drop('Survived', axis = 1)

train_data
survive_train = np.array(new_train['Survived'])

data_train = np.array(train_data)

data_test = np.array(new_test)

#Let's do some automatic parameter-specifications



specs = np.arange(500,2200, 100)

specs_dict = {}



for spec in specs:

    

    spec_dict[spec] = rf.fit(data_train, survive_train).predict(data_test)
test_data.shape
new_train
#Forgot to do it all in test...oops



test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

test.drop("Ticket", axis = 1, inplace = True)

test_embarked_dummies = pd.get_dummies(test['Embarked'])

test.drop('Embarked', axis = 1, inplace = True)

new_test = pd.concat([test, test_embarked_dummies], axis = 1)



new_test['Cabin'] = new_test['Cabin'].str[0]





test_cabin_dummies = pd.get_dummies(new_test['Cabin'])

new_test = pd.concat([new_test.drop('Cabin', axis =1), test_cabin_dummies], axis = 1)

new_test.dropna(inplace = True)
new_train.drop("T", axis = 1, inplace = True)
data = [new_train, new_test]



for  d in data:

    d.drop('PassengerId', axis = 1, inplace = True)
new_train_run = new_train.drop('Survived', axis = 1)

train_data = np.array(new_train_run)

survived_train = np.array(new_train['Survived'])

test_data = np.array(new_test)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()
specs = np.arange(500, 2200, 100)



specs_dict = {}





for spec in specs:

    

    specs_dict[spec] = rf.fit(train_data, survived_train).predict(test_data)



specs_dict[2100] # Here are the predictions. I forgot I can't really test these