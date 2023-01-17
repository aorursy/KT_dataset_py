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
import keras

# Binary Classification with Sonar Dataset: Standardized Smaller

import numpy

import pandas

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import timeit
# Massage data slightly

def prepareTitanicData(file):

    # load data

    data  = pd.read_csv(file)

    # We can see the age column has only 714 entries

    # Lets fill the missing values with the Average (median) age

    data["Age"] = data["Age"].fillna(data["Age"].median())

    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # lets deal with the embarked column

    # Find all the unique values for "Embarked".

    data["Embarked"] = data["Embarked"].fillna("S")

    data.loc[data["Embarked"] == "S", "Embarked"] = 0

    data.loc[data["Embarked"] == "C", "Embarked"] = 1

    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # drop the Name and Ticket columns

    data = data.drop(['Name','Ticket','Cabin'],axis=1)

    # one hot for embarked

    data2 = pd.get_dummies(data)

    #Copy survived column to end column - because I don't understand how to do it otherwise ;-)

    data2['Survived_1'] = data2['Survived']

    retDataframe = data2.drop(['Survived'],axis=1)

    # get the Numpy array

    dataset = retDataframe.values

    X=dataset[:,0:11] # Variables

    Y=dataset[:,11] # Result - target....

    return X,Y,retDataframe
#TrainX, TrainY, dataFrame = prepareTitanicData('/home/sean/Documents/ML_dataSets/Titanic/train.csv')

TrainX, TrainY, dataFrame = prepareTitanicData('../input/train.csv')
dataFrame.head()
print(TrainX[:5])

print(TrainY[:5])

# The NN model

def createNN():

    # create model

    model = Sequential()

    model.add(Dense(11, input_dim=11, init= 'normal' , activation= 'relu' ))

    model.add(Dense(6, init= 'normal' , activation= 'relu' ))

    model.add(Dense(1, init= 'normal' , activation= 'sigmoid' ))

    # Compile model

    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

    return model
#evaluate baseline model with standardized dataset

# Create the items for the pipeline pipeline and append the scalar and classifier

estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=createNN,nb_epoch=100,batch_size=10,verbose=0)))

# Create the pipeline passing in the estimators

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True,random_state=7 )
start = timeit.default_timer()

results = cross_val_score(pipeline, TrainX, TrainY, cv=kfold)

endTime = timeit.default_timer() - start

print("Calculation Duration = {}".format(endTime))

print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



#Calculation Duration = 221.14949154295027

#Standardized: 81.25% (3.71%)
# Next challenge for me is to take the created model and predict the outcome on the Test data...

# Once I figure out how I'll finalize this notebook