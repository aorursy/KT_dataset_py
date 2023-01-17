# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



seed=666



import numpy as np # linear algebra

np.random.seed(seed)

import tensorflow as tf

tf.set_random_seed(seed)

import random

random.seed(seed)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense



from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## Load Data
data_path = "../input/"

def loadData(path):

    train=pd.read_csv(path+"train.csv")

    test=pd.read_csv(path+"test.csv")

    gender_submissions=pd.read_csv(path+"gender_submission.csv")

    return train,test,gender_submissions



train, test, gender_submissions=loadData(data_path)
## train info
train.info()
train.isnull().sum()
## test info
test.info()
test.isnull().sum()
## gender submission info
gender_submissions.info()
gender_submissions.isnull().sum()
## The columns "Age", Fare", "Cabin" and "Embarked" contain null values
## Preprocess data
def cabinLevel(cabin):

    if cabin == None:

        return '0'

    elif isinstance(cabin, str):

        return cabin[0]

    else:

        return '0'

        

def try_parse(val, fail=None):

    try:

        if string.isnumeric():

            if val.isnan():

                return fail

            else:

                return val

        else:

            return float(val)

    except Exception:

        return fail;     



def cabinRoom(cabins):

    if cabins == None:

        return 0

    elif isinstance(cabins, str) and len(cabins)>0:

        all_cabins = cabins.split()

        for c in all_cabins:

#            print ("Cabin room: ", c)

            try:

                return float(c[1:])

            except Exception:

                continue

        return 0

    else:

        return try_parse(cabins,0)

    

import re

def ticketId(ticket):

    if ticket == None:

        return 'NONE'

    elif isinstance(ticket, str) and len(ticket)>0:

#        ticket_cleaned = ticket.translate(none, "./") # remove '.' and '/'

        ticket_cleaned = re.sub('[./]', '', ticket)

        all_tickets = ticket_cleaned.split()

        for t in all_tickets:

            try:

                num = float(t)

                continue       # continue with next token if string is a number

            except Exception:

                return t

    return 'NONE'



def ticketNo(ticket):

    if ticket == None:

        return 0

    elif isinstance(ticket, str) and len(ticket)>0:

        ticket_cleaned = re.sub('[./]', '', ticket)

        all_tickets = ticket_cleaned.split()

        last_item = all_tickets[-1]

        try:

            val = int(last_item)

#            print ("Val=", val)

            return val

        except Exception:

            return 0

    return 0



from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

def preprocessData(data):

    default_na_sex = 'U'

    default_na_age = 0 # data['Age'].mean()

    default_na_fare = 0.0

    default_na_embarked='X'

    # don't use PassengerId and Name

    # Encode labels replace null values

    nd=pd.DataFrame()

    nd['Pclass']=data['Pclass'].fillna(0)

    nd['Sex']=LabelEncoder().fit_transform(data['Sex'].fillna(default_na_sex))

    nd['Age']=data['Age'].fillna(default_na_age)

#    nd['Age2']=data['Age'].fillna(default_na_age)**2

#    nd['SibSp']=data['SibSp'].fillna(0)

    nd['Parch']=data['Parch'].fillna(0)

    nd['Fare']=data['Fare'].fillna(default_na_fare)

    nd['Cabin_Level']=LabelEncoder().fit_transform(data['Cabin'].apply(cabinLevel))

#    nd['Cabin_Room']=data['Cabin'].apply(cabinRoom)

#    nd['Embarked']=LabelEncoder().fit_transform(data['Embarked'].fillna(default_na_embarked))

#    nd['Ticket_Id']=LabelEncoder().fit_transform(data['Ticket'].apply(ticketId))

#    nd['Ticket_No']=data['Ticket'].apply(lambda x: ticketNo(x))

    return nd



# note: the uncommented features 'Age2', 'SibSp', 'Cabin_Room', 'Embarked', 'Ticket_Id' and 'Ticket_No' didn't add to a better score
X_train = preprocessData(train)

Y_train = train['Survived'].fillna(0)
X_train.head()
Y_train.head()
X_test = preprocessData(test)

Y_test = gender_submissions['Survived']
X_test.head()
Y_test.head()
## Normalize preprocessed data
def scaleData(data):

    return StandardScaler().fit_transform(data)



X_train = scaleData(X_train)

X_test = scaleData(X_test)
X_train
X_test
## create simple model with nin=X_train.shape[1], nhiddden=2, nout=1
nin = X_train.shape[1]

nhidden = 2

nout = 1

model = Sequential()

model.add(Dense(nhidden, activation='sigmoid', input_shape=(nin,)))

model.add(Dense(nout, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
## fit model
model.fit(X_train, Y_train,epochs=30, batch_size=1, verbose=0)
## predict values
threshold = 0.5

Y_pred = model.predict(X_test) >= threshold
model.predict(X_test)
Y_pred
## confusion matrix
confusion_matrix(Y_test, Y_pred)
## Accuracy Score
accuracy_score(Y_test, Y_pred)
## Precision Score
precision_score(Y_test, Y_pred)
## Recall Score
recall_score(Y_test, Y_pred)
## F1 Score
f1_score(Y_test,Y_pred)
## Kohen Kappa Score
cohen_kappa_score(Y_test, Y_pred)