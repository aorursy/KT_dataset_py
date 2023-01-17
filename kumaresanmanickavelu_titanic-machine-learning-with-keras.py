import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
Y_train = train["Survived"]
Y_train = np_utils.to_categorical(Y_train, 2)

full = train.append( test , ignore_index = True )

X_full = pd.DataFrame()

X_full['Fare']=(full['Fare'] - 32) / (500)
X_full['Pclass']=full['Pclass'] - 2
X_full['SibSp'] =full['SibSp']
X_full['Parch']= full['Parch']
X_full['Age']= full['Age'].fillna(0)  / 80
X_full['Sex'] = full['Sex'].map( {'female': 1, 'male': 0} )
X_full['Embarked'] = full['Embarked'].map( {'C': -1,'Q': 0, 'S': 1}).fillna(0)

cabin = full['Cabin'].fillna( 'U' )
cabin = cabin.map( lambda c : c[0] )
cabin = pd.get_dummies( cabin , prefix = 'Cabin' )


def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

# Extracting dummy variables from tickets:
ticket = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket , prefix = 'Ticket' )

title = pd.DataFrame()
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )


family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

X_full = pd.concat([X_full,cabin,ticket,title,family], axis=1) 

X_train = X_full[ 0:891 ]
X_test = X_full[ 891: ]

X_full.describe()


model = Sequential()
model.add(Dense(20, activation='relu', input_dim=len(X_train.columns)))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=300, verbose=0)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_classes(X_test, verbose=1)
submissions=pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": predictions})
submissions.to_csv("survived.csv", index=False, header=True)
