import numpy as np

import pandas as pd

from pandas import Series, DataFrame



import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score

import xgboost as xgb
titanic_df = pd.read_csv("../input/train.csv")

alldata = titanic_df.drop(

    ['PassengerId', 'Name', 'Ticket', 'Embarked', "Cabin"], axis=1)

testdata = pd.read_csv(

    "../input/test.csv").drop(['Name', 'Ticket', 'Embarked', "Cabin"], axis=1)



test_id = testdata['PassengerId']

testdata = testdata.drop('PassengerId', 1)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)



withnan = alldata.drop(['Sex', "Survived"], 1)

imp.fit(withnan)

withnan = imp.transform(withnan)



alldata['Age'] = withnan[:, 1]

alldata['Fare'] = withnan[:, 4]

testwithna = imp.transform(testdata.drop('Sex', 1))

testdata['Age'] = testwithna[:, 1]

testdata['Fare'] = testwithna[:, 4]



X_train = alldata.drop("Survived", axis=1)

Y_trian = alldata["Survived"]



testdata['gender'] = testdata['Sex'] == 'male'

testdata['gender'] = testdata['gender'].apply(lambda x: int(x))



testdata = testdata.drop('Sex', 1)



X_train['gender'] = X_train['Sex'] == 'male'

X_train['gender'] = X_train['gender'].apply(lambda x: int(x))



X_train = X_train.drop('Sex', 1)
logreg = LogisticRegression()

logreg.fit(X_train, Y_trian)

bools = testdata.isnull()

logreg_pred = logreg.predict(testdata)

#a = accuracy_score(grandT["Survived"], logreg_pred)

#print('logregression', a)
rf = RandomForestClassifier(criterion='gini',

                            n_estimators=700,

                            min_samples_split=10,

                            min_samples_leaf=1,

                            max_features='auto',

                            oob_score=True,

                            random_state=1,

                            n_jobs=-1)

rf.fit(X_train, Y_trian)

rf_pred = rf.predict(testdata)

#a = accuracy_score(grandT["Survived"], rf_pred)

#print('randomforest', a)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

gb = gbm.fit(X_train, Y_trian)

xgb_pred = gb.predict(testdata)
nn_clf = MLPClassifier(

    activation='relu',

    solver='adam',

    hidden_layer_sizes=(20, 10, 10, 10, 5),

    random_state=1,

    alpha=1e-5)

nn_clf.fit(X_train, Y_trian)

nn_pred = nn_clf.predict(testdata)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.utils.np_utils import to_categorical

model = Sequential()



model.add(Dense(20, input_dim=6, init='uniform'))

model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(10, init='uniform'))

model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(10, init='uniform'))

model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(10, init='uniform'))

model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(5, init='uniform'))

model.add(Activation('relu'))

# model.add(Dropout(0.5))



model.add(Dense(2, init='uniform'))

model.add(Activation('softmax'))



model.compile(optimizer='Adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

Y_trian_ke = to_categorical(Y_trian)

X_train_ke = np.array(X_train)

# print(type(X_train))

model.fit(X_train_ke, Y_trian_ke, batch_size=32, nb_epoch=20, verbose=0)

keras_pred = model.predict(np.array(testdata))

keras_nn_pred = (keras_pred[:, 1] - keras_pred[:, 0] > 0) - 0