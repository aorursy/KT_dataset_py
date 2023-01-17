import numpy as np

import pandas as pd

import seaborn as sb

import sklearn as skl

from matplotlib import pyplot as mpl



train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")



full_set = train_set.append(test_set)

full_set.head()



null_counts = full_set.isnull().sum()

null_counts
t_n = train_set.shape[0]

age_missing_survived = train_set['Survived'][train_set['Age'].isnull()]

cabin_missing_survived = train_set['Survived'][train_set['Cabin'].isnull()]

embarked_missing_survived = train_set['Survived'][train_set['Embarked'].isnull()]

known_survivors = train_set['Survived']#.where(pd.notnull(train_set['Survived']))

[[str(known_survivors.sum()) + ' / ' + str(known_survivors.shape[0]) + " known passengers survived or " + str(100 * round(1 - known_survivors.sum() / known_survivors.shape[0], 2)) + "% did not survive"],

[str(age_missing_survived.sum()) + ' / ' + str(age_missing_survived.shape[0]) + " passengers with missing ages survived or " + str(100 * round(1 - age_missing_survived.sum() / age_missing_survived.shape[0], 2)) + "% did not survive" ],

[str(cabin_missing_survived.sum()) + ' / ' + str(cabin_missing_survived.shape[0]) + " passengers with missing cabins survived or " + str(100 * round(1 - cabin_missing_survived.sum() / cabin_missing_survived.shape[0], 2)) + "% did not survive"],

[str(embarked_missing_survived.sum()) + ' / ' + str(embarked_missing_survived.shape[0]) + " passengers with missing embarkments survived or " + str(100 * round(1 - embarked_missing_survived.sum() / embarked_missing_survived.shape[0], 2)) + "% did not survive"]]
ages_present = full_set[pd.notnull(full_set['Age'])]['Age'].sort_values()

ages_present.plot.hist(100)
full_set['Age'] = full_set['Age'].interpolate()

full_set.sort_values(['Age'], ascending = True).head(3)
full_set[:]['Age'].plot.hist(100)
full_set = full_set.drop(['Cabin', 'Ticket', 'Name'], axis = 1)

full_set.isnull().sum()
miss_embark = full_set[full_set['Embarked'].isnull()]

miss_embark
embark_class = full_set[:][['Embarked', 'Pclass']].where(full_set['Survived'] == 1).where(full_set['Pclass'] == 1).where(full_set['Sex'] == 'female').groupby('Embarked').agg('count')

embark_class
full_set.loc[full_set[:]['Embarked'].isnull(), 'Embarked'] = ['S','C']

full_set['Fare'] = full_set['Fare'].interpolate()

full_set.isnull().sum()
from sklearn.preprocessing import LabelBinarizer

from collections import OrderedDict

import string





Emb_Coder = LabelBinarizer()

Emb_Vals = Emb_Coder.fit_transform(full_set['Embarked'])



Sex_Coder = LabelBinarizer()

Sex_Vals = Sex_Coder.fit_transform(full_set['Sex'])



PC_Coder = LabelBinarizer()

PC_Vals = PC_Coder.fit_transform(full_set['Pclass'])



data_dict = OrderedDict()

for i in range(3):

    data_dict["PC" + str(PC_Coder.classes_[i])] = PC_Vals[:, i]



for j in range(3):

    data_dict[str(Emb_Coder.classes_[j])] = Emb_Vals[:, j]

    

data_dict['gender_f/m'] = Sex_Vals[:,0]



data_dict

encodeVals = pd.DataFrame(data = data_dict)

encodeVals.head(10)

ind = [i for i in range(full_set.shape[0])]

full_set.insert(0, "ind_", ind)

encodeVals.insert(0, "ind_", ind)

full_set = full_set.drop(['Pclass', 'Sex', 'Embarked'], axis = 1)




full = full_set.join(encodeVals, on = 'ind_', how = "left", lsuffix = 'indx')



f_n = full.shape

t_n = train_set.shape



# collect training data

train_set = full[0:t_n[0]][full.columns[0:(f_n[1])]]

y_train = full_set[0:t_n[0]]['Survived']

p_train = full_set[0:t_n[0]]['PassengerId']

train_set = train_set.drop(["Survived", "ind_indx", "ind_", "PassengerId"], axis = 1)



# collect testing data

test_set = full[t_n[0]:full.size][full.columns[0:(f_n[1])]]

y_test = full_set[t_n[0]:f_n[0]]['Survived']

p_test = full_set[t_n[0]:f_n[0]]['PassengerId']

test_set = test_set.drop(["Survived", "ind_indx", "ind_", "PassengerId"], axis = 1)



train_set.head(10)
import tensorflow as tf



full_size = train_set.shape[0]

train_size = 700

dev_size = full_size - train_size



train_in_fn = tf.estimator.inputs.pandas_input_fn(train_set[0:train_size], 

                                                  y_train[0:train_size],  

                                                  num_epochs = 10,

                                                  batch_size = 100,

                                                  shuffle = True)



feature_names = train_set.columns

feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]



classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns, 

                                        hidden_units = [30,20,40,30,10])



classifier.train(input_fn = train_in_fn, steps = 10000)



classifier.evaluate(input_fn = train_in_fn, steps = 10000, name = "Train")

dev_in_fn = tf.estimator.inputs.pandas_input_fn(train_set[train_size:(train_size + dev_size)], 

                                                y_train[train_size:(train_size + dev_size)], 

                                                num_epochs = 10,

                                                batch_size = 100,

                                                shuffle = True)



classifier.evaluate(input_fn = dev_in_fn, steps = 10000, name = "Dev")
test_in_fn = tf.estimator.inputs.pandas_input_fn(test_set, 

                                                None,

                                                batch_size = 100,

                                                shuffle = True)

predictions = classifier.predict(input_fn = test_in_fn)

classes = []

for ind, prediction in enumerate(predictions):

    classes.append(int(prediction['class_ids'][0]))



submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": classes})

x = submit_ray.to_csv(index = False)



#Estimate 1

#print(x)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import SGD

from keras.utils import to_categorical



y_train = to_categorical(y_train)



model = Sequential()

model.add(Dense(36, activation = 'relu', input_shape = (11,)))

model.add(Dropout(0.25))

model.add(Dense(45, activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(36, activation = 'relu'))

model.add(Dense(2, activation = 'sigmoid'))



model.compile(loss='categorical_crossentropy',

              optimizer='adagrad',

              metrics=['accuracy'])



model.fit(train_set[0:train_size], y_train[0:train_size],

          epochs=1500,

          batch_size=20)
score = model.evaluate(train_set[train_size:full_size], y_train[train_size:full_size], batch_size=20)

score
prediction = model.predict(test_set, batch_size = 20)

pred_binary = []

for i in prediction:

    if i[0] > i[1]:

        pred_binary.append(0)

    else:

        pred_binary.append(1)



submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": pred_binary})

x = submit_ray.to_csv(index = False)



#Estimate 2

#print(x)
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(verbose = 1) 

RFC.fit(train_set[0:train_size], y_train[0:train_size])

RFC.score(train_set[0:train_size], y_train[0:train_size])

RFC.score(train_set[train_size:], y_train[train_size:])
RFCprediction = RFC.predict(test_set)

pred = []

for i in RFCprediction:

    if i[0] > i[1]:

        pred.append(0)

    else:

        pred.append(1)

len(pred)



RFC_submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": pred})

x = RFC_submit_ray.to_csv(index = False)



#Estimate 3

#print(x)