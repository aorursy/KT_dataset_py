import numpy

import pandas

from xgboost import XGBClassifier

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
seed = 7

numpy.random.seed(seed)







# load dataset

train_dataframe = pandas.read_csv(r"../input/train.csv")



X = train_dataframe.values[:,2:]

Y = train_dataframe.values[:,1]



# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.21, stratify=Y)





print("X shape", X.shape)



num_instances = len(X_Train)
xgmodel = XGBClassifier()

eval_set = [(X_Test, Y_Test)]

xgmodel.fit(X_Train, Y_Train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)



xgdpredictions = xgmodel.predict(X_Test)

xgdpredictions = [round(value) for value in xgdpredictions]



# eval

xgdaccuracy = accuracy_score(Y_Test, xgdpredictions)

print("XGBoost Accuracy: %.2f%%" % (xgdaccuracy * 100.0))



from keras import backend as K

from keras.layers import Activation

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from keras.constraints import maxnorm

from keras.utils.generic_utils import get_custom_objects





def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Activation(swish)})



# fix random seed for reproducibility

seed = 7
# create model

model = Sequential()

model.add(Dense(40, input_dim=57, init='uniform', activation='swish'))

model.add(Dropout(0.2))

model.add(Dense(20, init='uniform', activation='swish', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(10, init='uniform', activation='swish'))

model.add(Dense(1, init='uniform', activation='swish'))



# Compile model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_Train, Y_Train, epochs=10, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
