import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/data.csv",header=0)

data.drop("Unnamed: 32",axis=1,inplace=True)

data.drop("id",axis=1,inplace=True)



data.info()
from sklearn.model_selection import train_test_split 

train, test = train_test_split(data, test_size = 0.3)



prediction_var = list(data.columns[1:31])

outcome_var = "diagnosis"



train_X = train[prediction_var]# taking the training data input

train_y = train.diagnosis# This is output of our training data

# same we have to do for test

test_X = test[prediction_var] # taking test data inputs

test_y = test.diagnosis   #output value of test dat



train_X.info()
import matplotlib.pyplot as plt

n_train = np.array(train_X)

n_test = np.array(test_X)



plt.plot(n_train)

plt.show()
from keras.layers import Input, Dense

from keras.models import Model

import tensorflow as tf



input_dim = n_train.shape[1]

feature_dim = [25, 20, 15, 10]

print(input_dim)

inputs = Input(shape=(input_dim,))

encoded = inputs

encoded = Dense(feature_dim[0], kernel_initializer="uniform")(encoded)

encoded = Dense(feature_dim[1], kernel_initializer="uniform")(encoded)

encoded = Dense(feature_dim[2], kernel_initializer="uniform")(encoded)

encoded = Dense(feature_dim[3], kernel_initializer="uniform")(encoded)



decoded = encoded

decoded = Dense(feature_dim[2], kernel_initializer="uniform")(decoded)

decoded = Dense(feature_dim[1], kernel_initializer="uniform")(decoded)

decoded = Dense(feature_dim[0], kernel_initializer="uniform")(decoded)

decoded = Dense(input_dim, kernel_initializer="uniform")(decoded)





autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer='adadelta', loss='mse')



autoencoder.fit(n_train, n_train,

                verbose=0,

                epochs=150,

                batch_size=100,

                shuffle=True,

                validation_data=(n_test, n_test))



predict_vals = autoencoder.predict(n_train)

print(predict_vals.shape)

plt.plot(predict_vals)

plt.show()
from keras.models import Sequential



featuremodel = Sequential()

featuremodel.add(Dense(feature_dim[0], input_shape=(input_dim,), weights=autoencoder.layers[1].get_weights()))

featuremodel.add(Dense(feature_dim[1], weights=autoencoder.layers[2].get_weights()))

featuremodel.add(Dense(feature_dim[2], weights=autoencoder.layers[3].get_weights()))

featuremodel.add(Dense(feature_dim[3], weights=autoencoder.layers[4].get_weights()))



featuremodel.compile(optimizer='adadelta', loss='mse')



from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model



# classic svm

model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print("Accuracy svm: %s" % "{0:.3%}".format(metrics.accuracy_score(prediction, test_y)))



# classic svm with deep autoencoder

deepmodel = svm.SVC()

deepmodel.fit(featuremodel.predict(n_train),train_y)

deepprediction=deepmodel.predict(featuremodel.predict(n_test))

print("Accuracy d-svm: %s" % "{0:.3%}".format(metrics.accuracy_score(deepprediction, test_y)))



# linear svm

model = svm.SVC(kernel='linear')

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print("Accuracy lsvm: %s" % "{0:.3%}".format(metrics.accuracy_score(prediction, test_y)))



# linear svm with deep autoencoder

deepmodel = svm.SVC(kernel='linear')

deepmodel.fit(featuremodel.predict(n_train),train_y)

deepprediction=deepmodel.predict(featuremodel.predict(n_test))

print("Accuracy d-lsvm: %s" % "{0:.3%}".format(metrics.accuracy_score(deepprediction, test_y)))





########################

from sklearn.neighbors import KNeighborsClassifier

print("----------")



# classic knn

model = KNeighborsClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print("Accuracy knn: %s" % "{0:.3%}".format(metrics.accuracy_score(prediction, test_y)))



# classic knn with deep autoencoder

deepmodel = KNeighborsClassifier()

deepmodel.fit(featuremodel.predict(n_train),train_y)

deepprediction=deepmodel.predict(featuremodel.predict(n_test))

print("Accuracy d-nn: %s" % "{0:.3%}".format(metrics.accuracy_score(deepprediction, test_y)))
from sklearn.model_selection  import KFold

def classification_model(model,data,prediction_input,output):

    model.fit(data[prediction_input],data[output])

    predictions = model.predict(data[prediction_input])

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    

    kf = KFold(n_splits=5)

    error = []

    for train, test in kf.split(data):

        train_X = (data[prediction_input].iloc[train,:])

        train_y = data[output].iloc[train]

        model.fit(train_X, train_y)

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
model = svm.SVC()

classification_model(model,data,prediction_var,outcome_var)



# all data

np_data = np.array(data[prediction_var])

np_out = np.array(data[outcome_var])

data_prediction = featuremodel.predict(np_data)





# svm

model = svm.SVC()

deepmodel.fit(data_prediction,np_out)

deepprediction=deepmodel.predict(data_prediction)

print("Accuracy : %s" % "{0:.3%}".format(metrics.accuracy_score(deepprediction, np_out)))



# kfold

kf = KFold(n_splits=5)

error = []

for train, test in kf.split(data):

    tr_X = (data_prediction[train,:])

    tr_y = np_out[train]

    model.fit(tr_X, tr_y)

    te_x=data_prediction[test,:]

    te_y=np_out[test]

    error.append(model.score(te_x,te_y))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))