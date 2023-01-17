import warnings

warnings.filterwarnings('ignore')

import numpy as np

np.random.seed(42)

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from sklearn.metrics import recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../input/creditcard.csv')

data.head()

#Distributions for each feature

v_features = data.ix[:,1:29].columns



plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(data[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(data[cn][data.Class == 1], bins=50)

    sns.distplot(data[cn][data.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show() 
#Drop out the similar features

data = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V5','V6','V7','V8','Time','Amount'], axis =1)

#Only legal transactions are used to set Mu and Sigma

normal = data[data.Class == 0]

fraud = data[data.Class == 1]

#Splits the data in training_set and test_set

X_train = normal.sample(frac = 0.6)

X_CV = pd.concat([normal.sample(frac = 0.2),fraud.sample(frac = 0.50)], axis = 0)

X_test = pd.concat([normal.sample(frac = 0.2), fraud.sample(frac = 0.50)], axis = 0)

y_train, y_cv, y_test = X_train.pop('Class'), X_CV.pop('Class'), X_test.pop('Class')



m, n = X_train.shape

normal_y = normal.pop('Class')

fraud_y = fraud.pop('Class')
#Multivariable Gaussian Distribution

#Computes Mu

def mu(X):

    _mu = np.zeros((n, 1))

    for i in range(n):

        _mu[i] = (1 / m) * np.sum(X[:,i])

    return _mu



#Computes Sigma

def sigma(X):

    _sigma = np.zeros((n, 1))

    _mu = mu(X)

    for i in range(n):

        _sigma[i] = (1 / m) * np.sum(np.dot(X[:, i] - _mu, (X[:, i] - _mu).T))

    

    return _sigma

    

    

def p(X, Mu, Sigma):

    k = Mu.shape[0]

    if Sigma.shape[0] == 1 or Sigma.shape[1] == 1:

        Sigma = np.diag(Sigma[:, 0])

    

    X = X - Mu.T

    r = (2 * np.pi) ** (-k / 2) * np.linalg.det(Sigma) ** (-0.5)

    e = np.exp(-0.5 * np.sum(np.dot(X , np.linalg.pinv(Sigma)) * X, 1))

    

    return r * e

    
#Find the best value for epsilon

def selectThreshold(predictions_CV, Y_CV):

    bestEpsilon = 0

    bestF1 = 0

    F1 = 0

    stepsize = (np.max(predictions_CV) - np.min(predictions_CV)) / 1000

    

    for epsilon in np.arange(min(predictions_CV), max(predictions_CV), stepsize):

        pred = np.array(predictions_CV < epsilon, dtype = int)

        F1 = f1_score(pred, Y_CV, average = 'macro')

        if F1 > bestF1:

            bestF1 = F1

            bestEpsilon = epsilon

            

    print('Best F1 : ' ,F1, ' | Best Epsilon : ', bestEpsilon)

    

    return bestEpsilon

                
#Fit Mu and Sigma 

Mu = mu(X_train.as_matrix())

Sigma = sigma(X_train.as_matrix())

pred = p(X_CV.as_matrix(), Mu, Sigma)

epsilon = selectThreshold(pred, y_cv)
#Predictions for the test set

final_pred = p(X_test.as_matrix(), Mu, Sigma)

final_pred = np.array(final_pred < epsilon, dtype = int)

print(classification_report(y_test, final_pred))
#Predictions only for fraud transactions

fraud_pred = p(fraud.as_matrix(), Mu, Sigma)

fraud_pred = np.array(fraud_pred < epsilon, dtype = int)

print("Fraud Accuracy Score ", accuracy_score(fraud_y, fraud_pred) * 100 , "%")

#Confusion Matrix

cfm = confusion_matrix(y_test, final_pred)

ticks = ['Legal','Fraud']

sns.heatmap(cfm, annot = True,fmt = 'g', xticklabels = ticks, yticklabels= ticks)

sns.plt.suptitle('Multivariate Gaussian')

plt.show()

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder



data = pd.read_csv('../input/creditcard.csv')



#All the features are used for the neural network



#Normalization and features scaling

for f in data.columns.values:

    if f != 'Class':

        mean, std = data[f].mean(), data[f].std()

        data[f] = (data[f] - mean) / std



normal = data[data.Class == 0]

fraud = data[data.Class == 1]

#Split the data 



#20% of normal data + 50% of fraud

X_train = pd.concat([normal.sample(frac = 0.2),fraud.sample(frac = 0.50)], axis = 0)

X_test = pd.concat([normal.sample(frac = 0.2), fraud.sample(frac = 0.50)], axis = 0)





y_train, y_test = X_train.pop('Class'), X_test.pop('Class')

#One hot encoding

y_train = LabelEncoder().fit(y_train).transform(y_train)

y_train = to_categorical(y_train)

y_test = LabelEncoder().fit(y_test).transform(y_test)

y_test = to_categorical(y_test)



m, n = X_train.shape

normal_y = normal.pop('Class')

y_fraud = fraud.pop('Class')


input_size = X_train.shape[1]

output_size = 2

hidden_size = int((2 / 3) * input_size + output_size)



model = Sequential([Dense(hidden_size, activation = 'relu', input_dim = input_size),

                    Dropout(.15),

                    Dense(hidden_size, activation = 'relu'),

                    Dropout(.10),

                    Dense(hidden_size, activation = 'relu'),

                    Dropout(.5),

                    Dense(output_size, activation = 'softmax')

                    ])
#Training

model.compile(loss = 'categorical_crossentropy',

              optimizer = 'adam',

              metrics=['accuracy'])



model.fit(X_train.as_matrix(), y_train, batch_size = 128,  epochs = 100, verbose = 0);
#Predictions

pred = model.predict(X_test.as_matrix())

y_test = y_test.argmax(1)

pred = pred.argmax(1)



print(classification_report(y_test, pred))
#Predictions only for fraud transactions

fraud_pred = model.predict(fraud.as_matrix())



fraud_pred = fraud_pred.argmax(1)



print("Fraud Accuracy Score ", accuracy_score(y_fraud, fraud_pred) * 100 , "%")
#Confussion Matrix

cfm = confusion_matrix(y_test, pred)

sns.heatmap(cfm, annot = True,fmt = 'g',xticklabels = ticks, yticklabels= ticks)

sns.plt.suptitle('Neural Network')

plt.show()