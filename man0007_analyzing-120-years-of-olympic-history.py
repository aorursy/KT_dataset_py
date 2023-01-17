import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
dataset_athlet = pd.read_csv("../input/athlete_events.csv")
dataset_noc = pd.read_csv("../input/noc_regions.csv")
dataset_noc = dataset_noc.iloc[:,[0,1]]
dataset_final=pd.merge(dataset_athlet, dataset_noc, on = 'NOC',how = 'left')

X = dataset_final.iloc[:,[2,3,4,5,10,12,15]].values
Y = dataset_final.iloc[:,[14]].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:,1:4] = imputer.transform(X[:, 1:4])
for i in range(len(X[:,6])):
    if type(X[:,6][i]) != type('string'):
        X[:,6][i] = 'unknown'
for i in range(len(Y[:,0])):
    if type(Y[:,0][i]) != type('asd'):
        Y[:,0][i] = 'No Medals'

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
onehotencoder_X = OneHotEncoder(categorical_features = [4,5,6])
X = onehotencoder_X.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])
onehotencoder_Y = OneHotEncoder(categorical_features = [0])
Y = onehotencoder_Y.fit_transform(Y).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 180, kernel_initializer = 'uniform', activation = 'relu', input_dim = 278))
    # Adding the second hidden layer
    classifier.add(Dense(units = 180, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y=Y_train, cv = 5)
score = accuracies.mean()
variance = accuracies.std()

print('the score is:', score)
print('The variance is:', variance)