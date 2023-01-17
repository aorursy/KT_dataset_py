



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

dataset = pd.read_csv("../input/insurance/insurance.csv")

print(dataset.head())
X = dataset.iloc[:, 0:6].values

y = dataset.iloc[:, 6:7].values
print("Age     Gender  BMI     Kids    Smoker  Region    ")

s = [[str(e) for e in row] for row in X[0:5]]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))



print("\n")



print("Charge")

s = [[str(e) for e in row] for row in y[0:5]]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
from sklearn.preprocessing import LabelEncoder



labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_4 = LabelEncoder()

X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
shortenedX = X[0:10]

print("Age     Gender  BMI     Kids    Smoker  Region   ")

s = [[str(e) for e in row] for row in shortenedX]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

X = X[:, 1:]
shortenedX = X[0:10]

print("NW      SE      SW      Age     Gender  BMI     Kids    Smoker     ")

s = [[str(e) for e in row] for row in shortenedX]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler((0,1))

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



y_train = sc.fit_transform(y_train)

y_test = sc.transform(y_test)
shortenedX_train = X_train[0:5]

print("NW      SE      SW      Age                     Gender  BMI                     Kids    Smoker     ")

s = [[str(e) for e in row] for row in shortenedX_train]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
shortenedy_train = y_train[0:5]

print("Charge")

s = [[str(e) for e in row] for row in shortenedy_train]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
import keras

from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()

    

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 128, activation = 'relu'))

    

# Adding the second hidden layer

classifier.add(Dense(units = 64, activation = 'relu'))

    

classifier.add(Dense(units = 32, activation = 'relu'))

    

# Adding the output layer

classifier.add(Dense(units = 1, activation = 'linear'))

    

# Compiling the ANN

classifier.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    
History = classifier.fit(x = X_train, y = y_train, batch_size = 128, epochs = 150, verbose = 0)
plt.plot(History.history['mean_absolute_error'])

plt.title('Loss Function Over Epochs')

plt.ylabel('MAE value')

plt.xlabel('No. epoch')

plt.show()
y_pred = classifier.predict(X_test)



y_predInverse = sc.inverse_transform(y_pred)

y_testInverse = sc.inverse_transform(y_test)
combinedArray = np.column_stack((y_testInverse[0:10],y_predInverse[0:10]))

print("Actual Charge   Predicted Charge")

s = [[str(e) for e in row] for row in np.around(combinedArray, 2)]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))
def buildModel(optimizer):

    # Initialising the ANN

    classifier = Sequential()

    

    # Adding the input layer and the first hidden layer

    classifier.add(Dense(units = 128, activation = 'relu'))

    

    # Adding the second hidden layer

    classifier.add(Dense(units = 64, activation = 'relu'))

    

    

    classifier.add(Dense(units = 32, activation = 'relu'))

    

    # Adding the output layer

    classifier.add(Dense(units = 1, activation = 'linear'))

    

    # Compiling the ANN

    classifier.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    

    return classifier
from sklearn.model_selection import GridSearchCV 

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



classifier = KerasRegressor(build_fn = buildModel)

#What hyperparameter we want to play with

parameters = {'batch_size': [16, 32, 64, 128],

              'epochs': [100, 150],

              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'neg_mean_absolute_error',

                           cv = 5)

grid_search = grid_search.fit(X_train, y_train, verbose = 0)
best_parameters = grid_search.best_params_

best_score = grid_search.best_score_



print("Best Parameters: " + str(best_parameters))
bestClassifier = buildModel('adam')

HistoryBest = bestClassifier.fit(x = X_train, y = y_train, batch_size = 16, epochs =150 , verbose = 0)

plt.plot(History.history['mean_absolute_error'], label='Initial Parameters')

plt.plot(HistoryBest.history['mean_absolute_error'], label='GridSearchCV Best Parameters')

plt.title('Loss Function Over Epochs')

plt.ylabel('MAE value')

plt.xlabel('No. epoch')

plt.legend(loc="upper right")

plt.show()
from sklearn.metrics import mean_absolute_error 



print("Initial Classifier MAE: " + str(mean_absolute_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average')))

print("Best Classifier MAE: " + str(mean_absolute_error(y_test, bestClassifier.predict(X_test), sample_weight=None, multioutput='uniform_average')))
y_predBestInverse = sc.inverse_transform(bestClassifier.predict(X_test))



combinedArray = np.column_stack((y_testInverse[0:10],y_predInverse[0:10], y_predBestInverse[0:10]))

print("Actual Charge   Initial         Best ")

s = [[str(e) for e in row] for row in np.around(combinedArray, 2)]

lens = [max(map(len, col)) for col in zip(*s)]

fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)

table = [fmt.format(*row) for row in s]

print ('\n'.join(table))