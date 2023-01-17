import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelencoder_x_2=LabelEncoder()
X[:,4]=labelencoder_x_2.fit_transform(X[:,4])

X=X[:, 1:]

X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer with Dropout
classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
classifier.add(Dropout(p = 0.1))

# Adding Second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))# init = 'uniform'==> init weight randomly, activation Function = 'relu' 
classifier.add(Dropout(p = 0.1))

# Adding ouyput layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid' ))

# Compiing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,5000]])))
new_pred = (new_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))# init = 'uniform'==> init weight randomly, activation Function = 'relu' 
    classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid' ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.var()
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 
    classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# parameters = {'batch_size': range(25, 32),
#               'epochs': range(100, 501),
#               'optimizer': ['adam', 'rmsprop']}

parameters = {'batch_size': [25, 32],
              'epochs': [100, 103],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,y_train)
# # summarize results
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best accuracy: %f\nusing parameters : %s" % (best_accuracy,best_param))



# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
    