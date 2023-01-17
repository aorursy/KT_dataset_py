# import libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes

import numpy as np # Import Numpy for data statistical analysis 

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns # Statistical data visualization

# %matplotlib inline



# Import Cancer data drom the Sklearn library

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()





df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.shape



training_set = df_cancer.iloc[:, 0:30].values

test_set =  df_cancer.iloc[:, -1].values





# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(training_set, test_set, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 50 )

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()

variance = accuracies.std()


