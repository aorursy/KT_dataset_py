import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model

from IPython.display import SVG

from keras.utils import model_to_dot

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

import seaborn as sns



dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
def analyze(data):

    

  # View features in data set

  print("Dataset Features")

  print(data.columns.values)

  print("=" * 30)

    

  # View How many samples and how many missing values for each feature

  print("Dataset Features Details")

  print(data.info())

  print("=" * 30)

    

  # view distribution of numerical features across the data set

  print("Dataset Numerical Features")

  print(data.describe())

  print("=" * 30)

    

  # view distribution of categorical features across the data set

  print("Dataset Categorical Features")

  print(data.describe(include=['O']))

  print("=" * 30)
analyze(dataset)
sns.pairplot(dataset, hue="diagnosis", size= 2.5)
X = dataset.iloc[:,2:32] 

y = dataset.iloc[:,1] 
print("Earlier: ")

print(y[100:110])



labelencoder_Y = LabelEncoder()

y = labelencoder_Y.fit_transform(y)



print()

print("After: ")

print(y[100:110])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Scale values from faster convergence

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
def build_classifier(optimizer):

  classifier = Sequential()

  classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

  classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

  classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

  return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [1, 5],

               'epochs': [100, 120],

               'optimizer': ['adam', 'rmsprop']}
# Cross validation

grid_search = GridSearchCV(estimator = classifier,

                            param_grid = parameters,

                            scoring = 'accuracy',

                            cv = 10)
# Get best model

# Note: this may take some time

grid_search = grid_search.fit(X_train, y_train)
classifier = Sequential()
# Make the best classifier as we received earlier

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 100, verbose=1)
y_pred = classifier.predict(X_test)

# If probab is >= 0.5 classify as 1 or 0

y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]
# Finally use scikit-learn to build a confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
sns.heatmap(cm,annot=True)
# (True positive + True Negative)/Total

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

print("Accuracy: "+ str(accuracy*100)+"%")
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)