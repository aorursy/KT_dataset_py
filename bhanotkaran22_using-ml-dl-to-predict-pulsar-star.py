import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout



from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

dataset.head(5)
dataset.isnull().sum()
dataset.describe().T
plt.figure(figsize = (12, 8))

plot = sns.countplot(dataset['target_class'])

plot.set_title("Target Class count")

for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), xy = (p.get_x() + 0.35, p.get_height() + 40))
plt.figure(figsize = (12, 8))

sns.heatmap(dataset.corr(), annot = True, fmt = ".2f")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(dataset.iloc[:, :-1])

principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal component 1', 'Principal component 2'])

principalDf = pd.concat([principalDf, dataset.iloc[:, -1]], axis = 1)
plt.figure(figsize = (20, 12))

sns.scatterplot(x = 'Principal component 1', 

                y = 'Principal component 2', 

                data = principalDf,

               hue = 'target_class')
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], random_state = 0, test_size = 0.3)
def metrics(model, y_true, y_pred):

    print("The accuracy of the model {} is: {:.2f}%".format(model, accuracy_score(y_true, y_pred)*100))

    print("Confusion matrix for {}".format(model))

    print(confusion_matrix(y_true, y_pred))

    print("-"*40)
standardScaler = StandardScaler()

X_train = standardScaler.fit_transform(X_train)

X_test = standardScaler.transform(X_test)
# Support Vector Classifier

supportVectorClassifier = SVC(kernel = 'rbf')

supportVectorClassifier.fit(X_train, y_train)



# Random Forest Classifier

randomForestClassifier = RandomForestClassifier(n_estimators = 100)

randomForestClassifier.fit(X_train, y_train)



# Artificial Neural Network

artificialNeuralNetwork = Sequential()

artificialNeuralNetwork.add(Dense(units = 32, activation = 'relu', input_dim = 8))

artificialNeuralNetwork.add(Dropout(0.5))

artificialNeuralNetwork.add(Dense(units = 64, activation = 'relu'))

artificialNeuralNetwork.add(Dropout(0.5))

artificialNeuralNetwork.add(Dense(units = 128, activation = 'relu'))

artificialNeuralNetwork.add(Dropout(0.5))

artificialNeuralNetwork.add(Dense(units = 1, activation = 'sigmoid'))

artificialNeuralNetwork.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

artificialNeuralNetwork.fit(X_train, y_train, epochs = 50, shuffle = False, validation_split = 0.1, verbose = 0)
# Support Vector Classifier

metrics("Support Vector Classifier", y_test, supportVectorClassifier.predict(X_test)) 



# Random Forest Classifier

metrics("Random Forest Classifier", y_test, randomForestClassifier.predict(X_test)) 



# Artificial Neural Network

metrics("Artificial Neural Network", y_test, (artificialNeuralNetwork.predict(X_test) > 0.5))