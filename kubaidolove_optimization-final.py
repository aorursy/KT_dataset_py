import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score



dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values
#Encoding Categorical Data

from sklearn.preprocessing import OneHotEncoder

#cp

oneHotEncoder = OneHotEncoder(categorical_features=[2], n_values='auto')

oneHotEncoder.fit(X)

X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]

#restecg

oneHotEncoder = OneHotEncoder(categorical_features=[8], n_values='auto')

oneHotEncoder.fit(X)

X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]

#slope

oneHotEncoder = OneHotEncoder(categorical_features=[13], n_values='auto')

oneHotEncoder.fit(X)

X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]

#ca

oneHotEncoder = OneHotEncoder(categorical_features=[15], n_values='auto')

oneHotEncoder.fit(X)

X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]

#thal

oneHotEncoder = OneHotEncoder(categorical_features=[19], n_values='auto')

oneHotEncoder.fit(X)

X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]



from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()

X = scalerX.fit_transform(X)
from sklearn.model_selection import train_test_split

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(XTrain,yTrain)

yPred = classifier.predict(XTest)

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Logistic Regression :")

print("Accuracy = ", accuracy)

print(cm)
#K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

classifier.fit(XTrain,yTrain)

yPred = classifier.predict(XTest)

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("K Nearest Neighbors :")

print("Accuracy = ", accuracy)

print(cm)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(XTrain,yTrain)

yPred = classifier.predict(XTest)

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Gaussian Naive Bayes :")

print("Accuracy = ", accuracy)

print(cm)
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier as DT

classifier = DT(criterion='entropy', random_state=0)

classifier.fit(XTrain,yTrain)

yPred = classifier.predict(XTest)

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Decision Tree Classifier :")

print("Accuracy = ", accuracy)

print(cm)
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier as RF

classifier = RF(n_estimators=10, criterion='entropy', random_state=0)

classifier.fit(XTrain,yTrain)

yPred = classifier.predict(XTest)

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Random Forest Classifier :")

print("Accuracy = ", accuracy)

print(cm)
#Artificial Neural Network

from keras.models import Sequential

from keras.layers import Dense



#Initialising ANN

classifier = Sequential()



#Adding the first hidden layer or the input layer

classifier.add(Dense(activation='relu',

                     kernel_initializer='uniform',

                     input_dim=22,

                     units=12))

#Adding the second hidden layer

classifier.add(Dense(activation='relu',

                     kernel_initializer='uniform',

                     units=12))

#Adding the output layer

classifier.add(Dense(activation='sigmoid',

                     kernel_initializer='uniform',

                     units=1))



#Compiling the ANN

classifier.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

print(classifier.summary())



#Fitting the ANN

history = classifier.fit(XTrain, yTrain, batch_size=3, epochs=200, verbose=1)

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'],'green')

plt.plot(history.history['loss'],'red')

plt.title('Model Accuracy-Loss')

plt.xlabel('Epoch')

plt.legend(['Accuracy','Loss'])

plt.show()



#Predicting the Test set Results

yPred = classifier.predict(XTest)

yPred = (yPred>0.5) #Since output is probability

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Artificial Neural Network Classifier :")

print("Accuracy = ", accuracy)

print(cm)