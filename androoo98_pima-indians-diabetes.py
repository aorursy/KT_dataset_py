import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
#Load Data

data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

data.head()
sns.pairplot(data, hue='Outcome')
#First lets split the data

X = data.iloc[:, :-1].values

y = data.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Checking for nulls

data.isnull()

data.isnull().sum()

data.eq(0).any().any()
#Imputation

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values=0, strategy='mean', axis=0)

X_train = imputer.fit_transform(X_train)

X_test = imputer.fit_transform(X_test)
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)
def plot_corr(data, size):

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(size,size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)



plot_corr(data, data.shape[1])

data.corr()
sns.heatmap(data.corr(), annot=True)
#Fitting model

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
#Predicting test result

y_pred = classifier.predict(X_test)



#Predicting training result

y_pred_train = classifier.predict(X_train)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

true_pred = cm[1][1] + cm[0][0]

neg_pred = cm[1][0] + cm[0][1]

print(cm)

print('True prediction: ', true_pred, ' False prediction: ', neg_pred)

print('Accuracy of true prediction: ', (true_pred/X_test.shape[0])*100, '%')

print('Accuracy of false prediction: ', (neg_pred/X_test.shape[0])*100, '%')
#Performance on training data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_train, y_pred_train)))



#Performance on testing data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_test, y_pred)))
#Fitting model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train, y_train)
#Predicting test result

y_pred_knn = knn.predict(X_test)



#Predicting training result

y_pred_train_knn = knn.predict(X_train)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_knn)

true_pred = cm[1][1] + cm[0][0]

neg_pred = cm[1][0] + cm[0][1]

print(cm)

print('True prediction: ', true_pred, ' False prediction: ', neg_pred)

print('Accuracy of true prediction: ', (true_pred/X_test.shape[0])*100, '%')

print('Accuracy of false prediction: ', (neg_pred/X_test.shape[0])*100, '%')
#Performance on training data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_train, y_pred_train_knn)))



#Performance on testing data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_test, y_pred_knn)))
#Fitting model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
#Predicting test result

y_pred_tree = dt.predict(X_test)



#Predicting training result

y_pred_train_tree = dt.predict(X_train)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_tree)

true_pred = cm[1][1] + cm[0][0]

neg_pred = cm[1][0] + cm[0][1]

print(cm)

print('True prediction: ', true_pred, ' False prediction: ', neg_pred)

print('Accuracy of true prediction: ', (true_pred/X_test.shape[0])*100, '%')

print('Accuracy of false prediction: ', (neg_pred/X_test.shape[0])*100, '%')
#Display Tree

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO 

from IPython.display import Image 

from pydot import graph_from_dot_data

feature_names = data.columns[:-1]

dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, feature_names=feature_names)

(graph, ) = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
#Fitting model

from sklearn.naive_bayes import GaussianNB   

NB = GaussianNB()  

NB.fit(X_train, y_train)   
#Predicting test result

y_pred_nb = NB.predict(X_test)



#Predicting training result

y_pred_train_nb = NB.predict(X_train)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_nb)

true_pred = cm[1][1] + cm[0][0]

neg_pred = cm[1][0] + cm[0][1]

print(cm)

print('True prediction: ', true_pred, ' False prediction: ', neg_pred)

print('Accuracy of true prediction: ', (true_pred/X_test.shape[0])*100, '%')

print('Accuracy of false prediction: ', (neg_pred/X_test.shape[0])*100, '%')
#Performance on training data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_train, y_pred_train_nb)))



#Performance on testing data

from sklearn import metrics

print ("Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_test, y_pred_nb)))
from keras import Sequential

from keras.layers import Dense



classifierNN = Sequential()



#output = activation(dot(input, kernel) + bias)

#kernel is the weight matrix. kernel initialization defines the way to set the initial random weights of Keras layers.

#Random normal initializer generates tensors with a normal distribution.



#First Hidden Layer

classifierNN.add(Dense(8, activation='relu', kernel_initializer='random_normal', input_dim=8))



#Second Hidden Layer

classifierNN.add(Dense(8, activation='relu', kernel_initializer='random_normal'))



#Output Layer

classifierNN.add(Dense(1, activation='sigmoid',  kernel_initializer='random_normal'))



#Compiling the neural network

#We use binary_crossentropy to calculate the loss function between the actual output and the predicted output.

#Adam stands for Adaptive moment estimation. Adam is a combination of RMSProp + Momentum.

#Momentum takes the past gradients into account in order to smooth out the gradient descent.

#We use accuracy as the metrics to measure the performance of the model

classifierNN.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])





#Fitting the data to the training dataset

#We use a batch_size of 10. This implies that we use 10 samples per gradient update.3

#We iterate over 100 epochs to train the model. An epoch is an iteration over the entire data set.

classifierNN.fit(X_train,y_train, batch_size=10, epochs=100)
eval_model=classifierNN.evaluate(X_train, y_train)

eval_model
y_pred_NN=classifierNN.predict(X_test)

y_pred_NN =(y_pred_NN>0.5)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_NN)

true_pred = cm[1][1] + cm[0][0]

neg_pred = cm[1][0] + cm[0][1]

print(cm)

print('True prediction: ', true_pred, ' False prediction: ', neg_pred)

print('Accuracy of true prediction: ', (true_pred/X_test.shape[0])*100, '%')

print('Accuracy of false prediction: ', (neg_pred/X_test.shape[0])*100, '%')