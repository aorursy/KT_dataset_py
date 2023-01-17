import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

from tqdm import tqdm

import matplotlib.pyplot as plt

import itertools



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)

	plt.title(title)

	plt.colorbar()

	tick_marks = np.arange(len(classes))

	plt.xticks(tick_marks, classes, rotation=45)

	plt.yticks(tick_marks, classes)



	print(cm)



	thresh = cm.max() / 2.

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

		plt.text(j, i, cm[i, j],

				 horizontalalignment="center",

				 color="white" if cm[i, j] > thresh else "black")



	plt.tight_layout()

	plt.ylabel('True label')

	plt.xlabel('Predicted label')

	plt.show()

	



# import mushroom data

mushroom = pd.read_csv("../input/mushrooms.csv")
# setting up data in numpy array

lab = preprocessing.LabelEncoder()

for col in mushroom.columns:

    mushroom[col] = lab.fit_transform(mushroom[col])

y = mushroom["class"]

X = mushroom.drop("class", axis = 1)



# train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2215)

logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "Logistic Regression")
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "KNN")
svm = SVC()

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "SVM")

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "Decision Tree")
rf = RandomForestClassifier(n_estimators = 20)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "Random Forest")
gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = ["y_true", "y_pred"], title = "Gradient Boosting")
Y = pd.get_dummies(mushroom.iloc[:,0],  drop_first=False)

X = pd.DataFrame()

for each in mushroom.iloc[:,1:].columns:

    dummies = pd.get_dummies(mushroom[each], prefix=each, drop_first=False)

    X = pd.concat([X, dummies], axis=1)

	

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2215)



model = Sequential()

model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(0.1), metrics=['accuracy'])



n_epochs = 10

for k in tqdm(range(n_epochs)):

	history = model.fit(x_train.values, y_train.values, epochs=1, verbose=0, validation_data = (x_test.values, y_test.values))



# calculate the loss and accuracy

score = model.evaluate(x_test.values, y_test.values, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



y_pred = model.predict(x_test.values)

y_true = np.array([np.argmax(y_test.values[k]) for k in range(y_test.shape[0])])

y_pred = np.array([np.argmax(y_pred[k]) for k in range(y_pred.shape[0])])

print(classification_report(y_true, y_pred))

plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes = ["y_true", "y_pred"], title = "Neural Net")