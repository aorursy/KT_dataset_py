import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
dt = pd.read_csv("../input/heart.csv")
dt.head(10)
dt.dtypes
predicts = dt.iloc[:, 0:13].values
cla = dt.iloc[:,13].values
from sklearn.model_selection import train_test_split

predicts_train, predicts_test, class_train, class_test = train_test_split(predicts, cla, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB

classifier_NB = GaussianNB()

classifier_NB.fit(predicts_train, class_train)

predicts_NB = classifier_NB.predict(predicts_test)
from sklearn.metrics import confusion_matrix, accuracy_score

predict_NB = accuracy_score(class_test, predicts_NB)
matriz_NB = confusion_matrix(class_test, predicts_NB)
plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(matriz_NB,annot=True,cmap="Blues",fmt="d",cbar=False)
print(predict_NB)
from sklearn.tree import DecisionTreeClassifier

classifier_Tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)

classifier_Tree.fit(predicts_train, class_train)

predicts_Tree = classifier_Tree.predict(predicts_test)
predict_Tree = accuracy_score(class_test, predicts_Tree)
matriz_Tree = confusion_matrix(class_test, predicts_Tree)
plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(matriz_Tree,annot=True,cmap="Greens",fmt="d",cbar=False)
print(predict_Tree)
from sklearn.ensemble import RandomForestClassifier

classifier_RF = RandomForestClassifier(n_estimators=10, criterion= 'entropy', random_state=0)

classifier_RF.fit(predicts_train, class_train)

predicts_RF= classifier_RF.predict(predicts_test)
predict_RF = accuracy_score(class_test, predicts_RF)
matriz_RF = confusion_matrix(class_test, predicts_RF)
plt.title("Random Forest Confusion Matrix")

sns.heatmap(matriz_RF,annot=True,cmap="Reds",fmt="d",cbar=False)
print(predict_RF)
predicts_enc = dt.iloc[:, 0:13].values
cla_enc = dt.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

labelencoder_predicts = LabelEncoder()

predicts_enc[:,1] =  labelencoder_predicts.fit_transform(predicts_enc[:,1])

predicts_enc[:,2] =  labelencoder_predicts.fit_transform(predicts_enc[:,2])

predicts_enc[:,3] =  labelencoder_predicts.fit_transform(predicts_enc[:,3])

predicts_enc[:,4] =  labelencoder_predicts.fit_transform(predicts_enc[:,4])

predicts_enc[:,5] =  labelencoder_predicts.fit_transform(predicts_enc[:,5])

predicts_enc[:,6] =  labelencoder_predicts.fit_transform(predicts_enc[:,6])

predicts_enc[:,7] =  labelencoder_predicts.fit_transform(predicts_enc[:,7])

predicts_enc[:,8] =  labelencoder_predicts.fit_transform(predicts_enc[:,8])

predicts_enc[:,9] =  labelencoder_predicts.fit_transform(predicts_enc[:,9])

predicts_enc[:,10] =  labelencoder_predicts.fit_transform(predicts_enc[:,10])

predicts_enc[:,11] =  labelencoder_predicts.fit_transform(predicts_enc[:,11])

predicts_enc[:,12] =  labelencoder_predicts.fit_transform(predicts_enc[:,12])
onehotencode = OneHotEncoder(categories='auto')

predicts_enc = onehotencode.fit_transform(predicts_enc).toarray()
labelencoder_cla_enc = LabelEncoder()

cla_enc = labelencoder_cla_enc.fit_transform(cla_enc)
predicts_train_enc, predicts_test_enc, class_train_enc, class_test_enc = train_test_split(predicts_enc, cla_enc, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier_KNN.fit(predicts_train_enc, class_train_enc)

predicts_KNN = classifier_KNN.predict(predicts_test_enc)
predict_KNN = accuracy_score(class_test_enc, predicts_KNN)
matriz_KNN = confusion_matrix(class_test_enc, predicts_KNN)
plt.title("KNN Confusion Matrix")

sns.heatmap(matriz_KNN,annot=True,cmap="Purples",fmt="d",cbar=False)
print(predict_KNN)
from sklearn.linear_model import LogisticRegression

classifier_Regression = LogisticRegression(solver = 'lbfgs')

classifier_Regression.fit(predicts_train_enc, class_train_enc)

predicts_Regression = classifier_Regression.predict(predicts_test_enc)
predict_Regression = accuracy_score(class_test_enc, predicts_Regression)
matriz_Regression = confusion_matrix(class_test_enc, predicts_Regression)
plt.title("Regression Confusion Matrix")

sns.heatmap(matriz_Regression,annot=True,cmap="Oranges",fmt="d",cbar=False)
print(predict_Regression)
from sklearn.svm import SVC

classifier_SVM = SVC(kernel='linear', random_state=1)

classifier_SVM.fit(predicts_train_enc, class_train_enc)

predicts_SVM = classifier_Regression.predict(predicts_test_enc)
predict_SVM = accuracy_score(class_test_enc, predicts_SVM)
matriz_SVM = confusion_matrix(class_test_enc, predicts_SVM)
plt.title("SVM Confusion Matrix")

sns.heatmap(matriz_SVM,annot=True,cmap="Greys",fmt="d",cbar=False)
print(predict_SVM)
from sklearn.neural_network import MLPClassifier

classifier_Neural = MLPClassifier(verbose=True, max_iter=2000, tol=0.00002)

classifier_Neural.fit(predicts_train_enc, class_train_enc)

predicts_Neural = classifier_Neural.predict(predicts_test_enc)
predict_Neural = accuracy_score(class_test_enc, predicts_Neural)
matriz_Neural = confusion_matrix(class_test_enc, predicts_Neural)
plt.title("Neural Networks Confusion Matrix")

sns.heatmap(matriz_Neural,annot=True,cmap="Blues",fmt="d",cbar=False)
print(predict_Neural)
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier_Neural_Keras = Sequential()

classifier_Neural_Keras.add(Dense(units = 5, activation = 'relu', input_dim = 398))
classifier_Neural_Keras.add(Dense(units = 5, activation = 'relu'))

classifier_Neural_Keras.add(Dense(units = 1, activation = 'sigmoid'))
classifier_Neural_Keras.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
classifier_Neural_Keras.fit(predicts_train_enc, class_train_enc, batch_size = 8, epochs = 40)
predicts_Neural_Keras = classifier_Neural_Keras.predict(predicts_test_enc)
predict_Neural_Keras = accuracy_score(class_test_enc, predicts_Neural_Keras.round())
matriz_Neural_Keras = confusion_matrix(class_test_enc, predicts_Neural_Keras.round())
plt.title("Neural Networks_Keras Confusion Matrix")

sns.heatmap(matriz_Neural_Keras,annot=True,cmap="Greens",fmt="d",cbar=False)
print(predict_Neural_Keras)
plt.rcParams['figure.figsize'] = (16,5)

names_alg = ['Naive Bayes', 'Tree Classifier', 'Random Forest', 'KNN', 'Regression', 'SVM', 'Neural Networks', 'Keras']

result_alg = [predict_NB, predict_Tree, predict_RF, predict_KNN, predict_Regression, predict_SVM, predict_Neural, predict_Neural_Keras]

xs = [i + 0.5 for i, _ in enumerate(names_alg)]

plt.bar(xs,result_alg, color=('#8B0000','#FF6347','#CD6600','#8B8B00','#458B00','#53868B','#EE7942','#00FF33'))

plt.ylabel("Value")

plt.title("Algorithms")

plt.xticks([i + 0.5 for i, _ in enumerate(names_alg)], names_alg)

plt.show()