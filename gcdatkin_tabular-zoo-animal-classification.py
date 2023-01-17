import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/zoo-animal-classification/zoo.csv')
data
data.drop('animal_name', axis=1, inplace=True)
data.info()
data.isna().sum()
y = data['class_type']

X = data.drop('class_type', axis=1)
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression()



log_model.fit(X_train, y_train)
from sklearn.svm import SVC



svm_model = SVC(C=10.0)



svm_model.fit(X_train, y_train)
from sklearn.neural_network import MLPClassifier



nn_model = MLPClassifier(hidden_layer_sizes=(64, 64))



nn_model.fit(X_train, y_train)
log_acc = log_model.score(X_test, y_test)

svm_acc = svm_model.score(X_test, y_test)

nn_acc = nn_model.score(X_test, y_test)



print("Accuracy Results\n" + "*"*16)

print("      Logistic Model:", log_acc)

print("           SVM Model:", svm_acc)

print("Neural Network Model:", nn_acc)