import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
iris_dataset = sns.load_dataset("iris")
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.drop(["species"], axis=1), iris_dataset.species,test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=0.2, random_state=1)
print("Size of train set :", X_train.shape[0])
print("Size of validation set :", X_val.shape[0])
print("Size of test set :", X_test.shape[0])
print("Feature variables :", X_train.columns)
print("Target variables :", Y_train.name)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(X_train, Y_train)
Y_hat_val = clf.predict(X_val)
Y_hat_train = clf.predict(X_train)
from sklearn.metrics import accuracy_score

print("Accuracy on train set :", accuracy_score(Y_train, Y_hat_train))
print("Accuracy on validation set :", accuracy_score(Y_val, Y_hat_val))