import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

df.head()
dataset_dummies = pd.get_dummies(data=df, columns=['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage'],drop_first=True)

dataset_dummies.head(10)
df = dataset_dummies.drop('id', 1)
df = df.drop('Policy_Sales_Channel',1)
df.head()
y = df.loc[:, ["Response"]].values

df.drop(["Response"], axis=1, inplace=True)

X = df.iloc[:, 1:].values




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X)

y_train = y

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, class_weight='balanced')

classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'class_weight': ['balanced']}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
classifier_cv = LogisticRegression(random_state = 0, C=0.25, class_weight='balanced')

classifier_cv.fit(X_train, y_train)

y_pred = classifier_cv.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)*100
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))


from lightgbm import LGBMClassifier

classifier = LGBMClassifier(n_jobs=-1)

classifier.fit(X_train, y_train);



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))