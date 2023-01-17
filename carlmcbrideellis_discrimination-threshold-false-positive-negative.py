import pandas  as pd

import matplotlib.pyplot as plt

import seaborn as sn

from   sklearn.metrics import accuracy_score

from   sklearn.metrics import f1_score

from   sklearn.metrics import precision_score

from   sklearn.metrics import recall_score

from   sklearn.metrics import confusion_matrix

from   sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
#===========================================================================

# read in the dataset

#===========================================================================

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)



#===========================================================================

# split the data into train and test

#===========================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



#===========================================================================

# perform the classification

#===========================================================================

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10,max_features=1,random_state=42)

classifier.fit(X_train, y_train);
predictions = classifier.predict(X_test)



print("The precision score is: %.2f" % precision_score( y_test, predictions))

print("The recall score is: %.2f" % recall_score( y_test, predictions), "\n")

print("Accuracy score is: %.2f" % accuracy_score( y_test, predictions))

print("The F1 score is: %.2f" % f1_score( y_test, predictions))



cm = confusion_matrix( y_test , predictions )

plt.figure(figsize = (3,3))

sn.heatmap(cm, annot=True, annot_kws={"size": 25}, fmt="d", cmap="viridis", cbar=False)

plt.show()
discrimination_threshold = 0.25

predictions = classifier.predict_proba(X_test)

predictions = (predictions[::,1] > discrimination_threshold )*1



print("The recall score is: %.2f" % recall_score( y_test, predictions))

print("The precision score is: %.2f" % precision_score( y_test, predictions),"\n")

print("Accuracy score is: %.2f" % accuracy_score( y_test, predictions))

print("The F1 score is: %.2f" % f1_score( y_test, predictions))



cm = confusion_matrix( y_test , predictions )

plt.figure(figsize = (3,3))

sn.heatmap(cm, annot=True, annot_kws={"size": 25}, fmt="d", cmap="viridis", cbar=False)

plt.show()
discrimination_threshold = 0.75

predictions = classifier.predict_proba(X_test)

predictions = (predictions[::,1] > discrimination_threshold )*1



print("The precision score is: %.2f" % precision_score( y_test, predictions))

print("The recall score is: %.2f" % recall_score( y_test, predictions), "\n")

print("Accuracy score is: %.2f" % accuracy_score( y_test, predictions))

print("The F1 score is: %.2f" % f1_score( y_test, predictions))



cm = confusion_matrix( y_test , predictions )

plt.figure(figsize = (3,3))

sn.heatmap(cm, annot=True, annot_kws={"size": 25}, fmt="d", cmap="viridis", cbar=False)

plt.show()
from yellowbrick.classifier import DiscriminationThreshold

visualizer = DiscriminationThreshold(classifier, size=(1000, 500))



visualizer.fit(X_train, y_train)

visualizer.show();