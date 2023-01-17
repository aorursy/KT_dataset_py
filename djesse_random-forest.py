import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



# Load the training data

data = pd.read_csv("../input/binary-cooking/training_binary.csv")

y = data['cuisine']

x_frame = data.drop(['cuisine', 'id'],axis=1)



# Load the test data

test_data = pd.read_csv("../input/binary-cooking/test_binary.csv")

test_y = test_data['cuisine']

test_x_frame = test_data.drop(['cuisine', 'id'],axis=1)
from sklearn.metrics import precision_score, recall_score, accuracy_score



# Classify data with Random Forest classifier using default parameters

clf = RandomForestClassifier()

clf.fit(x_frame, y)



training_predictions = clf.predict(x_frame)

test_predictions = clf.predict(test_x_frame)



# Accuracy

print("Accuracy training: {}".format(accuracy_score(y, training_predictions)))

print("Accuracy test: {}".format(accuracy_score(test_y, test_predictions)))



# Precision

print("Average precision rate training: {}".format(precision_score(y, training_predictions, average='weighted')))

print("Average precision rate test: {}".format(precision_score(test_y, test_predictions, average='weighted')))



# Recall

print("Average recall rate training: {}".format(recall_score(y, training_predictions, average='weighted')))

print("Average recall rate test: {}".format(recall_score(test_y, test_predictions, average='weighted')))
from sklearn.metrics import confusion_matrix



# Confusion matrix training set using default parameters

pd.DataFrame(confusion_matrix(y,training_predictions,labels=list(set(y))),index=list(set(y)),columns=list(set(y)))
# Confusion matrix test set using default parameters

pd.DataFrame(confusion_matrix(test_y,test_predictions,labels=list(set(test_y))),index=list(set(test_y)),columns=list(set(test_y)))
# Fit Random Forest again using optimized parameters



opt_clf = RandomForestClassifier(n_estimators=80, max_depth=90)

opt_clf.fit(x_frame, y)



training_predictions = opt_clf.predict(x_frame)

test_predictions = opt_clf.predict(test_x_frame)



# Accuracy

print("Accuracy training: {}".format(accuracy_score(y, training_predictions)))

print("Accuracy test: {}".format(accuracy_score(test_y, test_predictions)))



# Precision

print("Average precision rate training: {}".format(precision_score(y, training_predictions, average='weighted')))

print("Average precision rate test: {}".format(precision_score(test_y, test_predictions, average='weighted')))



# Recall

print("Average recall rate training: {}".format(recall_score(y, training_predictions, average='weighted')))

print("Average recall rate test: {}".format(recall_score(test_y, test_predictions, average='weighted')))
# Confusion matrix training set using optimized parameters

pd.DataFrame(confusion_matrix(y,training_predictions,labels=list(set(y))),index=list(set(y)),columns=list(set(y)))
# Confusion matrix test set using optimized parameters

pd.DataFrame(confusion_matrix(test_y,test_predictions,labels=list(set(test_y))),index=list(set(test_y)),columns=list(set(test_y)))
import matplotlib.pyplot as plt



accuracy_train = []

accuracy_test = []



precision_train = []

precision_test = []



recall_train = []

recall_test = []



# Fit random forest classifier to find optimal max_depth in steps of 20 each

for i in range (1,220, 20):

    clf = RandomForestClassifier(n_estimators=80, max_depth=i)

    clf.fit(x_frame, y)

    

    # Predict on training and test data

    predictions = clf.predict(x_frame)

    t_predictions = clf.predict(test_x_frame)

    

    print(i)

    # Calculate accuracy

    accuracy_train.append(accuracy_score(y,predictions))

    accuracy_test.append(accuracy_score(test_y,t_predictions))

    

    # Calculate precision

    precision_train.append(precision_score(y, predictions, average='weighted'))

    precision_test.append(precision_score(test_y,t_predictions, average='weighted'))

    

    # Calculate recall

    recall_train.append(recall_score(y, predictions, average='weighted'))

    recall_test.append(recall_score(test_y,t_predictions, average='weighted'))

    print("===============")
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# Plot results of finding optimal tree depth

x = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

y = accuracy_train

fig, ax = plt.subplots()

ax.plot(x,accuracy_train, label="Training data")

ax.plot(x,accuracy_test, label="Test data")

start, end = ax.get_xlim()

ax.xaxis.set_ticks(np.arange(start + 8.95, end, 20))

ax.set_xlabel("Maximal tree depth")

ax.set_ylabel("Accuracy")

plt.legend()

plt.show()