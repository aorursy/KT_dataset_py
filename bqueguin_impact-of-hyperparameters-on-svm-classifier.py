%reset
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import data
train = pd.read_csv('../input/train.csv', index_col=0)
train.head()
# Transform species values to number label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(train.species)
train.species = le.transform(train.species)
all(train.species.value_counts() == 10) # There are 10 leaves for each species
# Define the target and the features
Y = train.iloc[:, 0]
X = train.iloc[:, 1:]
# Normalize the features X
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scale = scaler.transform(X)
# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.25, stratify=Y)
# stratify=y means that every kind of leaves will be present at the same proportion in the train and test set.
# Baseline with k-NN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
Y_prob = knn.predict_proba(X_test)

from sklearn.metrics import accuracy_score, log_loss
baseline_accuracy = accuracy_score(Y_test, Y_pred) 
baseline_logloss = log_loss(Y_test, Y_prob)
print(baseline_accuracy) # 95.2% of the leaves are well classified, this is our baseline
print(baseline_logloss)
import warnings
warnings.filterwarnings('ignore')

# Default multiclass SVM classifier (OvR and l2 penalty)
from sklearn.svm import LinearSVC
clf = LinearSVC(dual=False).fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print(accuracy_score(Y_test, Y_pred)) # Accuracy = 97.2%. This is better than with k-NN
# Multiclass SVM classifier with optimal hyper-parameters
from sklearn.model_selection import GridSearchCV

lsvm = LinearSVC(dual=False) # dual = False beacause we have more samples than variables
params = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2', 'l1'], 'multi_class': ['ovr', 'crammer_singer']}
# Here we will test 3 differents hyper-parameters:
    # C: the value of the coefficient of regularization
    # penalty: the norm used for regularization
    # multi_class: the method used for a multi-class problem, One vs rest or Crammer Singer
    # note: for Crammer Singer method, only l2 penalty is possible
gs = GridSearchCV(lsvm, params, cv=3)
# Training of the models
gs.fit(X_train, Y_train) # Takes few minutes ...
print(gs.best_params_) # It looks like the Crammer Singer method is the most efficient with C=0.01
# Compute the accuracy with optimal hyper-parameters
Y_pred = gs.predict(X_test)
print(accuracy_score(Y_test, Y_pred)) # The accuracy is now 98.4%, clearly better than the baseline
# Let's plot the evolution of the accuracy in terms of the hyper parameters

C = np.logspace(-4, 3, 15)

ovr_clf2 = LinearSVC(dual=False, penalty='l2', multi_class='ovr')
accuracy_ovr2 = []
for c in C:
    ovr_clf2.set_params(C=c)
    ovr_clf2.fit(X_train, Y_train)
    accuracy_ovr2.append(ovr_clf2.score(X_test, Y_test))

cs_clf2 = LinearSVC(dual=False, penalty='l2', multi_class='crammer_singer')
accuracy_cs2 = []
for c in C:
    cs_clf2.set_params(C=c)
    cs_clf2.fit(X_train, Y_train)
    accuracy_cs2.append(cs_clf2.score(X_test, Y_test))

ovr_clf1 = LinearSVC(dual=False, penalty='l1', multi_class='ovr')
accuracy_ovr1 = []
for c in C:
    ovr_clf1.set_params(C=c)
    ovr_clf1.fit(X_train, Y_train)
    accuracy_ovr1.append(ovr_clf1.score(X_test, Y_test))
plt.figure(figsize=(15, 10))
plt.plot([min(C), max(C)], [baseline_accuracy, baseline_accuracy], label='baseline (k-NN)', color='black', linestyle='--', linewidth=0.5)
plt.plot(C, accuracy_ovr2, label='OvsR, l2') 
plt.plot(C, accuracy_ovr1, label='OvsR, l1')
plt.plot(C, accuracy_cs2, label='Crammer Singer, l2')
plt.xscale('log')
plt.xlabel('Value of C')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.axis('tight')
plt.ylim([0.85, 1])
plt.show()
# RESULTS

# The Crammer Singer method is definitely more efficient than the One vs Rest method.
# We can see that the value of C don't really impact the accuracy for the Crammer Singer method.
# For the One vs Rest method the value of C impacts the accuracy.
# Finding good hyper-parameters can improve the score of multi-class SVM classifiers.