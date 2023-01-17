# Load library

# data analysis and wrangling
import pandas as pd
import numpy as np
import random

# machine learning
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
# Load data
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
combine = [train_df, test_df]
train_df.head()
train_df.info()
test_df.info()
# Prepare data
print("Before", train_df.shape, test_df.shape)
for dataset in [train_df, test_df]:
    dataset.dropna(inplace = True)
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int) # Convert Sex features to numerical feature
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, "Q" : 2} ).astype(int) # Convert Embarked features to numerical feature
    dataset.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace = True) # Delete unused features
print("After", train_df.shape, test_df.shape)
# Prepare training data
random.seed(1)
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df
def print_score(clf):
    '''
    Input: trained classifier
    Print Precision, Recall and F-score
    '''
    scoring = {'recall0': make_scorer(recall_score, average = None, labels = [0]), 
           'recall1': make_scorer(recall_score, average = None, labels = [1]),
           'precision0': make_scorer(precision_score, average = None, labels = [0]),
           'precision1': make_scorer(precision_score, average = None, labels = [1]),
           'recall': 'recall',
           'precision': 'precision'
          }

    scores = cross_validate(clf, X_train, Y_train, cv=5,scoring=scoring)
    scores['test_f-measure'] = 2*scores['test_recall']*scores['test_precision']/(scores['test_recall']+scores['test_precision'])
    scores['test_f-measure0'] = 2*scores['test_recall0']*scores['test_precision0']/(scores['test_recall0']+scores['test_precision0'])
    scores['test_f-measure1'] = 2*scores['test_recall1']*scores['test_precision1']/(scores['test_recall1']+scores['test_precision1'])

    print("Precision class 0: %0.2f (+/- %0.2f)" % (scores['test_precision0'].mean(), scores['test_precision0'].std() * 2))
    print("Precision class 1: %0.2f (+/- %0.2f)" % (scores['test_precision1'].mean(), scores['test_precision1'].std() * 2))
    print("Recall class 0: %0.2f (+/- %0.2f)" % (scores['test_recall0'].mean(), scores['test_recall0'].std() * 2))
    print("Recall class 1: %0.2f (+/- %0.2f)" % (scores['test_recall1'].mean(), scores['test_recall1'].std() * 2))
    print("F-measure class 0: %0.2f (+/- %0.2f)" % (scores['test_f-measure0'].mean(), scores['test_f-measure0'].std() * 2))
    print("F-measure class 1: %0.2f (+/- %0.2f)" % (scores['test_f-measure1'].mean(), scores['test_f-measure1'].std() * 2))
    print("Average F-measure: %0.2f (+/- %0.2f)" % (scores['test_f-measure'].mean(), scores['test_f-measure'].std() * 2))
# Decision Tree
from sklearn.metrics import make_scorer

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
print("Decision Tree\n")
print_score(decision_tree) # Print classifier score
Y_pred_tree = decision_tree.predict(X_test) # Predict test set
# Gaussian Naive Bayes

GNB = GaussianNB()
GNB.fit(X_train, Y_train)
print("Naive Bayes\n")
print_score(GNB) # Print classifier score
Y_pred_GNB = GNB.predict(X_test) # Predict test set
# Neural Network

MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
MLP.fit(X_train, Y_train)
print("Neural Network\n")
print_score(MLP) # Print classifier score
Y_pred_MLP = MLP.predict(X_test)