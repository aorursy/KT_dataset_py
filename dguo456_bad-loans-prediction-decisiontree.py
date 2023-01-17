%matplotlib inline
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import tree
from sklearn import metrics
data = pd.read_csv(os.path.join("../input", "loan_sub.csv"), sep=',')
data.head()
data.columns
data.dtypes
data['good_loans'] = data['bad_loans'].apply(lambda x : +1 if x==0 else -1)
data = data.drop('bad_loans', axis=1)
data['good_loans'].value_counts(normalize=True)
cols = ['grade', 'term','home_ownership', 'emp_length']
target = 'good_loans'

data = data[cols + [target]]
data.head()
data['good_loans'].value_counts()
# use the percentage of bad and good loans to downsample the good loans.
bad_ones = data[data[target] == -1]
good_ones = data[data[target] == 1]
percentage = len(bad_ones)/float(len(good_ones))

risky_loans = bad_ones
safe_loans = good_ones.sample(frac=percentage, random_state=33)

# combine two kinds of loans
data_set = pd.concat([risky_loans, safe_loans], axis=0)
data_set[target].value_counts(normalize=True)
def dummies(data, columns=['col_1','col_2','col_3', 'col_4']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
# one hot encoding
cols = ['grade', 'term','home_ownership', 'emp_length']
data_set = dummies(data_set, columns=cols)
data_set.head()
train_data, test_data = train_test_split(data_set, test_size=0.2, random_state=33)
trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])
testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(trainX, trainY)
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")
    
    if show_confussion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")
measure_performance(testX, testY, model)
import graphviz
dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 
graph = graphviz.Source(dot_data) 
#graph.render("loan") 
#graph.view()
graph
model_5 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)
model_5.fit(trainX, trainY)
measure_performance(testX, testY, model_5)