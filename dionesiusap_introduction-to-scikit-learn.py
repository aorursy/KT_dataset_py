# Importing neccessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import itertools

# Models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# Metrics for model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample datasets
from sklearn import datasets
import pandas as pd

with open('../input/weather.nominal.csv', 'r') as fo:
    data = pd.read_csv(fo)

# Preview weather data
data
# Data preprocessing and preview
iris = datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names'] + ['target'])

# Preview data
df.head()
# Data properties
print('PROPERTIES')
print('Data shape\t: {}'.format(df.shape))
print('Unique target\t: {}'.format(df['target'].unique()))

print()
# Checking null values in data
# For iris dataset, it is unnecessary as the data is guaranteed to be clean.
#     But for best practice, better doing it
print('NULL VALUES')
for attribute in df.columns:
    print('{}\t: {}'.format(attribute, df[attribute].isnull().sum()))
# Creating gaussian naive bayes classifier
clf_gnb = GaussianNB()

# Fitting the data
clf_gnb.fit(iris.data, iris.target)
# Showing model
# Splitting data for train and test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)
class_names = iris.target_names
# Function for plotting confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGn)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
# Fitting the data
clf_gnb.fit(X_train, y_train)

pred_gnb = clf_gnb.predict(X_test)

# Evaluating the model performance
print('PERFORMANCE')
print('Accuracy\t: {}'.format(accuracy_score(y_test, pred_gnb, normalize=True)))
# Calculating and showing the confusion matrix
cnf_matrix_gnb = confusion_matrix(y_test, pred_gnb)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_gnb, classes=class_names, normalize=True, title='GaussianNB Confusion Matrix')

plt.show()
# Splitting Data
kf = KFold(n_splits=10, random_state=False)
# Calculating the score from every batch
scores_gnb = cross_val_score(clf_gnb, iris.data, iris.target, cv=kf)

print('PERFORMANCE')
print('Cross Validate Scores\t: {}'.format(scores_gnb))
print('Average Accuracy\t: {}'.format(np.average(scores_gnb)))
# Creating the classifier
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=3)

# Fitting the data
clf_tree.fit(iris.data, iris.target)
data_tree = tree.export_graphviz(clf_tree, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(data_tree)
graph
# Fitting the data
clf_tree.fit(X_train, y_train)

pred_tree = clf_tree.predict(X_test)

# Evaluating the model performance
print('PERFORMANCE')
print('Accuracy\t: {}'.format(accuracy_score(y_test, pred_tree, normalize=True)))
# Calculating and showing the confusion matrix
cnf_matrix_tree = confusion_matrix(y_test, pred_tree)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_tree, classes=class_names, normalize=True, title='Decision Tree Confusion Matrix')

plt.show()
# Calculating the score from every batch
scores_tree = cross_val_score(clf_tree, iris.data, iris.target, cv=kf)

print('PERFORMANCE')
print('Cross Validate Scores\t: {}'.format(scores_tree))
print('Average Accuracy\t: {}'.format(np.average(scores_tree)))