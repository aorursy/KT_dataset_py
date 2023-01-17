%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

classifiers = [
    SGDClassifier(loss = 'log'),
    SVC(probability = True, kernel = 'rbf', C=1000, gamma=0.1),
    GaussianProcessClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier(activation = 'tanh', max_iter = 2000, solver = 'adam', hidden_layer_sizes = (50,50,50)),
    KNeighborsClassifier(),
    NuSVC(probability = True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

# Evaluation on the trained model
def evaluate_model(model, x_train, y_train, x_test, y_test):
    print('%s\nfitting...' % (model.__class__.__name__, ))
    model.fit(x_train, y_train)
    print('evaluating...')
    y_predict = model.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_predict)
    print('\taccuracy score: %f' % (accuracy_score, ))
    y_predict = model.predict_proba(x_test)
    log_loss = metrics.log_loss(y_test, y_predict)
    print('\tlog loss: %f\n\n' % (log_loss, ))
    return accuracy_score, log_loss
# Relative path for the train, test, and submission file
train_path = '../input/train.csv'
test_path = '../input/test.csv'
submission_path = '../input/sample_submission.csv'
submission_output = 'submit.csv'
# Load training data
train_data = pd.read_csv(train_path)
# Ignore the first column (id) and the second column (species) (pandas.DataFrame)
x = train_data.iloc[:, 2:]
# Convert the species to category type
y = train_data['species'].astype('category')
# Get the corresponding categories list for species (numpy.ndarray)
y = y.cat.codes.as_matrix()

# Load testing data
test_data = pd.read_csv(test_path)
# Load categories from submission file
submission_data = pd.read_csv(submission_path)

categories = submission_data.columns.values[1:]
n_class = len(categories)
categories_id = pd.Series(categories, dtype='category')

print('There %d classes' % (n_class,))
plt.hist(y, bins=n_class)
plt.title('Number of instances in each class')
plt.xlabel('class id')
plt.ylabel('number of instances')
plt.show()
# The folds are made by preserving the percentage of samples for each class
sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
for train_index, test_index in sss.split(x, y):
    # print('TRAIN:', train_index, 'TEST:', test_index)
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
accuracy_scores = list()
log_losses = list()

for clf in classifiers:
    accuracy_score, log_loss = evaluate_model(clf, x_train, y_train, x_test, y_test)
    accuracy_scores.append(accuracy_score)
    log_losses.append(log_loss)
log_cols = ['Classifier', 'Accuracy score', 'Log loss']
log = pd.DataFrame(columns = log_cols)
for i in range(0, len(classifiers)):
    log = log.append(
        pd.DataFrame([[classifiers[i].__class__.__name__, accuracy_scores[i], log_losses[i]]], 
                     columns = log_cols),
        ignore_index = True)

sns.barplot(x = 'Accuracy score', y = 'Classifier', data=log, color='r')
plt.xlabel('Accuracy score')
plt.title('Classifier accuracy score')
plt.show()

sns.barplot(x = 'Log loss', y = 'Classifier', data = log, color = 'b')
plt.xlabel('Log loss')
plt.title('Classifier log loss')
plt.show()
best_accuracy_classifier_id = np.argmax(accuracy_scores)
best_classifier = classifiers[best_accuracy_classifier_id]

best_classifier.fit(x_train, y_train)

test_y = best_classifier.predict_proba(test_data.iloc[:, 1:])

submission_data.iloc[:, 1:] = test_y
submission_data.tail()

f = open(submission_output, 'w')
f.write(pd.DataFrame(submission_data).to_csv(index = False))
f.close()
