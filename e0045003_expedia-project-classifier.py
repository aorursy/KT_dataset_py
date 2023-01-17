import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)

# Any results you write to the current directory are saved as output.
# Selected features which I think is more related to the outcome
features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'cnt']
X_train = train.loc[:, features]
Y_train = train.loc[:, 'target']
X_test = test.loc[:, features]
Y_test = test.loc[:, 'target']

# https://www.kaggle.com/dansbecker/handling-missing-values
# Imputation fills in the missing value with some number. 
# The imputed value won't be exactly right in most cases, 
# but it usually gives more accurate models than dropping the column entirely.
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)
from sklearn.tree import DecisionTreeClassifier

# Testing with DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score

Y_pred = clf.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

all_data = pd.read_csv('../input/all.csv')

# Slice necessary columns
# Selected features which I think is more related to the outcome
features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'cnt']
X_all = all_data.loc[:, features]
Y_all = all_data.loc[:, 'target'] 

# https://www.kaggle.com/dansbecker/handling-missing-values
# Imputation fills in the missing value with some number. 
# The imputed value won't be exactly right in most cases, 
# but it usually gives more accurate models than dropping the column entirely.
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_all = my_imputer.fit_transform(X_all)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

nFolds = 5 # Split data into 5 parts: use 4 parts for training, 1 part for testing. 
            # Rotating then average accuracy score at the end

# Add more type of classifier here
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    SVC(kernel="linear", C=0.025),
    MLPClassifier(early_stopping=True, hidden_layer_sizes=(2, 3))
]
# Accuracy matrix
accuracy = np.zeros(shape=(len(classifiers), nFolds))

kf = KFold(n_splits=nFolds)
fold_idx = 0
for train_index, test_index in kf.split(X_all):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = Y_all[train_index], Y_all[test_index]
    
    for clf_idx, clf in enumerate(classifiers):
        print('[Classifier {}]'.format(clf_idx))
        clf.fit(X_train, y_train)
        score = accuracy_score(y_test, clf.predict(X_test))
        accuracy[clf_idx][fold_idx] = score
        
    fold_idx += 1

print('# Accuracy matrix')
print(accuracy)

print('# Mean accuracy for each type classifier')
for clf, accuracy_mean in zip(classifiers, np.mean(accuracy, axis=1)):
    print(clf)
    print("-- Accuracy mean: {}".format(accuracy_mean))
