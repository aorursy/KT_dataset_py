import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
destinations = pd.read_csv('../input/destinations.csv')
print(train.shape)
print(test.shape)
print(destinations.shape)

# Any results you write to the current directory are saved as output.
#destinations file has 150 columns
#use PCA to downsize the columns into fewer features
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
dest_filtered = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_filtered = pd.DataFrame(dest_filtered)
dest_filtered["srch_destination_id"] = destinations["srch_destination_id"]

print(dest_filtered.shape)

# Selected features which I think is more related to the outcome
features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'cnt']
X_train = train.loc[:, features]
Y_train = train.loc[:, 'target']
X_test = test.loc[:, features]
Y_test = test.loc[:, 'target']

#add in the destination features to the feature sets
X_train = X_train.join(dest_filtered, on="srch_destination_id", how='left', rsuffix="dest")
X_train = X_train.drop("srch_destination_iddest", axis=1)
X_test = X_test.join(dest_filtered, on="srch_destination_id", how='left', rsuffix="dest")
X_test = X_test.drop("srch_destination_iddest", axis=1)

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
#use RandomForestClassififer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, Y_train)

Y_pred = rfc.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
#use SVM
from sklearn import svm
from sklearn.metrics import accuracy_score
svc = svm.SVC(kernel="linear",max_iter=500)
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LogReg_clf = LogisticRegression()
LogReg_clf.fit(X_train, Y_train)

Y_pred = LogReg_clf.predict(X_test)


print ('Accuracy = '+repr(accuracy_score(Y_pred, Y_test)))
#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=10)
abc.fit(X_train, Y_train)

Y_pred = abc.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
all_data = pd.read_csv('../input/all.csv')
#all_data = pd.read_csv('../input/test.csv')

# Slice necessary columns
# Selected features which I think is more related to the outcome
features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'cnt']
X_all = all_data.loc[:, features]
Y_all = all_data.loc[:, 'target'] 

#add in the destination features to the feature sets
X_all = X_all.join(dest_filtered, on="srch_destination_id", how='left', rsuffix="dest")
X_all = X_all.drop("srch_destination_iddest", axis=1)

# https://www.kaggle.com/dansbecker/handling-missing-values
# Imputation fills in the missing value with some number. 
# The imputed value won't be exactly right in most cases, 
# but it usually gives more accurate models than dropping the column entirely.
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_all = my_imputer.fit_transform(X_all)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

nFolds = 5 # Split data into 5 parts: use 4 parts for training, 1 part for testing. 
            # Rotating then average accuracy score at the end

# Add more type of classifier here
classifiers = [
    DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5),
    RandomForestClassifier(n_estimators=10),
    AdaBoostClassifier(n_estimators=10)
]
# Accuracy matrix
accuracy = np.zeros(shape=(len(classifiers), nFolds))

kf = KFold(n_splits=nFolds)
fold_idx = 0
for train_index, test_index in kf.split(X_all):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = Y_all[train_index], Y_all[test_index]
    
    for clf_idx, clf in enumerate(classifiers):
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