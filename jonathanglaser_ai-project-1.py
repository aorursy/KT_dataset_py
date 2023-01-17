# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_test_new = pd.read_csv("../input/new-datasets/X_test_new.csv")

X_train_new = pd.read_csv("../input/new-datasets/X_train_new.csv")

y_test_new = pd.read_csv("../input/new-datasets/y_test_new.csv")

y_train_new = pd.read_csv("../input/new-datasets/y_train_new.csv")
X_train_new.head()
X_train_new.describe()
y_train_new.surface.value_counts().plot(kind = 'bar')
import matplotlib.pyplot as plt

plt.figure(figsize=(26, 16))

for i, col in enumerate(X_train_new.columns[4:]):

    plt.subplot(3, 4, i + 1)

    plt.plot(X_train_new.loc[X_train_new['series_id'] == 0, col])

    plt.title(col)
columns=['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

def feature_data(X):

    new_data=pd.DataFrame()

    for col in columns:

        new_data[col+'_mean'] = X.groupby(['series_id'])[col].mean()

        new_data[col+'_median'] = X.groupby(['series_id'])[col].median()

        new_data[col+'_max'] = X.groupby(['series_id'])[col].max()

        new_data[col+'_min'] = X.groupby(['series_id'])[col].min()

        new_data[col + '_abs_max'] = X.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        new_data[col + '_abs_min'] = X.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        new_data[col + '_abs_avg'] = (new_data[col + '_abs_min'] + new_data[col + '_abs_max'])/2

        new_data[col+'_var'] = X.groupby(['series_id'])[col].var()

        new_data[col+'_std'] = X.groupby(['series_id'])[col].std()

        new_data[col + '_maxtoMin'] = new_data[col + '_max'] / new_data[col + '_min']

        new_data[col + '_range'] = new_data[col + '_max'] - new_data[col + '_min']

        new_data[col + '_mean_abs_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        new_data[col + '_abs_median_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.median(np.abs(np.diff(x))))

        new_data[col + '_abs_std_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.std(np.abs(x)))

        

    return new_data
X_train_new_2 = feature_data(X_train_new)

X_train_new_2.head()
X_test_new_2 = feature_data(X_test_new)

X_test_new_2.head()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import warnings  

warnings.filterwarnings('ignore')



lr = LogisticRegression(multi_class='multinomial')

lr.fit(X_train_new_2, y_train_new['surface'])

y_pred = lr.predict(X_test_new_2)

accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred)

print(accuracy)
# Combine X_train_new_2 and X_test_new_2, as well as y_train_new and y_test_new 

# in order to create X and y dataframes for performing k-fold cross validation



X_data = [X_train_new_2, X_test_new_2]

y_data = [y_train_new, y_test_new]



X = np.asarray(pd.concat(X_data))

y = np.asarray(pd.concat(y_data))
from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn import preprocessing

import statistics



def k_fold_cross_validation_logistic(k, X, y):

    kf = KFold(n_splits=k, random_state=21, shuffle=True)

    avg_accuracy = 0

    accuracies = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # scaling the data matrix:

        X_train = preprocessing.scale(X_train)

        X_test = preprocessing.scale(X_test)

        

        # Make prediction and determine accuracy for this fold:

        lr = LogisticRegression(multi_class='multinomial')

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)

        avg_accuracy += accuracy

        accuracies.append(accuracy)



    avg_accuracy = avg_accuracy / k

    stdev = statistics.stdev(accuracies)

    return  avg_accuracy, stdev



avg_accuracy, stdev = k_fold_cross_validation_logistic(10, X, y[:,3])

print("Average 10-fold CV accuracy = {:10.2f}, standard deviation = {:10.2f}".format(avg_accuracy, stdev))
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier()



tree.fit(X_train_new_2, y_train_new['surface'])

y_pred_test_tree = tree.predict(X_test_new_2) 

accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred_test_tree)

print(accuracy)
def k_fold_cross_validation_decisiontree(k, X, y):

    kf = KFold(n_splits=k, random_state=21, shuffle=True)

    avg_accuracy = 0

    accuracies = []

    

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # scaling the data matrix:

        X_train = preprocessing.scale(X_train)

        X_test = preprocessing.scale(X_test)

        

        # Make prediction and determine accuracy for this fold:

        tree = DecisionTreeClassifier()

        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_test) 

        accuracy = metrics.accuracy_score(y_test, y_pred)

        avg_accuracy += accuracy

        accuracies.append(accuracy)

        

    avg_accuracy = avg_accuracy / k

    stdev = statistics.stdev(accuracies)

    return  avg_accuracy, stdev





avg_accuracy, stdev = k_fold_cross_validation_decisiontree(10, X, y[:,3])

print("Average 10-fold CV accuracy = {:10.2f}, standard deviation = {:10.2f}".format(avg_accuracy, stdev))
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=600)



forest.fit(X_train_new_2, y_train_new['surface'])

y_pred_test_forest = forest.predict(X_test_new_2) 

accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred_test_forest)

print(accuracy)
def k_fold_cross_validation_randomforest(k, X, y):

    kf = KFold(n_splits=k, random_state=21, shuffle=True)

    avg_accuracy = 0

    accuracies = []

    

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # scaling the data matrix:

        X_train = preprocessing.scale(X_train)

        X_test = preprocessing.scale(X_test)

        

        # Make prediction and determine accuracy for this fold:

        forest = RandomForestClassifier(n_estimators=600)

        forest.fit(X_train, y_train)

        y_pred = forest.predict(X_test) 

        accuracy = metrics.accuracy_score(y_test, y_pred)

        avg_accuracy += accuracy

        accuracies.append(accuracy)

        

    avg_accuracy = avg_accuracy / k

    stdev = statistics.stdev(accuracies)

    return  avg_accuracy, stdev





avg_accuracy, stdev = k_fold_cross_validation_randomforest(10, X, y[:,3])

print("Average 10-fold CV accuracy = {:10.2f}, standard deviation = {:10.2f}".format(avg_accuracy, stdev))
from sklearn.svm import SVC



svm = SVC(decision_function_shape='ovo')

svm.fit(X_train_new_2, y_train_new['surface'])

y_pred_test_svm = svm.predict(X_test_new_2) 

accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred_test_svm)

print(accuracy)
def k_fold_cross_validation_svm(k, X, y):

    kf = KFold(n_splits=k, random_state=21, shuffle=True)

    avg_accuracy = 0

    accuracies = []

    

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # scaling the data matrix:

        X_train = preprocessing.scale(X_train)

        X_test = preprocessing.scale(X_test)

        

        # Make prediction and determine accuracy for this fold:

        svm = SVC(decision_function_shape='ovo')

        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)  

        accuracy = metrics.accuracy_score(y_test, y_pred)

        avg_accuracy += accuracy

        accuracies.append(accuracy)

        

    avg_accuracy = avg_accuracy / k

    stdev = statistics.stdev(accuracies)

    return  avg_accuracy, stdev





avg_accuracy, stdev = k_fold_cross_validation_svm(10, X, y[:,3])

print("Average 10-fold CV accuracy = {:10.2f}, standard deviation = {:10.2f}".format(avg_accuracy, stdev))
from sklearn.neighbors import KNeighborsClassifier



knn_accuracies = []

for i in range(1, 100):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train_new_2, y_train_new['surface'])

    y_pred_test_knn = knn.predict(X_test_new_2) 

    accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred_test_knn)

    knn_accuracies.append(accuracy)



print("Maximum accuracy:", max(knn_accuracies), "obtained at k =", knn_accuracies.index(max(knn_accuracies)))



def k_fold_cross_validation_knn(k, X, y):

    kf = KFold(n_splits=k, random_state=21, shuffle=True)

    avg_accuracy = 0

    accuracies = []

    

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # scaling the data matrix:

        X_train = preprocessing.scale(X_train)

        X_test = preprocessing.scale(X_test)

        

        # Make prediction and determine accuracy for this fold:

        knn = KNeighborsClassifier(n_neighbors=20)

        knn.fit(X_train, y_train['surface'])

        y_pred = knn.predict(X_test) 

        accuracy = metrics.accuracy_score(y_test['surface'], y_pred)        

        avg_accuracy += accuracy

        accuracies.append(accuracy)

        

    avg_accuracy = avg_accuracy / k

    stdev = statistics.stdev(accuracies)

    return  avg_accuracy, stdev





avg_accuracy, stdev = k_fold_cross_validation_svm(10, X, y[:,3])

print("Average 10-fold CV accuracy = {:10.2f}, standard deviation = {:10.2f}".format(avg_accuracy, stdev))
y_pred = pd.DataFrame(y_pred_test_tree)

y_pred
accuracy = metrics.accuracy_score(y_test_new['surface'], y_pred_test_forest)

print("Accuracy = ", accuracy)