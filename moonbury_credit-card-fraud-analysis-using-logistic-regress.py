# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print("The dataset contain 30 features and a resulting column 'Class'.\nAs said, this is a highly unbalanced problem. \nThere are 2 output values (0,1) and only 0.172% of positive value (1).")

data=pd.read_csv('../input/creditcard.csv')



import warnings

warnings.filterwarnings('ignore')
data.describe()
print("Prepare the X and y values. Since it is imbalanced data, use SMOTE to resample the data.")

print("If using SMOTE, we can achieve better recall (90%), otherwise, 54% recall.")



from sklearn.cross_validation import train_test_split

from collections import Counter

y = data['Class']

cols = set(data.columns)

cols.remove('Class')

cols.remove('Time')

cols.remove('Amount')

X = data[list(cols)]



print('Original dataset shape {}'.format(Counter(y)))

X_org = X

y_org = y



from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X, y)

print('Resampled dataset shape {}'.format(Counter(y_res)))
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



print("Apply Standaraization and Normailzation to the data")

centering = StandardScaler(with_mean=True, with_std=False)

X_res = centering.fit_transform(X_res)

scaling = MinMaxScaler(feature_range=(0,1))

X_res = scaling.fit_transform(X_res)



centering = StandardScaler(with_mean=True, with_std=False)

X = centering.fit_transform(X)

scaling = MinMaxScaler(feature_range=(0,1))

X = scaling.fit_transform(X)
print("Split the data into 80% training and 20% testing. I try other ratio but this gives me better result.\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=101)

print("Split the data into 80% training data and 20% testing data.")

print("# of training data:",  X_train.shape[0])

print("# of testing data:",  y_test.shape[0])



X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(X_res, y_res, train_size=0.80, random_state=101)

print("Split the data into 80% training data and 20% resampled testing data.")

print("# of training data:",  X_res_train.shape[0])

print("# of testing data:",  y_res_test.shape[0])
print("prepare the confusion matrix function")



import matplotlib.pyplot as plt

import itertools



def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    #print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import classification_report, accuracy_score,recall_score

from sklearn.metrics import confusion_matrix

import matplotlib.pylab as pylab



print("Train the logistic Regression with the sampled data and see.")

from sklearn.linear_model import LogisticRegression



regr = LogisticRegression()

regr.fit(X_res_train, y_res_train)

y_res_train_pred = regr.predict(X_res_train)

y_res_test_pred = regr.predict(X_res_test)



print("Training data accuracy:", accuracy_score(y_res_train, y_res_train_pred))

print("Training data recall:", recall_score(y_res_train, y_res_train_pred))

print("Testing data accuracy:", accuracy_score(y_res_test, y_res_test_pred))

print("Testing data recall:", recall_score(y_res_test, y_res_test_pred))

print("Classification Report")

print(classification_report(y_res_test, y_res_test_pred))



#print(np.mean(regr.predict_proba(X_test)))



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_res_test, y_res_test_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['True','False'])

plt.show()



print("The result is good on both training and testing data.")
print("Use the same function on original data and see.\n")

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)



print("Training data Accuracy:", accuracy_score(y_train, y_train_pred))

print("Training data recall:", recall_score(y_train, y_train_pred))

print("Testing data Accuracy:", accuracy_score(y_test, y_test_pred))

print("Testing data recall:", recall_score(y_test, y_test_pred))

print("Classification Report")

print(classification_report(y_test, y_test_pred))



from sklearn.metrics import confusion_matrix

import matplotlib.pylab as pylab



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_test_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['True','False'])



plt.show()



print("The result is not bad too on both training and testing data.")
print("next, I will use GridSearchCV to find the best hyperparameters.")
print("Use GridSearchCV to find the best paramters for logistic Regression and test on the data again.")

print("You can see the C value is larger than usual and I will like to give more penalty for false result.")

print("I find that C=100 can raise 1% recall.")



from sklearn.grid_search import GridSearchCV



parameters = {

    'tol': [0.00001, 0.0001, 0.001],

    'C': [1, 50, 100]

}



clfgs = GridSearchCV(LogisticRegression(random_state=101, n_jobs=1),

                     param_grid=parameters,

                     cv=3,

                     n_jobs=1,

                     scoring='recall'

                    )

clfgs.fit(X_res_train, y_res_train)

clf = clfgs.best_estimator_



print(clfgs.best_estimator_)

print("The best classifier score:",clfgs.best_score_)



y_res_train_pred = clf.predict(X_res_train)

y_res_test_pred = clf.predict(X_res_test)



print("Use the best classifer to run the sampled data")

#print("Print the classification Report")

#print(classification_report(y_test, y_test_pred))

print("Training sampled data accuracy:", accuracy_score(y_res_train, y_res_train_pred))

print("Training sampled data recall:", recall_score(y_res_train, y_res_train_pred))

print("Testing sampled data accuracy:", accuracy_score(y_res_test, y_res_test_pred))

print("Testing sampled data recall:", recall_score(y_res_test, y_res_test_pred))



cnf_matrix = confusion_matrix(y_test, y_test_pred)

plot_confusion_matrix(cnf_matrix,classes=['True','False'])

plt.show()



y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)



print("Use the best classifer to run the original data")

#print("Print the classification Report")

#print(classification_report(y_test, y_test_pred))

print("Training data accuracy:", accuracy_score(y_train, y_train_pred))

print("Training data recall:", recall_score(y_train, y_train_pred))

print("Testing data accuracy:", accuracy_score(y_test, y_test_pred))

print("Testing data recall:", recall_score(y_test, y_test_pred))



cnf_matrix = confusion_matrix(y_test, y_test_pred)

plot_confusion_matrix(cnf_matrix,classes=['True','False'])

plt.show()
print("I try couple tol and C and found that the best values are C=100 , tol=0.001")
print("Next, we will try the SGD classifier")



print("Train the SGDClassifier with the sampled data and see.")

from sklearn.linear_model import SGDClassifier



clf = SGDClassifier(random_state=101, n_jobs=1)

clf.fit(X_res_train, y_res_train)

y_res_train_pred = clf.predict(X_res_train)

y_res_test_pred = clf.predict(X_res_test)



print("Training data accuracy:", accuracy_score(y_res_train, y_res_train_pred))

print("Training data recall:", recall_score(y_res_train, y_res_train_pred))

print("Testing data accuracy:", accuracy_score(y_res_test, y_res_test_pred))

print("Testing data recall:", recall_score(y_res_test, y_res_test_pred))

print("Classification Report")

print(classification_report(y_res_test, y_res_test_pred))



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_res_test, y_res_test_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['True','False'])

plt.show()

print("The result for SGDC is worst than Logistic on recall.")
'''

print("Use GridSearchCV to find the best paramters for SGDClassifier")

from sklearn.grid_search import GridSearchCV



parameters = {

    #'loss': ('log','hinge'),

    'epsilon': [0.01,0.001,0.0001],

    'learning_rate': ('optimal','constant'),

    'eta0': [0.01,0.001,0.0001],

}



clfgs = GridSearchCV(SGDClassifier(random_state=101, n_jobs=1),

                     param_grid=parameters,

                     cv=3,

                     n_jobs=1,

                     scoring='recall'

                    )



clfgs.fit(X_res_train, y_res_train)

clf = clfgs.best_estimator_



print(clfgs.best_estimator_)

print("The best classifier score:",clfgs.best_score_)



y_res_test_pred = clf.predict(X_res_test)

y_test_pred = clf.predict(X_test)



print("Use the best classifer to run the test data")

print("Print the classification Report")

print(classification_report(y_test, y_test_pred))



print("Testing data Accuracy:", accuracy_score(y_test, y_test_pred))

print("Testing data recall:", recall_score(y_test, y_test_pred))



cnf_matrix = confusion_matrix(y_test, y_test_pred)

plot_confusion_matrix(cnf_matrix,classes=['True','False'])

plt.show()

'''
print("SGDClassifer is worst than Logistic Regression.\nIn general, I can achieve 90% reacll. Let's read more on how to improve the recall.")
print("let's examine the Time and Amount.")

import matplotlib.pyplot as plt

import plotly.plotly as py

fig, (ax1, ax2) = plt.subplots(2)



ax1.set_title('Amount and Time on Fraud')

ax1.scatter(data['Amount'], data['Class'])

ax2.scatter(data['Time'], data['Class'])

print("It seems that amount is an important feature.")



print(data.shape)

print(data[data['Amount'] < 2000].shape)
from sklearn.metrics import cohen_kappa_score

print("It is said that cohen kappa can give some insign on imbalanced data.")

print("Training data cohen kappa score:", cohen_kappa_score(y_train, y_train_pred))

print("Testing data cohen kappa score:", cohen_kappa_score(y_test, y_test_pred))

print("Resampled training data cohen kappa score:", cohen_kappa_score(y_res_train, y_res_train_pred))

print("Resampled testing data cohen kappa score:", cohen_kappa_score(y_res_test, y_res_test_pred))