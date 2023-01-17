import pandas as pd

import numpy as np



train_df = pd.read_csv('../input/train.csv')

train_df.head()
train_df.shape
X, y = train_df.iloc[:, 0:len(train_df.columns) - 1], train_df.iloc[:, -1]
test_df = pd.read_csv('../input/test.csv')

X_test, y_test = test_df.iloc[:, 0:len(test_df.columns) -1], test_df.iloc[:, -1]
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from sklearn.metrics import accuracy_score # for evaluation



classifiers = [

    DecisionTreeClassifier(),

    KNeighborsClassifier(7), # because there are 6 different labels

    SVC(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]



names = []

scores = []



for clf in classifiers:

    clf = clf.fit(X, y)

    y_pred = clf.predict(X_test)

    

    names.append(clf.__class__.__name__)

    scores.append(accuracy_score(y_pred, y_test))



score_df = pd.DataFrame({'Model': names, 'Score': scores}).set_index('Model')

score_df
import matplotlib.pyplot as plt

%matplotlib inline



ax = score_df.plot.bar()

ax.set_xticklabels(score_df.index, rotation=45, fontsize=10)
from sklearn.model_selection import GridSearchCV



parameters = {

    'kernel': ['linear', 'rbf'],

    'C': [100, 20, 1, 0.1]

}



selector = GridSearchCV(SVC(), parameters, scoring='accuracy') # we only care about accuracy here

selector.fit(X, y)



print('Best parameter set found:')

print(selector.best_params_)

print('Detailed grid scores:')

means = selector.cv_results_['mean_test_score']

stds = selector.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, selector.cv_results_['params']):

    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

    print()
clf = SVC(kernel='linear', C=1).fit(X, y)

y_pred = clf.predict(X_test)

print('Accuracy score:', accuracy_score(y_test, y_pred))