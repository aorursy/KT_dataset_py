import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn
data = pd.read_csv('../input/Wine.csv')
data.head(3)
data.columns = ['class','alcohol','malicAcid','ash','ashalcalinity','magnesium','totalPhenols','flavanoids','nonFlavanoidPhenols','proanthocyanins','colorIntensity','hue','od280_od315','proline']
data.head(3)
print('There are %d missing values in total.' % data.isna().sum().sum())
sn.countplot(data['class'], palette='Blues_d');
corr = data.corr()

fig, ax = plt.subplots(figsize=(10,10))

sn.heatmap(corr,ax=ax, cmap=sn.diverging_palette(20, 220, n=200), square=True, annot=True, cbar_kws={'shrink': .8})

ax.set_xticklabels(data.columns, rotation=45, horizontalalignment='right');
X = data.drop(['class'], axis=1)

Y = data['class']
from sklearn.model_selection import train_test_split
random_state = 2

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state, shuffle=True)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold
estimator = LogisticRegression(solver='liblinear', multi_class='auto')

selector = RFECV(estimator, step=1, cv = StratifiedKFold(10));

selector.fit(X, Y);
plt.figure()

plt.xlabel('Number of Features')

plt.ylabel('Cross Validation Score')

grid_scores = plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_, zorder = 3);

best_number = plt.scatter(selector.n_features_, np.max(selector.grid_scores_), color='red', zorder = 5);

plt.legend([best_number],['Optimal Number of Features'], loc='lower right');
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
classifiers = []

classifiers.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='auto')))

classifiers.append(('Support Vector Classifier', SVC(kernel='linear')))

classifiers.append(('GaussianNB', GaussianNB()))

classifiers.append(('K-Nearest Neighbors',KNeighborsClassifier(n_neighbors=3)))

classifiers.append(('Decision Tree', DecisionTreeClassifier()))

classifiers.append(('Multi-Layer Perceptron', MLPClassifier(hidden_layer_sizes=(15),solver='sgd',learning_rate_init=0.01,max_iter=500)))

classifiers.append(('eXtreme Gradient Boosting', XGBClassifier()))
from sklearn.model_selection import StratifiedKFold, cross_val_score
kfold = StratifiedKFold(n_splits=10, random_state=random_state)

cv_results = []

for name, classifier in classifiers:

    result = cross_val_score(classifier, X, Y, cv=kfold);

    cv_results.append((name, result));
results = pd.DataFrame(cv_results, columns=['classifier','cvscore'])

results['cvscore'] = [np.mean(i) for i in results['cvscore']]
sn.set_style('whitegrid')

ax = sn.barplot(x='cvscore',y='classifier', data=results.sort_values('cvscore'), palette='Blues_d')

ax.set(xlabel='Cross Validation Score', ylabel='');
print('The best performing model is: %s\nWith Cross-Validation Score of: %.2f' % (results.iloc[results['cvscore'].idxmax()][0], results.iloc[results['cvscore'].idxmax()][1]))
estimator = GaussianNB()

estimator.fit(X_train, Y_train)

Y_predict = estimator.predict(X_test)
from sklearn.metrics import accuracy_score

print('Prediction accuracy is: %.2f' % (100*accuracy_score(Y_predict, Y_test)))
X_test[Y_predict != Y_test]