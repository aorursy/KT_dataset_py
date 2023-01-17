import pandas as pd

import numpy as np

import os 

# Get some classifiers to evaluate

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.svm import SVC

# score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
print(os.listdir('../input'))

#read in the dataset

df = pd.read_csv('../input/diabetes_data.csv')



#take a look at the data

df.head()

#check dataset size

df.shape

#split data into inputs and targets

X = df.drop(columns = ['diabetes'])

y = df['diabetes']

print(y.shape)

print("first 10 labels")

print(y[:10])

#split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
seed = 1075

np.random.seed(seed)



# Create classifiers

rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_split=10)

et = ExtraTreesClassifier(n_estimators=150, max_depth=4, min_samples_split=10)

knn = KNeighborsClassifier()

svc = SVC(gamma='scale')

rg = RidgeClassifier()

gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_split=8)



clf_array = [rf, et, knn, svc, rg, gb]



for clf in clf_array:

    vanilla_scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)

    bagging_clf = BaggingClassifier(clf, max_samples=0.7, random_state=seed)

    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10, 

       n_jobs=-1)

    

    bag_model=BaggingClassifier(clf,bootstrap=True)

    bag_model=bag_model.fit(X_train,y_train)

    ytest_pred=bag_model.predict(X_test)

    print("score on test data:", accuracy_score(ytest_pred, y_test))

    

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [{0}]".format(clf.__class__.__name__, 

                                                              vanilla_scores.mean(), vanilla_scores.std()))

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [Bagging {0}]\n".format(clf.__class__.__name__, 

                                                                        bagging_scores.mean(), bagging_scores.std()))
# Example of hard voting 

from sklearn.ensemble import VotingClassifier

clf = [rf, et, knn, svc, rg, gb]

eclf = VotingClassifier(estimators=[('Random Forests', rf), ('Extra Trees', et), ('KNeighbors', knn), ('SVC', svc), ('Ridge Classifier', rg)], voting='hard')

for clf, label in zip([rf, et, knn, svc, rg, gb, eclf], ['Random Forest', 'Extra Trees', 'KNeighbors', 'SVC', 'Ridge Classifier', 'GradientBoosting', 'Ensemble']):

    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# Set up ensemble voting for bagging

ebclf_array = []



for clf in clf_array:

    ebclf_array.append(BaggingClassifier(clf, max_samples=0.7, random_state=seed))

for clf, label in (zip(ebclf_array, ['Bagging Random Forest', 'Bagging Extra Trees', 'Bagging KNeighbors',

                              'Bagging SVC', 'BaggingRidge Classifier', 'GradientBoostingBagged'])):

    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy',error_score='raise')

    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))

## Set up voting

v_eclf = VotingClassifier(estimators=[('Bagging Random Forest', ebclf_array[0]), ('Bagging Extra Trees', ebclf_array[1]), 

                                    ('Bagging KNeighbors', ebclf_array[2]), ('Bagging SVC', ebclf_array[3]), ('Bagging Ridge Classifier', ebclf_array[4])], voting='hard')

scores = cross_val_score(v_eclf, X, y, cv=10, scoring='accuracy',error_score='raise')

print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), 'Bagging Ensemble'))
from mlxtend.classifier import EnsembleVoteClassifier

import warnings

from xgboost import plot_importance

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier



warnings.filterwarnings('ignore')



# Create boosting classifiers

ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier()

xgb_boost = XGBClassifier()



boost_array = [ada_boost, grad_boost, xgb_boost, gb]



eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost, gb], voting='hard')



labels = ['Ada Boost', 'Grad Boost', 'XG Boost', 'Ensemble', 'Gradient Boosting']



for clf, label in zip([ada_boost, grad_boost, xgb_boost, eclf,], labels):

    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))
from mlens.ensemble import SuperLearner

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



lr = LogisticRegression()



seed = 1075



ensemble = SuperLearner(scorer = accuracy_score, 

                        random_state=seed, 

                        folds=10,

                        verbose = 2)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)



# Build the first layer

ensemble.add([rf, et, knn, rg])

# Attach the final meta estimator

ensemble.add_meta(lr)



ensemble.fit(X_train, y_train)

preds = ensemble.predict(X_test)

print("Fit data:\n%r" % ensemble.data)

print("Accuracy score:", accuracy_score(preds, y_test))
from itertools import combinations



names = ['Random Forest', 'Extra Trees', 'KNeighbors', 'SVC', 'Ridge Classifier']



def zip_stacked_classifiers(*args):

    to_zip = []

    for arg in args:

        temp_list = []

        for i in range(len(arg) + 1):

            temp = list(map(list, combinations(arg, i)))

            temp_list.append(temp)

        combined_items = sum(temp_list, [])

#         print(map(list(combinations(arg, 2))))

#         print (len(combined_items),combined_items)

#         combined_items = sum([map(list(), combinations(arg, i)) for i in range(len(arg) + 1)], [])

        combined_items = filter(lambda x: len(x) > 0, combined_items)

#         print (list(combined_items))

        to_zip.append(combined_items) 

#     print("to_zip[0]",list(to_zip[0]))

#     print("to_zip[1]",list(to_zip[1]))

    return zip(to_zip[0], to_zip[1])



stacked_clf_list = zip_stacked_classifiers(clf_array, names)

# for clf in stacked_clf_list:

#     print("clf", clf[1])

best_combination = [0.00, ""]



for clf in stacked_clf_list:

    

    ensemble = SuperLearner(scorer = accuracy_score, 

                            random_state = seed, 

                            folds = 10)

    ensemble.add(clf[0])

    ensemble.add_meta(lr)

    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)

    accuracy = accuracy_score(preds, y_test)

    

    if accuracy > best_combination[0]:

        best_combination[0] = accuracy

        best_combination[1] = clf[1]

    

    print("Accuracy score: ", accuracy, clf[1])



print("\nBest stacking model is {} with accuracy of: ",best_combination[1], best_combination[0])