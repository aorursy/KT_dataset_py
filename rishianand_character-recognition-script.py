import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder



%matplotlib inline





df = pd.read_csv('../input/data.csv')



df['character_class'] = LabelEncoder().fit_transform(df.character)

df.drop('character', axis=1, inplace=True)

df = df.astype(np.uint8)
from sklearn.linear_model import RidgeClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier



from sklearn.model_selection import cross_validate





df_sample = df.sample(frac=0.1, random_state=0)



names = ['RidgeClassifier', 'BernoulliNB', 'GaussianNB', 'ExtraTreeClassifier', 'DecisionTreeClassifier',

         'NearestCentroid', 'KNeighborsClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier']

classifiers = [RidgeClassifier(), BernoulliNB(), GaussianNB(), ExtraTreeClassifier(), DecisionTreeClassifier(),

                NearestCentroid(), KNeighborsClassifier(), ExtraTreesClassifier(), RandomForestClassifier()]

test_scores, train_scores, fit_time, score_time = [], [], [], []



for clf in classifiers:

    scores = cross_validate(clf, df_sample.iloc[:, :-1], df_sample.character_class)

    test_scores.append(scores['test_score'].mean())

    train_scores.append(scores['train_score'].mean())

    fit_time.append(scores['fit_time'].mean())

    score_time.append(scores['score_time'].mean())



pd.DataFrame({'Classifier': names,

              'Test_Score': test_scores,

              'Train_Score': train_scores,

              'Fit_Time': fit_time,

              'Score_Time': score_time})
from sklearn.model_selection import GridSearchCV





# K Nearest Neighbors

parameters = {'n_neighbors': np.arange(1, 25, 4)}

clf = GridSearchCV(KNeighborsClassifier(), parameters)



clf.fit(df_sample.iloc[:, :-1], df_sample.iloc[:, -1])

result = pd.DataFrame.from_dict(clf.cv_results_)



x, y = clf.best_params_['n_neighbors'], clf.best_score_

text = 'N Neighbors = {}, Score = {}'.format(x, y)



plt.figure()

plt.title('K Nearest Neighbors')

plt.xlabel('No. of Neighbors')

plt.ylabel('Accuracy Score')

plt.yticks(np.arange(0.6, 0.81, 0.02))



plt.plot(result.param_n_neighbors, result.mean_test_score, label='Mean Accuracy Score')

plt.plot(x, y, 'o', label=text)



plt.legend()

plt.show()
# Extremely Randomized Trees

parameters = {'n_estimators': np.arange(20, 310, 20)}

clf = GridSearchCV(ExtraTreesClassifier(), parameters)





clf.fit(df_sample.iloc[:, :-1], df_sample.iloc[:, -1])

result = pd.DataFrame.from_dict(clf.cv_results_)



x, y = clf.best_params_['n_estimators'], clf.best_score_

text = 'No. of Trees = {}, Score = {}'.format(x, y)



plt.figure()

plt.title('Extremely Randomized Trees Classification')

plt.xlabel('No. of Trees')

plt.ylabel('Accuracy Score')

plt.yticks(np.arange(0.6, 0.81, 0.02))



plt.plot(result.param_n_estimators, result.mean_test_score, label='Mean Accuracy Score')

plt.plot(x, y, 'o', label=text)



plt.legend()

plt.show()
# Random Forests

parameters = {'n_estimators': np.arange(20, 310, 20)}

clf = GridSearchCV(RandomForestClassifier(), parameters)



clf.fit(df_sample.iloc[:, :-1], df_sample.iloc[:, -1])

result = pd.DataFrame.from_dict(clf.cv_results_)



x, y = clf.best_params_['n_estimators'], clf.best_score_

text = 'No. of Trees = {}, Score = {}'.format(x, y)



plt.figure()

plt.title('Random Forests Classification')

plt.xlabel('No. of Trees')

plt.ylabel('Accuracy Score')

plt.yticks(np.arange(0.6, 0.81, 0.02))



plt.plot(result.param_n_estimators, result.mean_test_score, label='Mean Accuracy Score')

plt.plot(x, y, 'o', label=text)



plt.legend()

plt.show()
from sklearn.model_selection import learning_curve





df = df.sample(frac=1, random_state=0)



clf = ExtraTreesClassifier(n_estimators=256)



train_sizes, train_scores, test_scores = learning_curve(clf, df.iloc[:, :-1], df.character_class)

train_scores_mean = np.mean(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)



plt.figure()

plt.title('Learning Curve for Extra Trees Classification')

plt.xlabel('Training examples')

plt.ylabel('Score')



plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')

plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')



plt.legend()

plt.show()
clf = ExtraTreesClassifier(n_estimators=256)



scores = cross_validate(clf, df.iloc[:, :-1], df.iloc[:, -1])

print('Mean Accuracy Score:', scores['test_score'].mean())