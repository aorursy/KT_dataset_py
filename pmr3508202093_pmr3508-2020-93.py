import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import sklearn

import pandas_profiling as pp

import seaborn as sns
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        header=0,

        names=["Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Status",

        "Occupation", "Relationship", "Race", "Sex", "Gain", "Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

del adult["Id"]
adult.head()
# faz profiling da base

pp.ProfileReport(adult)
adult = adult.drop(['Country'], axis=1)



for column in ['Workclass', 'Occupation']:

    adult[column].fillna(adult[column].mode()[0], inplace=True)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

adult_corr = adult.copy()

categorical = list(adult_corr.select_dtypes(exclude=[np.number]).columns.values)

print(categorical)
for column in categorical:

  adult_corr[column] = adult_corr[column].astype('category')



adult_corr.info()
for column in categorical:

  adult_corr[column] = adult_corr[column].cat.codes



adult_corr.head()
adult_corr['Income'] = le.fit_transform(adult_corr['Income'])



adult_corr['Income'] = le.fit_transform(adult_corr['Income'])

mask = np.triu(np.ones_like(adult_corr.corr(), dtype=np.bool))



plt.figure(figsize=(13,13))



sns.heatmap(adult_corr.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap=plt.cm.RdYlGn_r)

plt.show()
adult_corr = adult_corr.drop(['fnlwgt','Workclass','Education','Occupation','Race','Status'], axis=1)

adult_corr.head()
Yadult = adult.Income

Xadult = adult_corr.drop(['Income'], axis=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
scores_mean = []

scores_std = []



k_lim_inf = 1

k_lim_sup = 30



folds = 10



k_max = None

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, metric = 'manhattan')

    

    score = cross_val_score(KNNclf, Xadult, Yadult, cv = folds)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% (best k = {2})'.format(k, max_acc*100, k_max))

print('\nBest k: {}'.format(k_max))
plt.figure(figsize=(15, 7))

plt.errorbar(np.arange(k_lim_inf, k_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, k_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('k')

plt.ylabel('accuracy')

plt.title('KNN performed on several values of k')

plt.axis([0, k_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])
scores_mean = []

scores_std = []



folds_lim_inf = 2

folds_lim_sup = 15



k = k_max



folds = None

max_acc = 0



i = 0

print('Finding best folds...')

for folds in range(folds_lim_inf, folds_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, metric = 'manhattan')

    

    score = cross_val_score(KNNclf, Xadult, Yadult, cv = folds)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        folds_max = folds

        max_acc = scores_mean[i]

    i += 1

    if not (folds%3):

        print('   folds = {0} | Best CV acc = {1:2.2f}% (best folds = {2})'.format(folds, max_acc*100, folds_max))

print('\nBest folds: {}'.format(folds_max))
plt.figure(figsize=(15, 7))

plt.errorbar(np.arange(folds_lim_inf, folds_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, folds_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('folds')

plt.ylabel('accuracy')

plt.title('KNN performed on several values of folds')

plt.axis([0, folds_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])
testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        header = 0,

        names=["Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Status",

        "Occupation", "Relationship", "Race", "Sex", "Gain", "Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



del testAdult["Id"]

testAdult = testAdult.drop(['Country','fnlwgt','Workclass','Education','Occupation','Race','Status'], axis=1)
# faz profiling da base

pp.ProfileReport(testAdult)
for column in [ "Age", "Education-Num",

        "Relationship", "Sex", "Gain", "Loss",

        "Hours per week"]:

    testAdult[column].fillna(testAdult[column].mode()[0], inplace=True)
le = LabelEncoder()



categorical = list(testAdult.select_dtypes(exclude=[np.number]).columns.values)



for column in categorical:

  testAdult[column] = testAdult[column].astype('category')



for column in categorical:

  testAdult[column] = testAdult[column].cat.codes
Xtest = testAdult

KNN = KNeighborsClassifier(n_neighbors=k_max)

KNN.fit(Xadult,Yadult)

Ypred = KNN.predict(Xtest)

Ypred
submission = pd.DataFrame()

submission[0] = Ypred

submission.columns = ['income']

submission
submission.to_csv('submission.csv', index=True, index_label='Id')