import pandas as pd

import numpy as np

import seaborn as sns



import sklearn

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.model_selection import cross_val_score

from sklearn import preprocessing



%matplotlib inline
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = "?")

adult.shape
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")

testAdult.shape
adult.head()
numeric = adult.describe(include = [np.number])

numericColumns = numeric.columns[1:]



numeric
categoric = adult.describe(exclude = [np.number])

categoricColumns = categoric.columns



categoric
nadult = adult.dropna()

nadult.shape
categoricAdult = adult[categoricColumns].apply(pd.Categorical)



for col in categoricColumns:

    adult[col + "_cat"] = categoricAdult[col].cat.codes
categoricTestAdult = testAdult[categoricColumns[:-1]].apply(pd.Categorical)



for col in categoricColumns[:-1]:

    testAdult[col + "_cat"] = categoricTestAdult[col].cat.codes
sns.pairplot(adult, vars = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 

                            'sex_cat', 'occupation_cat'], 

             hue = 'income')
sns.heatmap(adult.loc[:, [*numericColumns, 'income_cat']].corr().round(2), vmin = -1., vmax = 1., cmap = plt.cm.RdYlGn_r, annot = True)
fig, axes = plt.subplots(nrows = 3, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



adult.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0, 0], figsize = (20, 15))



relationship = adult.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = adult.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True, ax = axes[0, 1])



education = adult.groupby(['education', 'income']).size().unstack()

education['sum'] = adult.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True, ax = axes[1, 0])



occupation = adult.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = adult.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True, ax = axes[1, 1])



workclass = adult.groupby(['workclass', 'income']).size().unstack()

workclass['sum'] = adult.groupby('workclass').size()

workclass = workclass.sort_values('sum', ascending = False)[['<=50K', '>50K']]

workclass.plot(kind = 'bar', stacked = True, ax = axes[2, 0])



race = adult.groupby(['race', 'income']).size().unstack()

race['sum'] = adult.groupby('race').size()

race = race.sort_values('sum', ascending = False)[['<=50K', '>50K']]

race.plot(kind = 'bar', stacked = True, ax = axes[2, 1])
adult['native.country'].value_counts()
relevantNumericColumns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

relevantCategoricColumns = ['marital.status', 'occupation', 'relationship', 'race', 'sex']



Xadult = adult[relevantNumericColumns + relevantCategoricColumns]

numXadult = adult[relevantNumericColumns + list(map(lambda x: x + "_cat", relevantCategoricColumns))]



testXAdult = testAdult[relevantNumericColumns + relevantCategoricColumns]

testNumXadult = testAdult[relevantNumericColumns + list(map(lambda x: x + "_cat", relevantCategoricColumns))]



Yadult = adult.income
classifiers = {}

predictions = {}

scores = {}
%%time



scores['KNN'] = 0.0



for k in range(30, 35):

    knn = KNeighborsClassifier(k, metric = 'manhattan')

    score = np.mean(cross_val_score(knn, numXadult, Yadult, cv = 10))

    

    if score > scores['KNN']:

        bestK = k

        scores['KNN'] = score

        classifiers['KNN'] = knn



        

classifiers['KNN'].fit(numXadult, Yadult)

        

print("Best acc: {}, K = {}".format(scores['KNN'], bestK))
%%time



predictions['KNN'] = classifiers['KNN'].predict(testNumXadult)
%%time



classifiers['RandomForest'] = RandomForestClassifier(n_estimators = 750, max_depth = 12)



scores['RandomForest'] = np.mean(cross_val_score(classifiers['RandomForest'], numXadult, Yadult, cv = 10))



classifiers['RandomForest'].fit(numXadult, Yadult)



scores['RandomForest']
%%time



predictions['RandomForest'] = classifiers['RandomForest'].predict(testNumXadult)
%%time



classifiers['AdaBoost'] = AdaBoostClassifier(n_estimators = 500)



scores['AdaBoost'] = np.mean(cross_val_score(classifiers['AdaBoost'], numXadult, Yadult, cv = 10))



classifiers['AdaBoost'].fit(numXadult, Yadult)



scores['AdaBoost']
%%time



predictions['AdaBoost'] = classifiers['AdaBoost'].predict(testNumXadult)
bestScore = 0.0



for c in classifiers:

    print("Acurácia do classificador " + c + ": {}".format(scores[c]))

    

    if scores[c] > bestScore:

        bestScore = scores[c]

        bestClassifier = c

        

print("Classificador escolhido: " + bestClassifier)
id_index = pd.DataFrame({'Id' : list(range(len(predictions[bestClassifier])))})

income = pd.DataFrame({'income' : predictions[bestClassifier]})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')