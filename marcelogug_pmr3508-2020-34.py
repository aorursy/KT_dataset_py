import numpy as np

import seaborn as sns

import pandas as pd



import sklearn

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.model_selection import cross_val_score

from sklearn import preprocessing



%matplotlib inline
adult_train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = "?")

adult_train.shape

adult_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")

adult_test.shape
adult_train.head()
adult_train.describe()
adult_train.describe(exclude = [np.number])
n_adult = adult_train.dropna()

n_adult.shape
cat = adult_train.describe(exclude = [np.number]).columns



categoricAdult = adult_train[cat].apply(pd.Categorical)



for col in cat:

    adult_train[col + "_cat"] = categoricAdult[col].cat.codes

categoricTestAdult = adult_test[cat[:-1]].apply(pd.Categorical)



for col in cat[:-1]:

    adult_test[col + "_cat"] = categoricTestAdult[col].cat.codes
adult_train.head()
sns.pairplot(adult_train, vars=["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", 

                          "hours.per.week"], hue="income", diag_kws={'bw':"1.0"}, corner=True)

plt.show()
adult_train["native.country"].value_counts().plot(kind="pie", figsize = (8,8))

plt.show()
adult_copy = adult_train.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

adult_copy["income"] = le.fit_transform(adult_copy['income'])



#heat map:

plt.figure(figsize=(10,10))

mask = np.zeros_like(adult_copy.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(adult_copy.corr(), square=True, vmin=-1, vmax=1, annot = True, linewidths=.5, mask=mask)

plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



adult_train.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0, 0], figsize = (20, 15))



relationship = adult_train.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = adult_train.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True, ax = axes[0, 1])



education = adult_train.groupby(['education', 'income']).size().unstack()

education['sum'] = adult_train.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True, ax = axes[1, 0])



occupation = adult_train.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = adult_train.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True, ax = axes[1, 1])



princ_num_colum= ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

princ_cat_colum= ['occupation', 'relationship', 'sex','education']
X_train = adult_train[princ_num_colum + princ_cat_colum]

numX_train = adult_train[princ_num_colum + list(map(lambda x: x + "_cat", princ_cat_colum))]



X_test = adult_test[princ_num_colum + princ_cat_colum]

numXadul_test = adult_test[princ_num_colum + list(map(lambda x: x + "_cat", princ_cat_colum))]



Yadult = adult_train.income
classifiers = {}

scores = 0.0





for k in range(30, 35):

    knn = KNeighborsClassifier(k, metric = 'manhattan')

    score = np.mean(cross_val_score(knn, numX_train, Yadult, cv = 10))

    

    if score > scores:

        bestK = k

        scores = score

        classifiers['KNN'] = knn



        

classifiers['KNN'].fit(numX_train, Yadult)

        

print("Best acc: {}, K = {}".format(scores, bestK))
%%time



predictions = classifiers['KNN'].predict(numXadul_test)
id_index = pd.DataFrame({'Id' : list(range(len(predictions)))})

income = pd.DataFrame({'income' : predictions})

result = income
result.to_csv("submission.csv", index = True, index_label = 'Id')