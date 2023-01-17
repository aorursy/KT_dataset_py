import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import preprocessing
adult=pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",index_col=['Id'], na_values="?")

test_adult= pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",index_col=['Id'], na_values="?")
adult = adult.apply(lambda x:x.fillna(x.value_counts().index[0]))

test_adult=test_adult.apply(lambda x:x.fillna(x.value_counts().index[0]))
adult.columns

adult.head()

numeric = adult.describe(include = [np.number])

numericColumns = numeric.columns[1:]



numeric
categoric = adult.describe(exclude = [np.number])

categoricColumns = categoric.columns



categoric

    
numAdult = adult.apply(preprocessing.LabelEncoder().fit_transform)

test_adult=test_adult.apply(preprocessing.LabelEncoder().fit_transform)
plt.figure(figsize=(10,8))

adultNumTarget = numAdult.copy()





sns.heatmap(adultNumTarget.dropna().corr().round(2), vmin=-1, vmax=1, annot=True, cmap='viridis')

plt.show()
#Definition of target values and data columns

Yadult = numAdult.income

Xadult = numAdult[['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week','marital.status', 'occupation', 'relationship', 'race', 'sex']]
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



bestK=-1

bestMean=-1





for i in range(20,36):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xadult, Yadult, cv=10, n_jobs=-1)

    mean = np.mean(scores)

    print(i,":",mean)

    if(mean > bestMean):

        bestMean = mean

        bestK = i

        

print("Melhor opção: K = {} , score= {},".format(bestK,bestMean) )
knn = KNeighborsClassifier(n_neighbors=bestK)

knn.fit(Xadult,Yadult)
Xtest_adult = test_adult[['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week','marital.status', 'occupation', 'relationship', 'race', 'sex']]
YtestPred = knn.predict(Xtest_adult)
matches = np.where(YtestPred==0,"<=50K",">50K")

final = dict(enumerate(x.rstrip() for x in matches))

result = pd.DataFrame(final.items(), columns=['Id', 'income'])

result.to_csv ('submission.csv', index = False, header=True)

result.head()