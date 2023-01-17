import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?",index_col=['Id'])
adult.head()
adult.shape
adult.describe()
missing = adult.isnull().sum().sort_values(ascending = False)

missing.head()
adult['occupation'].value_counts()
adult['workclass'].value_counts()
adult['native.country'].value_counts()
adult["workclass"] = adult["workclass"].fillna('Private')

adult["occupation"] = adult["occupation"].fillna('Prof-specialty')

adult["native.country"] = adult["native.country"].fillna('United-States')
nadult = adult

nadult.drop_duplicates(inplace=True)

nadult.shape
for i in range (1,17):

    adult_ed = adult[adult['education.num']==i]

    adult_ed['education'].unique()

    print('education.num {0} = {1}'.format(i,adult_ed['education'].unique()[0]))
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Yadult = nadult.income
def Cross(folds, k_inf, k_sup, Xadult, Yadult):

    print('Cross Validation com {:d} folds:\n'.format(folds))

    for k in range (k_inf,k_sup+1):

        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

        scores = cross_val_score(knn, Xadult, Yadult, cv=folds)

        mean = scores.mean()*100

        std = scores.std()*100

        print('k={:d} --> Acurácia = {:.2f}% +/- {:.2f}%'.format(k,mean,std))
Cross(10,1,30,Xadult,Yadult)
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
plt.figure(figsize=(15,15))



sns.heatmap(numAdult.corr(), mask=np.triu(np.ones_like(numAdult.corr(), dtype=np.bool)), square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')

plt.show()
Xadult = numAdult.iloc[:,0:14]

Yadult = numAdult.income

Xadult.head()
Cross(10,1,30,Xadult,Yadult)
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult = numAdult.drop(['education','fnlwgt','hours.per.week'], axis=1)
Xadult = numAdult.iloc[:,0:11]

Yadult = numAdult.income

Xadult.head()
Cross(10,15,30,Xadult,Yadult)
nadult['native.country'].value_counts()
def boole(value, x):

    for v in x:

        if v == value:

            return 1

    return 0
country = ['United-States']

american = pd.DataFrame({'american': nadult['native.country'].apply(boole, args = [country])})
nadult2 = pd.concat([american, nadult], axis = 1)

nadult2.head()
numAdult2 = nadult2.apply(preprocessing.LabelEncoder().fit_transform)

numAdult2 = numAdult2.drop(['education','fnlwgt','native.country','hours.per.week'], axis=1)
Xadult = numAdult2.iloc[:,0:11]

Yadult = numAdult2.income

Xadult.head()
Cross(10,15,30,Xadult,Yadult)
knn = KNeighborsClassifier(n_neighbors=30,metric='manhattan')

knn.fit(Xadult,Yadult)
test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?",index_col=['Id'])

test.head()
missingTest = test.isnull().sum().sort_values(ascending = False)

missingTest.head()
test["workclass"] = test["workclass"].fillna('Private')

test["occupation"] = test["occupation"].fillna('Prof-specialty')

test["native.country"] = test["native.country"].fillna('United-States')
american = pd.DataFrame({'american': test['native.country'].apply(boole, args = [country])})

test2 = pd.concat([american, test], axis = 1)

test2.head()
Xtest = test2.drop(['education','fnlwgt','hours.per.week','native.country'], axis=1)

Xtest.head()
numTest = Xtest.apply(preprocessing.LabelEncoder().fit_transform)

numTest.head()
Ytest = knn.predict(numTest)
Ytest
Ytest = np.where(Ytest==0, "<=50K", Ytest) 

Ytest = np.where(Ytest=='1', ">50K", Ytest) 
Yfin = {'income': Ytest}

result = pd.DataFrame(Yfin)

result.head()
result.to_csv("result.csv", index=True, index_label='Id')