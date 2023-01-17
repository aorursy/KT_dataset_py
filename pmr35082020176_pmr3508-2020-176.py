#importando as libs

import numpy as np

import pandas as pd

import sklearn 

import matplotlib.pyplot as plt

import seaborn as sns
adult = pd.read_csv('../input/adult-pmr3508/train_data.csv', index_col='Id', na_values='?')

test_adult= pd.read_csv('../input/adult-pmr3508/test_data.csv', index_col='Id', na_values='?')



names_train = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

         "Hours per week", "Country", "Income"]

names_test = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

         "Hours per week", "Country"]



adult.columns = names_train

test_adult.columns=names_test
adult.shape
adult.head()
adult.isna().sum()
adult.Workclass.describe()

adult.Occupation.describe()

adult.Country.describe()
moda_workclass = adult.Workclass.describe().top

adult.Workclass =  adult.Workclass.fillna(moda_workclass)



moda_country= adult.Country.describe().top

adult.Country =  adult.Country.fillna(moda_country)



moda_occupation=adult.Occupation.describe().top

adult.Occupation=adult.Occupation.fillna(moda_occupation)



moda_workclass_test= test_adult.Workclass.describe().top

test_adult.Workclass =  test_adult.Workclass.fillna(moda_workclass_test)



moda_country_test = test_adult.Country.describe().top

test_adult.Country =  test_adult.Country.fillna(moda_country_test)



moda_occupation_test=test_adult.Occupation.describe().top

test_adult.Occupation=test_adult.Occupation.fillna(moda_occupation)



nadult=adult.dropna()

ntest_adult=test_adult.dropna()

nadult.shape
nadult.head()



nadult['Workclass'].value_counts().plot(kind='bar')
nadult['Education'].value_counts().plot(kind='bar')
nadult['Marital Status'].value_counts().plot(kind='bar')
nadult['Occupation'].value_counts().plot(kind='bar')
nadult['Relationship'].value_counts().plot(kind='bar')
nadult['Race'].value_counts().plot(kind='bar')
nadult['Sex'].value_counts().plot(kind='bar')
nadult['Country'].value_counts().plot(kind='bar')
fig,axes=plt.subplots(nrows = 4, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



adult.groupby(['Sex', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0, 0], figsize = (20, 15))

adult.groupby(['Relationship', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0,1], figsize = (20, 15))



adult.groupby(['Education', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[1,1], figsize = (20, 15))

adult.groupby(['Occupation', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[1,0], figsize = (20, 15))





adult.groupby(['Workclass', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[2,0], figsize = (20, 15))

adult.groupby(['Race', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[2,1], figsize = (20, 15))



adult.groupby(['Marital Status', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[3, 0], figsize = (20, 15))

adult.groupby(['Country', 'Income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[3, 1], figsize = (20, 15))
sns.pairplot(nadult,  hue='Income', diag_kws={'bw':'1.0'})

plt.tight_layout()

plt.show()
from sklearn import preprocessing

num_adult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

num_test_adult = ntest_adult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = num_adult.iloc[:,0:14]

Xtest_adult = num_test_adult.iloc[:,0:14]
Yadult = nadult.Income
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
best_score=0

best_k=0

min_k=5

max_k=35

for i in range(min_k,max_k,5):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  min_score=min(scores)

  

  if min_score>best_score:

    best_score=min_score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Age','Workclass','Education','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Age','Workclass','Education','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=25

max_k=35

for i in range(min_k,max_k,5):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  min_score=min(scores)

  

  if min_score>best_score:

    best_score=min_score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Marital Status','Age','Education','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Marital Status','Age','Education','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=19

max_k=25

for i in range(min_k,max_k,2):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  score=sum(scores)/len(scores)

  

  if score>best_score:

    best_score=score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Marital Status','Age','Education','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Marital Status','Age','Education','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=15

max_k=20

for i in range(min_k,max_k,2):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  score=sum(scores)/len(scores)

  

  if score>best_score:

    best_score=score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Marital Status','Age','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Marital Status','Age','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=21

max_k=27

for i in range(min_k,max_k,2):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  score=sum(scores)/len(scores)

  

  if score>best_score:

    best_score=score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Marital Status','Age','Education-Num','Occupation','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Marital Status','Age','Education-Num','Occupation','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=25

max_k=30

for i in range(min_k,max_k,2):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  score=sum(scores)/len(scores)

  

  if score>best_score:

    best_score=score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Relationship','Age','Education-Num','Occupation','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Relationship','Age','Education-Num','Occupation','Sex','Capital Gain','Capital Loss','Hours per week']]
best_score=0

best_k=0

min_k=25

max_k=30

for i in range(min_k,max_k,2):

  knn = KNeighborsClassifier(n_neighbors=i)

  knn.fit(Xadult,Yadult)

  scores = cross_val_score(knn, Xadult, Yadult, cv=10)

  score=sum(scores)/len(scores)

  

  if score>best_score:

    best_score=score

    best_k=i

    

print('best score=',best_score,'k used=',best_k)
Xadult=num_adult[['Race','Marital Status','Age','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]

Xtest_adult=num_test_adult[['Race','Marital Status','Age','Education-Num','Occupation','Relationship','Sex','Capital Gain','Capital Loss','Hours per week']]
knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(Xadult,Yadult)

Ytest_predict= knn.predict(Xtest_adult)

print(Ytest_predict)
submission = pd.DataFrame(Ytest_predict)

submission[1] = Ytest_predict

submission[0] = Xtest_adult.index

submission.columns = ['Id','Income']

submission.head()

submission.to_csv('submission.csv',index = False)