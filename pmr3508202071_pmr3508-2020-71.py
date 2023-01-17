import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

plt.style.use('seaborn')
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = "?")
adult.shape
adult.head()
testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values = "?")
testAdult.shape
testAdult.head()
total = adult.isnull().sum().sort_values(ascending = False)

percent = ((adult.isnull().sum()/adult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
print('occupation:\n')

print(adult['occupation'].describe())



print('\n\nworkclass:\n')

print(adult['workclass'].describe())



print('\n\nnative.country:\n')

print(adult['native.country'].describe())
value = adult['workclass'].describe().top

adult['workclass'] = adult['workclass'].fillna(value)



value = adult['native.country'].describe().top

adult['native.country'] = adult['native.country'].fillna(value)



value = adult['occupation'].describe().top

adult['occupation'] = adult['occupation'].fillna(value)
total = adult.isnull().sum().sort_values(ascending = False)

percent = ((adult.isnull().sum()/adult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
total = testAdult.isnull().sum().sort_values(ascending = False)

percent = ((testAdult.isnull().sum()/testAdult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
print('occupation:\n')

print(testAdult['occupation'].describe())



print('\n\nworkclass:\n')

print(testAdult['workclass'].describe())



print('\n\nnative.country:\n')

print(testAdult['native.country'].describe())
value = testAdult['workclass'].describe().top

testAdult['workclass'] = testAdult['workclass'].fillna(value)



value = testAdult['native.country'].describe().top

testAdult['native.country'] = testAdult['native.country'].fillna(value)



value = testAdult['occupation'].describe().top

testAdult['occupation'] = testAdult['occupation'].fillna(value)
total = testAdult.isnull().sum().sort_values(ascending = False)

percent = ((testAdult.isnull().sum()/testAdult.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
numeric_columns = ['age', 'fnlwgt', 'education.num','capital.gain','capital.loss','hours.per.week']

categoric_columns = ['workclass', 'education', 'marital.status', 'occupation',

                     'relationship', 'race', 'sex', 'native.country', 'income']
sns.pairplot(adult,hue='income',diag_kws={'bw':'1.0'})

plt.show()
adult.describe()
corr_mat = adult.corr()

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(corr_mat, annot=True, cmap='coolwarm')
def income_by(adult, variable, target_obj):



    targets = adult[target_obj].unique()



    for target in targets:

        plt.hist(adult[adult[target_obj]==target][variable],alpha = 0.7)



    plt.xlabel(variable)

    plt.ylabel('quantity')

    plt.legend(adult[target_obj].unique())

    plt.title('Histogram of '+ variable + ' by ' + target_obj)

    plt.show()
income_by(adult,'age','income')
income_by(adult,'education.num','income')
income_by(adult,'capital.gain','income')
income_by(adult,'capital.loss','income')
income_by(adult,'hours.per.week','income')
adult_less50 = adult[adult['income'] == '<=50K'] 

adult_greater50 = adult[adult['income'] == '>50K']
fig, axes = plt.subplots(nrows=1, ncols=2)

adult_less50['workclass'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K')

adult_greater50['workclass'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K')
fig, axes = plt.subplots(nrows=1, ncols=2)

adult_less50['education'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K')

adult_greater50['education'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K')
fig, axes = plt.subplots(nrows=1, ncols=2)

adult_less50['marital.status'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K')

adult_greater50['marital.status'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K')
fig, axes = plt.subplots(nrows=1, ncols=2)

adult_less50['occupation'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K')

adult_greater50['occupation'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K')
income_by(adult,'relationship','income')
fig, axes = plt.subplots(nrows=1, ncols=3)

adult_less50['sex'].value_counts().plot(kind = 'pie',ax=axes[0],title='<=50K')

adult_greater50['sex'].value_counts().plot(kind = 'pie',ax=axes[1],title='>50K')

adult['sex'].value_counts().plot(kind = 'pie',ax=axes[2],title='Total')
fig, axes = plt.subplots(nrows=1, ncols=2)

adult_less50['race'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K')

adult_greater50['race'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K')
adult["native.country"].value_counts()
income_by(adult,'hours.per.week','sex')
Xadult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = adult.income
XtestAdult = testAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores
scores.mean()
best_n, best_score = 0, 0

new_scores = []



for n in range(1,30):

    knn = KNeighborsClassifier(n_neighbors=n)

    n_score = np.mean(cross_val_score(knn, Xadult, Yadult, cv=10))

    print('KNN:',n,'\t Score:',n_score)

    new_scores.append(n_score)

    if n_score > best_score:

        best_score = n_score

        best_n = n
plt.plot(range(1,30), new_scores)

plt.xlabel("knn")

plt.ylabel("Score")

plt.show()
print('Best k =',best_n)

print('Score =',best_score)
knn = KNeighborsClassifier(n_neighbors = best_n)

knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

YtestPred
submission = pd.DataFrame()
submission[0] = XtestAdult.index

submission[1] = YtestPred

submission.columns = ["Id", "Income"]

submission.head()
submission.to_csv('submission.csv',index = False)