import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer
adult_train = pd.read_csv('../input/adult-pmr3508/train_data.csv', na_values='?')

adult_train.set_index('Id',inplace=True)



adult_test = pd.read_csv('../input/adult-pmr3508/test_data.csv', na_values='?')

adult_test.set_index('Id', inplace=True)



newnames = {

    "education.num" : "education_num",

    "marital.status" : "marital_status",

    "capital.gain" : "capital_gain",

    "capital.loss" : "capital_loss",

    "hours.per.week" : "hours_per_week",

    "native.country" : "native_country"

}



adult_train = adult_train.rename(columns = newnames)

adult_test = adult_test.rename(columns = newnames)



print(adult_train.shape)

print(adult_test.shape)
adult_train.head()
adult_test.head()
adult_train.isnull().sum().sort_values(ascending = False)
adult_train = adult_train.apply(lambda x:x.fillna(x.value_counts().index[0]))

adult_train.isnull().sum().sort_values(ascending = False)
adult_test.isnull().sum().sort_values(ascending = False)
adult_test = adult_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

adult_test.isnull().sum().sort_values(ascending = False)
print(adult_train.shape)

print(adult_test.shape)
adult_train.sample(5)
adult_train.info()
adult_train.describe()
adult_train['native_country'].value_counts()
is_us = adult_train['native_country'].map(lambda x: x=='United-States')

is_us.value_counts().plot(kind='pie',autopct='%1.0f%%')
adult_train['capital_gain'].value_counts()
adult_train['capital_loss'].value_counts()
cg0 = adult_train['capital_gain'].map(lambda x: x==0)

cg0.value_counts().plot(kind='pie',autopct='%1.0f%%')
cl0 = adult_train['capital_loss'].map(lambda x: x==0)

cl0.value_counts().plot(kind='pie',autopct='%1.0f%%')
sns.violinplot(x='sex', y='age', hue='income', data=adult_train, split=True)
sns.barplot(x='income', y='hours_per_week', data=adult_train)
sns.barplot(x='income', y='education_num', data=adult_train, hue='sex')
sns.barplot(x='income', y='education_num', data=adult_train)
def barplot_percent(x_axis,hue):

  x,y = hue, x_axis



  df1 = adult_train.groupby(x)[y].value_counts(normalize=True)

  df1 = df1.mul(100)

  df1 = df1.rename('percentage').reset_index()



  g = sns.catplot(x=y,y='percentage',hue=x,kind='bar',data=df1)

  g.ax.set_ylim(0,100)
barplot_percent('income','sex')
barplot_percent('income', 'race')
barplot_percent('income', 'occupation')
barplot_percent('income','workclass')
adult_test.sample(5)
adult_test.info()
adult_test.describe()
x = adult_train[['age','education_num','capital_gain', 'capital_loss', 'hours_per_week']]

y = adult_train['income']



SEED = 158020

np.random.seed(SEED)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25,

                                                         stratify = y)
knn = KNeighborsClassifier(n_neighbors=3)



scores = cross_val_score(knn, xtrain, ytrain, cv=10)

scores.mean()
knn.fit(xtrain,ytrain)

predict = knn.predict(xtest)

predict
accuracy_score(ytest,predict)
numadult_train = adult_train[['age','workclass','education_num','capital_gain',

                      'capital_loss', 'hours_per_week','marital_status',

                      'occupation','relationship','race','sex']].apply(preprocessing.LabelEncoder().fit_transform)

numadult_train
def predict_model(n):

  x = numadult_train

  y = adult_train['income']



  SEED = 158020

  np.random.seed(SEED)

  xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25,

                                                          stratify = y)

  

  scaler = StandardScaler()

  scaler.fit(xtrain)

  xtrain = scaler.transform(xtrain)

  xtest = scaler.transform(xtest)



  knn = KNeighborsClassifier(n_neighbors=n)

  knn.fit(xtrain,ytrain)

  scores = cross_val_score(knn, xtrain, ytrain, cv=10)

  return scores.mean()
x=[]

for i in range(1,40):

  x.append(predict_model(i))
plt.plot(x)
predict_model(18)
adult_test.sample(10)
numadult_test = adult_test[['age','workclass','education_num','capital_gain',

                      'capital_loss', 'hours_per_week','marital_status',

                      'occupation','relationship','race','sex']].apply(preprocessing.LabelEncoder().fit_transform)

numadult_test
x = numadult_train

y = adult_train['income']



SEED = 158020

np.random.seed(SEED)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25,

                                                          stratify = y)

  

scaler = StandardScaler()

scaler.fit(xtrain)

xtrain = scaler.transform(xtrain)

xtest = scaler.transform(xtest)



knn = KNeighborsClassifier(n_neighbors=18)

knn.fit(xtrain,ytrain)

predict = knn.predict(xtest)

accuracy_score(ytest,predict)
scaler = StandardScaler()

scaler.fit(numadult_test)

xadult_test = scaler.transform(numadult_test)



imputer = KNNImputer(n_neighbors=18)

imputer.fit(xtrain)

xadult_test = imputer.transform(xadult_test)
predict = knn.predict(xadult_test)

len(predict)
submission = pd.DataFrame()

submission[0] = adult_test.index

submission[1] = predict

submission.columns = ['Id','income']

submission['income'].value_counts()
submission.to_csv('my_submission.csv',index = False)