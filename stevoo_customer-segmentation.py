import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
#loading dataset and print information

df = pd.read_csv('../input/Mall_Customers.csv')

df.info()

df.head()
df.describe()
del df['CustomerID']

df.columns=['Gender', 'Age', 'AnnualIncome', 'SpendingScore']

df.head()
df['Gender'] = (df['Gender'] =='Male').astype(int)

df.head()
#map dataset from 4 to 2 dimensions

pca=PCA(n_components=2)

data=np.asarray(df)

pca.fit(data)

X=pca.transform(data)



plt.scatter(X[:,0], X[:,1])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)



fig, ax = plt.subplots(1, 1)

scatter=ax.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap=sns.diverging_palette(220, 10, as_cmap=True, center="dark"))
df['Group']=pd.Series(kmeans.labels_, df.index)

df.head()
df.groupby('Group').count().iloc[:,0].plot(kind='bar')
df.groupby('Group').mean()
ageGroup=[]



for age in df['Age']:

    if age<=25:

        index=0

    elif age<=40:

        index=1

    else:

        index=2

    ageGroup.append(index)



df['AgeGroup']=pd.Series(ageGroup, df.index)
df.groupby('AgeGroup').mean()
data=np.asarray(df[['AnnualIncome', 'SpendingScore']])

colors=['red','green', 'blue']

uniquePoints={}

fig, ax = plt.subplots(1, 1)

for idx, example in enumerate(data):

    label=ageGroup[idx]

    point=ax.scatter(example[0], example[1], c=colors[label], label=label)

    uniquePoints[label]=point

    



ax.legend([uniquePoints[key] for key in uniquePoints],['<26', '26-40', '41>'])

ax.set_ylabel("Spending score")

ax.set_xlabel("Annual Income")
fig, ax = plt.subplots(1, 2)

ax[0].bar([i for i in range(3)], df.groupby('AgeGroup').mean()['SpendingScore'].tolist(), tick_label=['Students', 'Adults', 'Elders'])

ax[0].set_title('Spending score')

ax[1].bar([i for i in range(3)], df.groupby('AgeGroup').mean()['AnnualIncome'].tolist(), tick_label=['Students', 'Adults', 'Elders'])

ax[1].set_title('Annual income')
spendingScore=df.groupby('Gender').mean()['SpendingScore'].tolist()

annualIncome=df.groupby('Gender').mean()['AnnualIncome'].tolist()



fig, ax = plt.subplots(1, 2)

ax[0].bar([i for i in range(2)], spendingScore, tick_label=['Women', 'Men'])

tot=np.sum(spendingScore)

for i in range(2):

    ax[0].text(i-0.2, spendingScore[i]+1, '{:.2f}'.format(spendingScore[i]/tot*100))

ax[0].set_title('Spending score')



ax[1].bar([i for i in range(2)], annualIncome, tick_label=['Women', 'Men'])

tot=np.sum(annualIncome)

for i in range(2):

    ax[1].text(i-0.2, annualIncome[i]+1, '{:.2f}'.format(annualIncome[i]/tot*100))

ax[1].set_title('Annual Income')
sns.pairplot(df[['Gender', 'Age', 'AnnualIncome', 'SpendingScore']])
#Variance of each feature

var = df[['Gender', 'Age', 'AnnualIncome', 'SpendingScore']].var()

print(var)
corr = df[['Gender', 'Age', 'AnnualIncome', 'SpendingScore']].corr()



sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True))
from sklearn.linear_model import LinearRegression

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import validation_curve

from sklearn.dummy import DummyRegressor



X=np.asarray(df[['Gender', 'Age', 'AnnualIncome']])

Y=np.asarray(df['SpendingScore'])

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
print('Dummy Regressor:')

dumModelout=cross_val_score(DummyRegressor(strategy='mean'), X, Y, cv=5)

#r^2 score

print('Dummy model score: {:.2f} (+/- {:.2f})'.format(dumModelout.mean(), dumModelout.std()**2))

print('\n')

print('Linear regression:')

outLR=cross_val_score(LinearRegression(normalize=True), X, Y, cv=5)

print('Linear model score: {:.2f} (+/- {:.2f})'.format(outLR.mean(), outLR.std()**2))

print('Support Vector Regression')

#Parameters to experiment with

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



clf = GridSearchCV(svm.SVR(), tuned_parameters, return_train_score=True, cv=5).fit(X,Y)

print('Best model\'s params:', clf.best_params_)

#print(clf.cv_results_.keys())

print('Train score: {:.2f} (+/- {:.2f})'.format(clf.cv_results_['mean_train_score'].mean(), clf.cv_results_['std_train_score'].mean()))

print('Test score: {:.2f}  (+/- {:.2f})'.format(clf.cv_results_['mean_test_score'].mean(), clf.cv_results_['std_test_score'].mean()))



optimised_clf= clf.best_estimator_