#A place for the imports

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

from sklearn import metrics



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

abalone = pd.read_csv('../input/abalone.csv')

abalone.columns=['Sex','Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight', 

                 'Shell weight','Rings']

abalone.sample(5)
abalone.info()
abalone.describe()
abalone[abalone.Height == 0]
abalone = abalone[abalone.Height > 0]

abalone.describe()
abalone.hist(figsize=(20,10), grid = False, layout=(2,4), bins = 30);
nf = abalone.select_dtypes(include=[np.number]).columns

cf = abalone.select_dtypes(include=[np.object]).columns
skew_list = stats.skew(abalone[nf])

skew_list_df = pd.concat([pd.DataFrame(nf,columns=['Features']),pd.DataFrame(skew_list,columns=['Skewness'])],axis = 1)

skew_list_df.sort_values(by='Skewness', ascending = False)
sns.set()

cols = ['Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight', 'Shell weight','Rings']

sns.pairplot(abalone[cols], height = 2.5)

plt.show();
data = pd.concat([abalone['Rings'], abalone['Height']], axis = 1)

data.plot.scatter(x='Height', y='Rings', ylim=(0,30));

abalone = abalone[abalone.Height < 0.4]

data = pd.concat([abalone['Rings'], abalone['Height']], axis = 1)

data.plot.scatter(x='Height', y='Rings', ylim=(0,30));
abalone.hist(column = 'Height', figsize=(20,10), grid=False, layout=(2,4), bins = 30);
corrmat = abalone.corr()

cols = corrmat.nlargest(8, 'Rings')['Rings'].index

cm = np.corrcoef(abalone[nf].values.T)

sns.set(font_scale=1.25)

plt.figure(figsize=(15,15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=nf.values, xticklabels=nf.values)

plt.show();
data = pd.concat([abalone['Rings'], abalone['Sex']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxenplot(x='Sex', y="Rings", data=abalone)

fig.axis(ymin=0, ymax=30);
abalone = pd.get_dummies(abalone)

abalone.head()
X = abalone.drop(['Rings'], axis = 1)

y = abalone['Rings']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)
from sklearn.linear_model import LinearRegression 

paramLin = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

LinearReg = GridSearchCV(LinearRegression(),paramLin, cv = 10)

LinearReg.fit(X = X_train,y= y_train)

LinearRegmodel = LinearReg.best_estimator_

print(LinearReg.best_score_, LinearReg.best_params_)
LinearReg.score(X_train,y_train)
LinearReg.score(X_test,y_test)
predictions = LinearReg.predict(X_test)

plt.scatter(y_test, predictions)

plt.xlabel('True Values')

plt.ylabel('Predictions')
from sklearn.linear_model import Ridge

paramsRidge = {'alpha':[0.01, 0.1, 1,10,100], 'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}



ridgeReg = GridSearchCV(Ridge(),paramsRidge, cv = 10)

ridgeReg.fit(X = X_train,y= y_train)

Rmodel = ridgeReg.best_estimator_

print(ridgeReg.best_score_, ridgeReg.best_params_)
ridgeReg.score(X_train,y_train)
ridgeReg.score(X_test,y_test)
predictions = ridgeReg.predict(X_test)

plt.scatter(y_test, predictions)

plt.xlabel('True Values')

plt.ylabel('Predictions')
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_std)

y_kmeans = kmeans.predict(X_std)
plt.scatter(X_std[:, 0], X_std[:, 1], c=y_kmeans, s=50, cmap='viridis');



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
corr_mat = np.corrcoef(X_std.T)
eigenvalues, eigenvectors = np.linalg.eig(corr_mat)

print('\nEigenvalues \n%s' %eigenvalues)
#eigenvalue and eigenvector pairs

pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]

pairs.sort(key = lambda x: x[0], reverse = True)
sorted_eigenval = []

for i in pairs:

    sorted_eigenval.append(i[0])

print(sorted_eigenval)
total = sum(eigenvalues)

variance_explained = [(i/total)*100 for i in sorted_eigenval]
variance_explained
cum_variance_explained = np.cumsum(variance_explained)

cum_variance_explained


#Plot variance explained by the principal components

with plt.style.context('fivethirtyeight'):

    plt.figure(figsize=(8, 6))

    plt.bar(range(10), variance_explained, alpha=0.7, align='center',

            label='individual explained variance')

    plt.step(range(10), cum_variance_explained, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout();
projection_mat = np.hstack((pairs[0][1].reshape(10,1),

                           pairs[1][1].reshape(10,1),

                           pairs[2][1].reshape(10,1)))
X_new = X_std.dot(projection_mat)

X_new.shape
abalone.head(5)
bins = [0,8,10,abalone['Rings'].max()]

group_names = ['young','medium','old']

abalone['Rings'] = pd.cut(abalone['Rings'],bins, labels = group_names)
dictionary = {'young':0, 'medium':1, 'old':2}

abalone['Rings'] = abalone['Rings'].map(dictionary)
abalone.head(10)
X = abalone.drop(['Rings'], axis = 1)

y = abalone['Rings']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

paramsKn = {'n_neighbors':range(1,30)}

Kneighbours = GridSearchCV(KNeighborsClassifier(),paramsKn, cv=10)



Kneighbours.fit(X=X_train,y=y_train)

Kmodel = Kneighbours.best_estimator_

print(Kneighbours.best_score_, Kneighbours.best_params_)
from sklearn.svm import SVC

paramsSvm = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],

                  'C':[0.1,1,10],'gamma':[0.01,0.1,0.5,1,2]}



Svm = GridSearchCV(SVC(),paramsSvm,cv=5)



Svm.fit(X_train,y_train)

model_svm = Svm.best_estimator_

print(Svm.best_score_,Svm.best_params_)