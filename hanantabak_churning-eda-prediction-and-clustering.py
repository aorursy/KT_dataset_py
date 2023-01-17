# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, chi2

from mlxtend.preprocessing import minmax_scaling

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import pylab as pl

from kmodes.kmodes import KModes



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/telecom-churn-datasets/churn-bigml-80.csv')

test = pd.read_csv('../input/telecom-churn-datasets/churn-bigml-20.csv')
train.sample(10)
train.describe()
test.sample(10)
test.describe()
train.info()
test.info()
train['State'].nunique()
train['State'].value_counts()
print('The percentage of customers churning from the company is: %{}'.format((train['Churn'].sum()) *100/train.shape[0]) ) # as the Churn column data type is boolean, every True value will be summed as '1'...I'll convert them later into binary 0's and 1's when I do the data cleaning part
plt.figure(figsize=(20,6))

sns.set_style('whitegrid')

sns.barplot(x='State',y='Churn', data=train)
sns.barplot(x='Churn', y='Customer service calls',data=train)
sns.barplot(x='Churn', y='Account length',data=train)
plt.hist(train['Account length'], bins=400)

plt.show()
churn_intl = train.groupby(['Churn','International plan']).size()

churn_intl.plot()

plt.show()

churn_voicem = train.groupby(['Churn','Voice mail plan']).size()

churn_voicem.plot()

plt.show()
train.head()
train['Total charge'] = train['Total day charge'] + train['Total eve charge'] + train['Total night charge'] + train['Total intl charge']

test['Total charge'] = test['Total day charge'] + test['Total eve charge'] + test['Total night charge'] + test['Total intl charge']
sns.boxplot(x='Churn',y='Total charge', data = train)
train2 = train.copy()

test2 = test.copy()

train2
train2['Churn'] = train2['Churn'].map({True:1,False:0}) # no need to do it for test dataset because Churn column will be dropped later.



train2['International plan'].replace(['No','Yes'],[0,1],inplace=True)

test2['International plan'].replace(['No','Yes'],[0,1],inplace=True)



# Now, I'll use the label encoder preprocessing technique:



encoder = LabelEncoder()

coded_voicem_train = encoder.fit_transform(train2['Voice mail plan'])

train2['Voice mail plan'] = coded_voicem_train

coded_voicem_test = encoder.transform(test2['Voice mail plan'])

test2['Voice mail plan'] = coded_voicem_test
train2.head()
test2.head()
train2.corr()
plt.figure(figsize=(15,15))

sns.heatmap(train2.corr() , annot =True)
train3 = train2.drop(['Total day minutes','Total eve minutes','Total night minutes', 'Total intl minutes'], axis=1)

features = ['International plan','Total charge','Customer service calls']

X_init = train3[features]

y = train3['Churn']

Xtest_init = test2[features]

ytest = test2['Churn']
X_init.head()
# mix-max scale the data between 0 and 1

X = minmax_scaling(X_init, columns = features)

Xtest = minmax_scaling(Xtest_init, columns = features)

Xtest
Xtrain,Xval,ytrain,yval = train_test_split(X,y,train_size=0.8)
Xtrain.shape
Xval.shape
ytrain.shape
yval.shape
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

results = []

names = []

for name,model in models:

    kfold = model_selection.KFold(n_splits=10)

    cv_result = model_selection.cross_val_score(model,Xtrain,ytrain, cv = kfold, scoring = "accuracy")

    names.append(name)

    results.append(cv_result)

for i in range(len(names)):

    print(names[i],results[i].mean())
chosen_model = KNeighborsClassifier()

param = {'n_neighbors': [1,2,3,4,5,6,7]}

grid = GridSearchCV(estimator= chosen_model, param_grid=param, cv=5)

grid.fit(Xtrain,ytrain)

print(grid.best_params_)

best_model = KNeighborsClassifier(n_neighbors=5)

best_model.fit(Xtrain,ytrain)

pred_val = best_model.predict(Xval)

pred = best_model.predict(Xtest)
print("Accuracy Score is:")

print(accuracy_score(ytest, pred))

print(accuracy_score(yval, pred_val))

print()
print("Classification Report:")

print(classification_report(ytest, pred))
conf = confusion_matrix(ytest,pred)

label = ["0","1"]

sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)

plt.show()
# train3 is fine for this.

clust_data = train3.drop(['Churn','State'], axis=1)

inertia = []

for i in range(1,11):

    clust_model = KMeans(n_clusters= i , init='k-means++', n_init=10)

    clust_model.fit(clust_data)

    inertia.append(clust_model.inertia_)



plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
clust_model = KMeans(n_clusters= 4 , init='k-means++', n_init=10)

clusters = clust_model.fit_predict(clust_data)

print(silhouette_score(clust_data, clusters))

train['clusters'] = pd.Series(clusters,index=train.index)

train
clust_churn = train.groupby('clusters').Churn.sum()

clust_churn
train['clusters'].value_counts()
train.head()
train['charge'] = train['Total charge']

charge_clust = train.groupby('clusters').charge.mean()

charge_clust