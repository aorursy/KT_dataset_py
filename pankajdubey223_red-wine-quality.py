# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('//kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
data.shape
data.describe(include = 'all')
data.info()
data['quality'].unique()
data['quality'] = data['quality'] - 3
data['quality'].unique()
data.hist(figsize=(20,20))

plt.show()
sns.pairplot(data, diag_kind = 'kde', hue = 'quality')

plt.show()
sns.distplot(data['total sulfur dioxide'])

plt.show()
data.groupby(['quality']).count()
df = data.copy()

X = df.iloc[:,:].values

from scipy.stats import zscore

zscore(X)

X.shape

plt.figure(figsize=(20,20))

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.grid(True)

plt.show()

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X[:,[0,11]])

plt.figure(figsize=(20,14))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'orange', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'black', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'green', label = 'Cluster 5')

plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 20, c = 'yellow', label = 'Cluster 6')







plt.title('Clusters of wine quality')

plt.xlabel('quality of wine in catageory')

plt.ylabel('fixed acidity')

plt.legend()

plt.show()
def replace(group):

    median, std = group.median(), group.std()

    outliers = (group - median).abs() > 2 * std

    group[outliers] = group.median()

    return group
data['total sulfur dioxide'].max()
data['total sulfur dioxide'] = replace(data['total sulfur dioxide'])

data['sulphates'] = replace(data['sulphates'])
sns.distplot(data['total sulfur dioxide'])

sns.distplot(data['sulphates'])
corr_matrix = data.corr()

plt.figure(figsize = (10,20))

ax = sns.heatmap(corr_matrix[['quality']].sort_values(by=['quality'],ascending=False),annot = True)
plt.figure(figsize = (15,15))

sns.heatmap(data.corr(),annot = True)
data['fixed acidity_per_volatiles acidity'] = data['fixed acidity']/data['volatile acidity']

data['residual sugar_per_volatiles acidity'] = data['residual sugar']/data['volatile acidity']

data['fixed acidity_per_residual sugar'] = data['fixed acidity']/data['residual sugar']

data['free sulfur dioxide_per_volatiles acidity'] = data['free sulfur dioxide']/data['volatile acidity']

data['total sulfur dioxide_per_PH'] = data['total sulfur dioxide']/data['pH']

data['pH_per_density'] = data['pH']/data['density']

data['sulphates_per_alcohol'] = data['sulphates']/data['alcohol']

data['alchol_per_residual sugar'] = data['alcohol']/data['residual sugar']

data['alcohol_per_chlorides'] = data['alcohol']/data['chlorides']

data['alcohol_per_density'] = data['alcohol']/data['density']

data['alcohol_per_fixed acidity'] = data['alcohol']/data['fixed acidity']

data['alcohol_per_volatile acidity'] = data['alcohol']/data['volatile acidity']

data['fixed acidity_per_density'] = data['fixed acidity']/data['density']

data['volatile acidity_per_density'] = data['volatile acidity']/data['density']

data['sulphates_per_density'] = data['sulphates']/data['density']

data['total sulfur dioxide_per_density'] = data['total sulfur dioxide']/data['density']

data['citric acid_per_density'] = data['citric acid']/data['density']

data.head()
#sns.pairplot(data,hue = 'quality')

plt.show()
X = data.drop('quality',axis = 1).copy()

Y = data['quality'].copy()

df = X.copy()

X.shape[1]



X = pd.DataFrame(X)

Y = pd.DataFrame(Y)
df = X

Y.head()

import time

from sklearn.manifold import TSNE
import time

from sklearn.manifold import TSNE



n_sne = 7000

time_start = time.time()

tsne = TSNE(n_iter = 1000,random_state=13)

tsne_result = tsne.fit_transform(df.values)

print('tsne done! Time elapsed {} seconds'.format(time.time()-time_start))
df['label'] = Y

%matplotlib inline

fig = plt.figure(figsize=(20,8))

plt.scatter(tsne_result[:,0],tsne_result[:,1],c = df['label'], cmap = plt.cm.get_cmap('Dark2',20),alpha = 0.9,linewidths = 5)

plt.clim(-0.5,9.5)

plt.colorbar(ticks = range(0,6))

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["quality"]):

    strat_train_set = data.loc[train_index]

    strat_test_set = data.loc[test_index]
strat_train_set=strat_train_set.reset_index(drop=True)

strat_train_set.head()
strat_test_set = strat_test_set.reset_index(drop=True)

strat_test_set.head()
X_train, y_train, X_test, y_test = heart = strat_train_set.drop("quality", axis=1),strat_train_set["quality"].copy(),strat_test_set.drop('quality',axis = 1),strat_test_set['quality'].copy()
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C = 1.0,solver = 'newton-cg', multi_class =  'multinomial')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn import metrics

metrics.accuracy_score(y_pred, y_test)
print(metrics.classification_report( y_test, y_pred))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot = True)
x = range(len(X_test[:20]))

plt.plot(x,classifier.predict(X_test[:20]),color = 'green', label = 'predicted values' )

plt.scatter(x,y_test[:20] , color = 'red', label = 'original values')

plt.grid(True)

plt.legend()
cat_features = list(range(0, X.shape[1]))

print(cat_features)



from catboost import CatBoostClassifier



clf = CatBoostClassifier(

    iterations=5, 

    learning_rate=0.1, 

    #loss_function='CrossEntropy'

)





clf.fit(X_train, y_train, 

        eval_set=(X_train, y_train), 

        verbose=False

)



print('CatBoost model is fitted: ' + str(clf.is_fitted()))

print('CatBoost model parameters:')

print(clf.get_params())

from catboost import CatBoostClassifier

clf = CatBoostClassifier(

    iterations=10,

#     verbose=5,

)



clf.fit(

    X_train, y_train,

    eval_set=(X_train, y_train),

)
y_pred = clf.predict(X_test)

from sklearn import metrics

metrics.accuracy_score(y_pred, y_test)
x = range(len(X_test[:20]))

plt.plot(x,clf.predict(X_test[:20]),color = 'green', label = 'predicted values' )

plt.scatter(x,y_test[:20] , color = 'red', label = 'original values')

plt.grid(True)

plt.legend()
import time

start_time = time.time()

from tpot import TPOTClassifier

tpot_clf = TPOTClassifier(verbosity=1,cv = 2,max_eval_time_mins=1)

tpot_clf.fit(X_train.values,y_train.values)

print(f'Time Elasped to done the whole process is {time.time()-start_time}')
automl.fit(X_train,y_train)

y_pred = automl.predict(X_test)

# Making the Confusion Matrix

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm,annot = True)

accuracy_score(y_pred,y_test)