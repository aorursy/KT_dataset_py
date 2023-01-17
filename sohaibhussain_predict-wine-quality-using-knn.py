# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random

import warnings

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")



from sklearn.preprocessing import StandardScaler # standardization

from sklearn.model_selection import train_test_split # Split dataset

from sklearn.neighbors import KNeighborsClassifier # KNN Model

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB # Naive Bayes Model

from sklearn.metrics import accuracy_score # Accuracy measurements

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine.head()
wine.info()



# Here we can see there is no null values in this dataset.
wine.isnull().sum()



# No null values present in data frame.
wine.shape



# Total data points are 1599
wine.columns



# List of features we have in this dataset.
wine['quality'].value_counts()
corr = wine.corr()

corr
# Generate a mask for the upper triangle

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



with sns.axes_style("white"):

    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(10, 8))

    ax = sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
wine.drop(["residual sugar",'free sulfur dioxide','pH'],axis = 1,inplace = True)

wine.head()
sns.distplot(wine['alcohol'])

plt.show()
bins = [0, 10, 12, 15]

labels = ["low","median","high"]

wine['alcohol_label'] = pd.cut(wine['alcohol'], bins=bins, labels=labels)

wine.drop('alcohol',axis =1, inplace = True)

wine.head()
sns.distplot(wine['quality'])

plt.show()
bins = [0, 4, 6, 10]

labels = ["poor","normal","excellent"]

wine['quality_label'] = pd.cut(wine['quality'], bins=bins, labels=labels)

wine.drop('quality',axis =1, inplace = True)

wine.head()
sns.pairplot(wine, hue="quality_label", palette="husl",diag_kind="kde")

plt.show()
sns.FacetGrid(wine,hue='quality_label', height=5).map(sns.distplot,'volatile acidity').add_legend()

plt.show()
sns.boxplot(x='quality_label',y='volatile acidity', data=wine)

plt.show()
sns.FacetGrid(wine,hue='quality_label', height=5).map(sns.distplot,'citric acid').add_legend()

plt.show()
sns.boxplot(x='quality_label', y='citric acid',data=wine)

plt.show()
wine.info()
wine['alcohol_label'].value_counts()
# Convert category values to numeric values by creating dummy featutes.

df_wine = pd.get_dummies(wine, columns=['alcohol_label'], drop_first=True)

df_wine.head()
result = df_wine['quality_label']

df_wine.drop(['quality_label'], axis=1, inplace=True)

print(df_wine.shape, result.shape)
# use 70% of the data for training and 30% for testing

X_train, X_test, Y_train, Y_test = train_test_split(df_wine, result, test_size=0.30, random_state=11)
# For KNN our dataset must have to standardised.

# No need standardised quality_label as it is the result column



scaler = StandardScaler()

scaler.fit(df_wine)

scaled_features = scaler.transform(df_wine)

df_wine_sc = pd.DataFrame(scaled_features, columns=df_wine.columns)
# use 70% of the data for training and 30% for testing

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_wine_sc, result, test_size=0.30, random_state=11)
# convert dataframe to nd numpy array

X_train_sc = X_train_sc.to_numpy()

y_train_sc = y_train_sc.to_numpy()
def apply_knn(neigh, weight='uniform'):

    knn = KNeighborsClassifier(n_neighbors=neigh, weights=weight)

    knn.fit(X_train_sc,y_train_sc)

    pred_knn = knn.predict(X_test_sc)

    return pred_knn
pred_knn_for_20 = apply_knn(20)

print('Accuracy of model at K=20 is', accuracy_score(y_test_sc, pred_knn_for_20))
clf = LogisticRegression(random_state=0)

clf.fit(X_train_sc, y_train_sc)

pred_lr = clf.predict(X_test_sc)

print('Accuracy of model is', accuracy_score(y_test_sc, pred_lr))
model = KNeighborsClassifier()



params = {'n_neighbors':list(range(1, 50, 2)), 'weights':['uniform', 'distance']}



gs = GridSearchCV(model, params, cv = 5, n_jobs=-1)



gs_results = gs.fit(X_train_sc, y_train_sc)
print('Best Accuracy: ', gs_results.best_score_)

print('Best Parametrs: ', gs_results.best_params_)
best_k = 13

best_weights = 'distance'

pred_knn_for_Best_k = apply_knn(best_k, best_weights)

print('Accuracy of model at K=13 is ', accuracy_score(y_test_sc, pred_knn_for_Best_k))
model = LogisticRegression(max_iter=10000)



params = [{'C': [10**-4, 10**-2, 10**0, 10**2, 10**4], 'penalty': ['l1', 'l2']}]



gs = GridSearchCV(model, params, cv=5, n_jobs=-1)



gs_results = gs.fit(X_train_sc, y_train_sc)
print('Best Accuracy: ', gs_results.best_score_)

print('Best Parametrs: ', gs_results.best_params_)
lr = LogisticRegression(C=1, penalty='l2', random_state=0)



lr.fit(X_train_sc, y_train_sc)



pred_lr_for_Best_param = lr.predict(X_test_sc)



print('Accuracy of model at C = 1 and Penalty = l2 is', accuracy_score(y_test_sc, pred_lr_for_Best_param))
knn = KNeighborsClassifier(n_neighbors=13,weights='distance')

scores_knn = cross_val_score(knn, X_train_sc, y_train_sc, cv=10, scoring='accuracy')
print(scores_knn)
print(scores_knn.mean())
lr = LogisticRegression(C=1, penalty='l2',random_state=0)

scores_lr = cross_val_score(lr, X_train_sc, y_train_sc, cv=10, scoring='accuracy')
print(scores_lr)
print(scores_lr.mean())
print(pd.DataFrame(y_test_sc)['quality_label'].value_counts())
cm = confusion_matrix(y_test_sc, pred_knn_for_Best_k)
names = ["excellent","normal","poor"]

print(pd.DataFrame(cm, index=names, columns=names))
cm = confusion_matrix(y_test_sc, pred_lr_for_Best_param)
names = ["excellent","normal","poor"]

print(pd.DataFrame(cm, index=names, columns=names))