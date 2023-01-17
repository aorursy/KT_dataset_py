# Loading a few important modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline

sns.set() #sets a style for the seaborn plots.

np.random.seed(64)
columns_names = ['Target', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids',

                 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']
# Loading the data from it's csv,

# and converting the 'Target' column to be a string so pandas won't infer it as a numeric value

data = pd.read_csv(os.path.join('..', 'input', 'Wine.csv'), header=None)

data.columns = columns_names

data['Target'] = data['Target'].astype(str)

data.head() # print the data's top five instances
data.info() # prints out a basic information about the data.
sns.countplot(data['Target']);
# This method prints us some summary statistics for each column in our data.

data.describe()
# box plots are best for plotting summary statistics.

sns.boxplot(data=data);
data_to_plot = data.iloc[:, 1:]

fig, ax = plt.subplots(ncols=len(data_to_plot.columns))

plt.subplots_adjust(right=3, wspace=1)

for i, col in enumerate(data_to_plot.columns):

    sns.boxplot(y=data_to_plot[col], ax = ax[i]);
columns_to_plot = list(data.columns)

columns_to_plot.remove('Target')

sns.pairplot(data, hue='Target', vars=columns_to_plot);

# the hue parameter colors data instances baces on their value in the 'Target' column.
sns.lmplot(x='Proline', y='Flavanoids', hue='Target', data=data, fit_reg=False);
sns.lmplot(x='Hue', y='Flavanoids', hue='Target', data=data, fit_reg=False);
# This is a good feature comination to separate the red ones (label 3)

sns.lmplot(x='Color_intensity', y='Flavanoids', hue='Target', data=data, fit_reg=False);
sns.boxplot(x=data['Target'], y=data['OD280/OD315_of_diluted_wines']);

# this is a vey good feature to separate label 1 and 3
plt.figure(figsize=(18,15))

sns.heatmap(data.corr(), annot=True, fmt=".1f");
sns.lmplot(x='Total_phenols', y ='Flavanoids', data=data, fit_reg=True);
from sklearn.model_selection import train_test_split

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



X = data.drop('Target', axis=1)

y = data['Target']

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



model = KNeighborsClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on training set:', model.score(x_train, y_train))

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



model = Pipeline(

    [

        ('scaler', StandardScaler()), # mean normalization

        ('knn', KNeighborsClassifier(n_neighbors=1))

    ]

)

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on training set:', model.score(x_train, y_train))

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
from sklearn.model_selection import learning_curve

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



def over_underfit_plot(model, X, y):

    plt.figure()



    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()



    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score");

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    plt.yticks(sorted(set(np.append(train_scores_mean, test_scores_mean))))

    

over_underfit_plot(model, x_train, y_train)
np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



X.drop('Total_phenols', axis=1, inplace =True) # delete one of the correlating features

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y) # split the data again



#fit the same model again and print the scores

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on training set:', model.score(x_train, y_train))

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
from sklearn.ensemble import RandomForestClassifier

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



model_feature_importance = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train).feature_importances_

feature_scores = pd.DataFrame({'score':model_feature_importance}, index=list(x_train.columns)).sort_values('score')

sns.barplot(feature_scores['score'], feature_scores.index)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel, RFECV

from sklearn.ensemble import RandomForestClassifier

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible.



model = Pipeline(

    [

        ('select', SelectFromModel(RandomForestClassifier(n_estimators=1000), threshold=0.06)),

        ('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier(n_neighbors=1))

    ]

)



model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on training set:', model.score(x_train, y_train))

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))