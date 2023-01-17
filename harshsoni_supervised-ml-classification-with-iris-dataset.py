# Python version

import sys

print('Python: {}'.format(sys.version))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# seaborn

import seaborn

print('seaborn: {}'.format(seaborn.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# Display plots within notebook

%matplotlib inline



# Import required python libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap  # For making our own cmap

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

# Classification Algorithms - 

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



# Hide FutureWarnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# set seaborn for attractive plots

sns.set(style='darkgrid')
# Load dataset into dataframe

path = '../input/Iris.csv'

feature_labels = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

iris_df = pd.read_csv(path)



#Drop Id column and change column names to desired names in above list

iris_df = iris_df.drop(columns='Id')

iris_df.columns = feature_labels



# keep unique classes

classes = iris_df['class'].unique()
iris_df.shape
iris_df.head(5)
iris_df.info()
iris_df.describe()
plt.figure(figsize=(12,8))

for i in range(1, len(feature_labels)):

    plt.subplot(2, 2, i)

    sns.distplot(iris_df[feature_labels[i-1]], kde=False)

    

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.figure(figsize=(6,5))

sns.boxplot(data=iris_df)
# Boxplot variant of violinplot makes, its same plot as violinplots as below but in box format

# plt.figure(figsize=(12,10))

# for i in range(1, len(feature_labels)):

#     plt.subplot(2, 2, i)

#     sns.boxplot(x='class', y=feature_labels[i-1], data=iris_df)

    

# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# Violinplot variant of boxplot

# plt.figure(figsize=(6,5))

# sns.violinplot(data=iris_df)
plt.figure(figsize=(12,8))

for i in range(1, len(feature_labels)):

    plt.subplot(2, 2, i)

    sns.violinplot(x='class', y=feature_labels[i-1], data=iris_df)

    

plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.pairplot(iris_df, hue='class')
test_size = 0.2

seed = 7



features_train, features_test, labels_train, labels_test = train_test_split(np.float64(iris_df.values[:, 0:4]), iris_df['class'], test_size=0.2, random_state=seed)
scaler = StandardScaler()

scaler.fit(features_train)

scaled_features_train = scaler.transform(features_train)

scaled_features_test = scaler.transform(features_test)



#### Visualise Boxplot for each scaled feature

scaled_df = pd.DataFrame(data=np.array(scaled_features_train), columns=feature_labels[0:4])

# scaled_df[feature_labels[4]] = le.inverse_transform(labels_train)

plt.figure(figsize=(6, 5))

sns.boxplot(data=scaled_df)
# SelectKBest using f_classif (ANOVA Test) as score function

bestfeatures = SelectKBest(score_func=f_classif, k=2)

fit = bestfeatures.fit(scaled_features_train, labels_train)



scores_df = pd.DataFrame(data=fit.scores_)

columns_df = pd.DataFrame(data=iris_df.columns)



feature_scores_df = pd.concat([columns_df,scores_df],axis=1)

feature_scores_df.columns = ['Features','Score']



# print(feature_scores_df.nlargest(4,'Score'))



plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

sns.barplot(x='Score', y='Features', order=feature_scores_df.nlargest(4,'Score')['Features'], data=feature_scores_df, palette=sns.cubehelix_palette(n_colors=4, reverse=True))

plt.subplot(1, 2, 2)

sns.heatmap(iris_df.corr(), annot=True, cmap=sns.cubehelix_palette(start=0, as_cmap=True))
transformed_features_train = scaled_features_train[:, [2, 3]]

transformed_features_test = scaled_features_test[:, [2, 3]]
clf_names = ['NB', 'SVC', 'KNN', 'CART', 'BAG', 'RF', 'AB']



models = [

    ('NB', GaussianNB()),

    ('SVC', SVC(C=1000, kernel='rbf', gamma=0.05)),

    ('KNN', KNeighborsClassifier(n_neighbors=5)),

    ('CART', DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=5)),

    ('BAG', BaggingClassifier(n_estimators=100)),

    ('RF', RandomForestClassifier(n_estimators=100, min_samples_split=20, min_samples_leaf=5)),

    ('AB', AdaBoostClassifier(n_estimators=100))

]
scoring = 'accuracy'



cv_results = []

for name, clf in models:

    kfold = KFold(n_splits=10, random_state=seed)

    cv_score = cross_val_score(clf, transformed_features_train, labels_train, cv=kfold, scoring=scoring)

    cv_results.append(cv_score)

    print('{}: {}, ({})' .format(name, cv_score.mean(), cv_score.std()))
plt.figure(figsize=(6, 5))

sns.boxplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))



# #### Alternate choice for boxplot - univariateplot -> boxplot + violinplot

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 2, 1)

# sns.boxplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))



# plt.subplot(1, 2, 2)

# sns.violinplot(data=pd.DataFrame(data=np.array(cv_results).transpose(), columns=clf_names))
x = transformed_features_train[:, [0]].ravel()

y = transformed_features_train[:, [1]].ravel()



le = LabelEncoder()

le.fit(classes)



# Encode all classes of Iris Flower species to values [0, 1, 2] to plot contour

target_labels_encoded = le.transform(iris_df['class'].ravel())

labels_train_encoded = le.transform(labels_train)



# color sequence

c = labels_train_encoded



models_ = models.copy()

trained_models = [(name, clf.fit(np.array(transformed_features_train), c)) for name, clf in models_]



titles = ('Input Data',

          'Gaussian NB',

          'SVM with RBF kernel',

          'KNN',

          'Decision Tree',

          'Bagging',

          'Random Forest',

          'AdaBoost',

         )



def mesh_grid(x, y, h=0.02):

    x_min, x_max = x.min(), x.max()

    y_min, y_max = y.min(), y.max()

    

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    

    return xx, yy



def plot_contour(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    return ax.contourf(xx, yy, Z, **params)



xx, yy = mesh_grid(x, y)



# custom cmap

my_cmap = ListedColormap(sns.color_palette().as_hex()[0:3])



fig, sub = plt.subplots(2, 4, figsize=(20, 8))



sub[0, 0].scatter(iris_df.values[:, [2]].ravel(), iris_df.values[:, [3]].ravel(), c=target_labels_encoded, cmap=my_cmap, s=20)

sub[0, 0].set_title(titles[0])



for clf, title, ax in zip(trained_models, titles[1:], sub.flatten()[1:]):

        plot_contour(ax, clf[1], xx, yy, cmap=my_cmap, alpha=1)

        ax.scatter(x, y, c=c, cmap=my_cmap, s=25, edgecolor='k')

        ax.set_xlim(xx.min(), xx.max())

        ax.set_ylim(yy.min(), yy.max())

        ax.set_xlabel('Petal length')

        ax.set_ylabel('Petal width')

        ax.set_xticks(())

        ax.set_yticks(())

        ax.set_title(title)



plt.show()
# Selecting and predicting using KNN

clf = models[2][1]

clf.fit(transformed_features_train, labels_train)

predictions = clf.predict(transformed_features_test)



print(accuracy_score(labels_test, predictions))

print(confusion_matrix(labels_test, predictions))

print(classification_report(labels_test, predictions))