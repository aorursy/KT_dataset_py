# Quick and easy overview testing a couple of ML supervised learning models

# as classifiers for the Mushroom dataset.



# The dataset itself consists of multiple categorical features so guess what,

# probably a tree will work wonders (don't need fancy deeplearning here)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.plotting import output_notebook, figure, show

from bokeh.models import HoverTool, ColumnDataSource

import matplotlib.pyplot as plt

from matplotlib import colors as mcolors



%matplotlib inline

output_notebook()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
df.shape
df.head()
df.describe()
df.groupby('class').size()
df.isnull().sum()
# encode the categories - don't use LabelEncoder as all features should be treated equal

# the below is a short cut for OneHotEncoding



for col in df.columns:

    df = pd.get_dummies(df,prefix=col, columns = [col], drop_first=True)   

 

df.head()
X = df.iloc[:,1:96]  # all rows, not col0 but all the remainng cols which are features

y = df.iloc[:, 0]  # all rows, label col only
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# find correlations to target

corr_matrix = df.corr().abs()

print(corr_matrix['class_p'].sort_values(ascending=False).head(11))
print(corr_matrix['class_p'].sort_values(ascending=True).head(11))
# first let's run all the data and setup models

# to find a model that works the best

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

resultsmean = []

resultsstddev = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    resultsmean.append(cv_results.mean())

    resultsstddev.append(cv_results.std())

resultsDf = pd.DataFrame(

    {'name': names,

     'mean': resultsmean,

     'std dev': resultsstddev

    }

)

resultsDf = resultsDf.sort_values(by=['mean'], ascending=False)

print(resultsDf)
# Make predictions using validation dataset using CART model

model1 = DecisionTreeClassifier()

model1.fit(X_train, Y_train)

predictions = model1.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
#plot graph of feature importances

feat_importances = pd.Series(model1.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
# a desire to understand if we can compress feature dimensionality

# using PCA

from sklearn.decomposition import PCA
# PCA

model3 = PCA(n_components=2)

pc = model3.fit_transform(X_train)
len(pca.components_)
principalDf = pd.DataFrame(data = pc, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['class_p']]], axis = 1)
plt.matshow(pca.components_,cmap='viridis')

#plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)

plt.colorbar()

plt.xticks(range(len(df.columns)),df.columns,rotation=65,ha='left')

plt.tight_layout()

plt.show()# 
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [1, 0]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['class_p'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
print('Explained variation per principal component: {}'.format(model3.explained_variance_ratio_))

from sklearn.feature_selection import SelectKBest, f_classif

# feature extraction

k_best = SelectKBest(score_func=f_classif, k=10)

# fit on train set

fit = k_best.fit(X_train, Y_train)

# transform train set

univariate_features = fit.transform(X_train)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.head()

featureScores.columns = ['Column','Score']  #naming the dataframe columns

print(featureScores.nlargest(10, 'Score'))  #print 10 best features