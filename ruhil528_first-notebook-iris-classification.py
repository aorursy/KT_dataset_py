# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score, KFold

%matplotlib inline
# read csv file and name columns
df = pd.read_csv('../input/iris.data.csv', names=["Sepal Len", "Sepal Wid", "Petal Len", "Petal Wid", "Type"])
df.describe()
# Print first five rows of the data
df.head()
# Data shuffle using pandas
df = df.sample(frac=1).reset_index(drop=True)
df.head()
# Separate iris dataset into features and response
X = df[["Sepal Len", "Sepal Wid", "Petal Len", "Petal Wid"]]
Y = df["Type"]
print(X.head())
print(Y.head())
# Scatter matrix plot
pd.plotting.scatter_matrix(X, diagonal='kde', alpha=0.4, figsize=(14,14))
plt.show()
# scatter plot using seaborn
sns.set(style='ticks')
sns.pairplot(df, hue='Type', diag_kind='kde')
plt.show()
# histograms
df.hist(figsize=(10,10))
plt.tight_layout()
plt.show()
# Data visualization - box plot
df.boxplot()
plt.ylabel('Measurement in cm', fontsize=14)
plt.title('Data Summary - Box Plot', fontsize=14)
plt.show()
# Numerical representation of Iris type
print(df.head())
df.loc[df['Type'] == 'Iris-setosa', 'Type'] = 0
df.loc[df['Type'] == 'Iris-versicolor', 'Type'] = 1
df.loc[df['Type'] == 'Iris-virginica', 'Type'] = 2
print(df.head())
# Assign target values
Y = df["Type"]
print(Y.head())
# Classification model comparison between KNN, SVC, LDA, QDA, Random Forest
print('Model Comparison and Selection: Logistic Reg., LDA, QDA, SVC, KNN, RandomForest')

# Prepare Models
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('SVC', svm.SVC(kernel='linear', C=1)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RFC', RandomForestClassifier(n_estimators=100, max_features=3)))

print(models)
# Evaluate each model in a loop
results = []

for name, model in models:
    #print(name, model)
    
    # Defining Kfold parameters
    k_fold = KFold(n_splits=5, shuffle=True, random_state=764)
    cv_results = cross_val_score(model, X, Y, cv=k_fold)
    
    # Append cross validation score to a list
    results.append(cv_results)
    
    # Print model performance 
    print("The performance of {} model".format(name))
    print("Accuracy of the cross_val_score (cv=5): ", cv_results)
    print("The mean score with 95-percent confidence interval: %0.2f (+/- %0.2f)" % (cv_results.mean(), 
                                                                                     cv_results.std() * 2), '\n')
