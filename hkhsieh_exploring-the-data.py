import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
test_df.head()
train_df.describe()
train = train_df[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',

                 'YearRemodAdd','MasVnrArea','BsmtFinSF1','WoodDeckSF','OpenPorchSF','EnclosedPorch',

                  'SaleCondition']]

test = test_df[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',

                 'YearRemodAdd','MasVnrArea','BsmtFinSF1','WoodDeckSF','OpenPorchSF','EnclosedPorch',

                  'SaleCondition']]
train.head()
test.head()
farms = [ train, test]

whole_data = pd.concat(farms)
len(whole_data)
whole_data = whole_data.replace(np.nan, whole_data.mean()).head(5)
from sklearn.model_selection import train_test_split



X = whole_data[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',

                 'YearRemodAdd','MasVnrArea','BsmtFinSF1','WoodDeckSF','OpenPorchSF','EnclosedPorch']]

y = whole_data[['SaleCondition']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

X_std = sc.transform(X)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')

knn.fit(X_train_std, y_train)
print(metrics.classification_report(y_test, knn.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, knn.predict(X_test_std),labels=['Normal','Abnorml']))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train_std, y_train)
print(metrics.classification_report(y_test, gnb.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, gnb.predict(X_test_std),labels=['Normal','Abnorml']))
# Scikit-Learn 官網作圖函式

print(__doc__)



import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(10,6))  #調整作圖大小

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt