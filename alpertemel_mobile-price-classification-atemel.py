# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")

test = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")
train.head()
train.dtypes
train.isnull().sum()/len(train)
train.describe()
train["price_range"].value_counts()
corr = train.corr().abs()

n_most_correlated = 10

most_correlated_feature = corr["price_range"].sort_values(ascending = False)[:n_most_correlated].drop("price_range")

most_correlated_feature_name = most_correlated_feature.index.values



f, ax = plt.subplots(figsize = (15, 6))

plt.xticks(rotation = "90")

sns.barplot(x = most_correlated_feature_name, y = most_correlated_feature)

plt.title("En fazle kolerasyona sahip değişkenler")
corr2 = train.corr()

f, ax = plt.subplots(figsize = (15, 6))

sns.heatmap(corr2)
most_correlated_feature2 = corr["price_range"].sort_values(ascending = False)[:10]

fair = most_correlated_feature2.index



train2 = pd.DataFrame()



for i in fair:

    train2 = pd.concat([train2, train[i]], axis = 1)



    

sns.pairplot(data = train2, hue = "price_range")
x = train.iloc[:, 0:20]

y = train.iloc[:, 20:21]



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 23)



#datanın train ve test setlerine bölümü.
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(x_train, y_train)

rf_tahmin = rf.predict(x_test)



print("Accuracy Score: ", accuracy_score(y_test, rf_tahmin))
import statsmodels.api as sm

X_1 = sm.add_constant(x)



model = sm.OLS(y,X_1).fit()

model.pvalues



cols2 = list(x.columns)

pmax = 1

while (len(cols2)>0):

    p= []

    X_1 = x[cols2]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols2)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols2.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols2

print(selected_features_BE)

len(selected_features_BE)





train2 = pd.DataFrame()



for i in selected_features_BE:

    train2 = pd.concat([train2, train[i]], axis = 1)

    

train2 = pd.concat([train2, train["price_range"]], axis = 1)



x2 = train2.iloc[:, 0:6]

y2 = train2.iloc[:, 6:7]



X_train, X_test, Y_train, Y_test = train_test_split(x2, y2, test_size =0.33, random_state =34)



rf2 = RandomForestClassifier()

rf2.fit(X_train, Y_train)

rf2_tahmin = rf2.predict(X_test)



print("FEATURE SELECTION İLE ACCURACY SCORE: ",accuracy_score(Y_test, rf2_tahmin))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

knn_tahmin = knn.predict(X_test)



accuracy_score(Y_test, knn_tahmin)
from sklearn.model_selection import GridSearchCV



knn_params = {

        "n_neighbors" : np.arange(2, 10, 1),

        "weights" : ["uniform", "distance"],

        "leaf_size" : (30, 40, 50, 20),

        "p" : (1, 2)

        }



knn_ = KNeighborsClassifier()



knn_grid = GridSearchCV(knn_, knn_params, cv = 10, n_jobs = -1, verbose = 2)

knn_grid.fit(X_train, Y_train)

knn_grid.best_params_



knn_tune = KNeighborsClassifier(leaf_size = 30, n_neighbors = 9, p = 2, weights = "uniform")

knn_tune.fit(X_train, Y_train)

knn_tune_tahmin = knn_tune.predict(X_test)



accuracy_score(Y_test, knn_tune_tahmin)
from sklearn.svm import SVC



svm = SVC(kernel = "linear")

svm.fit(X_train, Y_train)

svm_tahmin = svm.predict(X_test)



accuracy_score(Y_test, svm_tahmin)