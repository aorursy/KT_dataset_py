import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



from sklearn.cross_validation import train_test_split
data = pd.read_csv("../input/diabetes.csv")



data.head()
data.describe()
# convert all 0's to Boolean values

# sum the Trues along the columns

(data == 0).sum(axis=0)
corr = data.corr() # compute pairwise correlation of columns

_, ax = plt.subplots(figsize=(12, 10)) # returns a pyploy figure object and array of axes

cmap = sns.diverging_palette( 255, 10, as_cmap=True ) # return matplot lib colormap 

_ = sns.heatmap(

    corr, # use computed correlations

    cmap = cmap, # use our colormap

    square=True, # set axes to be same size

    cbar_kws={ 'shrink' : .9 }, # shrink color bar by 9/10ths

   # ax=ax, # axes in which to draw the plot

    annot = True, # put the correlation values in the plot

    annot_kws = { 'fontsize' : 14 } # 12 point font

)
# replace zeros of each column with their column's median

for col in data.columns[1:6]:

    data[col] = data[col].replace(0, data[col].median())

data.head()
# from Glucose to BMI, create 20 bins for the range of data

for col in data.columns[1:8]:

    data[col + "Band"] = pd.cut(data[col], 20)

data.head()
from operator import itemgetter 



for col in data.columns[1:8]: # Glucose to Age



    # turn the unique values of each band into a list

    # of lists of floats

    band_range = list()

    for band in data[col + "Band"].unique(): 

        band = band.strip('(] ').replace(' ', '').split(',')

        band = list(map(float,band))

        band_range.append(band)

        

    # sort the list of lists based on the first member of each list

    band_range = sorted(band_range, key=itemgetter(0))

    

    # find where our data points fall into a certain band

    # and assign them the band's respective index

    for i, b in enumerate(band_range):

        data.loc[(data[col] > b[0]) & (data[col] <= b[1]), col] = i

data.head()
# get rid of bands

for col in data.columns[9:]:

    data = data.drop(str(col), axis=1)

data.head()
target = data.Outcome

data = data.drop('Outcome', axis=1)

data.head()
X_train, X_test, Y_train, Y_test = train_test_split( data, target , train_size = .7 )



print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print(accuracy_score(Y_test, Y_pred))

# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

print(accuracy_score(Y_test, Y_pred))



# K-nearest Neighbors



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

print(accuracy_score(Y_test, Y_pred))





# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

print(accuracy_score(Y_test, Y_pred))





# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

print(accuracy_score(Y_test, Y_pred))





# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

print(accuracy_score(Y_test, Y_pred))



# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

print(accuracy_score(Y_test, Y_pred))



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

print(accuracy_score(Y_test, Y_pred))





# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

print(accuracy_score(Y_test, Y_pred))




