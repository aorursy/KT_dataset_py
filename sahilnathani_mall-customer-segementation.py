import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns  
df = pd.read_csv('../input/Mall_Customers.csv', 

                 encoding='ISO-8859-1')
df.columns
df['Annual Income'] = [x*1000 for x in df['Annual Income (k$)']]

df = df.drop(['Annual Income (k$)'], axis=1)
def box_plots(variable):

    cols = ['Annual Income', 'Spending Score (1-100)', 'Age']

    i=0

    plt.figure(figsize=(30, 20))

    for each in cols:

        i+=1

        plt.subplot(1, 3, i)

        sns.boxplot(x=variable, y=each, data=df)        
box_plots('Gender')
#Annual Income is Left-Skewed.

sns.countplot(x='Gender', data=df)
from sklearn.preprocessing import MinMaxScaler

cols = ['Age', 'Annual Income', 'Spending Score (1-100)']



mss = MinMaxScaler()

def scaler(value):

    x = np.array(df[value]).reshape(-1, 1)

    return mss.fit_transform(x)



for each in cols:

    df[each] = scaler(each)
df = df.drop(['CustomerID'], axis=1)

from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

df['Gender'] = lb.fit_transform(df['Gender'])
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



x = np.array(df.drop(['Gender'], axis=1))

y = np.array(df['Gender'])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)



from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.neural_network import MLPClassifier



names = ['knn', 'scv', 'rfc', 'abc', 'dtc', 'gnb', 'mnb', 'mlp']



classifiers = [KNeighborsClassifier(3), SVC(C=0.009, gamma=3), RandomForestClassifier(n_estimators=50), AdaBoostClassifier(), 

               DecisionTreeClassifier(), GaussianNB(), MultinomialNB(), MLPClassifier()]



for n, c in zip(names, classifiers):

    c.fit(x_train, y_train)

    score = c.score(x_test, y_test)



clf = [SVR(), RandomForestRegressor(), BaggingRegressor(), AdaBoostRegressor(), DecisionTreeRegressor(), LinearRegression()]

name = ['svr', 'randomForest', 'bagging', 'adaBoost', 'decisionTree', 'LinearRegression']



for nam, clf in zip(names, classifiers):

    clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)

    print('Score achieved by ', nam, ' is ', score*100)
x = np.array(df)

from sklearn.cluster import KMeans, MeanShift



kmeans = KMeans(n_clusters=4)

kmeans.fit(x)

y = kmeans.labels_

kmeans.cluster_centers_