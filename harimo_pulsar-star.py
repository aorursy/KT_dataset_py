# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import cm

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the Data

star = pd.read_csv('../input/pulsar_stars.csv')
# Head of the Data

star.head()
# Info

star.info()
# Describe

star.describe()
# Columns

star.columns
#-----Lets do some EDA--------

#Check the correlation between the various features

plt.figure(figsize=(10,6), dpi =100)

sns.heatmap(star.corr(), cmap="YlGnBu", annot=True)
plt.figure(figsize=(10,7), dpi =100)

sns.scatterplot(x=' Mean of the integrated profile', y= ' Skewness of the integrated profile',hue='target_class', data = star, palette='coolwarm',

markers=['o','v'])
plt.figure(figsize=(10,10), dpi =100)

sns.pairplot(data=star)
plt.figure(figsize=(100,6), dpi =100)

for col in star.columns.drop('target_class'):

    plt.figure(figsize=(10,6), dpi =100)

    #star[col].plot.kde()

    sns.distplot(star[col], bins=1000)
#-------------Machine learning-----------------

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_absolute_error,mean_squared_error 

#-- Models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
#Scaling the dataframe

scaler = MinMaxScaler()

scaler.fit(star.drop(['target_class'], axis = 1))

star_scaler =scaler.transform(star.drop(['target_class'], axis = 1))
#Train-test-split

X = star_scaler

y = star['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
#DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

dtree_pred = dtree.predict(X_test)

print(confusion_matrix(y_test,dtree_pred))

print(classification_report(y_test,dtree_pred))
from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydot 



features = list(star.columns[:-1])

features

dot_data = StringIO()  

export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png())  
#RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))
#LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)

log_pred = log.predict(X_test)

print(confusion_matrix(y_test,log_pred))

print(classification_report(y_test,log_pred))
#KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(X_test)

kmeans.cluster_centers_

kmeans.labels_
#KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

print(confusion_matrix(y_test,knn_pred))

print(classification_report(y_test,knn_pred))



error_rate = []

# Will take some time

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#Retrain with chosen K value

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

print(confusion_matrix(y_test,knn_pred))

print(classification_report(y_test,knn_pred))
#SVC

svc = SVC()

svc.fit(X_train,y_train)

svc_pred = svc.predict(X_test)

print(confusion_matrix(y_test,svc_pred))

print(classification_report(y_test,svc_pred))
#Grid Search

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)

print(grid.best_params_)

print(grid.best_estimator_)

grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test,grid_pred))

print(classification_report(y_test,grid_pred))
#Comparison of Regression models

fig,axes = plt.subplots(1,4,figsize=(17,4))

x = ['DTree','RFC','LogReg','KnnCl','SVC','Grid']



#Mean Absolute Error

axes[0].set_title("Mean Absolute Error")

y_mae = np.array([mean_absolute_error(y_test,dtree_pred),mean_absolute_error(y_test,rfc_pred),mean_absolute_error(y_test,log_pred),mean_absolute_error(y_test,knn_pred),mean_absolute_error(y_test,svc_pred),mean_absolute_error(y_test,grid_pred)])

axes[0].bar(x,y_mae)



#Mean Squared Error

axes[1].set_title("Mean Squared Error")

y_mse = np.array([mean_squared_error(y_test,dtree_pred),mean_squared_error(y_test,rfc_pred),mean_squared_error(y_test,log_pred),mean_squared_error(y_test,knn_pred),mean_squared_error(y_test,svc_pred),mean_squared_error(y_test,grid_pred)])

axes[1].bar(x,y_mse)



#Root Mean Squared Error

axes[2].set_title("Root Mean Squared Error")

y_rmse = np.array([np.sqrt(mean_squared_error(y_test,dtree_pred)),np.sqrt(mean_squared_error(y_test,rfc_pred)),np.sqrt(mean_squared_error(y_test,log_pred)),np.sqrt(mean_squared_error(y_test,knn_pred)),np.sqrt(mean_squared_error(y_test,svc_pred)),np.sqrt(mean_squared_error(y_test,grid_pred))])

axes[2].bar(x,y_rmse)



#Accuracy

axes[3].set_title("Accuracy Score")

y_acc = np.array([accuracy_score(y_test,dtree_pred)*100,accuracy_score(y_test,rfc_pred)*100,accuracy_score(y_test,log_pred)*100,accuracy_score(y_test,knn_pred)*100,accuracy_score(y_test,svc_pred)*100,accuracy_score(y_test,grid_pred)*100])

axes[3].bar(x,y_acc)
fig,axes = plt.subplots(2,3,figsize=(17,10))

model = ['DTree','RFC','LogReg','KnnCl','SVC','Grid']

i = 0

for pred in model:

    if pred == 'DTree':

        cm = confusion_matrix(y_test,dtree_pred)

    elif pred == 'RFC':

        cm = confusion_matrix(y_test,rfc_pred)

    elif pred == 'LogReg':

        cm = confusion_matrix(y_test,log_pred)

    elif pred == 'KnnCl':

        cm = confusion_matrix(y_test,knn_pred)

    elif pred == 'SVC':

        cm = confusion_matrix(y_test,svc_pred)

    elif pred == 'Grid':

        cm = confusion_matrix(y_test,grid_pred)

    c_mtx = axes.flatten()[i].matshow(cm)

    axes.flatten()[i].set_title(pred)

    axes.flatten()[i].set_ylabel('True Target')

    axes.flatten()[i].set_xlabel('Predicted Target')

    thresh = cm.max() / 2

    normalize = 0

    #print(range(cm.shape[0]), range(cm.shape[1]))

    for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        axes.flatten()[i].text(j, k, "{:,}".format(cm[k, j]),

                               horizontalalignment="center",

                               size = 30,

                               color="black" if cm[k, j] > thresh else "red")

    i = i+1

#ColorBar

fig.subplots_adjust(right=0.8)

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

fig.colorbar(c_mtx, cax=cbar_ax)
def plot_classification_report(cr,axes, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):



    lines = cr.split('\n')

    classes = []

    plotMat = []



    for line in lines[2 : 4]:

        #print(line)

        t = line.split()

        #print(t[0])

        classes.append(t[0])

        v = [float(x) for x in t[1: len(t) - 1]]

        #print(v)

        plotMat.append(v)

    

    axes.imshow(plotMat, interpolation='nearest', cmap=cmap)

    axes.set_title(title)

    #axes.colorbar()

    x_tick_marks = np.arange(3)

    #print(x_tick_marks)

    y_tick_marks = np.arange(len(classes))

    axes.set_xticklabels(['','precision','', 'recall','', 'f1-score',''])

    axes.set_yticklabels(['','',0,'','','',1,'',''])

    #axes.set_yticklabels(classes)

    #axes.tight_layout()

    axes.set_ylabel('Classes')

    axes.set_xlabel('Measures')

    for k, j in itertools.product(range(0,2), range(0,3)):

        axes.text(j, k, "{:,}".format(plotMat[k][j]),

                               horizontalalignment="center",

                               size = 20,

                               color="red")
fig,axes = plt.subplots(2,3,figsize=(17,10))

model = ['DTree','RFC','LogReg','KnnCl','SVC','Grid']

i = 0

for pred in model:

    if pred == 'DTree':

        plot_classification_report(classification_report(y_test,dtree_pred),axes = axes.flatten()[i],title = pred)

        

    elif pred == 'RFC':

        plot_classification_report(classification_report(y_test,rfc_pred),axes = axes.flatten()[i],title = pred)

        

    elif pred == 'LogReg':

        plot_classification_report(classification_report(y_test,log_pred),axes = axes.flatten()[i],title = pred)

        

    elif pred == 'KnnCl':

        plot_classification_report(classification_report(y_test,knn_pred),axes = axes.flatten()[i],title = pred)

        

    elif pred == 'SVC':

        plot_classification_report(classification_report(y_test,svc_pred),axes = axes.flatten()[i],title = pred)

        

    elif pred == 'Grid':

        plot_classification_report(classification_report(y_test,grid_pred),axes = axes.flatten()[i],title = pred)

        

    i = i+1