import numpy as np

import math

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import mpl_toolkits

import heapq

from mpl_toolkits.basemap import Basemap

from sklearn.linear_model import LogisticRegression

from sklearn import decomposition

from sklearn import svm
SF = pd.read_csv('../input/dratini_here.csv').drop(["Unnamed: 0"],1)

SF.head()
spawns = SF.drop(["num","lat","lng","distance","147"],1)
spawns.head()
percentage_appearance =  (spawns[spawns['dratini_here'] == 1].sum(axis=0) / spawns.sum(axis = 0))

percentage_appearance = percentage_appearance[~np.isnan(percentage_appearance)].drop(["dratini_here"])
percentage_appearance.sort_values(ascending=False).head()
train =SF.sample(frac=0.6)

rest = SF.drop(train.index)

cross = rest.sample(frac=0.5)

test = rest.drop(cross.index)

len(train),len(cross),len(test)
X_all = SF.drop(["num","lat","lng","distance","dratini_here","147"],1)

y_all = SF.dratini_here

X_train = train.drop(["num","lat","lng","distance","dratini_here","147"],1)

y_train = train.dratini_here

X_cross = cross.drop(["num","lat","lng","distance","dratini_here","147"],1)

y_cross = cross.dratini_here

X_test = test.drop(["num","lat","lng","distance","dratini_here","147"],1)

y_test = test.dratini_here
#a function to perform logistic regression given the solver and list of C values to try

#return a table of (train score, cv score) for each value of C



def LRcompare(solver,C,X_train,y_train,X_cross,y_cross):

    compare_table = pd.DataFrame(index=[solver], columns=C)

    for index, row in compare_table.iterrows():

        for C_value in C:

            compare_table[C_value].loc[index] = compute_scores(solver,C_value,X_train,y_train,X_cross,y_cross)

    

    return compare_table



def compute_scores(solver_choice,C_value,X_train,y_train,X_cross,y_cross):

    model = LogisticRegression(solver = solver_choice, C = C_value)

    model.fit(X_train,y_train)

    train_score = model.score(X_train,y_train)

    cv_score = model.score(X_cross,y_cross)

    

    return (train_score,cv_score)
C = [.001,.01,.1,.5,1,2,10,100,1000]

solver = "liblinear" #try different ones 

LR = LRcompare(solver,C,X_train,y_train,X_cross,y_cross)
LR
model = LogisticRegression(C=10)

model.fit(X_all,y_all)
coeffs = model.coef_[0]
heapq.nlargest(5,coeffs)
coeffs.argsort()[-5:][::-1]
list(X_all)[128], list(X_all)[23], list(X_all)[79], list(X_all)[88], list(X_all)[117] 
#a function to perform an SVM learning algorithm given a list of kernels and C values to try

#return a table of (train score, cv score) for each value of C and each kernel



def SVMcompare(kernels,C,X_train,y_train,X_cross,y_cross):

    compare_table = pd.DataFrame(index=kernels, columns=C)

    for index, row in compare_table.iterrows():

        for C_value in C:

            compare_table[C_value].loc[index] = compute_scores_SVM(index,C_value,X_train,y_train,X_cross,y_cross)

    

    return compare_table



def compute_scores_SVM(kernel_choice,C_value,X_train,y_train,X_cross,y_cross):

    model = svm.SVC(kernel = kernel_choice, C = C_value)

    model.fit(X_train,y_train)

    train_score = model.score(X_train,y_train)

    cv_score = model.score(X_cross,y_cross)

    

    return (train_score,cv_score)
#kernels = ["linear","poly","rbf","sigmoid"]  

#SVM = SVMcompare(kernels,C,X_train,y_train,X_cross,y_cross) 
#SVM
#use the best model 

model = svm.SVC(kernel = "rbf", C = 10)

model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)

cv_score = model.score(X_cross,y_cross)

test_score = model.score(X_test,y_test)
test_score
map_info = SF[['lat','lng']]

scores_all = model.decision_function(X_all)

map_info = map_info.assign(scores=scores_all)

map_info.head()
map = Basemap(projection='merc', lat_0 = 37.6, lon_0 = -122.5,

    resolution = 'h', area_thresh = 0.05,

    llcrnrlon=-122.8, llcrnrlat=37.2,

    urcrnrlon=-121.8, urcrnrlat=38.1)

 

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()



lons = map_info['lng'].tolist()

lats = map_info['lat'].tolist()

vals = map_info['scores'].tolist()

x,y = map(lons, lats)



map.scatter(x,y,c=vals,marker=".",cmap='afmhot', lw =0)

plt.colorbar() 

plt.show()