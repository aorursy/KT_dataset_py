# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import numpy.random as rd

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



colors = ["orange", "green", "blue", "violet", "red", "yellow", "magenta", "purple", "cyan"]

rd.seed(12)



def gen_data(n=1):

    set1_x = rd.normal(-3, 1, n*300)

    set1_y = rd.normal(0, 3, n*300)

    label1 = np.zeros(len(set1_x))



    set2_x = rd.normal(0, 1, n*200)

    set2_y = rd.normal(0, 3, n*200)

    label2 = np.ones(len(set2_x))



    set3_x = rd.normal(3, 1, n*100)

    set3_y = rd.normal(2, 1, n*100)

    label3 = np.array([2 for _ in set3_x])



    set4_x = rd.normal(3, 1, n*100)

    set4_y = rd.normal(-2, 1, n*100)

    label4 = np.array([3 for _ in set4_x])



    set_x = np.concatenate((set1_x, set2_x, set3_x, set4_x))

    set_y = np.concatenate((set1_y, set2_y, set3_y, set4_y))



    label = np.concatenate((label1, label2, label3, label4))



    data = np.transpose(np.array([set_x, set_y, label]))

    data = rd.permutation(data)

    return data



def show_data(data, nblabels):

    set_x, set_y = [[] for i in range(nblabels)], [[] for i in range(nblabels)]

    for [x, y, lab] in data:

        k = int(lab)

        set_x[k].append(x)

        set_y[k].append(y)

    plt.figure(figsize=(8, 6))

    for i in range(nblabels):

        plt.scatter(set_x[i], set_y[i], color=colors[i%nblabels])



    plt.xlabel('X1')

    plt.ylabel('X2')

    plt.xlim(-5,5)

    plt.ylim(-4,4)

    plt.show()

    

train_data = gen_data()

show_data(train_data, 4)    
from sklearn import tree



clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.05)

clf = clf.fit(train_data[:,:2], train_data[:,-1])
import graphviz



dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 



dot_data = tree.export_graphviz(clf, out_file=None,

                                feature_names=['X', 'Y'],          

                                class_names=['orange', 'vert', 'bleu', 'violet'],             

                                filled=True, rounded=True,                

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph
from sklearn.ensemble import RandomForestClassifier



rd.seed(12)



clf_y = []

vs = [] 

i = 0

data = gen_data()

while len(vs) < 100:

    if -3<data[i][0]<3:

        vs.append(data[i])

    i += 1

vs = np.array(vs)

for i in range(100):

    ts = gen_data(n=5)

    

    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(ts[:,:2], ts[:,-1])

    

    clf_y.append(clf.predict(vs[:,:2]))

    

clf_y = np.transpose(clf_y)



clf_var = [np.var(val) for val in clf_y]



fig, ax = plt.subplots(figsize=(8, 8))

ax.set_aspect(1)

for (i,var) in enumerate(clf_var):

    circle = plt.Circle((vs[i][0], vs[i][1]), var, color=colors[i%len(colors)])

    ax.add_artist(circle)

    

plt.grid()

plt.xlim(-3,3)

plt.ylim(-3,3)

plt.show()
from sklearn.ensemble import RandomForestClassifier

from scipy.interpolate import CubicSpline



rd.seed(12)

rf_acc = []

vs = [] 

vs = gen_data()

ts = gen_data(n=5)



for i in range(1,151):

    rf = RandomForestClassifier(n_estimators=i)

    rf = rf.fit(ts[:,:2], ts[:,-1])

    rf_acc.append(rf.score(vs[:,:2], vs[:,-1]))

  

x = np.arange(1, 151)

poly = np.polyfit(x, rf_acc,6) 

y = np.poly1d(poly)(x) 





plt.figure(figsize=(8, 6))

plt.plot(x, rf_acc)

plt.plot(x, y, 'r')

plt.xlabel("nombre d'arbres B")

plt.ylabel("score sur l'ensemble de validation")

plt.grid()

plt.show()
from sklearn.ensemble import RandomForestClassifier



rd.seed(12)



rf_y = []

vs = [] 

i = 0

data = gen_data()

while len(vs) < 100:

    if -3<data[i][0]<3:

        vs.append(data[i])

    i += 1

vs = np.array(vs)

for i in range(100):

    ts = gen_data(n=5)

    

    rf = RandomForestClassifier(n_estimators=100)

    rf = rf.fit(ts[:,:2], ts[:,-1])

    

    rf_y.append(rf.predict(vs[:,:2]))

    

rf_y = np.transpose(rf_y)



rf_var = [np.var(val) for val in rf_y]

    

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_aspect(1)

for (i,var) in enumerate(rf_var):

    circle = plt.Circle((vs[i][0], vs[i][1]), var, color=colors[i%len(colors)])

    ax.add_artist(circle)

    

plt.grid()

plt.xlim(-3,3)

plt.ylim(-3,3)

plt.show()
plt.bar(np.arange(0,100),np.array(clf_var)-np.array(rf_var))

plt.show()
plt.hist(np.array(clf_var)-np.array(rf_var), bins=50)

plt.show()
x = np.arange(0, 100)

poly1 = np.polyfit(x, rf_var,6) 

y1 = np.poly1d(poly1)(x) 

poly2 = np.polyfit(x, clf_var,6) 

y2 = np.poly1d(poly2)(x)



plt.figure(figsize=(8, 6))

plt.plot(rf_var, 'g', alpha=1, linewidth=0.5)

plt.plot(clf_var, 'r', alpha=1, linewidth=0.5)

plt.plot(x, y1, 'g', linewidth=3)

plt.plot(x, y2, 'r', linewidth=3)

plt.show()