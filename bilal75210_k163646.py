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

from sklearn import feature_selection

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn import metrics

import matplotlib.pyplot as plt 

from sklearn.model_selection import cross_val_score
col_names = ["letter", "x_box","y_box","width","hight","onpix","x_bar","y_bar","x2bar","y2bar","xybar","x2ybr","xy2br","x_ege","xegvy","y_ege","yegvx"]

dataframe = pd.read_csv('/kaggle/input/dataset1/letter-recognition.data',sep=',',names=col_names)

dataframe.head()
dataframe.info()
label = dataframe['letter']



features = dataframe.drop(['letter'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
x_train = X_train

x_test = X_test
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()



##After 80% accuracy, Model is going to overfit.as it is not increasing model accuracy neither it is performing well
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()

for x in range(10):

    kfold = KFold(n_splits=10)

    model = DecisionTreeClassifier()

    results = cross_val_score(model, features, label, cv=kfold)

    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
col_names = ["tls","tms","trs","mls","mms","mrs","bls","bms","brs","label"]

dataframe = pd.read_csv('/kaggle/input/dataset2/tic-tac-toe.data',sep=',',names=col_names)

dataframe.head()
dataframe.info()
label = dataframe['label']



features = dataframe.drop(['label'], axis = 1)

features = pd.get_dummies(features)

features.head()
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()



##After 90% accuracy, Model is going to overfit.as it is not increasing model accuracy neither it is performing well
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()
for x in range(10):

    kfold = KFold(n_splits=10)

    model = DecisionTreeClassifier()

    results = cross_val_score(model, features, label, cv=kfold)

    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
col_names = ["Alcohol", "Malic","Ash","Alcalinity","Magnesium","phenols","Flavanoids","Nonflavanoid","Proanthocyanins","Color","Hue","wines","Proline"]

dataframe = pd.read_csv('/kaggle/input/dataset3/wine.data',sep=',',names=col_names)

dataframe.head()
label = dataframe['Proline']



features = dataframe.drop(['Proline'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()



##After 90% accuracy, Model is going to overfit.as it is not increasing model accuracy neither it is performing well
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()
for x in range(10):

    kfold = KFold(n_splits=10)

    model = DecisionTreeClassifier()

    results = cross_val_score(model, features, label, cv=kfold)

    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
col_names = ["HEIGHT", "LENGTH","AREA","ECCEN","P_BLACK","P_AND","MEAN_TR","BLACKPIX","BLACKAND","WB_TRANS"]

dataframe = pd.read_csv('/kaggle/input/dataset4/page-blocks.data',sep=r'\s+',names=col_names)

dataframe.head()
label = dataframe['HEIGHT']



features = dataframe.drop(['HEIGHT'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()



##After 90% accuracy, Model is going to overfit.as it is not increasing model accuracy neither it is performing well
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,100):

 dtree = DecisionTreeClassifier(ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.015)

 dtree.fit(x_train, y_train)

 pred = dtree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 

 'acc_entropy':pd.Series(acc_entropy),

 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters

plt.plot('max_depth','acc_gini', data=d, label='gini')

plt.plot('max_depth','acc_entropy', data=d, label='entropy')

plt.xlabel('max_depth')

plt.ylabel('accuracy')

plt.legend()
for x in range(10):

    kfold = KFold(n_splits=10)

    model = DecisionTreeClassifier()

    results = cross_val_score(model, features, label, cv=kfold)

    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))