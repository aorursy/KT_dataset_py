import pandas as pd

import numpy as np

import math

import re

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from surprise import Reader, Dataset, SVD, evaluate

from pandas import read_csv

from datetime import datetime



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import ComplementNB



from sklearn import metrics

from sklearn.metrics import accuracy_score



from sklearn.metrics import precision_recall_curve  

from sklearn.metrics import classification_report

def parse(x):

    return datetime.strptime(x, '%Y %m %d %H')



dataset_train = pd.read_csv('../input/traindataset.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)





dataset_train.drop('No', axis=1, inplace=True)

# manually specify column names

dataset_train.columns = ['pm2.5', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

dataset_train.index.name = 'data'

dataset_train['pm2.5'].fillna(0, inplace=True)

dataset_train = dataset_train[24:]

#dataset_train = dataset_train[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pm2.5']]





def parse(x):

    return datetime.strptime(x, '%Y %m %d %H')

dataset_test1 = pd.read_csv('../input/testdataset1.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

dataset_test1.drop('No', axis=1, inplace=True)

# manually specify column names

dataset_test1.columns = ['pm2.5', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

dataset_test1.index.name = 'data'

dataset_test1['pm2.5'].fillna(0, inplace=True)

dataset_test1 = dataset_test1[24:]

#dataset_test1 = dataset_test1[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pm2.5']]



def parse(x):

    return datetime.strptime(x, '%Y %m %d %H')

dataset_test2 = pd.read_csv('../input/testdataset2.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

dataset_test2.drop('No', axis=1, inplace=True)



dataset_test2.columns = ['pm2.5', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

dataset_test2.index.name = 'data'

dataset_test2['pm2.5'].fillna(0, inplace=True)

dataset_test2 = dataset_test2[24:]



combine = [dataset_train, dataset_test1,dataset_test2]



dataset_train.head()
dataset_test2.head()
for dataset in combine:

    dataset['wnd_dir'] = dataset['wnd_dir'].map({'cv':0, 'NE':1, 'SE':2,'NW':3}).astype(int)

dataset_train.head()

#dataset_test1.head()
dataset_train.info()

print('_'*50)

dataset_test1.info()

print('_'*60)

dataset_test2.info()
dataset_train['pm2.5'] = dataset_train['pm2.5'].astype(np.int64)

dataset_train['press'] = dataset_train['press'].astype(np.int64)

dataset_train.info()



d_train = dataset_train



test1 = []

for row in d_train['pm2.5']:

    if row >= 50:

        test1.append('high')

    elif row <50:

        test1.append('low')

    elif row == "NA":

        test1.append("NA")



d_train['pm1'] = test1



print(d_train.head())

d_train = d_train.drop('pm2.5', 1)



#print(d_train.head())
d_train[["pm1", "dew"]].groupby(['pm1'], as_index=False).mean().sort_values(by='pm1', ascending=False)
d_train[["pm1", "snow"]].groupby(['pm1'], as_index=False).mean().sort_values(by='snow', ascending=False)
d_train[["pm1", "wnd_spd"]].groupby(['pm1'], as_index=False).mean().sort_values(by='wnd_spd', ascending=False)
d_train[["pm1", "rain"]].groupby(['pm1'], as_index=False).mean().sort_values(by='rain', ascending=False)
d_train[["pm1", "temp"]].groupby(['pm1'], as_index=False).mean().sort_values(by='temp', ascending=False)
d_train[["pm1", "press"]].groupby(['pm1'], as_index=False).mean().sort_values(by='press', ascending=False)
d_train
dataset_train = d_train

test1 = []

for row in d_train['pm1']:

    if row == 'high':

         test1.append(1)

    elif row == 'low':

         test1.append(0)

    

#print(test1)

#print(len(test1))

#print(len(d_train['pm2.5']))

dataset_train['pm1'] = test1

#print (test1)

print(dataset_train.head())



dataset_test1.describe()
dataset_test1['pm2.5'] = dataset_test1['pm2.5'].astype(np.int64)

dataset_test1['temp'] = dataset_test1['temp'].astype(np.int64)

dataset_test1.info()



#print(dataset_train['pm2.5'])



d_test1 = dataset_test1

#print(d_train)

test1 = []

for row in d_test1['pm2.5']:

    if row >= 50:

        test1.append('high')

    elif row <50:

        test1.append('low')

    elif row == "NA":

        test1.append("NA")

#print(test1)

#print(len(test1))

#print(len(d_train['pm2.5']))

d_test1['pm1'] = test1

#print (test1)

print(d_test1.head())

d_test1 = d_test1.drop('pm2.5', 1)

#df = df.drop('column_name', 1)

print(d_test1.head())
dataset_test1 = d_test1

test1 = []

for row in d_test1['pm1']:

    if row == 'high':

         test1.append(1)

    elif row == 'low':

         test1.append(0)

            

dataset_test1['pm1'] = test1

#print (test1)



print(dataset_test1.head())
dataset_test2['pm2.5'] = dataset_test2['pm2.5'].astype(np.int64)

dataset_test2.info()



#print(dataset_train['pm2.5'])



d_test2 = dataset_test2

#print(d_train)

test2 = []

for row in d_test2['pm2.5']:

    if row >= 50:

        test2.append('high')

    elif row <50:

        test2.append('low')

    elif row == "NA":

        test2.append("NA")

#print(test2)

#print(len(test2))

#print(len(d_train['pm2.5']))

d_test2['pm1'] = test2

#print (test2)

print(d_test2.head())

d_test2 = d_test2.drop('pm2.5', 1)

#df = df.drop('column_name', 1)

print(d_test2.head())





dataset_test2 = d_test2

test2 = []

for row in d_test2['pm1']:

    if row == 'high':

         test2.append(1)

    elif row == 'low':

         test2.append(0)

    

#print(test2)

#print(len(test2))

#print(len(d_train['pm2.5']))

dataset_test2['pm1'] = test2

#print (test2)



print(dataset_test2.head())
print(dataset_train.head())

print(dataset_test1.head())

print(dataset_test2.head())
print(d_train.head())

print(d_test1.head())

print(d_test2.head())
#X = dataset_train.iloc[:,dataset_train.columns != "pm1"]

#y = dataset_train.iloc[:,dataset_train.columns == "pm"]



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

X_train = dataset_train.drop("pm1", axis=1)

y_train = dataset_train["pm1"]

print(X_train.head())

y_train.head()
X_test = dataset_test1.drop("pm1", axis=1).copy()

y_test = dataset_test1["pm1"]

X_train.shape, y_train.shape#, X_test.shape

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train)*100, 3)

acc_decision_tree
X = dataset_test1.iloc[:,dataset_test1.columns != "pm1"]

y = dataset_test1.iloc[:,dataset_test1.columns == "pm1"]
#print(x_train)  

#print(answer)  

#print(y_train.head())  

#print(np.mean(answer == y_train))

#print(answer)





#precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(X_train))  

#print(precision.mean())

#print(recall.mean())

#answer = clf.predict_proba(X_train)

#print(answer)

#a = clf.predict_proba(X)

#print(a)

#print(classification_report(y_train, answer, target_names = ['high', 'low'])) 
from sklearn.metrics import accuracy_score

#1 tree

clf = DecisionTreeClassifier(random_state=69)

clf = clf.fit(X_train, y_train)



#score = cross_val_score(clf, X, y, cv=10).mean()

score_t = clf.score(X_train, y_train)



p_answer = clf.predict(X_train)

print("For Train_Dataset")

print(classification_report(p_answer, y_train, target_names = ['low', 'high']))





#accuracy_score(X_test, y_test)

#print(score)

print(score_t)

print('_'*60)



answer = clf.predict(X)

print("For Dataset_TEST_1")

print(classification_report(answer, y, target_names = ['low', 'high']))

score_t_1 = clf.score(X, y)

print(score_t_1)

print('_'*60)



from sklearn.metrics import accuracy_score

#1 model

clf = DecisionTreeClassifier()

#2 teleport

clf = clf.fit(X_train, y_train)



score_t = clf.score(X_train, y_train) #

score_tr= clf.score(X_test, y_test)

score = cross_val_score(clf,X,y,cv=10)#.mean() 

#cv : int, cross-validation generator or an iterable, optional

score_c = cross_val_score(clf,X,y,cv=10).mean()

#score1 = round(clf.score(X_test,y_test))

#score_recall = clf.metrics.recall_score(X_test, y_test, normalize=False) #y_true, y_pred.round(), normalize=False

#accuracy_score(X_test, y_test)

print('a1 :{}\na2 :{}\ncorss_validation: {}\ncorss_validation_by_mean: {}'.format(score_t,score_tr,score,score_c))

#print ((y_train, clf.predict(X_train)))

#print(score_recall)
import matplotlib.pyplot as plt





tr_gini = []

te_gini = []

#depth 25

for i in range(25):

   

    clf = DecisionTreeClassifier(random_state=69

                                ,max_depth=i+1

                                ,criterion='gini'#gini

                                ,splitter='random' #The strategy used to choose the split at each node.

                                ,min_samples_split =2

                                ,max_leaf_nodes = 7000 # Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

                                ) 

    clf.fit(X_train,y_train)

    score_tr = clf.score(X_train,y_train)

    score_te = cross_val_score(clf,X,y,cv=10).mean()    

    tr_gini.append(score_tr)

    te_gini.append(score_te)



    

fig, (ax1) = plt.subplots(1, figsize=(18, 6))    



ax1.plot(range(1,26),tr_gini,color='r',label='train')

ax1.plot(range(1,26),te_gini,color='blue',label='test')

ax1.set_xticks(range(1,26))

ax1.set_title('Gini')

ax1.legend()



print(max(te_gini))



print('Gini-besed: {}'.format(max(te_gini)))





print(classification_report(answer, y, target_names = ['low', 'high']))

X_2 = dataset_test2.iloc[:,dataset_test2.columns != "pm1"]

y_2 = dataset_test2.iloc[:,dataset_test2.columns == "pm1"]

print(len(y_2))
clf = DecisionTreeClassifier(random_state=69

                            ,criterion='gini'

                            ,max_depth=4

                            ,min_samples_leaf=1 #The minimum number of samples required to be at a leaf node.

                            ,splitter='random' #used to choose the split at each node.

                            ,min_samples_split =2

                            #,max_leaf_nodes = 7000

                            )

clf = clf.fit(X_train, y_train)

#cross_val_score(clf,X_2,y_2,cv=10).mean() 

#score_tr = clf.score(X_train,y_train)

#score_te = cross_val_score(clf,X_2,y_2,cv=10).mean()    

#tr_gini.append(score_tr)

#te_gini.append(score_te)



print("For Dataset_TEST_2")    

answer = clf.predict(X)



#print(x_train)  

print("data after predicting: ")

print(answer)  

print('original data:')

print(y['pm1'].values)

print(clf.score(X, y))

print(classification_report(answer, y, target_names = ['low', 'high']))

#print(y_2.head())  

#print(np.mean(answer == y_2))



#print(answer)

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_train)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 3)

acc_gaussian
gaussian = GaussianNB(var_smoothing=1e-03)

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_train)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian

#print(score_c)







#score1 = round(clf.score(X_test,y_test))

#score_recall = clf.metrics.recall_score(X_test, y_test, normalize=False) #y_true, y_pred.round(), normalize=False

#accuracy_score(X_test, y_test)

#print('pricision:{}\npricision:{}\ncorss_validation: {}\ncorss_validation_by_mean: {}'.format(score_t,score_tr,score,score_c))

#print ((y_train, clf.predict(X_train)))

#print(score_recall)

#score = cross_val_score(gclf,X,y,cv=10)#.mean() 

#cv : int, cross-validation generator or an iterable, optional

#score_c = cross_val_score(gclf,X,y,cv=10).mean()
#1 model

gclf = GaussianNB(var_smoothing=1e-03)

#2 teleport

gclf = gclf.fit(X_train, y_train)



score_t = gclf.score(X_train, y_train) #

score_tr= gclf.score(X, y)







print("For Train_Dataset")

p_n_answer1 = gclf.predict(X_train)

print(classification_report(p_n_answer1, y_train, target_names = ['low', 'high']))



print(score_t)



print('_'*60)

print("For TEST_Dataset_1")

p_n_answer = gclf.predict(X)

print(classification_report(p_n_answer, y, target_names = ['low', 'high']))



print(score_tr)

#1 model

gclf = GaussianNB(var_smoothing=1e-03)

#2 teleport

gclf = gclf.fit(X_train, y_train.values.ravel())



answer = gclf.predict(X_2)

print("For_Dataset_TEST2")

print('_'*60)

print("data after: ")

print(answer)

print('original data:')

print(y_2['pm1'].values)#print (df.loc[df.name == 'george', 'age'].values)





print(gclf.score(X_2, y_2))



print(classification_report(answer, y_2, target_names = ['low', 'high']))



#score_1 = cross_val_score(gclf,X_2,y_2,cv=10).mean()

#for dataset in combine:

#    dataset['wnd_dir'] = dataset['wnd_dir'].map({'cv':0, 'NE':1, 'SE':2,'NW':3}).astype(int)

#dataset_train.head()
