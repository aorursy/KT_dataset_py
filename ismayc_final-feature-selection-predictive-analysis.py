# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# this will remove warnings messages

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import KFold

from sklearn.svm       import SVC

from sklearn.ensemble  import RandomForestClassifier as RF

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import confusion_matrix

import itertools



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()



data1 = data

data2 = data

data3 = data

data4 = data
def run_cv(X,y,clf_class,**kwargs):

    # Construct a kfolds object

    kf = KFold(len(y),n_folds=5,shuffle=True)

    y_pred = y.copy()

    # Iterate through folds

    for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]

        y_train = y[train_index]

        # Initialize a classifier with key word arguments

        clf = clf_class(**kwargs)

        clf.fit(X_train,y_train)

        y_pred[test_index] = clf.predict(X_test)

    return y_pred



def accuracy(y_true,y_pred):

    return np.sum(y_true == y_pred)/float(y_true.shape[0])



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Greys,

                          print_confusion_matrix = False):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    fs = 20

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=fs)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=fs)

    plt.yticks(tick_marks, classes, fontsize=fs)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        statement = "Normalized confusion matrix "

    else:

        statement = 'Confusion matrix, without normalization'



    if print_confusion_matrix:

        print(statement)

        print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label', fontsize=fs)

    plt.xlabel('Predicted label', fontsize=fs)
#Preparing the data for usage/creating new variables

data1 = data1[pd.notnull(data1['nkill'])]

#print(data1.nkill.count())



total_nkill1 = data1.nkill.sum()

avg_kill1 = total_nkill1/data1.nkill.count()

#print(avg_kill1)



data1["avg_nkill"] = avg_kill1

data1["boolean_nkill"] = data1["nkill"] > data1["avg_nkill"]



col_names1 = data1.columns.tolist()

#print("Column names: ")

#print(col_names1)



#deleting NaN values

data1 = data1[pd.notnull(data1['imonth'])]

data1 = data1[pd.notnull(data1['country'])]

data1 = data1[pd.notnull(data1['region'])]

data1 = data1[pd.notnull(data1['vicinity'])]

data1 = data1[pd.notnull(data1['success'])]

data1 = data1[pd.notnull(data1['suicide'])]

data1 = data1[pd.notnull(data1['attacktype1'])]

data1 = data1[pd.notnull(data1['targtype1'])]

data1 = data1[pd.notnull(data1['individual'])]

data1 = data1[pd.notnull(data1['nperps'])]

data1 = data1[pd.notnull(data1['claimed'])]

data1 = data1[pd.notnull(data1['weaptype1'])]

data1 = data1[pd.notnull(data1['nkillter'])]

data1 = data1[pd.notnull(data1['nwound'])]



#reindexing the resulting data

data1.index = pd.RangeIndex(len(data1.index))



#isolate target data and converting it into 0/1 values

y1 = data1['boolean_nkill']

y1 = y1*1



# identifying data to drop, then dropping it

to_drop1 = ['iyear','nkill','boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data1_feat_space = data1.drop(to_drop1,axis=1)



# Pull out features for future use

features1 = data1_feat_space.columns

X1 = data1_feat_space.as_matrix().astype(np.float)



print(features1)



# Build a forest and compute the feature importances

forest1 = ExtraTreesClassifier(n_estimators=250,random_state=0)



forest1.fit(X1, y1)

importances1 = forest1.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest1.estimators_],

             axis=0)

indices1 = np.argsort(importances1)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X1.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices1[f], importances1[indices1[f]]))

    

# Plot the feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X1.shape[1]), importances1[indices1],

       color="r", yerr=std[indices1], align="center")

plt.xticks(range(X1.shape[1]), indices1)

plt.xlim([-1, X1.shape[1]])

plt.show()
#prepping the data

data2 = data2[pd.notnull(data2['nkill'])]

data2 = data2[pd.notnull(data2['imonth'])]

data2 = data2[pd.notnull(data2['suicide'])]

data2 = data2[pd.notnull(data2['nkillter'])]

data2 = data2[pd.notnull(data2['nwound'])]



print(data2.nkill.count())



total_nkill2 = data2.nkill.sum()

avg_kill2 = total_nkill2/data2.nkill.count()

print(avg_kill2)



data2["avg_nkill"] = avg_kill2

data2["boolean_nkill"] = data2["nkill"] > data2["avg_nkill"]



col_names2 = data2.columns.tolist()



#reindexing the resulting data

data2.index = pd.RangeIndex(len(data2.index))

#isolate target data

y2 = data2['boolean_nkill']

y2 = y2*1



# identifying data to drop, then dropping it

to_drop2 = ['iyear','country', 'region', 'vicinity', 'success', 'attacktype1', 'targtype1','individual', 'nperps', 'claimed', 'weaptype1', 'nwound',  'nkill', 'boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data2_feat_space = data2.drop(to_drop2,axis=1)



# Pull out features for future use

features2 = data2_feat_space.columns

X2 = data2_feat_space.as_matrix().astype(np.float)



#print(data)



#normalization

scaler2 = StandardScaler()

X2 = scaler2.fit_transform(X2)



print("Feature space holds %d observations and %d features" % X2.shape)

print("Unique target labels:", np.unique(y2))



print("Accuracy Support vector machines: %.3f" % accuracy(y2, run_cv(X2,y2,SVC)))

print("Random forest: %.3f" % accuracy(y2, run_cv(X2,y2,RF)))

print("K-nearest-neighbors: %.3f" % accuracy(y2, run_cv(X2,y2,KNN)))

  

y2 = np.array(y2)

class_names2 = np.unique(y2)



confusion_matrices2 = [

    ( "Support Vector Machines", confusion_matrix(y2,run_cv(X2,y2,SVC)) ),

    ( "Random Forest", confusion_matrix(y2,run_cv(X2,y2,RF)) ),

    ( "K-Nearest-Neighbors", confusion_matrix(y2,run_cv(X2,y2,KNN)) ),

]



plt.figure(figsize=(20,10))

for idx in range(3):

    plt.subplot(1,3,idx+1)

    plot_confusion_matrix(confusion_matrices2[idx][1],["False","True"],title=confusion_matrices2[idx][0])
#deleting NaN values

data3 = data3[pd.notnull(data3['nkill'])]

data3 = data3[pd.notnull(data3['country'])]

data3 = data3[pd.notnull(data3['region'])]

data3 = data3[pd.notnull(data3['vicinity'])]

data3 = data3[pd.notnull(data3['suicide'])]

data3 = data3[pd.notnull(data3['attacktype1'])]

data3 = data3[pd.notnull(data3['targtype1'])]

data3 = data3[pd.notnull(data3['nperps'])]

data3 = data3[pd.notnull(data3['weaptype1'])]



print(data.nkill.count())



#creating new variables

total_nkill3 = data3.nkill.sum()

avg_kill3 = total_nkill3/data3.nkill.count()

print(avg_kill3)



data3["avg_nkill"] = avg_kill3

data3["boolean_nkill"] = data3["nkill"] > data3["avg_nkill"]



#reindexing after removing rows with empty cells

data3.index = pd.RangeIndex(len(data3.index))

#isolate target data

y3 = data3['boolean_nkill']

y3 = y3*1



# We don't need these columns

to_drop3 = ['success', 'individual', 'claimed', 'nkillter', 'nwound', 'iyear','nkill','boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data3_feat_space = data3.drop(to_drop3,axis=1)



# Pull out features for future use

features3 = data3_feat_space.columns

X3 = data3_feat_space.as_matrix().astype(np.float)



print(features3)



# Build a forest and compute the feature importances

forest3 = ExtraTreesClassifier(n_estimators=250,random_state=0)



forest3.fit(X3, y3)

importances3 = forest3.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest3.estimators_],

             axis=0)

indices3 = np.argsort(importances3)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X3.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices3[f], importances3[indices3[f]]))





# Plot the feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X3.shape[1]), importances3[indices3],

       color="r", yerr=std[indices3], align="center")

plt.xticks(range(X3.shape[1]), indices3)

plt.xlim([-1, X3.shape[1]])

plt.show()

#removing NaN values

data4 = data4[pd.notnull(data4['nkill'])]

data4 = data4[pd.notnull(data4['imonth'])]

data4 = data4[pd.notnull(data4['suicide'])]

data4 = data4[pd.notnull(data4['targtype1'])]

data4 = data4[pd.notnull(data4['nperps'])]



print(data4.nkill.count())



#creating new variables

total_nkill4 = data4.nkill.sum()

avg_kill4 = total_nkill4/data4.nkill.count()

print(avg_kill4)



data4["avg_nkill"] = avg_kill4

data4["boolean_nkill"] = data4["nkill"] > data4["avg_nkill"]



#reindexing after removing rows

data4.index = pd.RangeIndex(len(data4.index))



#isolate target data

y4 = data4['boolean_nkill']

y4 = y4*1



# We don't need these columns

to_drop4 = ['nwound','nkillter','iyear','country', 'region', 'vicinity', 'success', 'attacktype1', 'individual', 'claimed', 'weaptype1', 'nwound',  'nkill', 'boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data4_feat_space = data4.drop(to_drop4,axis=1)



# Pull out features for future use

features4 = data4_feat_space.columns

X4 = data4_feat_space.as_matrix().astype(np.float)



# Scaling Features

scaler4 = StandardScaler()

X4 = scaler4.fit_transform(X4)



print("Feature space holds %d observations and %d features" % X4.shape)

print("Unique target labels:", np.unique(y4))



#printing the accuracy

print("Accuracy Support vector machines: %.3f" % accuracy(y4, run_cv(X4,y4,SVC)))

print("Random forest: %.3f" % accuracy(y4, run_cv(X4,y4,RF)))

print("K-nearest-neighbors: %.3f" % accuracy(y4, run_cv(X4,y4,KNN)))



#creating confusion matrices

y4 = np.array(y4)

class_names4 = np.unique(y4)



confusion_matrices4 = [

    ( "Support Vector Machines", confusion_matrix(y4,run_cv(X4,y4,SVC)) ),

    ( "Random Forest", confusion_matrix(y4,run_cv(X4,y4,RF)) ),

    ( "K-Nearest-Neighbors", confusion_matrix(y4,run_cv(X4,y4,KNN)) ),

]



plt.figure(figsize=(20,10))

for idx in range(3):

    plt.subplot(1,3,idx+1)

    plot_confusion_matrix(confusion_matrices4[idx][1],["False","True"],title=confusion_matrices4[idx][0])






