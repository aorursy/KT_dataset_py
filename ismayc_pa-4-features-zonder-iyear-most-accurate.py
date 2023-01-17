# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# this will remove warnings messages

import warnings

warnings.filterwarnings('ignore')



# import library for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()
data1 = data1[pd.notnull(data1['nkill'])]





#data = data[pd.notnull(data['iyear'])]

data1 = data1[pd.notnull(data1['imonth'])]

#data = data[pd.notnull(data['country'])]

#data = data[pd.notnull(data['region'])]

#data = data[pd.notnull(data['vicinity'])]

#data = data[pd.notnull(data['success'])]

data1 = data1[pd.notnull(data1['suicide'])]

#data = data[pd.notnull(data['attacktype1'])]

#data = data[pd.notnull(data['targtype1'])]

#data = data[pd.notnull(data['individual'])]

#data = data[pd.notnull(data['nperps'])]

#data = data[pd.notnull(data['claimed'])]

#data = data[pd.notnull(data['weaptype1'])]

data1 = data1[pd.notnull(data1['nkillter'])]

data1 = data1[pd.notnull(data1['nwound'])]

#data = data[pd.notnull(data['boolean_nkill'])]



print(data.nkill.count())



total_nkill = data.nkill.sum()

avg_kill = total_nkill/data.nkill.count()

print(avg_kill)



data["avg_nkill"] = avg_kill

data["boolean_nkill"] = data["nkill"] > data["avg_nkill"]

col_names = data.columns.tolist()

print("Column names: ")

print(col_names)
data.index = pd.RangeIndex(len(data.index))

#isolate target data

y = data['boolean_nkill']

y = y*1



# We don't need these columns

to_drop = ['iyear','country', 'region', 'vicinity', 'success', 'attacktype1', 'targtype1','individual', 'nperps', 'claimed', 'weaptype1', 'nwound',  'nkill', 'boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data_feat_space = data.drop(to_drop,axis=1)







# Pull out features for future use

features = data_feat_space.columns

X = data_feat_space.as_matrix().astype(np.float)



print(data)
# Scaling Features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)



print("Feature space holds %d observations and %d features" % X.shape)

print("Unique target labels:", np.unique(y))
from sklearn.cross_validation import KFold



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

from sklearn.svm       import SVC

from sklearn.ensemble  import RandomForestClassifier as RF

from sklearn.neighbors import KNeighborsClassifier as KNN



def accuracy(y_true,y_pred):

    return np.sum(y_true == y_pred)/float(y_true.shape[0])



print("Accuracy Support vector machines: %.3f" % accuracy(y, run_cv(X,y,SVC)))

print("Random forest: %.3f" % accuracy(y, run_cv(X,y,RF)))

print("K-nearest-neighbors: %.3f" % accuracy(y, run_cv(X,y,KNN)))
from sklearn.metrics import confusion_matrix

import itertools



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

    

y = np.array(y)

class_names = np.unique(y)



confusion_matrices = [

    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),

    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),

    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),

]



plt.figure(figsize=(20,10))

for idx in range(3):

    plt.subplot(1,3,idx+1)

    plot_confusion_matrix(confusion_matrices[idx][1],["False","True"],title=confusion_matrices[idx][0])