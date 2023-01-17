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
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from sklearn.linear_model import LogisticRegression

import datetime
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    

    labels = [0,1]

    # representing A in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    #plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    #plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    #plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
train.shape
train.head()
analyse = train.describe().T
# let's see how many features have same value till 75 percentile ?

# last row is TARGET, so don't count it

test_1 = analyse[analyse['min']==analyse['75%']].iloc[:-1]

print('Total features which have same value till 75 percentile is {0} out of {1}'.format(test_1.shape[0], train.shape[1]-2))
# drop the duplicated columns

train = train.T.drop_duplicates().T

print('after dropping duplicated columns, total number of features remain = ', train.shape[1])
train.head()
# check the columns and the total number of unique values

for col in train.columns:

    if(col=='TARGET'):

        continue

    print(col, ' : ', len(train[col].unique()))
# as we can see there are many columns only contain two distinct values

# let's check how many are there



take_col = []

for col in train.columns:

    if(col=='TARGET'):

        continue

    l = len(train[col].unique())

    if(l<=2):

        take_col.append([col, len(train[col].unique())])

        

print('Total number of columns contain only two or less unique value = ', len(take_col))
# let's check whether these columns combinely give any relevant information by univariate analysis

# we will apply logistic regression model on these features and compare with random model

# if logistic regression will give better result than we can say these features gives some good information

take_col = [x[0] for x in take_col]

x_train, x_test, y_train, y_test=train_test_split(train[take_col], train['TARGET'], test_size=0.3) #splitting the data
# total datapoints in y_test

y_test.value_counts()
# we need to generate 9 numbers and the sum of numbers should be 1

# one solution is to genarate 9 numbers and divide each of the numbers by their sum

# ref: https://stackoverflow.com/a/18662466/4084039

train_data_len = x_train.shape[0]

test_data_len = x_test.shape[0]



# we create a output array that has exactly same size as the CV data

train_predicted_y = np.zeros((train_data_len,2))

for i in range(train_data_len):

    rand_probs = np.random.rand(1,2)

    train_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Cross Validation Data using Random Model",log_loss(y_train,train_predicted_y, eps=1e-15))





# Test-Set error.

#we create a output array that has exactly same as the test data

test_predicted_y = np.zeros((test_data_len,2))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,2)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))



predicted_y =np.argmax(test_predicted_y, axis=1)

plot_confusion_matrix(y_test, predicted_y)
tscv=TimeSeriesSplit(n_splits=10)

penalty=['l2']

param={'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10], 'penalty':penalty}

clf=LogisticRegression()

clf=GridSearchCV(estimator=clf, param_grid=param, cv=tscv,  n_jobs=1, verbose=2)
start=datetime.datetime.now()

print(start)

clf.fit(x_train,y_train)

#pickle.dump(clf,open('bow_unigram.p','wb'))

end=datetime.datetime.now()

print('duration = ',(end-start))

c=clf.best_estimator_.get_params()['C']

penalty=clf.best_estimator_.get_params()['penalty']

print('best C=',c)

print('best penalty=',penalty)
y_pred=clf.predict(x_test)

plot_confusion_matrix(y_test, y_pred)
train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')

test = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')
def drop_duplicated_columns(train):

    train = train.T.drop_duplicates().T

    #test = test.T.drop_duplicates().T

    return (train)



def drop_duplicated_rows(train, test):

    train = train.drop_duplicates()

    test = test.drop_duplicates()

    return (train, test)



def drop_sparse_columns(train, test):

    take_col = []

    for col in train.columns:

        if(col=='TARGET'):

            continue

        l = len(train[col].unique())

        if(l<=2):

            take_col.append(col)

        

    train.drop(take_col, axis=1, inplace=True)

    col = train.columns.to_list()

    col.remove('TARGET')

    test = test[col]

    

    return train, test
train = drop_duplicated_columns(train)

train, test = drop_duplicated_rows(train, test)

train, test = drop_sparse_columns(train, test)
# we will see the columns and their percentile

for c in train.columns:

    percentile = []

    x = train[c].values

    for i in range(101):

        percentile.append(str(np.percentile(x, i)))

    print(c)

    print(','.join(percentile))

    print('***************************')
def unwanted_columns(train):

    remove_col = []

    for c in train.columns:

        percentile = []

        x = train[c].values

        if((np.percentile(x, 99.9)==0) & (min(x)>=-1)):

            remove_col.append(c)

    train.drop(remove_col, axis=1, inplace=True)    

    return train
train = unwanted_columns(train)

train.reset_index(drop=True, inplace=True)

min_var3 = train.loc[train['var3']>(-999999), 'var3'].min()

train.loc[train['var3']==(-999999), 'var3'] = min_var3
# on test data

col = train.columns.to_list()

col.remove('TARGET')
test = test[col]

test.loc[test['var3']==(-999999), 'var3'] = min_var3
test
# lets do some feature engineering
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
col = train.columns.to_list()

col.remove('TARGET')

col.remove("ID")
pca = PCA(n_components=2)

x_train_pca = pca.fit_transform(normalize(train[col]))

x_test_pca = pca.transform(normalize(test[col]))
train.insert(1, 'PCA1', x_train_pca[:, 0])

train.insert(1, 'PCA2', x_train_pca[:, 1])

test.insert(1, 'PCA1', x_test_pca[:, 0])

test.insert(1, 'PCA2', x_test_pca[:, 1])
from sklearn.cluster import KMeans

from tqdm import tqdm
for ncl in tqdm(range(2,11)):

    cls = KMeans(n_clusters=ncl)

    cls.fit_predict(train[col].values)

    train['kmeans_cluster'+str(ncl)] = cls.predict(train[col].values)

    test['kmeans_cluster'+str(ncl)] = cls.predict(test[col].values)

    #flist_kmeans.append('kmeans_cluster'+str(ncl))

col = train.columns.to_list()

col.remove('TARGET')

col.remove('ID')