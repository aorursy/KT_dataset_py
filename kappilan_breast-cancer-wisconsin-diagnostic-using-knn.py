# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn import metrics



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns',None)

bc_df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
bc_df.head(10)
bc_df.dtypes # Except for the diagnosis all the other columns are float values
bc_df.drop('Unnamed: 32',axis=1,inplace=True)
bc_df.shape
bc_df.isnull().sum() 
sns.pairplot(bc_df,hue='diagnosis')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

bc_df['diagnosis'] = le.fit_transform(bc_df['diagnosis']) 
corr_mat = bc_df.corr()
corr_mat['diagnosis'].sort_values(ascending=False) # Getting the highly correlated features with the target column
# Getting the columns that are having multi collinearity

# Creating a dataframe with correlated column, the correlation value and the source column to which it is correlated

# Filtering only those that are correlated more than 96%

multi_col_df = pd.DataFrame(columns=['corr_col','corr_val','source_col'])

for i in corr_mat:

    temp_df = pd.DataFrame(corr_mat[corr_mat[i]>0.96][i])

    temp_df = temp_df.reset_index()

    temp_df['source_col'] = i

    temp_df.columns = ['corr_col','corr_val','source_col']

    multi_col_df = pd.concat((multi_col_df,temp_df),axis=0)
multi_col_df[multi_col_df['corr_val']!=1]
# Lsting the columns with their correlation value with the target columnn in descending order

corr_mat['diagnosis'].sort_values(ascending=False)
X = bc_df.drop(['id','diagnosis'],axis=1)

y = bc_df['diagnosis']
X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.20,random_state=1234)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=0.30,random_state=1234)
X_train_z = zscore(X_train)  



X_val_z = zscore(X_val)



X_test_z = zscore(X_test)
knn_clfr = KNeighborsClassifier(n_neighbors=7,weights='distance')
knn_clfr.fit(X_train_z,y_train)
y_predict = knn_clfr.predict(X_val_z)

print(knn_clfr.score(X_train_z, y_train))

print(knn_clfr.score(X_val_z, y_val))

print(metrics.classification_report(y_val, y_predict))

print(metrics.confusion_matrix(y_val, y_predict))
y_predict = knn_clfr.predict(X_test_z)

print(knn_clfr.score(X_test_z, y_test))

print(metrics.classification_report(y_test, y_predict))

print(metrics.confusion_matrix(y_test, y_predict))
bc_df_new =  bc_df.drop(['radius_mean'], axis=1)

bc_df_new =  bc_df.drop(['perimeter_mean'], axis=1)

bc_df_new =  bc_df.drop(['radius_worst'], axis=1)

bc_df_new =  bc_df.drop(['area_worst'], axis=1)

bc_df_new =  bc_df.drop(['perimeter_se'], axis=1)
X = bc_df_new.drop(['id','diagnosis'],axis=1)

y = bc_df_new['diagnosis']
X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.20,random_state=1234)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=0.30,random_state=1234)
X_train_z = zscore(X_train)  



X_val_z = zscore(X_val)



X_test_z = zscore(X_test)
knn_clfr = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn_clfr.fit(X_train_z,y_train)
y_predict = knn_clfr.predict(X_val_z)

print(knn_clfr.score(X_train_z, y_train))

print(knn_clfr.score(X_val_z, y_val))

print(metrics.classification_report(y_val, y_predict))

print(metrics.confusion_matrix(y_val, y_predict))
y_predict = knn_clfr.predict(X_test_z)

print(knn_clfr.score(X_test_z, y_test))

print(metrics.classification_report(y_test, y_predict))

print(metrics.confusion_matrix(y_test, y_predict))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# creating odd list of K for KNN

myList = list(range(1,50))





# empty list that will hold cv scores

cv_scores = []

k_neighbors = []



# perform 10-fold cross validation

for k in myList:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train_z, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    k_neighbors.append(k)





MSE = [1 - x for x in cv_scores]

min(MSE)

MSE.index(min(MSE))

best_k = myList[MSE.index(min(MSE))]

print ("The optimal number of neighbors is %d" % best_k)
%matplotlib inline 

import matplotlib.pyplot as plt



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 18

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size



plt.xlim(0,25)





# plot misclassification error vs k

plt.plot(k_neighbors, MSE)







plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()