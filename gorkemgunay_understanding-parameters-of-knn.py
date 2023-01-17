# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data from csv file

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.info()
#Drop id and Unnamed: 32 columns

df.drop(['id','Unnamed: 32'],inplace = True, axis = 1)



#Convert Diagnosis from categorical to numeric

df.diagnosis = df.diagnosis.map({"M":1,"B":0})



#Show final info of dataframe

df.info()
#create X and Y objects

X = df.drop(["diagnosis"],axis = 1)

Y = df.diagnosis.values.reshape(-1,1)



#create test and train data sets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 0,test_size = 0.2)



print("X_train Shape:",X_train.shape)

print("X_test Shape:",X_test.shape)

print("Y_train Shape:",Y_train.shape)

print("Y_test Shape:",Y_test.shape)



result_train = {}

result_test = {}
#This function concatenate test and train scores in a dataframe

def prepare_dataframe(test_score_dict,train_score_dict,key,columns):

    df_test = pd.DataFrame(test_score_dict,index = ["Test Score"])

    df_train = pd.DataFrame(train_score_dict,index = ["Train Score"])

    np_result = np.concatenate([df_test,df_train],axis = 0)

    df_result = pd.DataFrame(np_result)



    df_result.index = ["Test Score","Train Score"]

    df_result.columns = [key + str(c) for c in columns]  

    return df_result
#Implementing KNN algoritms with different parameters

def knn_model(n_neighbors = 5,weights = 'uniform',algorithm = 'auto',p = 2):

    knn = KNeighborsClassifier(n_neighbors = n_neighbors,

                               weights = weights,

                               algorithm = algorithm,

                               p = p

                              )

    

    accuracies_train = cross_val_score(estimator = knn, X = X_train, y = Y_train, cv = 3)

    train_score = np.mean(accuracies_train)

    

    knn.fit(X_train,Y_train)

    test_score = knn.score(X_test,Y_test)

   

    return train_score, test_score

       

#KNN algorithm implementation with default parameters

train_score,test_score = knn_model()

result_train["Default-Train"] = train_score

result_test["Default-Test"] = test_score

print("Mean accuracy of train set:",train_score)

print("Mean accuracy of test set:", test_score) 
#KNN algorith implementation with different n_neighbors 

k_list = list(np.arange(1,285))

test_score_dict = {}

train_score_dict = {}



for k in k_list:

    train_score,test_score = knn_model(n_neighbors = k)

    train_score_dict[k] = (train_score)

    test_score_dict[k] = (test_score)

    

df_result = prepare_dataframe(test_score_dict,train_score_dict,"K = ",k_list)

df_result

#Plot score of different k values for test and train datas

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.plot(k_list,list(train_score_dict.values()))

plt.xlabel("K Value")

plt.ylabel("Train Score")



plt.subplot(1,2,2)

plt.plot(k_list,list(test_score_dict.values()))

plt.xlabel("K Value")

plt.ylabel("Test Score")

plt.show()
#Results stored in dictionaries

result_train["K-Best-Train"] = np.asarray(list(train_score_dict.values())).max()

result_test["K-Best-Test"] = np.asarray(list(test_score_dict.values())).max()
#KNN algorith implementation with different weights 

w_list = ['uniform','distance']



test_score_dict = {}

train_score_dict = {}



for w in w_list:

    train_score,test_score = knn_model(weights = w)

    train_score_dict[w] = (train_score)

    test_score_dict[w] = (test_score)



prepare_dataframe(test_score_dict,train_score_dict,"Weight = ",w_list)
#Results stored in dictionaries

result_train["Weight-Best-Train"] = np.asarray(list(train_score_dict.values())).max()

result_test["Weight-Best-Test"] = np.asarray(list(test_score_dict.values())).max()
#KNN algorith implementation with different algorithm

a_list = ['auto','ball_tree','kd_tree','brute']



test_score_dict = {}

train_score_dict = {}



for a in a_list:

    train_score,test_score = knn_model(algorithm = a)

    train_score_dict[a] = (train_score)

    test_score_dict[a] = (test_score)



prepare_dataframe(test_score_dict,train_score_dict,"Algorithm = ",a_list)
#Results stored in dictionaries

result_train["Algorithm-Best-Train"] = np.asarray(list(train_score_dict.values())).max()

result_test["Algorithm-Best-Test"] = np.asarray(list(test_score_dict.values())).max()
#KNN algorith implementation with different p

p_list = list(np.arange(1,11))





test_score_dict = {}

train_score_dict = {}



for p in p_list:

    train_score,test_score = knn_model(p = p)

    train_score_dict[p] = (train_score)

    test_score_dict[p] = (test_score)



prepare_dataframe(test_score_dict,train_score_dict,"P = ",p_list)
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.plot(p_list,list(train_score_dict.values()))

plt.xlabel("P Value")

plt.ylabel("Train Score")



plt.subplot(1,2,2)

plt.plot(p_list,list(test_score_dict.values()))

plt.xlabel("P Value")

plt.ylabel("Test Score")

plt.show()
#Results stored in dictionaries

result_train["P-Best-Train"] = np.asarray(list(train_score_dict.values())).max()

result_test["P-Best-Test"] = np.asarray(list(test_score_dict.values())).max()
#Implementation of GridSearch

grid = {'n_neighbors':np.arange(1,235),

        'p':np.arange(1,3),

        'weights':['uniform','distance'],

        'algorithm':['auto','ball_tree','kd_tree','brute']

       }

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X_train,Y_train)



print("Hyperparameters:",knn_cv.best_params_)

print("Train Score:",knn_cv.best_score_)

result_train["GridSearch-Best-Train"] = knn_cv.best_score_
#Results stored in dictionaries

result_test["GridSearch-Best-Test"] = knn_cv.score(X_test,Y_test)

print("Test Score:",knn_cv.score(X_test,Y_test))
#Result dataframe

columns = ["Default","K-Best","Weight-Best","Algorithm-Best","P-Best","GridSearchCV"]

prepare_dataframe(result_test,result_train,"",columns)
#Bar plot for showing result of parameters both train and test datas

plt.figure(figsize = (12,5))

X = np.arange(len(result_train))

ax = plt.subplot(111)

ax.bar(X, result_train.values(), width=0.2, color='b', align='center')

ax.bar(X-0.2, result_test.values(), width=0.2, color='g', align='center')

ax.legend(('Train Results','Test Results'))

plt.xticks(X, columns)

plt.title("Comparing Results of Parameters", fontsize=17)

plt.show()