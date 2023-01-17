# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Next, we'll load the Iris flower dataset, which is in the "../input/" directory

iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame



# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do

iris.head()



# Press shift+enter to execute this cell



# Another useful seaborn plot is the pairplot, which shows the bivariate relation

# between each pair of features

# 

# From the pairplot, we'll see that the Iris-setosa species is separataed from the other

# two across all feature combinations

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
import csv

import numpy

import random

reader=csv.reader(open("../input/Iris.csv","rt"),delimiter=',')

csvlist=list(reader)

csvlist.pop(0)



random.shuffle(csvlist)



m_attr = [0,1]



N = len(csvlist)

M = len(m_attr)





N_train_perc = 0.1

N_train = (int) (N * N_train_perc)

N_test = N - N_train



X_train = [[0 for x in range(M)] for y in range(N_train)] 

X_test = [[0 for x in range(M)] for y in range(N_test)]



T_train = [[0 for x in range(3)]for y in range(N_train)]

T_test = [[0 for x in range(3)]for y in range(N_test)]
n = 0

for i in range(N_train):

    m = 0

    for j in m_attr:

        if j == 0:

            X_train[n][m] = 1

        else:

            X_train[n][m] = csvlist[i][j]

        m = m + 1

        

    if(csvlist[i][5]=="Iris-setosa"):

        T_train[n][0] = 1

    elif(csvlist[i][5]=="Iris-versicolor"):

        T_train[n][1] = 1

    else:

        T_train[n][2] = 1

        

    n = n + 1

X_train=numpy.array(X_train).astype('double')    



#print(T_train)
n = 0

for i in range(N_train,N):

    m = 0

    for j in m_attr:

        if j == 0:

            X_test[n][m] = 1

        else:

            X_test[n][m] = csvlist[i][j]

        m = m + 1

    

    if(csvlist[i][5]=="Iris-setosa"):

        T_test[n][0] = 1

    elif(csvlist[i][5]=="Iris-versicolor"):

        T_test[n][1] = 1

    else:

        T_test[n][2] = 1

        

    n = n + 1

X_test=numpy.array(X_test).astype('double')    



#print(X_test)
W = np.dot(np.dot(numpy.linalg.inv(np.dot(X_train.T,X_train)),X_train.T),T_train)

print(W.shape)
#



N_correct = 0

for i in range(0,N_test):

    x = X_test[i]

    y = np.dot(W.T,x.T)

    

    t = T_test[i]

    

    if np.argmax(y) == np.argmax(t):

        N_correct = N_correct + 1

    



print(N_correct/N_test)

    

    
