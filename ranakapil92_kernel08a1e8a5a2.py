# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from csv import reader  

from pandas import DataFrame

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
inputdata = list()

with open('../input/hiring.csv', 'r') as file:

    csv_reader = reader(file)

    for row in csv_reader:

        if not row:

            continue

        inputdata.append(row)



rows= len(inputdata)



inputdata=inputdata[1:][:]

rows= len(inputdata)



for i in range(0,rows):

    if(inputdata[i][0]==""):

        inputdata[i][0]=0

    if(inputdata[i][0]=="five"):

        inputdata[i][0]=5

    if(inputdata[i][0]=="two"):

        inputdata[i][0]=2

    if(inputdata[i][0]=="seven"):

        inputdata[i][0]=7

    if(inputdata[i][0]=="three"):

        inputdata[i][0]=3

    if(inputdata[i][0]=="ten"):

        inputdata[i][0]=10

    if(inputdata[i][0]=="eleven"):

        inputdata[i][0]=11

inputdata[6][1]=0        
dataset=DataFrame(inputdata)

for i in range(1,4):

    dataset[i]= dataset[i].astype(float) 

    

X_main = dataset.iloc[:,:3].values

y_main = dataset.iloc[:, 3].values      
def mylinridgeregeval(X,weights):

    n=len(X)

    

    y_pred=np.empty(n)

    y_pred.fill(0.0)

    for j in range(0,n):

        y_pred[j]=np.dot(weights,X[j])   

        

    return y_pred  
def mylinridgereg(X,Y,Lambda):

    

    Xt=np.transpose(X)

    n=len(Xt)

    Lambda_I= Lambda*(np.identity(n))

    

    a = np.zeros(shape=(n,n))

    a=np.dot(Xt,X)

    a=np.add(a,Lambda_I)

    if(np.linalg.det(a)!=0):

        

        b=np.zeros(shape=(n,1))

        c=np.zeros(shape=(n,n))

        

        b=np.dot(Xt,Y)

        c=np.linalg.inv(a)

        c=np.dot(c,b)

        

        return c

    else:

        b=np.zeros(shape=(n,1))

        c=np.zeros(shape=(n,n))

        

        b=np.dot(Xt,Y)

        c=np.linalg.pinv(a)

        c=np.dot(c,b)

        

        return c
def meansquarederr(T,Tdash):

    error =0.0

    total_error=0.0

    

    error=np.subtract(T,Tdash)

    sq_error=np.square(error)

    total_error=np.sum(sq_error) 

    

    return total_error/float(len(T))  
def mean(attr):

    return sum(attr) / float(len(attr))        



def variance(attr, mean):

	return sum([(x-mean)**2 for x in attr])
error1=0.0

error2=0.0



for i in range(0,8):

    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.10,random_state=0)

    

    temp_X=np.copy(X_test)

    temp_y=np.copy(y_test)

    

    w =np.array([0.0,1.0,1.0,1.0])

    

    means=np.zeros((3,))

    variances=np.zeros((3,))

    

    for i in range(0,3):

        m=mean(X_train[:,i])

        means[i]=m

        v=variance(X_train[:,i],m)

        variances[i]=v

        X_train[:,i]=(X_train[:,i]-m)/v 

        

    w = mylinridgereg(X_train,y_train,0.000001)   

    

    t_train = mylinridgeregeval(X_train,w)

    error=meansquarederr(t_train,y_train)

    error1 = error1+error

    

    t_val = mylinridgeregeval(temp_X,w)

    error=meansquarederr(t_val,temp_y)

    error2 = error2+error

    

error1=error1/8

error2=error2/8



print(error1)

print(error2)

    