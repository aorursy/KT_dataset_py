# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/StudentsPerformance.csv")
df.info() ## Lets see what we have in the data
df.gender=[1 if each=="male" else 0for each in df.gender]
needed_data=df.drop(["race/ethnicity","parental level of education"

                    ,"lunch", "test preparation course"], axis=1)
needed_data.head() # This one includes features which we want.



male=needed_data[needed_data.gender==1]

female=needed_data[needed_data.gender==0]

#plt.plot(female["math score"])

#plt.show()



plt.plot(female["math score"],color="red")

plt.plot(male["math score"],color="blue")

plt.show()

y=needed_data.gender.values

x_Data=needed_data.drop(["gender"], axis=1)



x=(x_Data-np.min(x_Data))/(np.max(x_Data)-np.min(x_Data)) ## normalization

from sklearn.model_selection import train_test_split



x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T
def weights_and_bias(dimension): ## function for initializing of weight and bias

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b
def sigmoid (z):

    y_head = 1/(1+np.exp(-z))

    return y_head
def forwardAndBackwardProg (w,b,x_train,y_train):



    z=np.dot(w.T,x_train) + b

    y_head=sigmoid(z)



    loss=-(y_train*np.log(y_head)+(1-y_train)*np.log(1-y_head))

    cost=(np.sum(loss))/x_train.shape[1]



    derivative_weight=np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]



    gradients= {"d_weight" :derivative_weight , "d_bias" : derivative_bias}



    return gradients,cost

    
def update(w, b, x_train, y_train, learning_rate , numOfIteration):

    index=[]

    cost_list=[]

    cost_list2=[]

    

    for i in range(numOfIteration):

        gradients,cost=forwardAndBackwardProg (w,b,x_train,y_train)

        cost_list.append(cost)

        

        w=w-learning_rate * gradients["d_weight"]

        b=b-learning_rate * gradients["d_bias"]

        if i%100==0:

            

            index.append(i)

            cost_list2.append(cost)

            print("Cost after iteration %i : %f" %(i,cost))

        

    parameters={"weight" : w , "bias" : b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Iteration")

    plt.ylabel("cost")

    plt.title("Cost graph")

    plt.show()

    return parameters,gradients,cost_list



        

        

        
def predict(w,b, x_test):

    z=sigmoid(np.dot(w.T,x_test))

    y_prediction=np.zeros((1,x_test.shape[1]))

    

    for i in range (z.shape[1]):

        if z[0,i] >=0.5:

            y_prediction[0,i]=1

        else:

            y_prediction[0,i]=0

    return y_prediction

        
def logistic_reg(x_train,x_test,y_train,y_test,learning_rate,numOfIteration):

    dimension=x_train.shape[0]

    w,b= weights_and_bias(dimension)

    parameters,gradients,cost_list=update(w, b, x_train, y_train, learning_rate , numOfIteration) 

    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)

    print("test accuracy:{} %" .format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

logistic_reg(x_train,x_test,y_train,y_test,learning_rate=1,numOfIteration=300)

from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3) #n_neighbor is k

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

knn.score(x_test,y_test)
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



from sklearn.svm import SVC

svm=SVC(random_state=42)

svm.fit(x_train,y_train)

svm.score(x_test,y_test)

from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)

nb.score(x_test,y_test)
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

dt.score(x_test,y_test)
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test =train_test_split(x,y, test_size=0.15, random_state=42)



from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=200, random_state=42)   # 200 sub sample

rf.fit(x_train,y_train)

rf.score(x_test,y_test)
y_pred=knn.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.show()

y_pred=svm.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.show()



y_pred=nb.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.show()



y_pred=dt.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.show()



y_pred=rf.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.show()


