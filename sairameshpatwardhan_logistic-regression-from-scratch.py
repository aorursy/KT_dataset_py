import pandas as pd
from math import exp
import os
import numpy as np
import matplotlib.pyplot as plt 


df1= pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
X1= df1.iloc[0:768,0:8]
Y= df1.iloc[0:768,8]

X2= ((X1-np.mean(X1))/(np.max(X1)-np.min(X1)))
X_train= X2.iloc[0:614,:]
Y_train= Y.iloc[0:614]

X_test1= X2.iloc[614:768,:]
X_test= X_test1.reset_index(drop=True)
Y_test1= Y.iloc[614:768]
Y_test= Y_test1.reset_index(drop=True)
def predict(X,weights):
        predictions= np.dot(X,weights)
        predictions= -1* predictions
        predictions= np.exp(predictions)
        predictions= 1.0/ (1.0 + predictions)
        thr= 0.6
    
        for i in range(0,len(predictions)):
            if(predictions[i]>=thr):
                predictions[i]=1
            else:      
                predictions[i]=0 
        predictions= predictions.tolist()         
        return predictions
        


def accurate(predictions,Y):
        correct=0
        for i in range(0,len(Y)):
            if predictions[i]==Y[i]:
                correct+=1
        return (correct/ (len(Y)))*100        
        
def weights(X,Y,l_r,epochs,lamb):
        weight = np.zeros(len(X.columns)) 
        weight1= np.transpose(weight)
        weights = pd.Series(weight1) 
        
        for epoch in range(epochs):
            predictions= predict(X,weights)
            error= predictions- Y
            Xt= X.transpose()
            
            weights= pd.Series(weights)
            t1= np.dot(Xt,error)
            t2=lamb* weights
            
            fac= l_r/len(X)
            term= t1+ t2
                   
            weights= weights- fac*(term)
            weights= weights.tolist()
        return weights
epochs = 300
l_r= 0.1
lamb= [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]
# lamb= [0]
accuracy1= []
accuracy2= []
accuracy3= []
    
for i in lamb:
    weights1= weights(X2, Y,l_r,epochs,i)
    predictions1= predict(X2,weights1)
    accuracy= accurate(predictions1,Y)
    accuracy1.append(accuracy)
        
   
    predictions2= predict(X_test,weights1)
    accuracy= accurate(predictions2,Y_test)
    accuracy2.append(accuracy)
        
    
plt.plot(lamb, accuracy1, label = "train")  
plt.plot(lamb, accuracy2, label = "test") 
# plt.plot(lamb, accuracy3, label = "test")
plt.ylabel('accuracy') 
plt.title('lamb choosing')   
plt.legend() 
plt.show() 
# print(accuracy1)
# print(accuracy2)


