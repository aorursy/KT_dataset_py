# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import seaborn as sn

import matplotlib.pyplot as plt



from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



def Graph_Confusion_Matrix(CM, labels):

    #np.fill_diagonal(CM,0)



    plt.figure(figsize = (5,5))

    sn.set(font_scale=1.4)#for label size

    sn.heatmap(CM, annot=True,annot_kws={"size": 16}, fmt='g'

               ,xticklabels = labels

               ,yticklabels = labels)# font size

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()

    

class anomaly_detection():

    

    def __init__(self, df_X):

         self.df_X = df_X

        

         m,n = df_X.values.shape



         self.sigma = np.zeros((n,n))

         self.mu = np.zeros((n))



    def Guassion_Parameters(self):

        ##

        #Obtain the mean and covariance matrix need for the Guassian curve

        ##

        X = self.df_X.values

        X_mean = X.mean(axis=0)

        

        m, n = X.shape # number of training examples, number of features

        

        #Creates the covariance Matrix

        Sigma = np.dot((X - X_mean).T, (X - X_mean))

        

        Sigma = Sigma * (1.0/m)

        

        self.mu = X_mean

        self.sigma = Sigma





    def multivariateGaussian(self, df_X):

        ##

        #Calculates the P-Value based on the parameters found above.

        #This is the vectorised form of the equation.

        ##

        

        X = df_X

        

        m, n = X.shape # number of training examples, number of features

    

        X = X.values - self.mu.reshape(1,n) # (X - mu)

    

        # vectorized implementation of calculating p(x) for each m examples: p is m length array

        p = (1.0 / (math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(self.sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(self.sigma)) * X, axis=1))

    

        return p



    def Best_Epsilon(self, Yval ,Pred):

        ##

        #Obtains the best epsilon when the F1 score is maxamised.

        ##

        

        

        bestF1 = 0

        bestEpsilon = 0

    

        stepsize = (max(Pred) - min(Pred)) / 1000

        for epsilon in np.arange(min(Pred), max(Pred), stepsize):

            

            predictions = (Pred < epsilon).astype(int)

            

            F1 = f1_score(Yval, predictions)

            

            if F1 > bestF1:

                bestF1 = F1

                bestEpsilon = epsilon

                

        return(bestEpsilon, bestF1)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")

df.head(5)
%matplotlib inline



for i, col in enumerate(df.columns):

    plt.figure(i, figsize=(5,5))

    sn.distplot(df[col].loc[df['Class'] == 1], color="red" ,label="Fraud 1")

    sn.distplot(df[col].loc[df['Class'] == 0], color="blue" ,label="Authorised 0")

    plt.legend(fontsize=12)

    plt.title(col, fontsize= 12)

    #print(i,col)
df_base = df[["Class", "V3", "V4", "V10", "V11", "V12" , "V14" , "V16", "V17", "V18"]] 
#Seporates the two classes

df_fraud = df_base.loc[df["Class"] == 1]

df_authorised = df_base.loc[df["Class"] == 0]



#Resets the index

df_fraud = df_fraud.reset_index(drop=True)

df_authorised = df_authorised.reset_index(drop=True)



#splits into three groups

train, test, validation = np.split(df_authorised, [int(0.7* len(df_authorised)),int(0.85* len(df_authorised)) ])

post_V, post_T = np.split(df_fraud ,2)



test = test.append(post_T).sample(frac=1).reset_index(drop=True)

validation = validation.append(post_V).sample(frac=1).reset_index(drop=True)



for i in range (1,len(df_base.columns)):

        

    model_1 = anomaly_detection(train.iloc[:,[0,i]].drop(["Class"], axis=1))

    

    #Fits the model

    model_1.Guassion_Parameters()



    pred_Val = model_1.multivariateGaussian(validation.iloc[:,[0,i]].drop(["Class"], axis=1))



    #Use validation data to find the best threshold for P Value

    best_E, Best_F1 = model_1.Best_Epsilon(validation["Class"], pred_Val)



    yval = validation["Class"]





    CM_Val = confusion_matrix(yval,  (pred_Val < best_E).astype(int), [1,0])

    Graph_Confusion_Matrix(CM_Val, [1,0])

    print("Columns Name: ", train.iloc[:,[0,i]].columns)

    print("F1 Score: ",Best_F1)
model_1 = anomaly_detection(train[["V12", "V17"]])



#Fits the model

model_1.Guassion_Parameters()



pred_Val = model_1.multivariateGaussian(validation[["V12", "V17"]])



#Use validation data to find the best threshold for P Value

best_E, Best_F1 = model_1.Best_Epsilon(validation["Class"], pred_Val)



yval = validation["Class"]





CM_Val = confusion_matrix(yval,  (pred_Val < best_E).astype(int), [1,0])

Graph_Confusion_Matrix(CM_Val, [1,0])

print("Columns Name: ", train.iloc[:,[0,i]].columns)

print("F1 Score: ",Best_F1)
pred_Test = model_1.multivariateGaussian(test[["V12", "V17"]])



ytest = test["Class"]

CM_Test = confusion_matrix(ytest,  (pred_Test < best_E).astype(int),  [1,0])

Graph_Confusion_Matrix(CM_Test,  [1,0])
