import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



import os

print(os.listdir("../input"))

print("Reading the csv file and looking at the first five rows :\n")

headbrain = pd.read_csv("../input/headbrain.csv")

print(headbrain.head())

#y=mx+c

print(headbrain.shape)
print("HeadBrain Info :\n")

print(headbrain.info())
print("Checking for any null values:\n")

print(headbrain.isnull().any())
print("Checking for unique values in each column:\n")

print(headbrain.nunique())
plt.figure(figsize=(10,10))

sns.scatterplot(y='Brain Weight(grams)',x='Head Size(cm^3)',data=headbrain)

plt.show()
headbrain  = headbrain.values



X = headbrain[:,2]

Y = headbrain[:,3]



X.shape,Y.shape
def Linear_Regression(X,Y):

    mean_x = np.mean(X)

    mean_y = np.mean(Y)



    n = len(X)

    numerator= 0

    denominator=0

    for i in range(n):

        numerator   += ((X[i] - mean_x) * (Y[i] - mean_y))

        denominator += ((X[i] - mean_x) ** 2)



    m = numerator/ denominator

    c = mean_y - m * mean_x

    

    return(m,c)



def predict(X,m,c):

    pred_y=[]

    for i in range(len(X)):

        pred_y.append(c + m * X[i])



    return(pred_y)

def r2score(y_obs,y_pred):

    yhat = np.mean(y_obs)

    

    ss_res = 0.0

    ss_tot = 0.0

    

    for i in range(len(y_obs)):

        ss_tot += (y_obs[i]-yhat)**2

        ss_res += (y_obs[i]-y_pred[i])**2

        

    r2 = 1 - (ss_res/ss_tot)



    return r2

plt.title("Linear Regression Plot of HeadSize Vs Brain Weight")



X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)



m,c = Linear_Regression(X_train,y_train)

print("slope = ", m)

print('intercept = ',c)

      

y_pred =  predict(X_test,m,c)



print("R-squared :",r2score(y_test,y_pred))



plt.plot(X_test,y_pred,color='red',label='Linear Regression')

plt.scatter(X_train,y_train,c='b',label='Scatter Plot')

plt.xlabel("Head Size")

plt.ylabel("Brain Weight")

plt.legend()

plt.show()
#Reshape the input data into 2D array

X = X.reshape(len(X),1)



X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)



reg = LinearRegression()

reg.fit(X_train,y_train)



y_predictions = reg.predict(X_test)



print("R-squared :",r2_score(y_test, y_predictions))
