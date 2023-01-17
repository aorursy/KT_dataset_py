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
df = pd.read_csv("/kaggle/input/szeged-weather/weatherHistory.csv")

df.head()


df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

df.head()


df
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from sklearn import preprocessing



#x = df[['GRE Score','CGPA']]

y = df['Chance of Admit ']

# showcasing the relationship between all the columns and the label

#loop throught our feature columns

for col in df.columns:

    #we want every feature column except for chance of admit itself 

    #remember, the variable col contains a list with column name as string it it 

    if(col != ['Chance of Admit ']):

        

        #create a scatter plot with current feature column, 

        #chance of admit(y), always remains on the y axis 

        plt.scatter(df[col],y)

        

        #set the x, y axis lables as respective column names, 

        #again admission chance of admission always remains in the y axis

        plt.xlabel(col)

        plt.ylabel('Chance of Admit')

        

        #display our graphs 

        plt.show()




x = np.array(df[['GRE Score','CGPA']])

y = np.array(df['Chance of Admit '])



#x=preprocessing.scale(x)

#y=preprocessing.scale(y)

x[:5],' ' ,y[:5]
# ax = sns.scatterplot(x=x, y=y)
#for any given x, this mothod uses linear regression to predict a value



def predict(x,m,b):

    r =  0 

    for i in range(0,len(x)):

        r = r +(m[i]*x[i]) 

    return r+b
#using our trained m and b, the coeffiecients, this method tells us how accurate our model is by returning the sum of all errors

def error(b, m, X,Y):

    totalError = 0

    for i in range(0, len(X)):

        x = X[i]

        y = Y[i]

        for n in range(0,len(m)):

            totalError += (y - (m[n] * x[n] + b)) ** 2

    return totalError / float(len(X))
#this is the gradient decent method, 

#this method will be used over thousands of iterations to train our coefficents

#this takes our two coefficents, calculates the gradient, 

## it then uses the gradient to update the coefficients 

## these updated coeffiencents are exported



#by interating many times this method, will keep adjusting the coeffienects, based on their respecive gradients

## by doing so after a certain amoint of trainin we can make accurate predictions 



def gradient_dec(b_current, m_current, X,Y, lr):

    #initialise the variables that will store gradients

    b_gradient = 0

    m_gradient = np.zeros(m_current.shape)

    

    N = float(len(X))

    

    #loop through all entries in the dataset

    for i in range(0, len(X)):

        x = X[i]#data feature

        y = Y[i]#data target

        

        for n in range(0, len(x)):

            #calculate the GRADIENT for our two coefficents respectively 

            b_gradient += -(2/N) * (y - ((m_current[n] * x[n]) + b_current))

        

            m_gradient[n] += -(2/N) * x[n] * (y - ((m_current[n] * x[n]) + b_current))

    

    #new variables to strore the adjusted coeffcients

    new_b = b_current - (lr * b_gradient)

    new_m = m_current - (lr * m_gradient)

    #print(new_b,new_m)

    #return the adjusted coeffecients 

    return [new_b, new_m]

# np.array(X.loc[1]),initial_m
#this is the training method, 

#this method runs the GD method multiple times to get the trained coeffecients



#this method accepts our features, target, initial coeffecients = usually 0 , learning rate and number of training interations



def train(X,Y, initial_b, initial_m, lr, num_iter):

    #variables to strore the coeffecients, initiazed here,

    b = initial_b

    m = initial_m

    

    #run the training loopas many times as specified in number of iterations

    for i in range(num_iter):

        #use the gradient_dec method get our slightly more accurate coefficents

        bx, mx = gradient_dec(b, m, X,Y, lr)

        

        #update coeff with new coeff

        b = bx

        m = mx

        

    #after training is done for the number of loops return coeffecients     

    return [b, m]
X =x

Y =y

lr = 0.000001 #learning rate 

initial_b = 0 # initial y-intercept guess

initial_m = np.zeros(X.shape[1]) # initial slope guess

num_iter = 500# number of 'epochs'





# #calculate y-intercept, and slope by training our model 

[b, m] = train(X,Y, initial_b, initial_m, lr, num_iter)
[b, m]
#y[56]
error(b, m, X,Y)
y[56]
predict(x[56],m,b)


# pred_for = [i for i in range(0,22,1)]

# predictions =[]

# for yhat in pred_for:

#     predictions.append(predict(yhat,m,b))





# ax = sns.scatterplot(x=x, y=y)



# plt.scatter(x=pred_for, y=predictions, color='r')
