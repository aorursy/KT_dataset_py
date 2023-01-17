import os

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame

import scipy.optimize as opt  

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
f=pd.read_csv("../input/mushrooms.csv")

df=DataFrame(f)

df.head()[:2]
df.dtypes
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df.columns:

    df[col]=labelencoder.fit_transform(df[col])

df.head()[:2]
count_var=[]

for col in df.columns:

    count_var.append(df[col].unique().sum())

size=np.arange(len(count_var))

fig=plt.figure(figsize = (15,10))

ax=fig.add_subplot(1,1,1, axisbg='red')

ax.bar(size,count_var, color = 'k')

ax.set(title="Unique elements per column",

      ylabel='No of unique elements',

      xlabel='Features')
df.corr()
import seaborn as sns

plt.figure(figsize = (10,10))

sns.heatmap(df.corr(), cmap = 'inferno',square=True)
#Separating the train and target values.Lets select only two features so that things can be intuitively easy.So lers select the most corellated features to 'class' . 

target=df['class']

train=df[['gill-size','gill-color']]

print(train.shape)

print(target.shape)
#Count of the classes

fig=plt.figure(figsize = (15,10))

ax=fig.add_subplot(1,1,1, axisbg='blue')

pd.value_counts(target).plot(kind='bar', cmap = 'cool')

plt.title("Class distribution")
def sigmoid(theta,X):  

    X = np.array(X)

    theta = np.asarray(theta)

    return((1/(1+math.e**(-X.dot(theta)))))
# Function for the cost function of the logistic regression.

def cost(theta, X, Y):

    first = np.multiply(-Y, np.log(sigmoid(theta,X)))

    second = np.multiply((1 - Y), np.log(1 - sigmoid(theta,X)))

    return np.sum(first - second) / (len(X))
# It calculates the gradient of the log-likelihood function.

def log_gradient(theta,X,Y):

    first_calc = sigmoid(theta, X) - np.squeeze(Y).T

    final_calc = first_calc.T.dot(X)

    return(final_calc.T)
# This is the function performing gradient descent.

def gradient_Descent(theta,X,Y,itr_val,learning_rate=0.00001):

    cost_iter=[]

    cost_val=cost(theta,X,Y)

    cost_iter.append([0,cost_val])

    change_cost = 1

    itr = 0

    while(itr < itr_val):

        old_cost = cost_val

        theta = theta - (0.01 * log_gradient(theta,X,Y))

        cost_val = cost(theta,X,Y)

        cost_iter.append([i,cost])

        itr += 1

    return theta
def pred_values(theta,X,hard=True):

    X = (X - np.mean(X,axis=0))/np.std(X,axis=0)

    pred_prob = sigmoid(theta,X)

    pred_value = np.where(pred_prob >= .5 ,1, 0)

    return pred_value
theta = np.zeros((train.shape)[1])

theta = np.asmatrix(theta)

theta = theta.T

target = np.asmatrix(target).T

y_test = list(target)
params = [10,20,30,50,100]

for i in range(len(params)):

    th = gradient_Descent(theta,train,target,params[i])

    y_pred = list(pred_values(th, train))

    score = float(sum(1 for x,y in zip(y_pred,y_test) if x == y)) / len(y_pred)

    print("The accuracy after " + '{}'.format(params[i]) + " iterations is " + '{}'.format(score))
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(train, target)

clf.score(train, target)