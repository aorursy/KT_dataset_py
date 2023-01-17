# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

from scipy.sparse import csc_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#!/usr/bin/env python3

# -*- coding: utf-8 -*-
data = pd.read_csv('../input/spam.csv',encoding='latin-1')

data.head()
#data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data = data.rename(columns={"v1":'label', "v2":'text'})

data.head()
num = pd.value_counts(data['label'],sort=True).sort_index()

num.plot(kind='bar')

plt.title('histogram')

plt.xlabel('spam or not')

plt.ylabel('number')
print(data.shape)



data.replace(('spam', 'ham'), (1, 0), inplace=True)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.3)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer()

vect.fit(x_train)

x_train_df = vect.transform(x_train)

x_test_df = vect.transform(x_test)



ham_words = ''

spam_words = ''

spam = data[data.label == 1]

ham = data[data.label == 0]
import nltk



for val in spam.text:

    text = val.lower()

    tokens = nltk.word_tokenize(text)

    for words in tokens:

        spam_words = spam_words + words + ' '

        

for val in ham.text:

    text = val.lower()

    tokens = nltk.word_tokenize(text)

    for words in tokens:

        ham_words = ham_words + words + ' '
from wordcloud import WordCloud



spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)

ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)



plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()



plt.figure(figsize=(10,8), facecolor='k')

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l1')

lr.fit(x_train_df,y_train)

lrscore = lr.score(x_test_df,y_test)

print('Logistic regression score ',lrscore)

print(' ')



from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(max_depth=60)

rfc.fit(x_train_df,y_train)

rfcscore = rfc.score(x_test_df,y_test)

print('Random Forest score ',rfcscore)

print(' ')



from sklearn.naive_bayes import MultinomialNB

mn = MultinomialNB()

mn.fit(x_train_df,y_train)

mnscore = mn.score(x_test_df,y_test)

print('Multinomial score ',mnscore)

print(' ')
print('Implementation of logistic regression ')



x_train_df = x_train_df.todense()

x_test_df = x_test_df.todense()

def sigmoid(z):

    return float(1.0 / float((1.0 + math.exp(-1.0*z))))
def hypothesis(theta,x):

    z = 0

    for i in range(len(theta)):

        xi = x[i]

        z += np.dot(xi,theta[i].transpose())

    return sigmoid(z)
def cost_function(theta,m):

    sumOfErrors = 0

    for i in range(m):

        hi = hypothesis(theta,x_train_df[i])

        if y_train.iloc[i] == 1:

            error = y_train.iloc[i] * math.log(hi)

        elif y_train.iloc[i] == 0:

            error = (1-y_train.iloc[i]) * math.log(1-hi)

        sumOfErrors += error

    const = -1/m

    j = const * sumOfErrors

    return j

def cost_function_derivative(theta,j,m,alpha):

    sumErrors = 0

    for i in range(m):

        xi = x_train_df[i]

        hi = hypothesis(theta,xi)

        yt = y_train.iloc[i]

        hit = np.subtract((float)(hi),float(yt))

        error = np.dot(hit,x_train_df[i][j])

        sumErrors += error

    m = len(y_test)

    constant = float(alpha)/float(m)

    j = constant * sumErrors

    return j
def gradient_descent(theta,m,alpha):

    new_theta = []

    for j in range(len(theta)):

        new_theta_value = theta[j] - cost_function_derivative(theta,j,m,alpha)

        new_theta.append(new_theta_value)

    return new_theta
def logistic_regression(alpha,theta,num_iters):

    m = len(y_train)

    for x in range(num_iters):

        new_theta = gradient_descent(theta,m,alpha)

        theta = new_theta

        if x % 100 == 0:

            cost_function(theta,m)

            print ('theta ', theta)	

            print ('cost is ', cost_function(theta,m))

    score = 0

    length = len(x_test)

    for i in range(length):

        prediction = round(hypothesis(theta,x_test_df[i]))

        answer = y_test.iloc[i]

        if prediction == answer:

            score += 1

    my_score = float(score) / float(length)

    print ('Your score: ', my_score)
def ReLU(z):

    if z > 0:

        return float(z)

    return float(0.01*z)
def hypothesisrelu(theta,x):

    z = 0

    for i in range(len(theta)):

        xi = x[i]

        z += np.dot(xi,theta[i].transpose())

    return ReLU(z)

    

def cost_functionrelu(theta,m):

    sumOfErrors = 0

    for i in range(m):

        hi = hypothesisrelu(theta,x_train_df[i])

        if y_train.iloc[i] == 1:

            error = y_train.iloc[i] * abs(hi-1)

        elif y_train.iloc[i] == 0:

            error = (1-y_train.iloc[i]) * abs(hi)

        sumOfErrors += error

    const = 1/m

    j = const * sumOfErrors

    return j



def cost_function_derivativerelu(theta,j,m,alpha):

    sumErrors = 0

    for i in range(m):

        xi = x_train_df[i]

        hi = hypothesisrelu(theta,xi)

        yt = y_train.iloc[i]

        hit = np.subtract((float)(hi),float(yt))

        error = np.dot(hit,x_train_df[i][j])

        sumErrors += error

    m = len(y_test)

    constant = float(alpha)/float(m)

    j = constant * sumErrors

    return j

    

def gradient_descentrelu(theta,m,alpha):

    new_theta = []

    for j in range(len(theta)):

        new_theta_value = theta[j] - cost_function_derivativerelu(theta,j,m,alpha)

        new_theta.append(new_theta_value)

    return new_theta



def logistic_regression_relu(alpha,theta,num_iters):

    m = len(y_train)

    for x in range(num_iters):

        new_theta = gradient_descentrelu(theta,m,alpha)

        theta = new_theta

        if x % 100 == 0:

            cost_functionrelu(theta,m)

            print ('theta ', theta)	

            print ('cost is ', cost_functionrelu(theta,m))

    score = 0

    length = len(x_test)

    for i in range(length):

        prediction = round(hypothesisrelu(theta,x_test_df[i]))

        answer = y_test.iloc[i]

        if prediction == answer:

            score += 1

    my_score = float(score) / float(length)

    print ('Your score with ReLu: ', my_score)
def Tanh(z):

    return float(math.tanh(z))
def hypothesistanh(theta,x):

    z = 0

    for i in range(len(theta)):

        xi = x[i]

        z += np.dot(xi,theta[i].transpose())

    return Tanh(z)

    

def cost_functiontanh(theta,m):

    sumOfErrors = 0

    for i in range(m):

        hi = hypothesistanh(theta,x_train_df[i])

        if y_train.iloc[i] == 1:

            error = y_train.iloc[i] * (1-hi*hi)

        elif y_train.iloc[i] == 0:

            error = (1-y_train.iloc[i]) * (1-hi*hi)

        sumOfErrors += error

    const = 1/m

    j = const * sumOfErrors

    return j



def cost_function_derivativetanh(theta,j,m,alpha):

    sumErrors = 0

    for i in range(m):

        xi = x_train_df[i]

        hi = hypothesistanh(theta,xi)

        yt = y_train.iloc[i]

        hit = np.subtract((float)(hi),float(yt))

        error = np.dot(hit,x_train_df[i][j])

        sumErrors += error

    m = len(y_test)

    constant = float(alpha)/float(m)

    j = constant * sumErrors

    return j

    

def gradient_descenttanh(theta,m,alpha):

    new_theta = []

    for j in range(len(theta)):

        new_theta_value = theta[j] - cost_function_derivativetanh(theta,j,m,alpha)

        new_theta.append(new_theta_value)

    return new_theta



def logistic_regression_tanh(alpha,theta,num_iters):

    m = len(y_train)

    for x in range(num_iters):

        new_theta = gradient_descenttanh(theta,m,alpha)

        theta = new_theta

        if x % 100 == 0:

            cost_functiontanh(theta,m)

            print ('theta ', theta)	

            print ('cost is ', cost_functiontanh(theta,m))

    score = 0

    length = len(x_test)

    for i in range(length):

        prediction = round(hypothesistanh(theta,x_test_df[i]))

        answer = y_test.iloc[i]

        if prediction == answer:

            score += 1

    my_score = float(score) / float(length)

    print ('Your score with tanh: ', my_score)
initial_theta = np.zeros(x_test_df[1].shape)

alpha = 0.3

iterations = 1000

logistic_regression(alpha,initial_theta,iterations)

print(' ')

initial_theta = np.zeros(x_test_df[1].shape)

logistic_regression_relu(alpha,initial_theta,iterations)

print(' ')

initial_theta = np.zeros(x_test_df[1].shape)

logistic_regression_tanh(alpha,initial_theta,iterations)

print(' ')