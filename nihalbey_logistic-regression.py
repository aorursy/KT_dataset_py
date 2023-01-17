# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#before of all we read the data that is we will use

data = pd.read_csv("../input/data.csv")
#we can see class/feature of our data

data.columns
data.head()

#id and unnamed32 is unnecessary for our model and we change M(malignant) and B(benign) replace 0,1
#firstly we drop two columns

data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)

#we change value of diognose M = 0 and B = 1(we use this diversity in logistic regression)

data["diagnosis"].replace("M",0,inplace = True)

data["diagnosis"].replace("B",1,inplace = True)
#we select x,y axis and we normalize our data

y = data.diagnosis.values

x_data = data.drop("diagnosis",axis=1)

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#we separate train and test data with sklearn selection model

#You can thnk this x_train for learn and y_train is answer of x_train and finally we testing our data with x_test andy_test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y)

#You can see shape of our data it is available for preparing model

print("xtrain:{}".format((x_train).shape))

print("y_train:{}".format((y_train).shape))

print("xtest:{}".format((x_test).shape))

print("ytest:{}".format((y_test).shape))
#We did 

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(max_iter = 200)

lgr.fit(x_train,y_train)

print("our accuracy is:{}".format(lgr.score(x_test,y_test)))
#We can evaluate our model so and we have y_predict and y_true(y_test)

from sklearn.metrics import confusion_matrix

y_true = y_test 

y_pred = lgr.predict(x_test) #Predict data for eveluating 

cm = confusion_matrix(y_true,y_pred)





#We draw heatmap for showing confusion matrix

import matplotlib.pyplot as plt

import seaborn as sns

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax)