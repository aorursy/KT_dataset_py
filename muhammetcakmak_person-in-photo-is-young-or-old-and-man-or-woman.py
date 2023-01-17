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
# Here we import dataset to python and assign it to data variable

data = pd.read_csv("../input/list_attr_celeba.csv")
# We have a quick look into the data

data.head()
# This codes gives general information about the number of data and the type of data

data.info()
# We make list of young and old celebrities.

young=[]

old=[]

for i in data.Young.values:

    if i == 1:

        young.append(i)

    else:

        old.append(i)

        

# Bar chart that shows the number of young and old celebrity

top=[('Young',young.count(1)),('Old',old.count(-1))]

labels, ys = zip(*top)

xs = np.arange(len(labels)) 

width = 1

plt.bar(xs, ys, width, align='center')

plt.xticks(xs, labels) 

plt.yticks(ys)
# we drop the "image_id" colomn because we dont need it while coding

data.drop(["image_id"],axis=1,inplace = True)
# we assign Young feature as y variable which is our target data her.

y = data.Young.values

# we drop Young feature from data set and assign it as x_data 

x_data = data.drop(["Young"],axis=1)
# Normalization

# Normalization is required if there is big diffirence between your features data

# Here there is not big difference but I prefer it to keep program in safe

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# Train-Test Split

# Here we split the data 80 percent for training and 20 percent for testing

# we write random_state=42 because if you rerun code, it splits by using same rule

# As a result of using random_state=42, we can get same accuracy.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
# Logistic regression

# Here we import and use logistic regression to make and test model.

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

# score predicts and gives the accuracy of model 

print("test accuracy {}".format(lr.score(x_test,y_test)))
# Here we import dataset to python and assign it to data1 variable

data1 = pd.read_csv("../input/list_attr_celeba.csv")
# We make list of men and women celebrities.

Man=[]

Woman=[]

for i in data.Male.values:

    if i == 1:

        Man.append(i)

    else:

        Woman.append(i)
# Bar chart that shows the number of man and woman celebrity

top=[('Man',Man.count(1)),('Woman',Woman.count(-1))]

labels, ys = zip(*top)

xs = np.arange(len(labels)) 

width = 1

plt.bar(xs, ys, width, align='center')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels

plt.yticks(ys)
# we drop the "image_id" colomn because we dont need it while coding

data1.drop(["image_id"],axis=1,inplace = True)
# we assign Young feature as y1 variable which is our target data her.

y1 = data.Male.values

# we drop Young feature from data set and assign it as x_data1 

x_data1 = data.drop(["Male"],axis=1)
# Normalization

# Normalization is required if there is big diffirence between your features data

# Here there is not big difference but I prefer it to keep program in safe

x1 = (x_data1 - np.min(x_data1))/(np.max(x_data1)-np.min(x_data1)).values
# Train-Test Split

# Here we split the data 70 percent for training and 30 percent for testing

# we write random_state=42 because if you rerun code, it splits by using same rule

# As a result of using random_state=42, we can get same accuracy.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x1,y1,test_size = 0.3,random_state=42)
# Logistic regression

# Here we import and use logistic regression to make and test model.

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() 

lr.fit(x_train,y_train)

# score predicts and gives the accuracy of model  

print("test accuracy {}".format(lr.score(x_test,y_test)))