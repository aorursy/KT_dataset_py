# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # drawing graph



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/learn-together/train.csv')

print("Size of Train dataframe is: {}".format(train_df.shape))

test_df =  pd.read_csv('/kaggle/input/learn-together/test.csv')

print("Size of Test dataframe is: {}".format(test_df.shape))
def Diff(a,b):

    return(list(set(a)-set(b)))



train = train_df.columns

test = test_df.columns

target = Diff(train,test)

print('The target is',target[0])
categories = train_df[target[0]].unique()

val = []

for i in categories:

    temp = len(train_df[train_df[target[0]]==i])

    val.append(temp)

labels=categories

sizes=val

colors=['green','red','orange','blue','darkorange','grey','pink']

plt.axis('equal')

plt.title('target classes distribution')

plt.pie(sizes, explode=(0,0,0,0,0,0,0), labels=labels,colors=colors,autopct='%1.2f%%', shadow=True, startangle=90)

plt.show()                        
print('We have following categorical features:')

print()

cat = []

cont = []

for i in test[1:]:

    temp1 = train_df[i].unique()

    temp2 = test_df[i].unique()

    if len(temp1) == len(temp2):

        print(i,':',len(temp1),'unique values')

        cat.append(i)

    else:

        cont.append(i)

print()

print('And we have',len(cont), 'of following continuous features:') 

print(cont)
train_df[cont].hist(bins=20, figsize=(15,15), color = 'orange')

plt.suptitle("Histogram for each train numeric input variable")

plt.show()
test_df[cont].hist(bins=20, figsize=(15,15), color = 'darkorange')

plt.suptitle("Histogram for each test numeric input variable")

plt.show()