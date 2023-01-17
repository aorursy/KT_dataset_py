# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import accuracy_score

from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#data

df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
#useful functions

def basic_info(data):

    print("----------Top-5- Record----------")

    print(data.head(5))

    print("----------Bottom-5- Record----------")

    print(data.tail(5))

    print("-----------Information-----------")

    print(data.info())

    print("-----------Data Types-----------")

    print(data.dtypes)

    print("----------Missing value-----------")

    print(data.isnull().sum())

    print("----------Null value-----------")

    print(data.isna().sum())

    print("----------Shape of Data----------")

    print(data.shape)
basic_info(df_train)
basic_info(df_test)
#assign batch_no for every row

df_train['batch'] = [i//500000 for i in df_train.index]

df_test['batch'] = [i//500000 for i in df_test.index]
print("unique batch no in train data:", df_train.batch.unique()) 

print("unique batch no in test data:",df_test.batch.unique())

print("unique openchannels in train data:",df_train.open_channels.unique())
#lineplot for every train data



fig = plt.figure(figsize=(21, 7))



ax2 = fig.add_subplot(3,1,1)

plt.plot(df_train.open_channels)

plt.title("Lineplot of open_channels",fontsize = 10,color='red')



ax1 = fig.add_subplot(3,1,2)

plt.plot(df_train.signal)

plt.title("Lineplot of signal",fontsize = 10,color='black')



ax3 = fig.add_subplot(3,1,3)

plt.plot(df_test.signal)

plt.title("Lineplot of signal in test data",fontsize = 10,color='green')



plt.show() 
df_train["time_last"] = [round(x % 50, 4)  for x in df_train.time]

df_test["time_last"] = [round(x % 50, 4)  for x in df_test.time]
#plot target distribution for every batch



batch_count = len(df_train.batch.unique())



fig = plt.figure(figsize=(20, 30))

i = 1

for b in df_train.batch.unique():

    ax = fig.add_subplot(batch_count/2,2,i)

    sns.countplot(x='open_channels', data=df_train[df_train.batch==b], ax=ax)

    i += 1

    plt.title("batch=%s" % b,fontsize = 10,color='black')

    

plt.show()  
#the relation between signal and open channels



batch_count = len(df_train.batch.unique())



fig = plt.figure(figsize=(20, 30))

i = 1

for b in df_train.batch.unique():

    ax = fig.add_subplot(batch_count/2,2,i)

    sns.scatterplot(y='open_channels', x='signal', data=df_train[df_train.batch==b], ax=ax)

    i += 1

    plt.title("batch=%s" % b,fontsize = 10,color='black')

    

plt.show() 
#scatter plot for total dataset

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111)

plt.scatter(y='open_channels', x='signal', data=df_train)

plt.show()
#distplot of signal in each batch

batch_count = len(df_train.batch.unique())



fig = plt.figure(figsize=(20, 50))

i = 1

for b in df_train.batch.unique():

    ax1 = fig.add_subplot(batch_count,2,i)

    sns.distplot(df_train[df_train.batch==b].signal, ax=ax1)

    plt.title("batch=%s" % b,fontsize = 10,color='black')

    

    ax2 = fig.add_subplot(batch_count,2,i+1)

    sns.distplot(df_train[df_train.batch==b].open_channels, ax=ax2, kde=False)

    plt.title("batch=%s" % b,fontsize = 10,color='red')

    i += 2

    

plt.show() 
#distplot of signal by open channels



openC_count = len(df_train.open_channels.unique()) + 1



fig = plt.figure(figsize=(20, 30))

i = 1

for o in np.sort(df_train.open_channels.unique()):

    ax = fig.add_subplot(openC_count/2,2,i)

    sns.distplot(df_train[df_train.open_channels==o].signal, ax=ax)

    i += 1

    plt.title("open channels=%s" % o,fontsize = 10,color='red')

    

plt.show() 
#boxplot of signal by open channels

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1,1,1)

sns.boxplot(y=df_train.signal, x=df_train.open_channels, ax=ax,palette="Blues")



plt.title("signal vs. open channels",fontsize = 10,color='red')

plt.show() 
#distplot of signal in test data

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1,1,1)

sns.distplot(df_test.signal, ax=ax,color="red")

sns.distplot(df_train.signal, ax=ax, color = "blue")



plt.title("distribution of signal in test&train data",fontsize = 10,color='red')

plt.show() 
x_columns = ["time_last","signal"]

X = df_train[x_columns]

y = df_train['open_channels']



x_train, x_test, y_train, y_test = train_test_split(X, y)
#model fitting

model = DecisionTreeClassifier()

model.fit(x_train, y_train)
#validation

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



acc_train = model.score(x_train, y_train)

acc_test = model.score(x_test, y_test)



print("accuracy for train data is %.2f" % acc_train)

print("accuracy for test data is %.2f" % acc_test)
submission_csv = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
X_test = df_test[x_columns]



submission_csv["open_channels"] = model.predict(X_test).astype(int)

submission_csv['time'] = [format(submission_csv.time.values[x], '.4f') for x in range(2000000)]

submission_csv.to_csv("submission.csv", index=False)