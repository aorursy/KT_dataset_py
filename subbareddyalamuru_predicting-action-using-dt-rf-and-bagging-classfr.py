# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from catboost import datasets

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/amazon-employee-access-challenge/train.csv')

test_data = pd.read_csv('/kaggle/input/amazon-employee-access-challenge/test.csv')
train_data.head()
df = train_data
""" iterate through all the columns of a dataframe and modify the data type

    to reduce memory usage.        

"""

start_mem = df.memory_usage().sum() / 1024**2

print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



for col in df.columns:

    col_type = df[col].dtype



    if col_type != object:

        c_min = df[col].min()

        c_max = df[col].max()

        if str(col_type)[:3] == 'int':

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)  

        else:

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)

    else:

        df[col] = df[col].astype('category')



end_mem = df.memory_usage().sum() / 1024**2

print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))    
df.isna().sum()
plt.figure(figsize=(30,20))

for i in range(1,10):

    plt.subplot(5,2,i)

    plt.hist(df[df.columns[i]])

    plt.xlabel(df.columns[i])

    plt.ylabel("Frequency")

plt.show() 
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidth=1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
models = [DecisionTreeClassifier(),RandomForestClassifier(),BaggingClassifier()]
y = df['ACTION']
x = df.drop(['ACTION'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lst = []

for i in models:

    dt = i

    print(type(i))

    dt.fit(X_train,y_train)

    lst.append([type(i),confusion_matrix(y_test, dt.predict(X_test)),accuracy_score(y_test, dt.predict(X_test))])

    print(cross_val_score(i,X_train,y_train,cv=10))

    
scores = pd.DataFrame(lst)
scores
test_data.head()
df1 = test_data.drop(['id'],axis=1)
df1.head()
lst = []

dt = BaggingClassifier()

dt.fit(x,y)

lst.append(dt.predict(df1))
lst[0]
test_data['Action'] = lst[0].T
test_data