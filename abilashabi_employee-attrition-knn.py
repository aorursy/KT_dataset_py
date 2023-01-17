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
import pandas as pd

import numpy as np

import matplotlib.pyplot as  plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import pandas_profiling
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()


pandas_profiling.ProfileReport(df)

num_cols = df.select_dtypes(include=np.number).columns

cat_cols = df.select_dtypes(exclude=np.number).columns
df[cat_cols].apply(lambda x:print(x.value_counts()))
df.OverTime.replace({"Yes":1,"No":0},inplace = True)
df.Over18.replace({"Y":1},inplace = True)
df_onehot = pd.get_dummies(df[cat_cols.drop(["Attrition","Over18","OverTime"])])
df_onehot.head()
df_final = pd.concat([df_onehot,df[num_cols],df["Attrition"],df["Over18"],df["OverTime"]],axis = 1)
df_final.head()
X = df_final.drop(columns=['Attrition'])

X.head()
Y = df_final[['Attrition']]
Y.head()
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X
from sklearn.model_selection import train_test_split 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state= 42)
from sklearn.neighbors import KNeighborsClassifier 

from math import sqrt

from sklearn import metrics
length = round(sqrt(df.shape[0])) 

length
accuracy_dict = {}

accuracy_list = []

for k in range(1,length+1):

    model = KNeighborsClassifier(n_neighbors = k,weights='uniform', algorithm='auto').fit(X_train,Y_train)

    Y_predict = model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test,Y_predict)

    accuracy_dict.update({k:accuracy})

    accuracy_list.append(accuracy)

    print("Accuracy ---> k = {} is {}" .format(k,accuracy))
key_max = max(accuracy_dict.keys(), key=(lambda k: accuracy_dict[k]))



print( "The Accuracy value is ",accuracy_dict[key_max], "with k= ", key_max)
elbow_curve = pd.DataFrame(accuracy_list,columns = ['accuracy'])
elbow_curve.plot()