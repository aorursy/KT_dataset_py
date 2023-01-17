# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn import preprocessing

%matplotlib inline

import matplotlib.pyplot as plt



df=pd.read_csv("../input/data.csv",header = 0)
df.head()
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()
data_df= list(df.columns[1:31]) 

data_df_main = df.loc[:,data_df]

data_df_main
X = data_df_main

y = df['diagnosis']
svm_model = SVC()



parameters = [

              {'C': [1, 10, 100, 1000], 

               'kernel': ['linear']

              },

              

 ]
grid_svm = GridSearchCV(svm_model, parameters, cv=20, scoring="accuracy")

grid_svm.fit(X,y)
print(grid_svm.best_score_)
