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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_in  = pd.read_csv('../input/train.csv')

df_train = df_in.iloc[0:100000,:]

df_test = df_in.iloc[100001:,:]
df_in.shape

#df_test.shape

#df_train.head()
from sklearn import linear_model as lm





df_train.describe()



reg = lm.LinearRegression()

X_cols=['cont1','cont2','cont3','cont4','cont6'] 

y_cols = ['loss']



X = df_train[X_cols]

y = df_train[y_cols]

reg.fit(X,y)



X_test = df_test[X_cols]

y_test = df_test[y_cols]

y_pred = reg.predict(X_test)



error = y_test - y_pred

centerd = y_test - np.mean(y_test)





R2 = 1-error.T.dot(error)/(centerd.T.dot(centerd))

R2
import matplotlib.pyplot as plt



plt.scatter(X,y)