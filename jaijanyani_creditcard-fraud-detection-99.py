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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt     

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score





dataframe = pd.read_csv('../input/creditcard.csv')



array = dataframe.values        #Converting Pandas_DataFrame --> Numpy_Array



x_train  = array[ :  , 0 : 29]    #Splitting Input and Output

y_train  = array[ :  , 30]



scaler = MinMaxScaler(feature_range=(0, 1))    #Normalizing The Data Set

x_train2 = scaler.fit_transform(x_train)

# y_train2 = scaler.fit_transform(y_train)



x_train = x_train2

# y_train = y_train2





x1, x2, y1, y2 = train_test_split(x_train, y_train, test_size = 0.01 , random_state = 42)  #Spliting Data into Test and Train data

x1, x3, y1, y3 = train_test_split(x_train, y_train, test_size = 0.99 , random_state = 42)  #Spliting Data into Test and Train data





clf = svm.SVC()

clf.fit(x3,y3)

p = clf.predict(x2)



# print(accuracy_score(y2 , p))

print(p)

#p = pd.DataFrame(p, columns=['p']).to_csv('prediction.csv')