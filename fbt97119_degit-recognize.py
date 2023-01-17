# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all' 

from sklearn.tree import plot_tree
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head()
y=train.iloc[:,0]

x=train.drop('label',axis=1)

del train
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.25,random_state=42)
Knn=KNeighborsClassifier()

Knn.fit(x_train,y_train)

y_pre=Knn.predict(x_test)
sum(y_pre==y_test)/y_test.shape[0]
result=Knn.predict(test) #19:26  about half hours 

result=pd.DataFrame(result)

result.set_index(np.arange(1,28001),inplace=True)

result.index.names=['ImageId']

result.columns=['Label']
result.to_csv('submission.csv')

result