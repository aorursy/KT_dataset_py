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
data=pd.read_csv('/kaggle/input/apndcts/apndcts.csv')

data.head()
data.info()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.tree import DecisionTreeClassifier
y=data.pop('class')

x=data



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)



model=DecisionTreeClassifier()

model.fit(x_train,y_train)



y_pre=model.predict(x_test)



print('Accu: ',model.score(x_test,y_test))

print('Matrix\n',confusion_matrix(y_test,y_pre))

print('Report\n',classification_report(y_test,y_pre))