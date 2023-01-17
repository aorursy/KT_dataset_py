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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split 
df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

print(df.shape)

df.head(5)
# Selecting from Third to Thirty Second Column

X= df.iloc[:,2:32]

# Selecting second Columnn

y=df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

#create an instance and fit the model 

logmodel = LogisticRegression(max_iter=5000)

logmodel.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

nc=np.arange(0.1,0.9,0.1)

acc=np.empty(8)

i=0

for k in np.nditer(nc):

    y_predp = (logmodel.predict_proba(X_test)[:,1] >= k).astype(bool)

    y_pred=np.full(y_pred.size,'B')

    y_pred[y_predp]='M'

    acc[i]=accuracy_score(y_test, y_pred)

    i = i + 1

acc
x=pd.Series(acc,index=nc)

x.plot()

# Add title and axis names

plt.title('Class 1 Probability vs Accuracy')

plt.xlabel('Class 1 Probability')

plt.ylabel('Accuracy')

plt.show() 