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
import seaborn as sns
x= pd.read_csv("../input/heart.csv")
x.head()
x.describe()
x.columns
sns.boxplot(x="age",y="chol",data=x,palette = "hls")
sns.boxplot(x="thalach",y="age",data=x,palette = "hls")
sns.pairplot(data=x)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(x,test_size = 0.3)
test.head()
train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]
# kernel = linear

help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)