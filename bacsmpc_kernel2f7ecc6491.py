# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas

import numpy

import matplotlib.pyplot

%matplotlib inline

import seaborn

seaborn.set() # setting seaborn default for plots



from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder ##convert test data into numbers

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier as dc
data = pandas.read_csv("../input/and-or-xor/and.csv")
data.head()
X = data.drop("0.2",axis = True)

Y = data["0.2"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)
model = dc()

model.fit(X,Y)
newdata = [1,1]

model.predict([newdata])
newdata = [0,0]

model.predict([newdata])