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
#importing the modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
#importing the dataset

data = pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")

data
#counting the number of columns to set X and y

len(data. columns)
#find the missing values and dropping them

data.isna().sum()
data.shape
data.dropna(inplace=True)
#selecting x and y

X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
model = LogisticRegression(solver='lbfgs')

model.fit(X_train,y_train)
y_preds= model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_preds)

print(cm)
import seaborn as sns

sns.heatmap(cm,annot = True)
sns.heatmap(cm/np.sum(cm), annot=True, fmt = '.2%', cmap='Blues')
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_preds))