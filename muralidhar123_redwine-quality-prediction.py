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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv("/kaggle/input/winequalityred/winequality-red.csv",sep=",")
df.head()
df.describe()
df.isnull().sum()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

new_data= scaler.fit_transform(df.drop(labels = ['quality'],axis = 1))



columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']
new_df = pd.DataFrame(data = new_data , columns = columns)

X = new_df
y = df["quality"]

sns.pairplot(df,hue='quality',height=3.2)
plt.plot(X,y)

plt.show()
sns.heatmap(df.corr())
df.plot()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.85 , random_state=150)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)
model.predict(X_test)
score = accuracy_score(y_test,model.predict(X_test))
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,20,50,100,500,1000],'gamma':[1,0.01,0.1,1.5,2.5,0.001,0.89],'degree':range(1,10)}
grid = GridSearchCV(SVC(),param_grid,verbose=5)
grid.fit(X_train,y_train)
grid.best_params_
model_new = SVC(C=1,degree=6,gamma=0.1)

model_new.fit(X_train,y_train)
model_new.predict(X_test)
score2 = accuracy_score(y_test,model_new.predict(X_test))

score2