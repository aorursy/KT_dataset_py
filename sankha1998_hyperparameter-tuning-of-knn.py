# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sample=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample.head()
train.head()
X = train.iloc[:,1:].values

y = train.iloc[:,0]
def ch(X,y,i):   

    print(y[i])

    plt.imshow(X[i].reshape(28,28))

    (X[i].reshape(1,784))

    plt.show()

    
ch(X,y,305)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
param_dict={

    'max_depth':[5,10,15,20],

    'splitter':['best', 'random'],

}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(clf,param_grid=param_dict,cv=10)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
grid.best_score_
grid.best_params_
Xf = test.iloc[:,:].values
y_pred = grid.predict(Xf)
data=pd.DataFrame()
data['ImageId']=sample.ImageId

data['Label']=y_pred
data.to_csv('submission.csv',index=False)



