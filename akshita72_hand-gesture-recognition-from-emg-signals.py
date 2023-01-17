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
df0 = pd.read_csv("/kaggle/input/emg-4/0.csv", header=None )

df1 = pd.read_csv("/kaggle/input/emg-4/1.csv", header=None )

df2 = pd.read_csv("/kaggle/input/emg-4/2.csv", header=None )

df3 = pd.read_csv("/kaggle/input/emg-4/3.csv", header=None )

df = pd.concat([df0,df1,df2,df3], axis = 0)
df.head()
x = df.loc[:,0:63]

y = df[64]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = pd.DataFrame(sc.fit_transform(x_train))

x_test = pd.DataFrame(sc.transform(x_test))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



lr_grid = {'max_depth' : [4,8,16,32,64,128],

           'criterion' : ['entropy','gini']}



clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)



gs = GridSearchCV(estimator = clf, param_grid=lr_grid,cv = 5)

gs.fit(x_train,y_train)

y_pred = gs.predict(x_test)
gs.best_params_
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



print('Classification Report: \n', classification_report(y_test,y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test,y_pred))