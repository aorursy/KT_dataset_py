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

from sklearn.model_selection import train_test_split,GridSearchCV,KFold

from sklearn.tree import DecisionTreeClassifier



train_data = pd.read_csv("../input/titanic/train_data.csv")

trainSetX = train_data[["Sex","Age","Pclass_1","Pclass_2","Pclass_3"]]

trainSetY = train_data["Survived"]



depth_val =np.arange(2,10,step=3)

min_leaf  = np.arange(10,30,step=5)



model = DecisionTreeClassifier()



X_train,X_test,y_train,y_test =train_test_split(trainSetX,trainSetY)



grid_para = [{"max_depth":depth_val,"min_samples_leaf":min_leaf}]

tree_cv = GridSearchCV(model,grid_para,cv=KFold())

tree_cv.fit(X_train,y_train)



depth = tree_cv.best_params_["max_depth"]

min_leaf=tree_cv.best_params_["min_samples_leaf"]

model = DecisionTreeClassifier(criterion="entropy",max_depth=depth,min_samples_leaf=min_leaf)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

x_pred = model.predict(X_test)

x_pred