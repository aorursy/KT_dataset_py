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
train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

train.shape, test.shape
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train.isnull().any().describe()


test.isnull().any().describe()
import seaborn as sns

g = sns.countplot(Y_train)
X_train = X_train / 255.0

test = test / 255.0
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



grid_params = {

    "n_neighbors": [1, 2, 3, 4, 5]

}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, n_jobs=-1, cv=3, verbose=10)

gs_result = gs.fit(X_train, Y_train)
gs_result.best_score_
gs_result.best_estimator_
gs_result.best_params_
preds = gs_result.best_estimator_.predict(test)
np.savetxt('submission.csv', 

           np.c_[range(1,len(test)+1),preds], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')