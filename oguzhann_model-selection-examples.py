# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.datasets import load_iris

import numpy as np

import pandas as pd

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# show head of df

iris = load_iris()



data = iris.data

feature_names = iris.feature_names



df = pd.DataFrame(data,columns = feature_names)

df.head()
df.info()
# load iris data

x = iris.data

y = iris.target



x = x[:100,:]

y = y[:100] 
#normalization

x = (x-np.min(x))/(np.max(x)-np.min(x))
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3)
#Grid search CV with logistic regression



from sklearn.linear_model import LogisticRegression



grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge



logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv = 10)

logreg_cv.fit(x_train,y_train)



print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)

print("accuracy: ",logreg_cv.best_score_)
#calculate score value

logreg2 = LogisticRegression(C=1,penalty="l1")

logreg2.fit(x_train,y_train)

print("score: ", logreg2.score(x_test,y_test))
