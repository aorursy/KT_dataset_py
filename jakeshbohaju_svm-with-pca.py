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
df= pd.read_csv("/kaggle/input/brain-tumor/bt_dataset_t3.csv")
df.head(5)
df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.mean())

y = df['Target']

X = df.drop(['Target','Image'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

steps = [('scaler', StandardScaler()),('pca',PCA()),('clf',SVC(kernel='rbf'))]

parameters = {

    'pca__n_components' :[2,3,4],

    'clf__C':[0.001,0.1,0.01,1,10,100,10e5],

    'clf__gamma':[1,0.1,0.01,0.001]

}

pipeline = Pipeline(steps)
cv=5

grid = GridSearchCV(pipeline,param_grid=parameters,cv=cv)

grid.fit(X_train,y_train)

print("Score for %d fold : = %f"%(cv,grid.score(X_test,y_test)))

print("Parameters : ",grid.best_params_)
y_pred_test = grid.predict(X_test)

print("Accuracy : ", grid.best_score_)