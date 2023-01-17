# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder





# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

train.head()

#(Initial code borrowed from this notebook: https://www.kaggle.com/uluumy/mercedez-baseline-2)
X_train = train.drop('y', axis = 1)

Y_train = train['y']
import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings("ignore")



from sklearn import linear_model, decomposition, datasets

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import make_scorer

    

r2 = make_scorer(r2_score)

n_components = [100, 200, 300, 350]



regr = linear_model.Ridge()

pca = decomposition.PCA()

pipe = Pipeline(steps=[('pca', pca), ('linear', regr)])



estimator = GridSearchCV(pipe,dict(pca__n_components=n_components, linear__alpha = [0.0, 1.0, 2.0, 4.0, 16.0, 32.0, 64.0, 128.0, 256.0]),verbose = 1, scoring = r2)



estimator.fit(X_train, Y_train)



plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,

            linestyle=':', label='n_components chosen')

plt.legend(prop=dict(size=12))

plt.show()
y_pred = estimator.predict(test)
ID = list(test['ID'])

y_pred = list(y_pred)

print (y_pred[:5], ID[:5])
outputfile = open('result.csv', "w+")

outputfile.write("ID,y\n")

print (len(ID), len(y_pred))

for i in range(len(ID)):

    outputfile.write(str(ID[i])+ "," + str(y_pred[i])+"\n" )

outputfile.close()    

    
estimator.cv_results_