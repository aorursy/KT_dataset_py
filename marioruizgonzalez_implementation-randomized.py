# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":



    dataset = pd.read_csv('/kaggle/input/felicidad/felicidad.csv')



    print(dataset)



    X = dataset.drop(['country', 'rank', 'score'], axis=1)

    y = dataset[['score']]



    reg = RandomForestRegressor()



    parametros = {

        'n_estimators' : range(4,16),

        'criterion' : ['mse', 'mae'],

        'max_depth' : range(2,11)

    }



    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)



    print(rand_est.best_estimator_)

    print(rand_est.best_params_)

    print(rand_est.predict(X.loc[[0]]))