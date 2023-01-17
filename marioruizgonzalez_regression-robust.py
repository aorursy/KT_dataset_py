# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd



from sklearn.linear_model import (

    RANSACRegressor, HuberRegressor

)

from sklearn.svm import SVR



from sklearn.model_selection import train_test_split

from sklearn.metrics  import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":

    dataset =  pd.read_csv('/kaggle/input/felicidad_corrupt.csv')

    print(dataset.head(5))



    X = dataset.drop(['country', 'score'], axis=1)

    y = dataset[['score']]



    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)



    estimadores = {

        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),

        'RANSAC' : RANSACRegressor(),

        'HUBER' : HuberRegressor(epsilon=1.35)

    }

    

    



    for name, estimador in estimadores.items():

        estimador.fit(X_train, y_train)

        predictions = estimador.predict(X_test)

        print("="*64)

        print(name)

        print("MSE: ", mean_squared_error(y_test, predictions))