# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import sklearn



from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":

    

    dt_heart = pd.read_csv('/kaggle/input/hearts/heart.csv')



    print(dt_heart['target'].describe())



    X = dt_heart.drop(['target'], axis=1)

    y = dt_heart['target']



    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)



    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)

    boost_pred = boost.predict(X_test)

    print("="*64)

    print(accuracy_score(boost_pred, y_test))



#implementacion_boosting