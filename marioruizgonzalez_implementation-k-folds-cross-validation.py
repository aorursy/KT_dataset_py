# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor



from sklearn.model_selection import (

    cross_val_score, KFold

)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":



    dataset = pd.read_csv('/kaggle/input/felicidad.csv')



    X = dataset.drop(['country', 'score'], axis=1)

    y = dataset['score']



    model = DecisionTreeRegressor()

    score = cross_val_score(model, X,y, cv= 3, scoring='neg_mean_squared_error')

    print(np.abs(np.mean(score)))



  
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for train, test in kf.split(dataset):

    print(train)

    print(test)



#implementacion_cross_validation