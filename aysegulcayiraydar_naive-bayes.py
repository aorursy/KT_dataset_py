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
data = pd.read_csv('/kaggle/input/epirecipes/epi_r.csv' )
type(data['rating'][0])
data.info()

data.fillna(data.mean(),inplace = True)

x =data.iloc[:,2:6].values

y = data['rating'].values.reshape(-1,1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x)

print(y)
from sklearn.naive_bayes import GaussianNB





# instantiate the model

gnb = GaussianNB()





# fit the model

gnb.fit(x_train, y_train.ravel())