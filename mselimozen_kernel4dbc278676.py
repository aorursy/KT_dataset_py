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

import matplotlib.pyplot as plt



veri = pd.read_csv('../input/sandp500/all_stocks_5yr.csv')



veri.drop(['date'], axis = 1, inplace = True)





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

name = veri.iloc[:,5:6].values

name = le.fit_transform(name)



veri.drop(['Name'], axis = 1, inplace = True)

name = pd.DataFrame(data = name, index = range(619040), columns = ['Name'])



tam_veri = pd.concat([veri, name], axis = 1)



kapanıs = tam_veri.iloc[:,3:4]

tam_veri.drop(['close'], axis = 1, inplace = True)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tam_veri, kapanıs, test_size = 0.33, random_state = 0)







from sklearn.linear_model import LinearRegression





x_train = x_train.fillna(x_train.mean())

y_train = y_train.fillna(y_train.mean())

reg = LinearRegression()

reg.fit(x_train, y_train)

x_test = x_test.fillna(x_test.mean())

y_pred = reg.predict(x_test)

y_pred = pd.DataFrame(data = y_pred, index = range(204284), columns = ['y_pred'])

plt.plot(y_test, y_pred, 'blue')

plt.show()
