# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
test_data = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

train_data = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
y_train = train_data.loc[:,train_data.keys()[-1]]

x_train = train_data.loc[:,train_data.keys()[1:5]]
knn = KNeighborsRegressor(n_neighbors=100, weights="distance") #n_neighbors가 커지면 오버피팅이 될 수 있지만 리더보드에서 고득점이 나오길래 일단 100으로 하였습니다...!
x_test = test_data.loc[:,test_data.keys()[1:,]]
knn.fit(x_train, y_train)
predict = knn.predict(x_test)
for i in range(len(predict)):

  submit['Expected'][i]=predict[i].item()

submit
submit.to_csv("submit4.csv",mode='w',header=True,index=False)