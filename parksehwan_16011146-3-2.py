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
import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
test = pd.read_csv("../input/mlregression-cabbage-price/test_cabbage_price.csv")

train = pd.read_csv("../input/mlregression-cabbage-price/train_cabbage_price.csv")
train_x = train.loc[:,train.keys()[1:-1]]

train_y = train.loc[:,train.keys()[-1]]

test_x = test.loc[:,test.keys()[1:,]]
knn = KNeighborsRegressor(n_neighbors=700,weights = "distance")

knn.fit(train_x,train_y)
testresult = knn.predict(test_x)
testresult
testresult = testresult.astype(np.int32)

id = np.array([i for i in range(0,731)])

sub = np.hstack([id.reshape(731,1),testresult.reshape(731,1)])

submission = pd.DataFrame(sub,columns=["ID","Expected"])

submission.to_csv("submission.csv",mode='w',header=True,index=False)
submission