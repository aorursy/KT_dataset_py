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

import os

from sklearn.neighbors import KNeighborsClassifier  #KNN 불러오기

train_csv=pd.read_csv("/kaggle/input/sejongai-challenge-pretest-1/train.csv")
train_csv
train_x=train_csv.loc[:,train_csv.keys()[1:-1]]

train_y=train_csv.loc[:,train_csv.keys()[-1]]
knn=KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(train_x,train_y)
test_csv=pd.read_csv("/kaggle/input/sejongai-challenge-pretest-1/test_data.csv")
test_csv
test_x=test_csv.loc[:,test_csv.keys()[1:]]
predict=knn.predict(test_x)
predict
##### Make CSV

predict=predict.astype(np.uint32)

id=np.array([i for i in range(predict.shape[0])]).reshape(-1,1).astype(np.uint32)

result=np.hstack([id,predict.reshape(-1,1)])

df=pd.DataFrame(result,columns=["ID","Label"])

df.to_csv("result.csv",index=False)