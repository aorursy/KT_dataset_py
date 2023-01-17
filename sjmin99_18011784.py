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
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=5, p=2)
pd_x = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/train.csv")

pd_y = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/test_data.csv")



pd_y = pd_y.dropna(axis=1)



pd_x_train = pd_x.iloc[:,1:-1]

pd_x_test = pd_x.iloc[:,-1]



pd_y = pd_y.iloc[:,1:]



x_train = np.array(pd_x_train)

x_test = np.array(pd_x_test)

y_data = np.array(pd_y)

# y_test = np.array(pd_y_test)|



print(x_train.shape, y_data.shape)
knn.fit(x_train, x_test)
predict = knn.predict(y_data)
ID = np.array([i for i in range(len(predict))]).reshape(-1,1)

Label = predict.reshape(-1,1)



result = np.hstack((ID,Label))
df = pd.DataFrame(result, columns=("ID","Label"),dtype=int)

df.to_csv("result.csv",index=False)