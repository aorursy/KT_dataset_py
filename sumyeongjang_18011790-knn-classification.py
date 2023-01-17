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

import torch

import pandas as pd



torch.manual_seed(1)



xy=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv', header=None)



x_data=xy.loc[1:, 1:8]

y_data=xy.loc[1:,9]

x_data=np.array(x_data)

y_data=np.array(y_data)



x_train=torch.FloatTensor(x_data)

y_train=torch.FloatTensor(y_data)



test=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv', header=None)

x_data=test.loc[1:, 1:8]



x_data=np.array(x_data)

x_test=torch.FloatTensor(x_data)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9,p=2)

knn.fit(x_train, y_train)
predict=knn.predict(x_test)
submission=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

for i in range(len(predict)):

  submission["Label"][i]=int(predict[i])

submission["Label"]=submission["Label"].astype(int)

submission.to_csv('my_submission.csv', index=False, header=True)
submission