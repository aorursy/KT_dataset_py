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
X_train = pd.read_csv('../input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/X_train.txt',delim_whitespace=True,header=None).to_numpy()

y_train = pd.read_csv('../input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/y_train.txt',delim_whitespace=True,header=None).to_numpy().reshape(-1)

X_test = pd.read_csv('../input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/X_test.txt',delim_whitespace=True,header=None).to_numpy()
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(C=2.0)



import time

start = time.time()



logistic.fit(X_train,y_train)



end = time.time()



print("Time Required to train: {}".format(end-start))



start = time.time()



y_pred = logistic.predict(X_test)



end = time.time()



print("Time Required to test: {}".format(end-start))
import pandas as pd

submit = pd.read_csv('../input/uci-har/sample_submit.csv')
for i in range(len(y_pred)):

    submit['Label'][i]=y_pred[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)