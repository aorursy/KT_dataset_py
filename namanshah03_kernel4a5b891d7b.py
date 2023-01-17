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




train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')

test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')



train_df.head()




X_train = train_df.drop(['Id', 'label', 'Soil'], axis=1)

Y_train = train_df['label']



X_test = test_df.drop(['Id', 'Soil'], axis=1)

X_test.head()
from sklearn.ensemble import RandomForestClassifier as RFC

rfc_b = RFC()

rfc_b.fit(X_train,Y_train)

y_pred = rfc_b.predict(X_train)

y_pred_test = rfc_b.predict(X_test)
submission_df = pd.DataFrame()

submission_df['Id'] = test_df['Id']
submission_df['Predicted'] = y_pred_test.tolist()
submission_df.tail()
submission_df.to_csv('20180066.csv',index=False)
!ls