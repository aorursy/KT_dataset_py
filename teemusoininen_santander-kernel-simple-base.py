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
train_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
train_data.head()
test_data=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
test_data.head()
test_x=test_data.copy().drop(columns=['ID_code'])
test_x.head()
train_y=train_data['target']
train_x=train_data.copy().drop(columns=['target','ID_code'])
train_x.head()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
test_y=clf.predict(test_x)
sub_df=pd.DataFrame({"ID_code":test_data["ID_code"].values})
sub_df["target"]=test_y
sub_df.to_csv("submission.csv",index=False)