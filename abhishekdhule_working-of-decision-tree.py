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
!pip install pydotplus
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from IPython.display import Image
import pydotplus
data=pd.read_csv('/kaggle/input/data.csv')
data.head()
X= data[['age','bp']]
y=data[['diabetes']]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
model1=tree.DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=6,max_depth=5)
model1.fit(X_train,y_train)
y_pred_train=model1.predict(X_train)
print('train:',accuracy_score(y_train,y_pred_train))
y_pred_test=model1.predict(X_test)
print('test:',accuracy_score(y_test,y_pred_test))

dot_data=tree.export_graphviz(model1,out_file=None,feature_names=X.columns,class_names=str(y['diabetes'].unique()))

graph=pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
bp_lessthanequal_79 = X_train[X_train['bp']<=79.0].shape[0]
bp_greaterthan_79 = X_train.shape[0]-bp_lessthanequal_79

print('Number of samples with bp<=79 :',bp_lessthanequal_79)
print('Number of samples with bp>79 :',bp_greaterthan_79)
print('number of patients having diabetes are : ',(y_train['diabetes']==1).sum())
print('number of patients not having diabetes are : ',(y_train['diabetes']==0).sum())
bp_lessthanequal_70=X_train[X_train['bp']<=70]
bp_greaterthan_70=X_train[X_train['bp']>70]
number_lessthanequal_70=bp_lessthanequal_70.shape[0]
number_bp_greaterthan_70=bp_greaterthan_70.shape[0]

print('number of patients with bp<=70 are : ',number_lessthanequal_70)
print('number of patients with bp>70 are : ',number_bp_greaterthan_70)
print('number of patients having diabetes are : ',(y_train['diabetes']==1).sum())
print('number of patients not having diabetes are : ',(y_train['diabetes']==0).sum())
positives_on_left=np.logical_and(X_train['bp']<=70,y_train['diabetes']==1).sum()
negatives_on_left=np.logical_and(X_train['bp']<=70,y_train['diabetes']==0).sum()

print('Number of patients with bp<=70 and have diabetes are : ',positives_on_left)
print('Number of patients with bp>70 and have diabetes are : ',negatives_on_left)
positives_on_right=np.logical_and(X_train['bp']>70,y_train['diabetes']==1).sum()
negatives_on_right=np.logical_and(X_train['bp']>70,y_train['diabetes']==0).sum()

print('Number of patients with bp<=70 and have diabetes are : ',positives_on_right)
print('Number of patients with bp>70 and have diabetes are : ',negatives_on_right)