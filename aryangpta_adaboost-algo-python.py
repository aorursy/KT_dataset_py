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
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import random
credit_data = pd.read_csv('../input/dataset2/bankloans.csv')
credit_data.head()
credit_data.shape
credit_data = credit_data.fillna(0)
target = credit_data['default']
credit_data.head()
credit_data['weightage'] = 1/850
credit_data.head(5)
def adaboost_trees(data,target,model):

    DT = DecisionTreeClassifier(max_depth = 1)
    DT.fit(data, target)
    model.append(DT)
    pred = DT.predict(data)
    return pred

num_estimators = 3
pred_list = []
counter = 0
sgn = []
model = []
train = credit_data.drop(['default'], axis=1)
for i in range(num_estimators):
    tgt = credit_data['default']
    pred = adaboost_trees(train,tgt,model)
    pred_list.append(pred)
    arr_index = train.index.to_list()
    arr_index = np.array(arr_index)
    predn = pd.DataFrame({'index':arr_index, 'prediction':pred})
    predn = predn.set_index('index')
    for j in range(765):
        if pred[j] == target[j]:
            index = arr_index[j]
            predn = predn.drop(index)
    total_error = 0
    list_index = predn.index.to_list()
    for i in list_index:
        weight = train.at[i, 'weightage']
        total_error = total_error + weight
    significance = 0.5*math.log((1-total_error)/total_error)
    sgn.append(significance)
    print(total_error)
    print(significance)
    #train_x['new_wt'] = 0
    for i in arr_index:
        if i in list_index:
            weight = train.at[i,'weightage']*math.exp(significance) 
            train.at[i, 'new_wt'] =  weight
            
        else:
            weight = train.at[i,'weightage']*math.exp(-significance)
            train.at[i,'new_wt'] = weight
    wt_sum = float(train['new_wt'].agg(['sum']))
    print(wt_sum)
    train['new_wt_norm'] = train['new_wt']/wt_sum
    train['weightage'] = train['new_wt_norm']
    
train.head()
sgn
final_pred1 = []
final_pred1 = pred_list[1]*sgn[1]
final_pred0 = []
final_pred0 = pred_list[0]*sgn[0]
final_pred2 = []
final_pred2 = pred_list[2]*sgn[2]
final = final_pred1 + final_pred2 + final_pred0
final
