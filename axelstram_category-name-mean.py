# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def calculate_mean_deal_prob(dic):
    probabilities = list(dic.values())
    probs = []
    for p in probabilities:
        probs.append(p['deal_probability'])
        
    return sum(probs) / float(len(probs))
def get_predictions(df, dic, mean):
    y_pred = []
    
    for row in df.itertuples():
        category_name = row.category_name

        if category_name in dic:
            y_pred.append(dic[category_name]['deal_probability'])
        else:
            y_pred.append(mean)
    
    return y_pred
def rmse(x,y): return math.sqrt(((x-y)**2).mean())
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train = df[df.activation_date <= '2017-03-24']
df_val = df[df.activation_date > '2017-03-24']
len(df_train)
len(df_val)
groupby = df_train.groupby(by=['category_name'])
deal_prob_by_category_name = (groupby.agg({'deal_probability': 'sum'})) / (groupby.agg({'deal_probability': 'count'}))
dic = deal_prob_by_category_name.to_dict(orient='index')
len(dic)
mean = calculate_mean_deal_prob(dic)
mean
y = df_val['deal_probability'].as_matrix().ravel()
y_pred = get_predictions(df_val, dic, mean)
rmse(y, y_pred)
y_pred = get_predictions(df_test, dic, mean)
df_test['deal_probability'] = y_pred
df_test[['item_id','deal_probability']].to_csv('submission.csv', index=False)