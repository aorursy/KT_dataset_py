# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
def combine_data(x):
    return " ".join(map(str, x))
cols_combined = train.iloc[:,2:].apply(lambda x : combine_data(x), axis = 0)
data = pd.DataFrame({'cols' : train.columns[2:],'combined' : cols_combined})
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer( binary = True)
count_model = model.fit_transform(data.combined)
sim = np.dot(count_model,count_model.T)
sim.todense().shape
diagonalval = sim.diagonal()
column_sim = pd.DataFrame(sim / diagonalval, index = train.columns[2:], columns = train.columns[2:])
column_sim.sort_values(by = 'f190486d6', ascending = False).index[:40] 
train['combined'] = train.iloc[:,2:].apply(lambda x : combine_data(x), axis = 1)
data = train[['ID','target','combined']]
data.head()
model_2 = CountVectorizer(binary = True)
count_model_2 = model_2.fit_transform(data.combined)
sim = np.dot(count_model_2,count_model_2.T)
sim.todense().shape
diagonalval = sim.diagonal()
row_sim = pd.DataFrame(sim / diagonalval)
row_sim.iloc[:,2071].sort_values(ascending=False)
