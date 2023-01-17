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
import time
start_time = time.time()
import pandas as pd
from IPython.display import display
import numpy as np
df_train = pd.read_json('../input/train.json')
display(df_train)
trainIngredients = df_train.drop(['cuisine','id'], axis=1)
display(trainIngredients)
trainID = df_train.id
trainCuisine = df_train.cuisine
df_test = pd.read_json('../input/test.json')
display(df_test)
testIngredients = df_test.drop(['id'], axis=1)
testID = df_test.id
ing_set = set()
cuisine_set = set()
for i,row in trainIngredients.iterrows():
    for food in row[0]:
        ing_set.add(food)
for i,row in testIngredients.iterrows():
    for food in row[0]:
        ing_set.add(food)
print(ing_set)
for cus in trainCuisine:
    cuisine_set.add(cus)
print(cuisine_set)