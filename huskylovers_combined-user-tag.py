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
train_data = pd.read_csv("../input/data-preprocessing/treemodel_train.csv")

test_data = pd.read_csv("../input/data-preprocessing/treemodel_test.csv")

train_data.head()
tag_data = pd.read_csv("../input/tag-process2/user_tag.csv")

tag_data.head()
del tag_data['tag_label']

del tag_data['user_id']
tag_data.head()
for column in tag_data.columns:

    tag_data[column] = tag_data[column].apply(lambda x:int(np.random.binomial(1,x,1)))
index = len(train_data)

train_data = pd.concat([train_data,tag_data[:index]],axis=1)

test_data = pd.concat([test_data,tag_data[index:]],axis=1)
train_data.to_csv('train_user_tag.csv',index=False)

test_data.to_csv('test_user_tag.csv',index=False)