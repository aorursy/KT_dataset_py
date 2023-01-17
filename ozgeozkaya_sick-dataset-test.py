# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director
data = pd.read_csv('../input/sick-dataset/SICK_train.txt', sep='\t')



display(data)
data_ids = pd.DataFrame(columns = ['sentences'])



data_ids['sentences'] = pd.concat([data['sentence_A'], data['sentence_B']])
data_ids['id'] = data_ids.groupby(data_ids.columns.tolist(), sort=False).ngroup() + 1



data_ids.reset_index(drop=True, inplace=True)
split_point = (len(data['sentence_A']) - 1)

id1 = data_ids['id'].loc[:split_point]

id2 = data_ids['id'].loc[split_point + 1:]

score = data['relatedness_score'].tolist()



formatted_data = pd.DataFrame({'id1': id1.tolist(), 'id2': id2.tolist(), 'score': score}) 



display(formatted_data)