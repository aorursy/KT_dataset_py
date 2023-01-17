# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
def pad_labels(data_input):
    results = data_input.split(' ', 9)
    input_value = list(map(int, results))
    padded_order = [0,1,2,3,4,5,6,7,8]
    final_list = ['N','N','N','N','N','N','N','N','N']
    for index, val in enumerate(padded_order):
        if val in input_value:
            final_list[index] = 'Y'
    return ' '.join(e for e in final_list)
train.labels.apply(str)
train=train.fillna({'labels':'0'})
train['padded_labels'] = train['labels'].apply(pad_labels)
train['Lunch'],train['Dinner'], train['Reservations'], train['Outdoor'], train['Expensive'], train['Alcohol'], train['TableService'], train['Classy'], train['Kids'] = train['padded_labels'].str.split(' ', 9).str
normalized_data = train
normalized_data = normalized_data.drop(['labels', 'padded_labels'], axis=1)
plt.style.use('ggplot')
graph_df = normalized_data.Lunch.value_counts().rename('Lunch').to_frame()\
               .join(normalized_data.Dinner.value_counts().rename('Dinner').to_frame()) \
               .join(normalized_data.Alcohol.value_counts().rename('Alcohol').to_frame()) \
               .join(normalized_data.TableService.value_counts().rename('TableService').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))
normalized_data[(normalized_data['Lunch']=='Y') & (normalized_data['Dinner'] == 'Y')].Lunch.value_counts().item()
normalized_data[(normalized_data['Dinner'] == 'Y') & (normalized_data['Reservations'] == 'Y')].Dinner.value_counts().item()
normalized_data[(normalized_data['Alcohol']=='Y') & (normalized_data['Kids'] == 'Y') & (normalized_data['Outdoor'] == 'Y')].Alcohol.value_counts().item()