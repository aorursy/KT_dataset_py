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
#Boris kakoy orex blyatS

with open('Ai academy 2019\dota2_skill_train.jsonlines') as fin:

    for line in tqdm.tqdm_notebook(fin):

        record = json.loads(line)

        trainx.at[record['id'],'all_item_list'] = [i['item_id'] for i in record['item_purchase_log']]

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

trainx = trainx.join(pd.DataFrame(mlb.fit_transform(trainx.pop('all_item_list')),

                          columns=mlb.classes_,

                          index=trainx.index))