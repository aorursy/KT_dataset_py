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
%%time
path_train = '../input/train.json'
path_test  = '../input/test.json'
path_validation = '../input/validation.json'
%%time
def datapre(path):
    df = pd.read_json(path)
    if 'annotations' in list(df.columns):
        dg = pd.read_json(df['annotations'].to_json()).T.sort_values(by = ['image_id'])
        dg1 = pd.read_json(df['images'].to_json()).T.sort_values(by = ['image_id'])
        del df
        dg['url'] = dg1['url'].str[0]
        del dg1
        return dg
    else:
        dg1 = pd.read_json(df['images'].to_json()).T.sort_values(by = ['image_id'])
        del df
        dg1['url'] = dg1['url'].str[0]
        return dg1
%%time
df_train = datapre(path_train)
df_validation  = datapre(path_validation)
df_test = datapre(path_test)
df_train.head()
df_test.head()
df_validation.head()
