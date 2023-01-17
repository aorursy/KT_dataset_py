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
df = pd.read_excel('../input/data.xlsx')
df_ref = pd.read_excel('../input/ref_data.xlsx')
df
ref_list = list(df['id'][df['ref'].isnull()])
df_new = df_ref[df_ref['id'].isin(ref_list)]
df_dict = df_new.set_index('id').T.to_dict('list')
for key, value in df_dict.items():
    df['ref'][df['id'] == key] = value
df