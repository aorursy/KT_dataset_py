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
from clustergrammer2 import net

df = {}
df['ini'] = pd.read_csv('../input/StudentsPerformance.csv')

print(df['ini'].shape)

df['ini'].head()
rows = df['ini'].index.tolist()
keep_cols = df['ini'].columns.tolist()[5:]

mat = df['ini'][keep_cols].get_values()

mat.shape
rows[0]
new_rows = [('S' + str(x),

             'Gender: ' + df['ini']['gender'][x],

             'Education: ' + df['ini']['parental level of education'][x], 

             'Lunch: ' + df['ini']['lunch'][x], 

             'Test Prep: ' + df['ini']['test preparation course'][x] 

            ) for x in rows]
new_rows[:3]
clean_cols = [x.replace(' score','') for x in keep_cols]

df['cat'] = pd.DataFrame(data=mat, columns=clean_cols, index=new_rows).transpose()

df['cat'].head()
net.load_df(df['cat'])

net.normalize(axis='row', norm_type='zscore')

net.widget()