# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_dict = {'category' : ['good', 'bad', 'good', 'bad', 'good',

                      'bad', 'good', 'bad', 'good', 'bad'],

            'level' : [1, 1, 2, 3, 4, 4,

                      2, 2, 1, 3],

            'feature_1': np.random.randn(10),

            'feature_2': np.random.randn(10)}

data_df = pd.DataFrame(data_dict)

print(data_df)
group_1 = data_df.groupby('category')

print(list(group_1))
group_2 = data_df.groupby(['category', 'level'])

print(list(group_2))
def get_letter_type(letter):

    return int(letter//2)

group_3 = data_df.groupby(get_letter_type, axis=0)

print(list(group_3))
def get_letter_type(letter):

    if 'feature' in letter:

        return 'feature'

    else:

        return 'other'

group_3 = data_df.groupby(get_letter_type, axis=1)

print(list(group_3))