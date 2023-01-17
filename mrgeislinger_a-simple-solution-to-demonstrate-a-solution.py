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
# Reading in the data file

df = pd.read_csv('../input/fis-pt012120-mod2-project-warmup/test.csv')

df.head()
# Grabbing one feature (going to overwrite it as `price`)

df_submit = df[['id','reviews_per_month']]

df_submit.head()
df_submit = df_submit.rename(columns={'reviews_per_month':'price'})
# All predictions are just $27.01

df_submit['price'] = 27.01

df_submit.head()
# Makes sure it's just `id` and `price` (leave out the index)

df_submit.to_csv('my_other_submission.csv',index=False)
# Check out the file was created!

!ls