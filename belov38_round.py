# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import modf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Get submission from:

# https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50 

df = pd.read_csv('/kaggle/input/subm4898/submission 4898.csv')

print(df.shape)

df.head()
def part_round(param):

    fract, whole = modf(param)

    if fract < 0.1:

        return whole

    elif fract > 0.9:

        return whole+1

    else:

        return param
for column in [f'F{n+1}' for n in range(28)]:

    df[column] = df[column].apply(part_round)



df.to_csv('submission.csv', index=False)