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

        absolute_file_path = os.path.join(dirname, filename)

        print(absolute_file_path)

        if filename == 'train.csv':

            df_train = pd.read_csv(absolute_file_path)

        elif filename == 'test.csv':

            df_test = pd.read_csv(absolute_file_path)



# Any results you write to the current directory are saved as output.
df_train.head()
df_train.describe()