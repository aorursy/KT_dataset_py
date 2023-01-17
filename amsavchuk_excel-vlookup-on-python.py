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
def create_df(data):

    df = pd.DataFrame(data, columns = ['Id', 'Name']) 

    return df
data_a = [[1, 'Tom'], [2, 'Nick'], [1, 'Juli'], [3, 'Alex']] 

a = create_df(data_a)
data_b = [[1, 'Michael'], [3, 'David'], [3, 'Antony']] 

b = create_df(data_b)
a_found_in_b = a[a['Id'].isin(b['Id'])]

a_found_in_b.head()
b_found_in_a = b[b['Id'].isin(a['Id'])]

b_found_in_a.head()