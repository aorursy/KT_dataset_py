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
n = 200000
%%time

series_1 = np.random.randint(low = 1,high = 1000,size = n)

series_1_T = series_1.reshape(n,1)

series_2  = np.random.randint(low = 1,high = 1000,size = n)

series_2_T = series_2.reshape(n,1)
%%time

def differ(x):

    count = 0

    tabel_1 = series_1 + series_1_T[x:x+2000]

    tabel_2 = series_2 + series_2_T[x:x+2000]

    diff= tabel_1[tabel_1>tabel_2].shape[0]

    count += diff

    return count
arr = pd.DataFrame(data = np.arange(0,n,2000),columns = ["numbers"])
%%time

count_each_run = arr["numbers"].apply(differ)
count_each_run.sum()