# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# create full dataframe

full = pd.concat([train,test])



"""I'm not sure if there's an Python equivalent to the str function in r, here's a function 

instead to accomplish the same task, returns the size of the dataframe, unique values of each column, 

along with data types for each column"""

def rstr(df): 

    return df.shape, df.apply(lambda x: [x.unique()]), df.dtypes



print (rstr(full))
sex_group_survival = full['Survived'].groupby('Sex').sum()

sex_group_survival.plot(kind = 'bar')