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

import pandas as pd



# Any results you write to the current directory are saved as output.
fields = ['CONTROL', 'TUITIONFEE_IN']

new_df = pd.read_csv("/kaggle/input/tuition-analysis-20172018/MERGED2017_18_PP.csv",usecols=fields)



new_df=new_df.loc[new_df['CONTROL'] == 1]

new_df=new_df.dropna()

new_df = new_df[new_df['TUITIONFEE_IN'] != 0]
temp_df=new_df.sample(100)
#temp_df.mean(axis=0,'TUITIONFEE_IN')

print("Mean: ", temp_df['TUITIONFEE_IN'].mean())

#newdf["page"].mean(axis=1)

print("Standard Error",temp_df['TUITIONFEE_IN'].sem()) 