# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/cities_r2.csv')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
t_pop=df[['population_total','state_name']].groupby('state_name').sum()
t_pop.plot(kind='bar')
liter=df[['state_name','population_female','literates_female']].groupby('state_name').sum()
liter['prop']=liter['literates_female']/liter['population_female']
liter.plot(kind='bar')