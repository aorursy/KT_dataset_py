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
population = pd.read_csv('../input/world-population-19602018/population_clean.csv')

train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
pop_list=[]

for coun in train['Country/Region'].unique():

    if len(population[population['Country Name']==coun]['2018'].values)== 0:

        pop_list.append(None)

    else:

        pop_count = population[population['Country Name']==coun]['2018'].values[0]

        pop_list.append(pop_count)
pop_list_pd = pd.DataFrame({'country':train['Country/Region'].unique(),'pop':pop_list})
pop_list_pd[pop_list_pd['pop'].isna()]
pop_list_pd.to_csv('world_population.csv',index=False)