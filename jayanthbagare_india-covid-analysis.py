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
india_data = pd.read_csv("/kaggle/input/indiacovid/COVID-19 - India_Growth_Rate.csv")
india_data.rename(columns={'Date':'date','Number of Affected People':'infected_count','Difference %':'percent_difference'},inplace=True)
india_data
india_data['growth_rate'] = 0
for i in range(1, len(india_data)):

    if((i != 1) & (india_data.loc[i-1,'infected_count'] != 0)):

        india_data.loc[i,'growth_rate'] = india_data.loc[i,'infected_count'] / india_data.loc[i-1,'infected_count']
india_data