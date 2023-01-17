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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service.csv')
mask = df['Year'] == 2020
df.loc[mask, 'Whole year'] = df[mask].iloc[:,2:14].sum(axis=1)
sector = df.groupby('Year')
whole_year = sector['Whole year'].sum()
whole_year.plot()
sector.agg('sum').sum(axis=0)[:-1].plot(kind='bar')