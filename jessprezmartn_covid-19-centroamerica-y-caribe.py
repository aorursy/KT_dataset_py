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
pd.set_option('display.max_columns', 500)
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df['Country/Region'].unique()
acumulates = df[(df['Country/Region'] == 'Mexico')]

acumulates
df2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
# cases = df2[(df2['Country/Region'] == 'Dominican Republic')]

# cases = df2[(df2['Country/Region'] == 'Jamaica')]

# cases = df2[(df2['Country/Region'] == 'Mexico')]

# cases = df2[(df2['Country/Region'] == 'The Bahamas')]

# cases = df2[(df2['Country/Region'] == 'Guadeloupe')]

cases = df2[(df2['Country/Region'] == 'Cuba')]

cases
pd.options.display.max_columns