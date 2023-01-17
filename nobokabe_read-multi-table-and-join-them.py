# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# read files

df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)



state_latlong = pd.read_csv('../input/statelatlong.csv')

state_gdp = pd.read_csv('../input/US_GDP_by_State.csv')

df_train.head()
state_latlong.head()
state_gdp.head()
# SELECT *

# FROM df_train t

# INNER JOIN state_latlong l ON (t.addr_state = l.State)

df_train = df_train.reset_index().merge(state_latlong.rename(columns={"State":"addr_state"})).set_index(df_train.index.names)

df_test = df_test.reset_index().merge(state_latlong.rename(columns={"State":"addr_state"})).set_index(df_test.index.names)



df_train.head()
# SELECT *

# FROM state_gdp g

# WHERE g.year = 2013

state_gdp = state_gdp[state_gdp.year == 2013]

state_gdp.head()

# SELECT *

# FROM df_train t

# INNER JOIN state_gdp g ON (t.City = l.State)

df_train = df_train.reset_index().merge(state_gdp.rename(columns={"State":"City"})).set_index(df_train.index.names)

df_test = df_test.reset_index().merge(state_gdp.rename(columns={"State":"City"})).set_index(df_test.index.names)

df_train.head()