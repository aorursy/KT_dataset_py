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

data = '../input/suicides-in-india/Suicides in India 2001-2012.csv'

df = pd.read_csv(data)

print(df.head())



# we will see how the dataset is organised

print(df.describe())



print("Lets see for which states do we have the data. For the sake of simplicity we will only be looking at the states and the total number of suicides that are happening in those states. So we group by the 'State' key and aggregate the 'Total' column.")

df_state = df.drop('Year', axis=1)

df_state = df_state.groupby('State', as_index=False).agg({'Total': 'sum'})

print(df_state.head())



print('Dropping the non relevant fields')

print('Shape before dropping the fields {}'.format(df_state.shape))



df_state = df_state.drop(df_state.index[df_state.State == 'Total (All India)'])

df_state = df_state.drop(df_state.index[df_state.State == 'Total (States)'])

df_state = df_state.drop(df_state.index[df_state.State == 'Total (Uts)'])



print('Shape after dropping the fields {}'.format(df_state.shape))



print('Changing to jsonstr to feed to html file')

jsonstr = df_state.to_json(orient='records')

print('this can be now pushed to google facets')