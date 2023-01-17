# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#For reading data

import pandas as pd



#For visualizations

import plotly.express as px

import plotly.graph_objects as go



#To ignore warnings

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/indian-migration-history/IndianMigrationHistory.csv')

df.head()
rows, cols = df.shape

print("Rows:", rows, '\nColumns:', cols)
df.drop(labels=['Country Origin Name', 'Country Origin Code', 'Migration by Gender Code', 'Country Dest Code'], axis=1, inplace=True)

df.head()
df['Country Dest Name'].value_counts()
df['Migration by Gender Name'].value_counts()
tot = df[df['Migration by Gender Name'] == 'Total']

not_tot = df[df['Migration by Gender Name'] != 'Total']
fig = px.sunburst(not_tot, path=['Migration by Gender Name', 'Country Dest Name'], values='1960 [1960]', title='Migration to countries based on Gender(1960)')

fig.show()
req_cols = ['1960 [1960]', '1970 [1970]', '1980 [1980]', '1990 [1990]', '2000 [2000]']

t_cols = ['1980 [1980]', '1990 [1990]', '2000 [2000]']

not_tot['Total Migration[1960-2000]'] = not_tot[req_cols].sum(axis=1)

not_tot['Final Migration[1980-2000]'] = not_tot[t_cols].sum(axis=1)
mig_df = not_tot.sort_values("Total Migration[1960-2000]", ascending=False)

m = mig_df[mig_df['Migration by Gender Name'] == 'Male']

f = mig_df[mig_df['Migration by Gender Name'] == 'Female']

cols_used = m['Country Dest Name'].tolist()
fig = go.Figure(data=[

    go.Bar(name='Male', x=cols_used, y=m['Total Migration[1960-2000]'].tolist()),

    go.Bar(name='Female', x=cols_used, y=m['Total Migration[1960-2000]'].tolist())

])

fig.update_layout(title="Total Migration from 1960-2000")

fig.update_yaxes(type='log') #Makes the country viewing a little easy :)

fig.show()
fin_df = not_tot.sort_values('Final Migration[1980-2000]', ascending=False)
fig = px.bar(fin_df, x="Country Dest Name", y="Final Migration[1980-2000]", color="Migration by Gender Name", title="Total Migration from 1980-2000")

fig.update_yaxes(type='log')

fig.show()