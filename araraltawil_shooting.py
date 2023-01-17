# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month


data.dropna(inplace=True)






import plotly.express as px

killing_by_race_data=data[data['manner_of_death']=='shot']
fig = px.histogram(killing_by_race_data,x='race',color='race')
fig.show()




unarmed_shootout_data = data[data['armed']=='unarmed']['race']
fig = px.histogram(unarmed_shootout_data,x='race',color='race')
fig.show()



shootout_by_states_data = data['state'].value_counts()[:10]
shootout_by_states_data = pd.DataFrame(shootout_by_states_data)
shootout_by_states_data=shootout_by_states_data.reset_index()
fig = px.pie(shootout_by_states, values='state', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()




import plotly.figure_factory as ff
np.random.seed(1)
x = data['age']
hist_data = [x]
group_labels = ['Age']
fig = ff.create_distplot(hist_data, group_labels)
fig.show()