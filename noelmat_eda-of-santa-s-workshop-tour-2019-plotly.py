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
path = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
family_data=pd.read_csv(path)

family_data.head()
import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go
#lets have a look at the families on basis on the choices

plt.figure(figsize=(10,10))

sns.countplot(x='choice_0',data=family_data)
import gc

def choiceToDate(x,col):

    for i in x.family_id:

        x.iloc[i,1]=x['choice'].iloc[i] - pd.DateOffset(family_data[col].iloc[i])



choice_plot= go.Figure()

date = '2019-12-25'



for col in family_data.columns[1:-1]:

    choice = pd.DataFrame({'family_id':family_data.family_id,'choice': [date]*len(family_data)})

    choice['choice']=pd.to_datetime(choice.choice)

    choiceToDate(choice,col)

    choice = choice.set_index('choice').resample('D').count()

    choice.columns = ['family_count']

    choice_plot.add_trace(go.Scatter(x=choice.index,

                    y=choice.family_count,

                    mode='lines+markers',                                     

                    name=col))

    del(choice)

    gc.collect()

choice_plot.update_layout(title='Choice of Dates to attend Santa\'s Workshop ({}st Preference)'.format(int(col[-1])+1),

                           xaxis_title='Date',

                           yaxis_title='No of families')

choice_plot.show()
n_people_plot = go.Figure(go.Pie(values=family_data.n_people,labels=family_data.n_people))

n_people_plot.update_layout(title='Pie Chart of number of persons in families')

n_people_plot.show()
plt.figure(figsize=(10,6))

sns.countplot('n_people',data=family_data)

plt.show()