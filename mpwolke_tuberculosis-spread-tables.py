# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/determine-the-pattern-of-tuberculosis-spread/tuberculosis_data_WHO.csv')



head = ff.create_table(df.head())

py.iplot(head)
describe = ff.create_table(df.describe())

py.iplot(describe)
missing_count = pd.DataFrame({'missing': df.isnull().sum().sort_values(ascending=False)})



missing_count = ff.create_table(missing_count)

py.iplot(missing_count)
missing_percentage = pd.DataFrame({'missing': df.isnull().mean().sort_values(ascending=False)})



missing_percentage = ff.create_table(missing_percentage)

py.iplot(missing_percentage)
#https://www.kaggle.com/prestonfan/data-visualization-palmer-archipelago-penguins/notebook

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

df.iloc[:,:] = imputer.fit_transform(df)



head = ff.create_table(df.head())

py.iplot(head)
missing_count = pd.DataFrame({'missing': df.isnull().sum().sort_values(ascending=False)})



missing_count = ff.create_table(missing_count)

py.iplot(missing_count)
sns.countplot(df['World Bank income group'], data=df, palette='bone')

plt.title('World Bank Income Group')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
island_percentage = df['World Bank income group'].value_counts()



labels = island_percentage.index

values = island_percentage.values



colors = ['green', 'red', 'yellow']



island = go.Pie(labels = labels,

                         values = values,

                         marker = dict(colors = colors),

                         name = 'World Bank Income Group', hole = 0.3)



df = [island]



layout = go.Layout(

           title = 'Percentage - World Bank income group')



fig = go.Figure(data = df,

                 layout = layout)



py.iplot(fig)