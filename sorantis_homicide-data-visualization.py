# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from bokeh.plotting import figure, output_notebook, show

from bokeh.charts import Bar, Histogram, Line, output_file, show

from bokeh.charts.attributes import cat, color

from bokeh.charts.operations import blend, stack

pd.options.display.max_columns = 20

output_notebook()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
homicides = pd.read_csv('../input/database.csv')

homicides = (homicides.drop(['Record ID', 'Agency Code','Crime Solved', 'Agency Name', 'Agency Type',

       'Record Source'],axis=1))

homicides = homicides.replace('Unknown', np.nan)

homicides = homicides[homicides['Victim Age'] <= 100]
df = pd.DataFrame()

df['Black'] = homicides[(homicides['Victim Race'] == 'Black')]['Year'].value_counts()

df['White'] = homicides[(homicides['Victim Race'] == 'White')]['Year'].value_counts()

df['Native'] = homicides[(homicides['Victim Race'] == 'Native American/Alaska Native')]['Year'].value_counts()

df['Asian'] = homicides[(homicides['Victim Race'] == 'Asian/Pacific Islander')]['Year'].value_counts()

df['Total'] = homicides[homicides['Victim Race'] != np.nan]['Year'].value_counts()

df = df.reset_index().rename(columns={'index': 'Year'})

show(Bar(df,

          values=blend('Black', 'White', 'Asian', 'Native', name='victims', labels_name='victim'),

          label=cat(columns='Year', sort=True),

          stack=cat(columns='victim', sort=True),

          color=color(columns='victim', palette=['Green', 'Blue', 'Black','Red'], sort=False),

          legend='top_right',

          title="Victims per Race"))
top10_relationships = homicides.groupby('Relationship').size().sort_values(ascending=False).to_frame('Total').reset_index().head(10)

rel_chart = Bar(top10_relationships,

             values='Total',

             label=cat(columns='Relationship', sort=False),

             color=color(columns='Relationship', palette=['Green'], sort=False),

             legend=False,

             title="Relationships")



show(rel_chart)
top10_cities = homicides.groupby('City').size().sort_values(ascending=False).to_frame('Total').reset_index().head(10)

cities = Bar(top10_cities,

             values='Total',

             label=cat(columns='City', sort=False),

             color=color(columns='City', palette=['Blue'], sort=False),

             legend=False,

             title="Cities")



show(cities)