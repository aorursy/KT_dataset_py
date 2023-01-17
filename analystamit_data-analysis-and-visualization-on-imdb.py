import pandas as pd

import numpy as np

%matplotlib inline

pd.set_option("display.max_rows",10)

df = pd.read_csv('../input/movie_metadata.csv')

df1 = df[df.title_year.notnull()]

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1['title_year'].groupby(df1['title_year']).size().plot()
# Removing the records before 1990 year. 

df2 = df1[df1['title_year'] > 1990]
from bokeh.charts import Scatter, Histogram

from bokeh.models import HoverTool

from bokeh.io import output_notebook, show



output_notebook()
s = Scatter(data = df2, x = 'title_year', y = 'duration', height = 400)

show(s)
movie_count_by_director = df2['director_name'].groupby(df2['director_name']).count().sort_values(ascending = False)
directors = movie_count_by_director[movie_count_by_director > 15].index.values.tolist()
s = Scatter(data = df2[df2.director_name.isin(directors)], 

            x = 'title_year', y = 'duration', color = 'director_name', 

            height = 400, width = 800,

            title = "Movies duration for movies created by director who directed maximum movoies",

            tools='hover, box_zoom, lasso_select, save, reset')

hover = s.select(dict(type = HoverTool))

hover.tooltips = [('Director-Likes','@director_facebook_likes'),

                 ('Movie Name','@title_name')]

            

show(s)