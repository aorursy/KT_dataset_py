import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing essential libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import Standard

import os # accessing directory structure

import plotly

import plotly.express as px
#Enter your code here

#movies =pd.read_csv('movie_plotly.csv')

movies = pd.read_csv("/kaggle/input/MovieAssignmentData.csv") 

movies.head()
#movies.dtypes
# Subtask 1.3 - Try adding multiple facets to the scatter plot, so that we could view three genres - 

#'Action','Drama','Biography' side by side

import plotly.express as px

plt.figure(figsize=[300,320])

fig = px.scatter(movies, x="Runtime", y="IMDb_rating", 

                 animation_group="genre_1",size="actor_1_facebook_likes",

                 color="genre_1",facet_col="genre_1", 

                 hover_name="genre_1",log_x=True, size_max=45, range_x=[100,150], range_y=[7.4,8.9])

fig.show()
#Sub task 1.1 marginal plots with an histogram

import plotly.express as px

plt.figure(figsize=[100,100])

movies = pd.read_csv("/kaggle/input/MovieAssignmentData.csv")

fig = px.scatter(movies, x="budget", y="Gross", color="genre_1", marginal_y="rug", marginal_x="histogram")

fig
#Subtask 1.1 - Try customizing your scatter plot with addition of marginal plots such as box plot or violin plot

import plotly.express as px

plt.figure(figsize=[100,100])

movies = pd.read_csv("/kaggle/input/MovieAssignmentData.csv")

fig = px.scatter(movies, x="budget", y="Gross", color="genre_1", marginal_y="violin",

           marginal_x="box", trendline="ols")

fig
#Subtask 1.4 - Try creating a bubble plot between IMDb rating and runtime and see if linking 'actor's Facebook likes' 

#to the size of the marker bring out any additional insight from the plot.
#What is the insight that you see  to get by adding the third dimension of 'actor's Facebook likes'
#movies.head().dtypes
import plotly.express as px

plt.figure(figsize=[300,320])

fig = px.scatter(movies, x="Runtime", y="IMDb_rating", 

                 animation_group="genre_1",size="actor_1_facebook_likes", 

                 color="genre_1",facet_col="genre_1", 

                 hover_name="genre_1",log_x=True, size_max=45, range_x=[100,150], range_y=[7.4,8.9])

fig.show()
df = px.data.gapminder()

df.head()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

           size="gdpPercap", color="continent", hover_name="country", facet_col="continent",

           log_x=True, size_max=45, range_x=[100,1000000], range_y=[30,90])

fig.show()
df = px.data.tips()

df.head(5)
#Subtask 1- Create a comparison plot among different days of a week with respect to time of the day,

#total bill value and tip amount.Also, try color coding the client based on gender
import plotly.express as px

df = px.data.tips()

fig = px.scatter(df, x="total_bill", y="tip", facet_row="time", facet_col="day", color="smoker", trendline="ols",

          category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

fig.show()
import plotly.graph_objects as go



categories = ['Bitterness','Froth','Price',

              'Content', 'Mouthfeel']



fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=[1, 5, 2, 2, 3],

      theta=categories,

      fill='toself',

      name='Product A'

))

fig.add_trace(go.Scatterpolar(

      r=[4, 3, 2.5, 1, 2],

      theta=categories,

      fill='toself',

      name='Product B'

))



fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 5]

    )),

  showlegend=False

)



fig.show()
df = px.data.gapminder()

df
import plotly.express as px

df = px.data.gapminder()

fig = px.area(df, x="year", y="pop", color="continent", line_group="country")

fig.show()
df = px.data.tips()

df
import plotly.express as px

df = px.data.tips()

fig = px.box(df, x="day", y="total_bill", color="sex", notched=True)

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(df, y="total_bill", x="day", color="sex", box=True, points="all", hover_data=df.columns)

fig.show()
df = px.data.tips()

df
#importing essential libraries

 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import Standard

import os # accessing directory structure

import plotly

import plotly.express as px
#Plotly Version

plotly.__version__
import sys; 

print(sys.version)

import plotly

print(plotly.__version__)
#Install plotly

!pip install plotly==4.8.1 --trusted-host pypi.org --trusted-host files.pythonhosted.org
import sys; 

print(sys.version)

import plotly

print(plotly.__version__)
import sys; 

print(sys.version)

import plotly

print(plotly.__version__)
#Install plotly

!pip install plotly==4.8.1 --trusted-host pypi.org --trusted-host files.pythonhosted.org
import sys; 

print(sys.version)

import plotly

print(plotly.__version__)
import plotly.express as px

import plotly.graph_objects as py

import numpy as np

df = px.data.gapminder().query("year == 2007")

fig = px.sunburst(df, path=['continent', 'country'], values='pop',

                  color='gdpPercap', hover_data=['iso_alpha'],

                  color_continuous_scale='RdBu',

                  color_continuous_midpoint=np.average(df['gdpPercap'], weights=df['pop']))

fig.show()
px.data.gapminder().dtypes
df=px.data.gapminder()

df
import plotly.express as px

plt.figure(figsize=[150,125])

df = px.data.gapminder()

fig = px.choropleth(df, locations="country", color="lifeExp", hover_name="country",animation_frame="country", 

                    range_color=[20,80])

fig.show()
df = px.data.gapminder()

df
import plotly.express as px

plt.figure(figsize=[300,320])

df = px.data.gapminder()

fig = px.scatter_geo(df, locations="iso_alpha", color="continent", hover_name="country", size="pop",

               animation_frame="year", projection="natural earth")

fig.show()
df = px.data.gapminder()

df