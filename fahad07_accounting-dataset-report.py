import pandas as pd
df= pd.read_html('../input/accounting-dataset/sampletable.html')
df
dfn= df[0]
df[0]
df[0][(df[0]['Age']>='60')&(df[0]['Office']=='Tokyo')&(df[0]["Salary"])]
df[0][(df[0]['Age']>='60')|(df[0]['Office']=='Tokyo')|(df[0]["Salary"])]
df[0]['Office'].unique()
df[0]['Office'].value_counts().plot(kind ='bar')
dfn
df[0]['Office'].value_counts().plot(kind ='bar')
# adapted from https://plot.ly/python/plotly-express/
import plotly.express as px
iris = px.data.iris()
fig = px.scatter(iris, x="sepal_width", y="sepal_length")
fig.show()
# adapted from https://plot.ly/python/plotly-express/
import plotly.express as px
gapminder = px.data.gapminder()
fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()