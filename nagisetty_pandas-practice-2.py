import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/gapminder.csv')
df.head()
type(df)
df.shape
df.info()
country_df = df['country']
df[['country', 'year']].head()
df.columns
list(df.columns)
df.loc[0]
df.iloc[-1]
df.loc[[0,99,999]]
df.loc[[9,99,999],['country','year']]
df.iloc[[9,99,999],[1,2]]
#mean()
le_mean = df['lifeExp'].mean()
le_mean
#condition
df.loc[df['lifeExp']> le_mean,'country'].head(10)
#groupby
df.groupby('year')
df.groupby('year')[['lifeExp','gdpPercap']].mean()
df.groupby(['year','continent'])[['lifeExp','gdpPercap']].mean().head(20)
df.groupby('continent')['country'].nunique()
gyle = df.groupby('year')['lifeExp'].mean()
gyle.plot()
from numpy import NaN, NAN, nan

visited = pd.read_csv('../input/survey_visited.csv')
survey = pd.read_csv('../input/survey_survey.csv')
survey
ebola = visited = pd.read_csv('../input/ebola_country_timeseries.csv')
ebola.head()

ebola['Cases_Guinea'].value_counts(dropna=False).head()

ebola.fillna(method='bfill').head()
ebola['Cases_Guinea'].sum(skipna=True)
pew = pd.read_csv('../input/pew.csv')
pew.head()
pd.melt(pew, id_vars='religion').head(20)
pd.melt(pew, id_vars='religion',value_name='count',var_name='income').head(10)
ebola.head()
ebola_melt = pd.melt(ebola,
                    id_vars= ['Date', 'Day'],
                    var_name = 'cd_country',
                    value_name = 'count'
                              )
var_split = ebola_melt['cd_country'].str.split('_')
status_values = var_split.str.get(0)
country_values = var_split.str.get(1)
ebola_melt['status'] = status_values
ebola_melt['country'] = country_values
ebola_melt.head()
variable_split = ebola_melt['cd_country'].str.split('_',expand=True)
variable_split.head()
variable_split.columns = ['status1','country1']
ebola_clean = pd.concat([ebola_melt,variable_split],axis=1)
ebola_clean.head()
weather = pd.read_csv("../input/weather.csv")
weather.head()
weather_melt = pd.melt(weather,
                      id_vars=['id','year','month','element'],
                      var_name = 'day',
                      value_name = 'temp'
                      )
weather_melt.head()
weather_tidy = weather_melt.pivot_table(
    index=['id','year','month','day'],
    columns ='element',
    values = 'temp').reset_index()
weather_tidy.head()

