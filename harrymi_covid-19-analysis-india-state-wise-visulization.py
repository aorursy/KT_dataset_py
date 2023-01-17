import pandas as pd

import numpy as np

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go
india_df=pd.read_csv('../input/covid19-data/india.csv')
china_df=pd.read_csv('../input/covid19-data/China.csv')
united_df=pd.read_csv('../input/covid19-data/United_state.csv')
southkoria_df=pd.read_csv('../input/covid19-data/South Korea.csv')
itely_df=pd.read_csv('../input/covid19-data/Italy.csv')
pd.set_option('display.max_rows', 1222)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
india_df.head(1)
india_df.columns
china_df.head(2)
united_df.head(2)
from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=5, cols=1,subplot_titles=("India", "China ", "United State", "South Korea","Itely"))



fig.append_trace(go.Scatter(

    x=india_df['date'],

    y=india_df['new_cases'], name='India'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=china_df['date'],

    y=china_df['new_cases'], name='China'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=united_df['date'],

    y=united_df['new_cases'], name='United State'

), row=3, col=1)



fig.append_trace(go.Scatter( 

    x=southkoria_df['date'],

    y=southkoria_df['new_cases'], name='South Korea'

), row=4, col=1)

fig.append_trace(go.Scatter( 

    x=itely_df['date'],

    y=itely_df['new_cases'], name='Itely'

), row=5, col=1)





fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

#fig.show()
fig = make_subplots(rows=5, cols=1,subplot_titles=("India", "China ", "United State", "South Korea"))



fig.append_trace(go.Bar(

    x=india_df['date'],

    y=india_df['new_cases'], name='India'

), row=1, col=1 )



fig.append_trace(go.Bar( 

    x=china_df['date'],

    y=china_df['new_cases'], name='China'

), row=2, col=1)



fig.append_trace(go.Bar( 

    x=united_df['date'],

    y=united_df['new_cases'], name='United State'

), row=3, col=1)



fig.append_trace(go.Bar( 

    x=southkoria_df['date'],

    y=southkoria_df['new_cases'], name='South Korea'

), row=4, col=1)

fig.append_trace(go.Bar( 

    x=itely_df['date'],

    y=itely_df['new_cases'], name='Itely'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
fig = make_subplots(rows=2, cols=2,subplot_titles=("India", "China ", "United State", "South Korea"))



fig.append_trace(go.Bar(

    x=india_df['date'],

    y=india_df['new_cases'], name='India'

), row=1, col=1 )



fig.append_trace(go.Bar( 

    x=china_df['date'],

    y=china_df['new_cases'], name='China'

), row=1, col=2)



fig.append_trace(go.Bar( 

    x=united_df['date'],

    y=united_df['new_cases'], name='United State'

), row=2, col=1)



fig.append_trace(go.Bar( 

    x=southkoria_df['date'],

    y=southkoria_df['new_cases'], name='South Korea'

), row=2, col=2)





fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
fig = make_subplots(rows=5, cols=1,subplot_titles=("India", "China ", "United State", "South Korea",'itely'))



fig.append_trace(go.Scatter(

    x=india_df['date'],

    y=india_df['total_cases'], name='India'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=china_df['date'],

    y=china_df['total_cases'], name='China'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=united_df['date'],

    y=united_df['total_cases'], name='United State'

), row=3, col=1)



fig.append_trace(go.Scatter( 

    x=southkoria_df['date'],

    y=southkoria_df['total_cases'], name='South Korea'

), row=4, col=1)



fig.append_trace(go.Scatter( 

    x=itely_df['date'],

    y=itely_df['total_cases'], name='Itely'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
fig = make_subplots(rows=5, cols=1,subplot_titles=("India", "China ", "United State", "South Korea"))



fig.append_trace(go.Bar(

    x=india_df['date'],

    y=india_df['total_cases'], name='India'

), row=1, col=1 )



fig.append_trace(go.Bar( 

    x=china_df['date'],

    y=china_df['total_cases'], name='China'

), row=2, col=1)



fig.append_trace(go.Bar( 

    x=united_df['date'],

    y=united_df['total_cases'], name='United State'

), row=3, col=1)



fig.append_trace(go.Bar( 

    x=southkoria_df['date'],

    y=southkoria_df['total_cases'], name='South Korea'

), row=4, col=1)

fig.append_trace(go.Bar( 

    x=itely_df['date'],

    y=itely_df['total_cases'], name='Itely'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
all_data=pd.concat([india_df,united_df,china_df,southkoria_df,itely_df])
px.line(all_data, x='date', y='total_cases', color='location')
px.line(all_data, x='date', y='new_cases', color='location')
all_data.head(2)
all_data['Mortality_Rate']=all_data['total_deaths']/all_data['total_cases']
px.line(all_data, x='date', y='Mortality_Rate', color='location')
a=all_data['date'].max()
specific_data=all_data.loc[all_data['date']==a]
specific_data
fig = make_subplots(rows=5, cols=1,subplot_titles=("Total_Cases_Per_Million", "New_Cases_Per_Million ", "Total_Deaths_Per_million", "Mortality_Rate",'new_deaths_per_million'))



fig.append_trace(go.Bar(

    x=specific_data['total_cases_per_million'],

    y=specific_data['location'] ,  orientation='h'

), row=1, col=1 )



fig.append_trace(go.Bar( 

    x=specific_data['new_cases_per_million'],

    y=specific_data['location'],  orientation='h'

), row=2, col=1)



fig.append_trace(go.Bar( 

    x=specific_data['total_deaths_per_million'],

    y=specific_data['location'],  orientation='h'

), row=3, col=1)



fig.append_trace(go.Bar( 

    x=specific_data['Mortality_Rate'],

    y=specific_data['location'], orientation='h'

), row=4, col=1)

fig.append_trace(go.Bar( 

    x=specific_data['new_deaths_per_million'],

    y=specific_data['location'],  orientation='h'

), row=5, col=1)







fig.update_layout(height=800, width=800, title_text="Stacked Subplots")

fig.show()
#px.bar(specific_data, x='Mortality_Rate', y='location' ,orientation='h')
#px.bar(specific_data, x='total_cases_per_million', y='location' ,orientation='h')
#px.bar(specific_data, x='total_deaths_per_million', y='location' ,orientation='h')
fig = make_subplots(rows=5, cols=1,subplot_titles=("total_cases_per_million", "total_cases_per_million ", "total_cases_per_million", "South Korea",'itely'))



fig.append_trace(go.Scatter(

    x=india_df['date'],

    y=india_df['total_cases_per_million'], name='India'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=china_df['date'],

    y=china_df['total_cases_per_million'], name='China'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=united_df['date'],

    y=united_df['total_cases_per_million'], name='United State'

), row=3, col=1)



fig.append_trace(go.Scatter( 

    x=southkoria_df['date'],

    y=southkoria_df['total_cases_per_million'], name='South Korea'

), row=4, col=1)



fig.append_trace(go.Scatter( 

    x=itely_df['date'],

    y=itely_df['total_cases_per_million'], name='Itely'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
fig = make_subplots(rows=5, cols=1,subplot_titles=("India", "China ", "United State", "South Korea",'itely'))



fig.append_trace(go.Scatter(

    x=india_df['date'],

    y=india_df['total_deaths_per_million'], name='India'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=china_df['date'],

    y=china_df['total_deaths_per_million'], name='China'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=united_df['date'],

    y=united_df['total_deaths_per_million'], name='United State'

), row=3, col=1)



fig.append_trace(go.Scatter( 

    x=southkoria_df['date'],

    y=southkoria_df['total_deaths_per_million'], name='South Korea'

), row=4, col=1)



fig.append_trace(go.Scatter( 

    x=itely_df['date'],

    y=itely_df['total_deaths_per_million'], name='Itely'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
#import os 

#os.chdir('C:\\Users\HarryMi\Desktop\Covid19')
df=pd.read_csv('../input/covid19-data/india.csv')
#df.drop('Sno',axis=1,inplace=True)
df.head(5)
df['date']=pd.to_datetime(df['date'], errors='raise', dayfirst=True, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
g=df.groupby('date')
india=g.sum()
india.reset_index(inplace = True, drop = False)
india.tail()
#india["new_case"]=india["Confirmed"]-india["Confirmed"].shift(1)
india['Mortality']=india['total_deaths']/india['total_cases']
fig = make_subplots(rows=3, cols=1,subplot_titles=("Total Cases day by day", "Mortality Rate Day By day ", "New Cases Trend Day By day", ))



fig.append_trace(go.Bar(

    x=india['date'],

    y=india['total_cases'], name='Total Cases'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['total_cases'], name='Total cases'

), row=1, col=1)



fig.append_trace(go.Bar( 

    x=india['date'],

    y=india['Mortality'], name='Mortality Rate'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['Mortality'], name='Mortality Rate'

), row=2, col=1)



fig.append_trace(go.Bar( 

    x=india['date'],

    y=india['new_cases'], name='New Cases'

), row=3, col=1)

fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['new_cases'], name='New Cases'

), row=3, col=1)







fig.update_layout(height=800, width=600, title_text="Stacked Subplots")

fig.show()
#fig = px.bar(india, x="date", color="total_cases")

#fig.show()
#fig = px.bar(india, x="date", color="Mortality")

#fig.show()
#fig = px.line(india, x="date", y="Mortality")

#fig.show()
#fig = px.bar(india, x="date", color="new_cases")

#fig.show()
fig = make_subplots(rows=5, cols=1,subplot_titles=("per Million total cases", "per Million new cases ", "per Million total death", "per Million new death",'Total test per Million'))



fig.append_trace(go.Scatter(

    x=india['date'],

    y=india['total_cases_per_million'], name='per Million total cases'

), row=1, col=1 )



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['new_cases_per_million'], name='per Million new cases'

), row=2, col=1)



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['total_deaths_per_million'], name='per Million total death'

), row=3, col=1)



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['new_deaths_per_million'], name='per Million new death'

), row=4, col=1)



fig.append_trace(go.Scatter( 

    x=india['date'],

    y=india['total_tests_per_thousand'], name='Total Test per Thousend'

), row=5, col=1)







fig.update_layout(height=600, width=600, title_text="Stacked Subplots")

fig.show()
age_df=pd.read_csv('../input/age-data/age.csv')
g=age_df.iloc[:,[4,5,10]]
g['Current Status'].unique()
g.head()
g.dropna(inplace=True)
px.histogram(g, x='Age Bracket', color="Gender", nbins=10)
px.histogram(g, x='Gender')
px.histogram(g, x='Gender', color="Current Status" , nbins=10)
px.histogram(g, x='Age Bracket', color="Current Status" , nbins=10)




harsh=pd.read_csv('../input/covid19-data/covid_19_india.csv')

harsh.drop('Sno',axis=1,inplace=True)
harsh['Date']=pd.to_datetime(harsh['Date'], errors='raise', dayfirst=True, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
newx=harsh.groupby(['Date','State/UnionTerritory'])
data_new=newx.sum()
data_new.reset_index(inplace = True, drop = False)
df5=data_new[data_new["Deaths"]>10]
df5
df5['Mortality']=df5['Deaths']/df5['Confirmed']
fig = px.line(df5, x="Date", y="Mortality",color='State/UnionTerritory')

fig.show()
fig = px.line(df5, x="Date", y="Confirmed",color='State/UnionTerritory')
fig.show()
testing_data=pd.read_csv('../input/covid19-data/testing.csv')
testing_data.tail(1)
testing_data['Updated On']=pd.to_datetime(testing_data['Updated On'], errors='raise', dayfirst=True, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
a=testing_data['Updated On'].max()

a
test=testing_data.loc[testing_data['Updated On']==pd.Timestamp('2020-04-25 00:00:00')]
test.head(1)
px.bar(test, x='State', y='Test positivity rate')
testing_data=pd.read_csv('../input/covid19-data/icmrtest.csv')
testing_data.head()
testing_data['Update Time Stamp']=pd.to_datetime(testing_data['Update Time Stamp'], errors='raise', dayfirst=True, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
testing_data.isnull().sum()
fig = px.bar(testing_data, x="Update Time Stamp", y="Total Samples Tested")
fig.show()
fig = px.line(testing_data, x="Update Time Stamp", y="Test positivity rate")

fig.show()
fig = px.bar(testing_data, x="Update Time Stamp", y="Test positivity rate")

fig.show()
fig = px.bar(testing_data, x="Update Time Stamp", y="Tests Per Confirmed Case")

fig.show()