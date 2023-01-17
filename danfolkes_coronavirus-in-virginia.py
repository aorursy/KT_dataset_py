# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.express as px

df=pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')
dfpop=pd.read_csv('/kaggle/input/us-pop/co-est2019-alldata.csv', encoding = "ISO-8859-1")

df = df[df.state == 'Virginia']
df = df[df.deaths > 1]
dfpop = dfpop[dfpop.STATE == 51]
dfpop["FIPS"] = '000000000' + dfpop["COUNTY"].astype(str)
dfpop["FIPS"] = dfpop["FIPS"].str[-3:]
dfpop["FIPS"] = dfpop["STATE"].astype(str) + dfpop["FIPS"]

dfpop = dfpop[['FIPS','POPESTIMATE2019', 'CTYNAME', 'STATE','STNAME','COUNTY']]
dfpop[dfpop.CTYNAME.str.contains("James")].head()
df['FIPS'] = df['fips'].astype(str).str[:5]



dfnew = pd.merge(df, dfpop, on='FIPS', how='outer')
dfnew = dfnew[dfnew.county != 'Unknown']
#print(dfnew.deaths.sum())
dfnew = dfnew[dfnew.deaths > 2]
dfnew = dfnew[dfnew.cases > 2]

dfnew['death2pop'] = (dfnew['deaths']/dfnew['POPESTIMATE2019']) * 100
dfnew['death2cases'] = (dfnew['deaths']/dfnew['cases']) * 100
dfnew['cases2pop'] = (dfnew['cases']/dfnew['POPESTIMATE2019']) * 100

#dfnew[dfnew.county == 'James City'].head(1)
fig = px.line(dfnew[dfnew.date > '2020-04-15'], x="date", y="deaths", title='deaths by county', color="county", hover_name="cases")
fig.show()
fig = px.bar(dfnew[dfnew.date > '2020-04-05'], x='date', y='death2cases',
             hover_data=['cases', 'deaths'], color='county',
             labels={'pop':'population of Canada'}, height=400)
#fig.update_layout(barmode='stack')

fig.show()
fig = px.line(dfnew[dfnew.date > '2020-04-05'], x="date", y="death2pop", title='deaths per capita', color="county", hover_data=['cases', 'deaths'] )
fig.show()
fig = px.line(dfnew[dfnew.date > '2020-04-05'], x="date", y="death2cases", title='deaths per case', color="county", hover_name="deaths")
fig.show()
fig = px.line(dfnew[dfnew.date > '2020-04-15'], x="date", y="cases2pop", title='cases to population', color="county", hover_name="deaths")
fig.show()