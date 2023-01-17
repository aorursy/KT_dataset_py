import numpy as np

import pandas as pd 

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import matplotlib.pyplot as plt

init_notebook_mode(connected=True)

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/NCHS_-_Leading_Causes_of_Death__United_States.csv')
df.head()
df.dropna()

df.columns = ['Year','X','Cause','State','Deaths','Death Rate']
AL = df.loc[(df['Year']) & (df['State']=='Alabama') & (df['Deaths']) & (df['Cause']), ['Year','State' , 'Deaths', 'Cause']]

AL.head(100)

AL.sort_values(by=['Year'])



Y=[]

for i in AL['Year'].unique():

    byyear = AL.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

Deathtotal = Y

years = np.unique(AL.Year)



plt.plot(years,Deathtotal)

plt.title('Death Rate in Alabama')

plt.xlabel('years')

plt.ylabel('Death')

plt.show()
FL = df.loc[(df['Year']) & (df['State']=='Florida') & (df['Deaths']) & (df['Cause']), ['Year','State' , 'Deaths', 'Cause']]

FL.head(100)

FL.sort_values(by=['Year'])



Y=[]

for i in FL['Year'].unique():

    byyear = FL.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

Deathtotal = Y

years = np.unique(AL.Year)



plt.plot(years,Deathtotal)

plt.title('Death Rate in Florida')

plt.xlabel('years')

plt.ylabel('Death')

plt.show()
Category = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        Category.append(i)

        

tots = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        cause = df.loc[(df['Year']==1999) & (df['Cause']==i) & (df['Deaths']), ['Year','Cause','Deaths']]

        tots.append(cause['Deaths'].sum())



plt.bar(Category, tots)

plt.xlabel('Category')

plt.ylabel('Total')

plt.yticks([50000,150000, 250000, 350000, 650000, 800000],

           ['50K','150K','250K','350K','6500K','800K'])

plt.xticks(Category, fontsize=7, rotation=30)

plt.title('Total deaths 1999')

plt.show()



Category = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        Category.append(i)

        

tots = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        cause = df.loc[(df['Year']==2009) & (df['Cause']==i) & (df['Deaths']), ['Year','Cause','Deaths']]

        tots.append(cause['Deaths'].sum())



plt.bar(Category, tots)

plt.xlabel('Category')

plt.ylabel('Total')

plt.yticks([50000,150000, 250000, 350000, 650000, 800000],

           ['50K','150K','250K','350K','6500K','800K'])

plt.xticks(Category, fontsize=7, rotation=30)

plt.title('Total deaths 2009')

plt.show()



Category = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        Category.append(i)

        

tots = []

for i in df['Cause'].unique():

    if i == 'All causes':

        pass

    else:

        cause = df.loc[(df['Year']==2016) & (df['Cause']==i) & (df['Deaths']), ['Year','Cause','Deaths']]

        tots.append(cause['Deaths'].sum())

        



plt.bar(Category, tots)

plt.xlabel('Category')

plt.ylabel('Total')

plt.yticks([50000,150000, 250000, 350000, 650000, 800000],

           ['50K','150K','250K','350K','6500K','800K'])

plt.xticks(Category, fontsize=7, rotation=30)

plt.title('Total deaths 2016')

plt.show()
US = df.loc[(df['Year']) & (df['State']=='United States') & (df['Deaths']) & (df['Cause']), ['Year','State' , 'Deaths', 'Cause']]

US.head(100)

US.sort_values(by=['Year'])



Y=[]

for i in US['Year'].unique():

    byyear = US.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

Deathtotal = Y

years = np.unique(US.Year)



#[1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015]



plt.plot(years,Deathtotal)

plt.title('Death Rate in United States')

plt.xlabel('years')

plt.xticks([1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015],

           ['1999','2001','2003','2005','2007','2009', '2011', '2013', '2015'])

plt.ylabel('Death')

plt.show()
Y=[]

for i in FL['Year'].unique():

    byyear = FL.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

States = []

for i in df['State'].unique()[:50]:

    bystate = df.loc[(df['Year']==1999) & (df['State']==i) & (df['Cause']) & (df['Deaths']), ['Year','State','Cause','Deaths']]

    bystate = bystate['Deaths'].sum()

    States.append(bystate)



len(States)

z = States
data = dict(

    type = 'choropleth', 

    locations = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],

    locationmode = 'USA-states', 

    colorscale = [

        [1.0, 'rgb(180, 180, 180)'],

        [0, 'rgb(0, 0, 0)']

    ],

    text = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], 

    z = z, 

    colorbar = {'title':'Death Total by State: 1999'}

)



layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [data],layout=layout)

iplot(choromap)
Y=[]

for i in FL['Year'].unique():

    byyear = FL.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

States = []

for i in df['State'].unique()[:50]:

    bystate = df.loc[(df['Year']==2007) & (df['State']==i) & (df['Cause']) & (df['Deaths']), ['Year','State','Cause','Deaths']]

    bystate = bystate['Deaths'].sum()

    States.append(bystate)



len(States)

z = States
data = dict(

    type = 'choropleth', 

    locations = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],

    locationmode = 'USA-states', 

    colorscale = [

        [1.0, 'rgb(180, 180, 180)'],

        [0, 'rgb(0, 0, 0)']

    ],

    text = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], 

    z = z, 

    colorbar = {'title':'Death Total by State: 2007'}

)



layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [data],layout=layout)

iplot(choromap)
Y=[]

for i in FL['Year'].unique():

    byyear = df.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    Y.append(byyear)

    

States = []

for i in df['State'].unique()[:50]:

    bystate = df.loc[(df['Year']==2016) & (df['State']==i) & (df['Cause']) & (df['Deaths']), ['Year','State','Cause','Deaths']]

    bystate = bystate['Deaths'].sum()

    States.append(bystate)



len(States)

z = States
data = dict(

    type = 'choropleth', 

    locations = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],

    locationmode = 'USA-states', 

    colorscale = [

        [1.0, 'rgb(180, 180, 180)'],

        [0, 'rgb(0, 0, 0)']

    ],

    text = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], 

    z = z, 

    colorbar = {'title':'Death Total by State: 2016'}

)



layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [data],layout=layout)

iplot(choromap)
#Linear Regression



D=[]

for i in US['Year'].unique():

    byyear = US.loc[(df['Year']==i) & (df['Cause']) & (df['Deaths']), ['Year','Cause','Deaths']]

    byyear = byyear['Deaths'].sum()

    D.append(byyear)

    

years = np.unique(US.Year)





from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
x = np.unique(years).reshape(-1,1) 

y = D



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state=42)



reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
plt.scatter(x_test, y_test,  color='black')

plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xlabel('Years')

plt.ylabel('US Death Rate')

plt.xticks([1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015])



plt.show()