# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Importing all the necessary tools of plotly

import plotly.io as pio

pio.renderers.default = "svg"



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import *

from sklearn.linear_model import LinearRegression

import xgboost as xgb

import lightgbm as lgb

import plotly.offline as pl

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings(action='ignore')
data = pd.read_csv(r'/kaggle/input/time-series-19-covid-combined_csv23_sep.csv')

#/kaggle/input/datahub-covid19/time-series-19-covid-combined_11_augcsv
data.head()
data['Datenew'] = pd.to_datetime(data.Date)
data.rename(columns={'Country/Region':'Country'},inplace=True)

x=pd.DataFrame(data.Date.str.split('-',expand=True)).rename({0:'Year',1:'Month',2:'Day'},axis=1).astype('int')

data=pd.concat([data,x],axis=1)

data.drop(columns='Date',inplace=True)
a = list(set(data.Country))



data2 = pd.DataFrame(columns=data.columns)

data1 = pd.DataFrame(columns=data.columns)



for i in range(1,188):

    

    if (data[data.Country==a[i]].Lat.nunique() >=2):

        data2 = pd.concat([data2,data[data.Country==a[i]]],axis=0)

                      

    else:

        data1 = pd.concat([data1,data[data.Country==a[i]]],axis=0)



data2.reset_index(drop=True,inplace=True)

data1.reset_index(drop=True,inplace=True)
data2.dropna(inplace=True)

data2.reset_index(drop=True,inplace=True)
coun = sorted(list(set(data2.Country)))



data2_au = data2[data2.Country==coun[0]].reset_index(drop=True)

data2_ch = data2[data2.Country==coun[1]].reset_index(drop=True)

data2_de = data2[data2.Country==coun[2]].reset_index(drop=True)

data2_fr = data2[data2.Country==coun[3]].reset_index(drop=True)

data2_ne = data2[data2.Country==coun[4]].reset_index(drop=True)

data2_uk = data2[data2.Country==coun[5]].reset_index(drop=True)
data2_fr.nunique()
upd=226 ### Total no. of days around which dataset is utilized i.e from 22nd January 2020 to 3rd September 2020
data2_uk['Con']=0

data2_uk['Dea']=0

data2_uk['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_uk.Lat.nunique():

        x = x + data2_uk['Confirmed'][j+i]

        y = y + data2_uk['Deaths'][j+i]

        z = z + data2_uk['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_uk.loc[i,'Con'] = x

    data2_uk.loc[i,'Dea'] = y

    data2_uk.loc[i,'Rec'] = z



data2_uk['Days']=0

for i in range(1,len(data2_uk)+1):

    data2_uk.loc[i-1,'Days'] = i

    

data2_uk.drop(index=list(range(upd,data2_uk.shape[0])),inplace=True)

    

data2_ne['Con']=0

data2_ne['Dea']=0

data2_ne['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_ne.Lat.nunique():

        x = x + data2_ne['Confirmed'][j+i]

        y = y + data2_ne['Deaths'][j+i]

        z = z + data2_ne['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_ne.loc[i,'Con'] = x

    data2_ne.loc[i,'Dea'] = y

    data2_ne.loc[i,'Rec'] = z



data2_ne['Days']=0

for i in range(1,len(data2_ne)+1):

    data2_ne.loc[i-1,'Days'] = i

    

data2_ne.drop(index=list(range(upd,data2_ne.shape[0])),inplace=True)



    

data2_ch['Con']=0

data2_ch['Dea']=0

data2_ch['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_ch.Lat.nunique():

        x = x + data2_ch['Confirmed'][j+i]

        y = y + data2_ch['Deaths'][j+i]

        z = z + data2_ch['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_ch.loc[i,'Con'] = x

    data2_ch.loc[i,'Dea'] = y

    data2_ch.loc[i,'Rec'] = z



data2_ch['Days']=0

for i in range(1,len(data2_ch)+1):

    data2_ch.loc[i-1,'Days'] = i

    

data2_ch.drop(index=list(range(upd,data2_ch.shape[0])),inplace=True)



    

data2_de['Con']=0

data2_de['Dea']=0

data2_de['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_de.Lat.nunique():

        x = x + data2_de['Confirmed'][j+i]

        y = y + data2_de['Deaths'][j+i]

        z = z + data2_de['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_de.loc[i,'Con'] = x

    data2_de.loc[i,'Dea'] = y

    data2_de.loc[i,'Rec'] = z



data2_de['Days']=0

for i in range(1,len(data2_de)+1):

    data2_de.loc[i-1,'Days'] = i

    

data2_de.drop(index=list(range(upd,data2_de.shape[0])),inplace=True)



    

data2_au['Con']=0

data2_au['Dea']=0

data2_au['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_au.Lat.nunique():

        x = x + data2_au['Confirmed'][j+i]

        y = y + data2_au['Deaths'][j+i]

        z = z + data2_au['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_au.loc[i,'Con'] = x

    data2_au.loc[i,'Dea'] = y

    data2_au.loc[i,'Rec'] = z



data2_au['Days']=0

for i in range(1,len(data2_au)+1):

    data2_au.loc[i-1,'Days'] = i

    

data2_au.drop(index=list(range(upd,data2_au.shape[0])),inplace=True)

    

    

data2_fr['Con']=0

data2_fr['Dea']=0

data2_fr['Rec']=0



for i in range(upd):

    x,y,z=0,0,0

    n=1

    j=0

    while n!=data2_fr.Lat.nunique():

        x = x + data2_fr['Confirmed'][j+i]

        y = y + data2_fr['Deaths'][j+i]

        z = z + data2_fr['Recovered'][j+i]

        j=j+upd

        n=n+1

        

    data2_fr.loc[i,'Con'] = x

    data2_fr.loc[i,'Dea'] = y

    data2_fr.loc[i,'Rec'] = z



data2_fr['Days']=0

for i in range(1,len(data2_fr)+1):

    data2_fr.loc[i-1,'Days'] = i

    

data2_fr.drop(index=list(range(upd,data2_fr.shape[0])),inplace=True)
data2 = pd.concat([data2_uk,data2_ne,data2_ch,data2_de,data2_au,data2_fr],axis=0).reset_index(drop=True)

data2['Confirmed'] = data2['Con']

data2['Deaths'] = data2['Dea']

data2['Recovered'] = data2['Rec']

data2.drop(columns = ['Con','Dea','Rec'],inplace=True)
data1.shape
data1['Days'] = 0



for i in range(0,data1.shape[0],upd):

    for j in range(1,upd+1):

        data1.loc[i+j-1,'Days'] = j
data_updated = pd.concat([data1,data2],axis=0)

data_updated.reset_index(drop=True,inplace=True)

data_updated.drop(columns=['Province/State','Lat','Long'],inplace=True)

data_updated.rename(columns={'Day':'Day_of_the_month'},inplace=True)
data_updated['Continent'] = 0



def continent(x):

      

    if x in ['Afghanistan','Armenia','Azerbaijan','Bahrain','Bangladesh','Bhutan','Burma','Brunei','Korea, South','Cambodia','China','Cyprus','Georgia','India','Indonesia','Iran','Iraq','Israel','Japan','West Bank and Gaza','Jordan','Kazakhstan','Kuwait','Kyrgyzstan','Taiwan*','Laos','Lebanon','Malaysia','Maldives','Mongolia','Myanmar','Nepal','North Korea','Oman','Pakistan','Palestine','Philippines','Qatar','Russia','Saudi Arabia','Singapore','South Korea','Sri Lanka','Syria','Taiwan','Tajikistan','Thailand','Timor-Leste','Turkey','Turkmenistan','United Arab Emirates','UAE','Uzbekistan','Vietnam','Yemen']:

        return('Asia')

    

    elif x in ['Albania','Andorra','Holy See','Bosnia and Herzegovina','Armenia','Austria','Azerbaijan','Belarus','Belgium','Bosnia','Herzegovina','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Georgia','Germany','Greece','Hungary','Iceland','Ireland','Italy','Kazakhstan','Kosovo','Latvia','Liechtenstein','Lithuania','Luxembourg','Malta','Moldova','Monaco','Montenegro','Netherlands','North Macedonia','Macedonia','Norway','Poland','Portugal','Romania','San Marino','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine','United Kingdom','Vatican City']:

        return ('Europe')

    

    elif x in ['Algeria','Angola','Benin','Botswana','Congo (Kinshasa)','Burkina Faso','Burundi','Cabo Verde','Cameroon','Central African Republic','CAR','Chad','Comoros','Congo','Djibouti','Egypt','Western Sahara','Equatorial Guinea','Eritrea','Eswatini','Swaziland','Ethiopia','Gabon','Gambia','Ghana','Guinea','Guinea-Bissau','Kenya','Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique','Namibia','Niger','Nigeria','Rwanda','Sao Tome and Principe','Senegal','Seychelles','Congo (Brazzaville)','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe']:

        return ('Africa')

    

    elif x in ['Antigua and Barbuda','Bahamas','Barbados',"Cote d\'Ivoire",'Belize','Canada','Costa Rica','Cuba','Dominica','Dominican Republic','El Salvador','Grenada','Guatemala','Haiti','Honduras','Jamaica','Mexico','Nicaragua','Panama','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','US']:

        return ('North America')

    

    elif x in ['Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Guyana','Paraguay','Peru','Suriname','Uruguay','Venezuela']:

        return 'South America'

    

    elif x in ['Australia','Fiji','Kiribati','Marshall Islands','Micronesia','Nauru','New Zealand','Palau','Papua New Guinea','Samoa','Solomon Islands','Tonga','Tuvalu','Vanuatu']:

        return 'Australia'

    

    elif x in ['Diamond Princess','MS Zaandam']:

        return 'Cruise ship'

    

    else:

        pass

    

data_updated['Continent'] = data_updated['Country'].apply(lambda x : continent(x))    
data_updated.Confirmed = data_updated.Confirmed.astype('int64')

data_updated.Recovered = data_updated.Recovered.astype('int64')

data_updated.Deaths = data_updated.Deaths.astype('int64')

data_updated.Days = data_updated.Days.astype('int64')

data_updated.Year = data_updated.Year.astype('int64')

data_updated.Month = data_updated.Month.astype('int64')
data_updated['Day_of_Week'] = data_updated.Datenew.dt.dayofweek

data_updated['Week'] = data_updated.Datenew.dt.week
data_updated['Day_of_Week'].replace({0:'Sunday', 1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'},inplace=True)
plt.figure(figsize=(15,11))

sns.distplot(np.log1p(data_updated.Confirmed),kde=True)
plt.figure(figsize=(15,11))

sns.distplot(np.log1p(data_updated.Deaths))
plt.figure(figsize=(15,11))

sns.distplot(np.log1p(data_updated.Recovered))
plt.figure(figsize = (15,8))

sns.heatmap(data_updated.corr())
data_updated.head()
## Here I have preprocessed and extracted the US data



data_us = data_updated[data_updated.Country=='US']
tr1 = go.Scatter(                           # 'go' is taken from 'import plotly.graph_objs as go' which has been executed already

 

    

                x=data_us.Days,             # data to be assign along the x-axis

    

                y=data_us.Confirmed,        # data to be assigned along the y-axis

    

                mode='lines',               # mode helps to assgn the plot which we are interested in. 

                                            # mode has 'lines','markers' and 'lines+markers' parameters, 'line+markers' gives more clear and precised visualization by adding dots at every specific coordinate

    

                name = 'Confirmed',         # name defines the name of which plot is this

    

                text = 'U.S'              # text is usually printed whenever you put the mouse cursor on that plot to see to which data the plot is associated with



)            

data1=[tr1]        # this is used to store the trace in the list. For more than one plot, we add their trace in this list only



layout1 = dict(title = 'Confirmed cases vs Days',xaxis=dict(title='Days'),yaxis=dict(title='No. of cases'))        # we design layout to provide title of the plot and to give specific name to the x-axis and y-axis



fig = dict(data=data1,layout=layout1)          # further we form a dictionary having the srored trace and the layout



iplot(fig)
tr1 = go.Scatter(

                x=data_updated[data_updated.Country=='US'].Days,

    

                y=data_updated[data_updated.Country=='US'].Confirmed,

    

                mode='lines',

    

                name = 'US',

    

                text = 'Day vs Confirmed in U.S.A')



tr2 = go.Scatter(

    

    x=data_updated[data_updated.Country=='Russia'].Days,

    

    y=data_updated[data_updated.Country=='Russia'].Confirmed,

    

    mode = 'lines',

    

    name='Russia',

    

    text = 'Day vs Confirmed in Russia'

)



tr3 = go.Scatter(

    

    x=data_updated[data_updated.Country=='Brazil'].Days,

    

    y=data_updated[data_updated.Country=='Brazil'].Confirmed,

    

    mode = 'lines',

    

    name='Brazil',

    

    text = 'Day vs Confirmed in Brazil'

)



tr4 = go.Scatter(

    

    x=data_updated[data_updated.Country=='India'].Days,

    

    y=data_updated[data_updated.Country=='India'].Confirmed,

    

    mode = 'lines',

    

    name='India',

    

    text = 'Day vs Confirmed in India'

)







layout1 = dict(title = 'Confirmed cases vs Days in diff. countries',xaxis=dict(title='Days'),yaxis=dict(title='No. of cases'))



data1=[tr1,tr2,tr3,tr4]



fig = dict(data=data1,layout=layout1)



iplot(fig)
## Using plotly.express



fig1 = px.line(data_us, x="Days", y="Confirmed", color="Month")



#fig2 = px.line(data_us, x="Days", y="Deaths", color="Month")

#fig3 = px.line(data_us, x="Days", y="Recovered", color="Month")



iplot(fig1)

# Similarly you can plot for fig2 and fig3 specifically
# Here, for the ease of understanding, I just plotted 70 data points only



tr1 = go.Scatter(

                x=data_us.Confirmed,

    

                y=data_us.Deaths,

    

                mode='markers',

    

                name = 'Confirmed with deaths',

    

                text = 'Confirmed with deaths in U.S.A'



)



tr2 = go.Scatter(

                x=data_us.Confirmed,

    

                y=data_us.Recovered,

    

                mode='markers',

    

                name = 'Confirmed with recovered',

    

                text = 'Confirmed with recovered in U.S.A'



)



layout1 = dict(title = 'Confirmed deaths vs Confirmed recovered ',xaxis=dict(title='Confirmed cases'),yaxis=dict(title='Deaths and Recovered cases'))



data1=[tr1,tr2]



fig = dict(data=data1,layout=layout1)



iplot(fig)
fig = px.scatter(data_us[:226], x="Confirmed", y="Deaths", color="Month",title='Confirmed deaths')

iplot(fig)
fig1 = px.bar(data_us[:226], x='Days',y = 'Deaths',color = 'Month', title = 'Daywise representation')

iplot(fig1)
data_overall = data_updated.groupby(['Country'])[['Country','Confirmed','Deaths','Recovered']].agg('max').sort_values(by='Confirmed')[:]



data_overall.reset_index(drop = True,inplace = True)        
## Using plotly.graph_objs



tr1 = go.Bar(

    ## Plotted for top 15 countries

    x = data_overall.Country[173:],      

    y = data_overall.Confirmed[173:],

    name = 'Confirmed',

  #  text = data_overall.Country[173:]

)



tr2 = go.Bar(

    ## Plotted for top 15 countries

    x = data_overall.Country[173:],      

    y = data_overall.Deaths[173:],

    name = 'Confirmed',

  #  text = data_overall.Country[173:]

)





data1 = [tr1]

layout1 = go.Layout(barmode='group')

fig = go.Figure(data=data1 , layout=layout1)

iplot(fig)
## Using plotly.express



## Here again I have used US data



## Additionally I have added more parameters for better understanding



x = px.bar(data_updated[data_updated.Country=='US'],x = 'Month', y = 'Confirmed',color = 'Deaths')

iplot(x)
figure = px.scatter(data_overall[:60],                    # DataFrame which are going to visualize

                 

                 x='Confirmed',                   # Factor along x-axis

                 

                 y="Deaths",                      # Factor along y-axis

                 

                 size='Recovered',                # Size of the bubble. The more the value of size, the more will be the size of bubble and vice versa

                 

                 size_max=35,                    # Referencing the maximum size of thr bubble

                 

                 title = ' Bubble Chart representing the overall summary  of confirmed, deaths and recovered cases ',

                 

                 color = 'Recovered',

                 

                )

                 

figure.show()
fig = px.box(data_updated[:640],

             x="Month",

             y="Confirmed")

fig.show()
data_updated.info()
figure = px.scatter_geo(

                     data_updated,               ## Main dataframe

    

                     locations="Country",           ## The factor which includes all the details

    

                     locationmode='country names',  ## Assigned location area to be plotted on the map

    

                     color="Continent",             ## Assigned color for better understanding

    

                     hover_name="Country",          ## Detail associated with the area

    

                     size = 'Confirmed',              ## Variation in size is assgned to a factpr which is dependent on the frame factor 

    

                     animation_frame="Days",        ## the frame which we want to assgn on which the entire plot will vary

    

                     projection="natural earth",     ## you can aso add 'equirectangular'`, `'mercator'`, `'orthographic'`,`'kavrayskiy7'`,etc also

                    

                     title = 'Variation of the Confirmed cases with the rise in the no, of days in the animated form'

    

)

figure.show()
fig = px.choropleth(

                    data_updated,

    

                    locations='Country',

    

                    locationmode='country names',

    

                    color='Confirmed',

    

                    hover_name='Country',

    

                    animation_frame='Days',

        

    

    ###  Here, plotly express has projection parameter which is used to cahnge the orientation of the chart. Default is 'natural earth'. You may also use 'orthographic' 

    

            

                    title='Choropleth representation of the data'

)

fig.show()
figure = px.scatter(data_updated, 

                    

                    x='Confirmed', 

                    

                    y='Recovered', 

                    

                    animation_frame="Days",

                    

                    animation_group='Country',

                    

                    size="Deaths", 

                    

                    color="Continent", 

                    

                    hover_name="Country",

                    

                    size_max=55  

                    

                    

                    )



########################################################################################################################################################################

##### If you do not want manual control over the chart then drop the below line. Actually doing this will activate the updatemenus parameter in the layout method 

########################################################################################################################################################################

#figure["layout"].pop("updatemenus")

figure.show()
a1=data_updated.groupby(['Country']).get_group(('China'))

a2=data_updated.groupby(['Country']).get_group(('Russia'))

a3=data_updated.groupby(['Country']).get_group(('US'))

a4=data_updated.groupby(['Country']).get_group(('Brazil'))

a5=data_updated.groupby(['Country']).get_group(('India'))

a6=data_updated.groupby(['Country']).get_group(('Italy'))

a7=data_updated.groupby(['Country']).get_group(('France'))

a8=data_updated.groupby(['Country']).get_group(('Germany'))

a9=data_updated.groupby(['Country']).get_group(('Iran'))

a10=data_updated.groupby(['Country']).get_group(('Pakistan'))

a13=data_updated.groupby(['Country']).get_group(('Saudi Arabia'))

a14=data_updated.groupby(['Country']).get_group(('Mexico'))

a15=data_updated.groupby(['Country']).get_group(('South Africa'))

a0 = pd.concat([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a13,a14,a15],axis=0)



##### SELECTED TOP INFECTED COUNTRIES AND SOME LESS INFECTED ONES ######



figure = px.bar(a0, 

                    

                    x='Confirmed',     #### factor along x axis

                    

                    y='Country',    #### Category on which we are interested to see the case

                    

                    animation_frame="Days",          ### frame associated

                    

                    animation_group='Country',

                    

                    color="Continent",

                labels = 'Months',

                

    ########### To have verical bar chart, apply orientation = 'v'

                

                 barmode='overlay',

                

                                

                    )

figure.show()
figure = px.bar(a0, 

                    

                    x='Deaths',     #### factor along x axis

                    

                    y='Country',    #### Category on which we are interested to see the case

                    

                    animation_frame="Days",          ### frame associated

                    

                    animation_group='Country',

                    

                    color="Continent",

                labels = 'Months',

                

    ########### To have verical bar chart, apply orientation = 'v'

                

                 barmode='overlay',

                

                                

                    )

figure.show()
figure = px.bar(a0, 

                    

                    x='Recovered',     #### factor along x axis

                    

                    y='Country',    #### Category on which we are interested to see the case

                    

                    animation_frame="Days",          ### frame associated

                    

                    animation_group='Country',

                    

                    color="Continent",

                labels = 'Months',

                

    ########### To have verical bar chart, apply orientation = 'v'

                

                 barmode='overlay',

                

                                

                    )

figure.show()
data_updated.describe()
data = data_updated
data
data.drop(columns=['Month','Datenew','Day_of_the_month'], inplace=True)
data = pd.concat([data,pd.get_dummies(data[['Country','Continent','Day_of_Week','Week']])],axis=1)
data.drop(columns=data[['Country','Continent','Day_of_Week','Week']],inplace=True)
data.drop(columns=['Year'],inplace=True)
data = data.sample(frac=1).reset_index(drop=True)
from sklearn.metrics import mean_squared_error
label = data.Recovered

data.drop(columns=['Recovered'],inplace=True)
def model(m,train,label,c,rnd):

    l1=list()

    l2=list()

    splitters = KFold(n_splits = c,random_state=rnd)

    for tr_index,te_index in splitters.split(train,label):

        

        xtrain,xval = train.loc[tr_index,:],train.loc[te_index,:]

        ytrain,yval = label.loc[tr_index],label.loc[te_index]

        

        m.fit(xtrain,ytrain)

        tr_pred = m.predict(xtrain)

        te_pred = m.predict(xval)

        

        print('Training loss: {}               , Validation loss: {}'.format(np.sqrt(mean_squared_error(ytrain,tr_pred)),np.sqrt(mean_squared_error(yval,te_pred))))

        l1.append(np.sqrt(mean_squared_error(ytrain,tr_pred)))

        l2.append(np.sqrt(mean_squared_error(yval,te_pred)))

    

    print(np.mean(l1),np.mean(l2))

    

    return(l1,l2)
x,y = model(GradientBoostingRegressor(random_state=29385),data,np.log1p(label),5,76344)
plt.figure(figsize=(15,9))

plt.plot(list(np.arange(0,1,0.2)),x,'r--')

plt.plot(list(np.arange(0,1,0.2)),y,'b')

plt.legend(loc='best')

plt.ylim(0.5,0.7)

plt.title('Training/Validation Loss Summary')