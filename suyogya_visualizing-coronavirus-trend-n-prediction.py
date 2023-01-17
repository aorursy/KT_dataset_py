# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Please download the Visualizing_coronavirusTrend_n_prediction.ipynb from input section


# ## Coronavirus Trend and its future forecasting



# This notebook is divided into 3 parts :

# 

# Part 1 : Trend ( Spreadness Nature ) of COVID19 all around the world [ Exploratory Data Analysis ]

# 

# Part 2 : Prediction of virus in coming days

# 

# Part 3 : Comparison between COVID19 and other pandemics ( SARS, Ebola )

# 

# I hope you will find it helpful !!! :) 

# 

# Note : all datas are taken from source (Data Repository by Johns Hopkins CSSE) : https://github.com/CSSEGISandData/COVID-19



#Import basic lib

import pandas as pd

import numpy as np

import re

from datetime import datetime



#visualization lib

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



#prediction lib

from fbprophet import Prophet #Time series predictions based on seasonality and trend





# In[2]:





#Import dataset

corona = pd.read_csv('covid19_2020.csv') 





# In[3]:





corona





# In[4]:





#Lets understand which columns have more null values

corona.isnull().sum()





# In[5]:





#cols = ['Confirmed','Recovered','Deaths']

corona['Confirmed'].fillna(0,inplace=True)

corona['Recovered'].fillna(0,inplace=True)

corona['Deaths'].fillna(0,inplace=True)





# In[6]:





corona.tail()





# In[7]:





corona.isnull().sum()





# In[8]:





#Since Province/State has higher NAN values, lets drop it as we have country

corona.drop('Province/State',axis=1,inplace=True)





# In[9]:





corona.head()





# In[10]:





corona.isnull().sum()





# In[11]:





#Lets fill latitude and longitude with mean values 

corona.describe()





# In[12]:





corona['Latitude'].fillna(29.853541,inplace=True)

corona['Longitude'].fillna(28.894062,inplace=True)





# In[13]:





corona.isnull().sum()





# In[14]:





corona





# In[15]:





corona





# In[16]:





# As an alternative we can just drop NAN values of all columns at first.

#data.dropna(subset=['Confirmed', 'Recovered', 'Deaths', 'Longitude', 'Latitude'], inplace=True)





# ## Part 1 : Trend [ Spreadness Nature ]around the world [ Exploratory Data Analysis ]



# In[17]:





corona.columns = ['Country','Latitude','Longitude','Confirm','Recover','Death','Date']





# In[18]:





corona





# In[19]:





#Get countries name

Countries = corona['Country'].unique().tolist()





# In[20]:





Countries





# In[21]:





len(Countries)





# In[22]:





#Using mercator to define the world map

def mercator(data, lon="Longitude", lat="Latitude"):

    """Converts decimal longitude/latitude to Web Mercator format"""

    k = 6378137

    data["x"] = data[lon] * (k * np.pi/180.0)

    data["y"] = np.log(np.tan((90 + data[lat]) * np.pi/360.0)) * k

    return data





# In[23]:





data = mercator(corona)





# In[24]:





data





# In[25]:





from bokeh.plotting import figure

from bokeh.io import output_notebook,show

from bokeh.models import WMTSTileSource





# In[26]:





output_notebook()





# In[27]:





url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'

p = figure(tools='pan,wheel_zoom',x_axis_type='mercator',y_axis_type='mercator')





# In[28]:





p.add_tile(WMTSTileSource(url=url))

p.circle(x=data['x'],y=data['y'],fill_color='orange',size=5)

show(p)





# ## Lets cluster the 182 countries as per their continent



# In[29]:





Europe = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria', 'Lithuania', 'Croatia', 'Luxembourg',

                'Cyprus', 'Malta', 'Czechia', 'Netherlands', 'Denmark', 'Poland', 'Estonia', 'Portugal',

                'Finland', 'Romania', 'France', 'Slovakia', 'Germany', 'Slovenia', 'Greece', 'Spain',

                'Hungary', 'Sweden', 'Ireland', 'UK']





# In[30]:





Non_Europe = ['Albania', 'Belarus', 'Bosnia', 'Herzegovina', 'Kosovo', 'Macedonia', 'Moldova',

          'Norway', 'Russia', 'Serbia', 'Switzerland', 'Ukraine', 'Turkey']





# In[31]:





Asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh' 'Bhutan',

                  'Brunei', 'Cambodia', 'Mainland China', 'Cyprus', 'Georgia','Hong Kong',

                  'India' 'Indonesia',

                  'Iran', 'Iraq', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan'

                  , 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar',

                  'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines',

                  'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka','Syria',

                  'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan',

                  'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']





# In[32]:





Africa = ['Liberia', 'Tanzania', 'Eritrea','Ethiopia', 'Cameroon', 'Ghana','South Africa', 'Kenya', 

                    'Rwanda','Nigeria', 'Gabon', 'Tunisia','Senegal', 'Algeria', 

                    'Ivory Coast','Uganda', 'Morocco', 'Zimbabwe','Egypt']





# In[33]:





America = ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic',

                   'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico',

                   'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'US']





# In[34]:





#Lets separate the continent

Data_Europe = corona[corona['Country'].isin(Europe)]

Data_Non_Europe = corona[corona['Country'].isin(Non_Europe)]

Data_Asia = corona[corona['Country'].isin(Asia)]

Data_Africa = corona[corona['Country'].isin(Africa)]

Data_America = corona[corona['Country'].isin(America)]

Data_Australia = corona[(corona['Country']=='New Zealand') | (corona['Country']=='Australia')]

Data_Others = corona[corona['Country']=='Others']





# In[35]:





Data_Europe





# In[36]:





Data_Australia





# In[37]:





Data_Others





# ## Lets check confirm and death number by continent now



# In[38]:





#Asia Confirm 

Data_Asia['Confirm'].max()

#Data_Asia['Death'].max()





# In[39]:





Total_Confirm = [Data_Asia['Confirm'].max(),Data_Europe['Confirm'].max(),

                 Data_Non_Europe['Confirm'].max(),Data_Africa['Confirm'].max(),Data_America['Confirm'].max(),

                 Data_Australia['Confirm'].max(),Data_Others['Confirm'].max()]

Total_Confirm





# In[40]:





Total_Death = [Data_Asia['Death'].max(),Data_Europe['Death'].max(),

                 Data_Non_Europe['Death'].max(),Data_Africa['Death'].max(),Data_America['Death'].max(),

                 Data_Australia['Death'].max(),Data_Others['Death'].max()]

Total_Death





# In[41]:





Areas = ['Asia','Europe','Non_Europe','Africa','America','Australia','Others']





# In[42]:





#Framing the continent in pandas dataframe 

Continents = pd.DataFrame({'Confirm' : Total_Confirm , 'Death' : Total_Death},index=Areas)





# In[43]:





Continents





# In[44]:





sns.set()



plt.figure(figsize=(12,6),dpi=300)



position = np.arange(len(Areas))

width= 0.4



#Plotting main attributes

plt.bar(position - (width/2),(Continents['Confirm']/Continents['Confirm'].sum())*100,width=width,label='Confirm')

plt.bar(position + (width/2),(Continents['Death']/Continents['Death'].sum())*100,width=width,label='Death')



#Tweaking

plt.xticks(position,rotation=10)

plt.yticks(np.arange(0,101,10))



#Extra

ax = plt.gca()

ax.set_xticklabels(Areas)

ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

ax.set_yticks(np.arange(0,100,5),minor=True)

ax.yaxis.grid(which='major')

ax.yaxis.grid(which='minor',linestyle='--')

plt.title('Confirm vs Death in different continents')

plt.legend()





# In[45]:





plt.figure(figsize=(10,6))



plt.plot('Date','Confirm', data = Data_Asia,label='Asia')

plt.plot('Date','Confirm', data = Data_Europe, label = 'Europe')

plt.plot('Date','Confirm', data = Data_Non_Europe, label = 'Non Europe')

plt.plot('Date','Confirm', data = Data_Africa, label = 'Africa')

plt.plot('Date','Confirm', data = Data_America, label = 'America')

plt.plot('Date','Confirm', data = Data_Australia, label = 'Australia')

plt.plot('Date','Confirm', data = Data_Others, label = 'Others')

plt.xticks(np.arange(0, 60, 2), rotation=70)

plt.legend()

plt.title('Confirm cases in different continent')





# In[46]:





plt.figure(figsize=(10,6))



plt.plot('Date','Death', data = Data_Asia,label='Asia')

plt.plot('Date','Death', data = Data_Europe, label = 'Europe')

plt.plot('Date','Death', data = Data_Non_Europe, label = 'Non Europe')

plt.plot('Date','Death', data = Data_Africa, label = 'Africa')

plt.plot('Date','Death', data = Data_America, label = 'America')

plt.plot('Date','Death', data = Data_Australia, label = 'Australia')

plt.plot('Date','Death', data = Data_Others, label = 'Others')

plt.xticks(np.arange(0, 60, 2), rotation=70)

plt.yticks(np.arange(0, 3001, 500))

plt.legend()

plt.title('Death cases in different continent')





# In[47]:





#Observation so far :

#It seems that death rate has decreased exponentially in Asia by the second week of this month

#and Europe have exponential increase.





# ## Part 1 ( Continued ):

# Since we get latest data , we are trying to analysis in this data as well. It has 16k rows.



# In[48]:





corona = pd.read_csv('covid_19_clean_complete.csv',parse_dates=['Date']).fillna(0)





# In[49]:





corona.info()





# In[50]:





corona.head()





# In[51]:





#No NAN values

corona.isnull().sum()





# What is the total number of Confirmed, Deaths and Recovered cases so far?



# In[52]:





#recent stat

recent_stat = corona.groupby('Date',as_index=False)['Confirmed','Deaths','Recovered'].sum()

recent_stat





# In[53]:





#Lets sort it by Latest report

sorted_stat = recent_stat.sort_values(by='Date',ascending=False) 

sorted_stat 





# In[54]:





#Lets break it down to look at increase trend

first_date = sorted_stat.iloc[-1]

recent_date = sorted_stat.iloc[0]





# In[55]:





first_date





# In[56]:





recent_date





# In[57]:





#Calculate the Death ratio 

def calculate_death_ratio(confirmed,death):

    return "{0:.2f}%".format((death/confirmed)*100)





# In[58]:





print("The Death Ratio is : {}".format(calculate_death_ratio(recent_date['Confirmed'],recent_date['Deaths'])))





# Enough with data, lets start the fun part : VISUALIZATION



# In[59]:





fig = go.Figure()

#Confirmed line

fig.add_trace(go.Scatter(

                x=recent_stat['Date'],

                y=recent_stat['Confirmed'],

                name = 'Confirmed'))



#Deaths line

fig.add_trace(go.Scatter(

                x=recent_stat['Date'],

                y=recent_stat['Deaths'],

                name = 'Deaths'))



#Recovered Line

fig.add_trace(go.Scatter(

                x=recent_stat['Date'],

                y=recent_stat['Recovered'],

                name = 'Recovered'))



fig.update_layout(title_text = 'Overview of confirmed , deaths and recovered cases around the world')



fig.show()





# Lets break it down GEOGRAPHICALLY and see which countries have been effected most?



# In[60]:





#China and United States Data

corona['Country/Region'].replace({"Mainland China":"China", "US":"United States"},inplace=True)





# In[61]:





corona_by_country = corona.groupby(["Country/Region","Province/State"],as_index=False)['Confirmed','Deaths','Recovered','Lat','Long']





# In[62]:





corona_by_country





# In[63]:





recent_geo_country = corona_by_country.last().groupby(['Country/Region']).sum()





# In[64]:





recent_geo_country





# In[65]:





#Import ISO country codes mapping to visualize the countries in maps

country_codes = pd.read_csv('wikipedia-iso-country-codes.csv',usecols = ['English short name lower case','Alpha-3 code'])





# In[66]:





country_codes.columns = ['Name','Code']





# In[67]:





country_codes





# In[68]:





#Lets map it with our data

recent_geo_country = recent_geo_country.merge(country_codes,left_on='Country/Region',right_on='Name')





# In[69]:





recent_geo_country.sort_values('Confirmed',ascending=False)





# In[130]:





#Lets visualize country wise

map_fig = px.choropleth(recent_geo_country,

                    locations="Code",

                    color="Confirmed",

                    hover_name="Name",

                    )

map_fig.update_layout(title = "Most affected ares by Geography")

map_fig.show()





# In[71]:





#Lets dive deeper into China case

china_case = corona[corona['Country/Region'] == 'China']

china_province_case = china_case.groupby('Province/State',as_index=False).last()





# In[72]:





china_case





# In[73]:





china_province_case





# In[74]:





china_fig = go.Figure(data=go.Scattergeo(

            lon = china_province_case['Long'],

            lat = china_province_case['Lat'],

            text = china_province_case['Province/State'],

            mode = 'markers',

            marker_color = china_province_case['Confirmed'],

            marker = dict(

            size = 6,

            reversescale = True,

            autocolorscale = False,

            colorscale = 'Bluered_r',

            cmin = 0,

            color = china_province_case['Confirmed'],

            cmax = china_province_case['Confirmed'].max(),

            colorbar_title="Confirmed cases"

            )))

china_fig.update_layout(title="Most affected provinces in China",geo_scope="asia")

china_fig.show()





# China vs Rest of World 



# In[75]:





#Get the count of confirmed cases for CHINA alone

china = recent_geo_country['Name']=='China'





# In[76]:





china_cases = recent_geo_country[china]['Confirmed'].iloc[0]





# In[77]:





china_cases





# In[78]:





#Get the count of confirmed cases for REST of world 

world_cases = recent_geo_country[~china]['Confirmed'].sum()





# In[79]:





world_cases





# In[80]:





#Plotting the bar graph for them

bar_graph = go.Figure([go.Bar(x=['China','Rest of World'],y=[china_cases,world_cases])])

bar_graph.update_layout(title="China vs Rest of World Confirmed Cases")

bar_graph.show()





# In[81]:





#Lets look at Italy, Spain, Germany and Iran closer as they are at top 5 position

recent_geo_country.sort_values('Confirmed',ascending=False)





# In[82]:





italy = corona['Country/Region'] == 'Italy'

spain = corona['Country/Region'] == 'Spain'

germany = corona['Country/Region'] == 'Germany'

iran = corona['Country/Region'] == 'Iran'

nepal = corona['Country/Region'] == 'Nepal'





# In[83]:





italy_case = corona[italy]

spain_case = corona[spain]

germany_case = corona[germany]

iran_case = corona[iran]

nepal_case = corona[nepal]





# In[84]:





i_figure = go.Figure()

i_figure.add_trace(go.Scatter(

            x=italy_case['Date'],

            y=italy_case['Confirmed'],

            name='Italy'

            ))



i_figure.add_trace(go.Scatter(

            x=spain_case['Date'],

            y=spain_case['Confirmed'],

            name='Spain'

            ))



i_figure.add_trace(go.Scatter(

            x=germany_case['Date'],

            y=germany_case['Confirmed'],

            name='Germany'

            ))



i_figure.add_trace(go.Scatter(

            x=iran_case['Date'],

            y=iran_case['Confirmed'],

            name='Iran'

            ))

                  

i_figure.add_trace(go.Scatter(

            x=nepal_case['Date'],

            y=nepal_case['Confirmed'],

            name='Nepal'

            ))

                   

i_figure.update_layout(title_text='Confirmed cases of COVOD19 in Italy, Spain, Germany, Iran and Nepal')

i_figure.show()





# Checking the people and its prediction



# In[85]:





people_info = pd.read_csv('People_Info.csv').fillna('NA')

people_info.head()





# What is gender ratio of effected people?

# What are the common symptoms for the effected people?



# In[86]:





gender_fig = px.histogram(people_info,x='gender')

gender_fig.show()





# In[87]:





#Lets separate the symptoms and get insights

symptoms = pd.DataFrame(data=people_info['symptom'].value_counts().head(10)[1:])

symptoms





# In[88]:





words = symptoms.index

words





# In[89]:





weights = symptoms.symptom

weights





# In[90]:





word_cloud_data = go.Scatter(x=[4,2,2,3, 1.5, 5, 4, 4,0],

                 y=[2,2,3,3,1, 5,1,3,0],

                 mode='text',

                 text=words,

                 marker={'opacity': 0.5},

                 textfont={'size': weights, 'color':["red", "green", "blue", "purple", "black", "orange", "blue", "black"]})

word_cloud_layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},

                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})





# In[91]:





word_cloud = go.Figure(data = [word_cloud_data], layout=word_cloud_layout)

word_cloud.update_layout(title_text='Word cloud of most common symptoms by frequency')

word_cloud.show()





# We can see FEVER and COUGH are distinguishing symptoms among all the symptoms in our 26 thousand records.



# Lets see how AGE matters for RECOVERY and DEATH?



# In[92]:





people_info





# In recovery and death column, data of DATE is present in some rows. To make it TRUE ( 1 ), lets build a function and plot it



# In[93]:





def is_date(value):

    ##will return 1 if date is present

    regex = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')

    return bool(regex.match(value))





# In[94]:





people_info['clean_recovered'] = people_info['recovered'].apply(lambda x : '1' if is_date(x) else x)

people_info['clean_recovered'] = people_info['clean_recovered'].astype('category')





# In[95]:





people_info['clean_recovered']





# In[96]:





people_info['clean_deaths'] = people_info['death'].apply(lambda x : '1' if is_date(x) else x)

people_info['clean_deaths'] = people_info['clean_deaths'].astype('category')





# In[97]:





people_info['clean_deaths']





# In[98]:





rec_age_fig = make_subplots(rows=1,cols=2,subplot_titles=('Age vs. Recovered','Age vs. Death'))

rec_age_fig





# In[99]:





#Plot the Age vs Recovered.

rec_age_fig.add_trace(go.Box(

                            x=people_info['clean_recovered'],

                            y=people_info['age'],

                            name='Recovered'),

                            row=1,

                            col=1

                            )

                        





# In[100]:





#Plot the Age vs Death

rec_age_fig.add_trace(go.Box(

                                x=people_info['clean_deaths'],

                                y=people_info['age'],

                                name='Deaths'),

                        row=1,

                        col=2                  

                        )





# In[101]:





rec_age_fig.update_traces(boxpoints='all')

rec_age_fig.update_layout(title_text="Subplots of age in relation to recovery and death")

rec_age_fig.show()





# How many people travelled to or from Wuhan ( origin place of COVID19 )? 



# In[102]:





total_num = len(people_info)

visiting_or_from_wuhan = people_info['visiting Wuhan'].value_counts()[1]+people_info['from Wuhan'].value_counts()[1]

not_visiting_or_from_wuhan = total_num - visiting_or_from_wuhan

wuhan_info = pd.DataFrame([visiting_or_from_wuhan,not_visiting_or_from_wuhan],columns=['Total'],index=['Visiting/from Wuhan','Not Visiting/from Wuhan'])





# In[103]:





wuhan_info





# In[104]:





pie_fig = go.Figure(data=[go.Pie(labels=wuhan_info.index,values = wuhan_info['Total'])])

pie_fig.show()





# ## Part 2 : Prediction of virus in coming days 



# Preparation of Data



# In[105]:





time_series = corona[['Date','Confirmed']].groupby('Date',as_index=False).sum()





# In[106]:





time_series





# In[107]:





time_series.columns = ['ds','y']





# In[108]:





time_series.ds = pd.to_datetime(time_series.ds)





# In[109]:





time_series





# Dividing into train set and test sets



# In[110]:





train_range = np.random.rand(len(time_series)) < 0.8

train_range





# In[111]:





train_ts = time_series[train_range]

train_ts





# In[112]:





test_ts = time_series[~train_range]

test_ts = test_ts.set_index('ds')

test_ts





# Prophet Model



# In[113]:





#Train Model

prophet_Model = Prophet()

prophet_Model.fit(train_ts)





# In[114]:





#Test Model

future = pd.DataFrame(test_ts.index)

predict = prophet_Model.predict(future)

forecast = predict[['ds','yhat','yhat_lower','yhat_upper']]

forecast = forecast.set_index('ds','','')





# In[115]:





future





# In[116]:





predict





# In[117]:





forecast





# In[118]:





#Lets visualize

test_fig = go.Figure()



test_fig.add_trace(go.Scatter(

                        x=test_ts.index,

                        y=test_ts.y,

                        name="Actual Cases",

                        mode='lines',

                        line_color= "deepskyblue"

                        ))



test_fig.add_trace(go.Scatter(

                x= forecast.index,

                y= forecast.yhat,

                name= "Prediction",

                mode = 'lines'))



test_fig.add_trace(go.Scatter(

                x= forecast.index,

                y= forecast.yhat_lower,

                name= "Prediction Lower Bound",

                mode = 'lines',

                line = dict(color='gray', width=2, dash='dash')))



test_fig.add_trace(go.Scatter(

                x= forecast.index,

                y= forecast.yhat_upper,

                name= "Prediction Upper Bound",

                mode= 'lines',

                line = dict(color='royalblue',width=2,dash='dash')))



test_fig.update_layout(title_text="Prophet Model's Test Prediction",xaxis_title="Date",yaxis_title="Cases")

test_fig.show()





# Evaluate our model



# In[119]:





def calculate_mse(actual,predicted):

    ##Calculate Mean Squared Error for given predicted and actual values

    errors = 0

    n = len(actual)

    for i in range(n):

        errors += (actual[i] - predicted[i]) **2

    return errors/n





# In[120]:





print("The MSE for the Prophet time series model is {}".format(calculate_mse(test_ts.y,forecast.yhat)))





# Lets fit our whole data into model keep MSE in mind



# In[121]:





prophet_model_full = Prophet()

prophet_model_full.fit(time_series)





# In[122]:





future_full = prophet_model_full.make_future_dataframe(periods=150)

future_full





# In[123]:





forecast_full = prophet_model_full.predict(future_full)

forecast_full = forecast_full.set_index('ds')

forecast_full





# In[124]:





pred_fig = go.Figure()



pred_fig.add_trace(go.Scatter(

                    x=time_series.ds,

                    y=time_series.y,

                    name="Actual",

                    line_color="red",

                    opacity=0.8

                    ))



pred_fig.add_trace(go.Scatter(

                    x=forecast_full.index,

                    y=forecast_full.yhat,

                    name="Prediction",

                    line_color="deepskyblue",

                    opacity=0.8

                    ))



pred_fig.update_layout(title_text="Prophet Model Forecasting",xaxis_title="Date",yaxis_title="Cases")

pred_fig.show()





# ## Part 3 : Comparison between COVID19 and other pandemics ( SARS, Ebola, H1N1 ) 



# In[125]:





epidemics_data = pd.DataFrame({

    'epidemic' : ['COVID19','SARS','EBOLA','MERS','H1N1'],

    'start_year' : [2019,2003,2014,2012,2009],

    'end_year' : [2020,2004,2016,2017,2010],

    'confirmed' : [corona['Confirmed'].sum(),8096,28646,2494,6724149],

    'deaths' : [corona['Deaths'].sum(),774, 11323, 858, 19654]

                            })





# In[126]:





epidemics_data['mortality_rate'] = round((epidemics_data['deaths']/epidemics_data['confirmed'])*100,2)





# In[127]:





epidemics_data





# In[128]:





#visualization

temp = epidemics_data.melt(

                            id_vars='epidemic',

                            value_vars=['confirmed','deaths','mortality_rate'],

                            var_name='Case',

                            value_name='Value')

temp





# In[129]:





fig = px.bar(

            temp,

            x="epidemic",

            y="Value",

            text="Value",

            color="epidemic",

            facet_col="Case",

            color_discrete_sequence = px.colors.qualitative.Bold)



fig.update_traces(textposition="outside")

fig.update_layout(uniformtext_minsize=8,uniformtext_mode='hide')

fig.update_yaxes(showticklabels=False)

fig.layout.yaxis2.update(matches=None)

fig.layout.yaxis3.update(matches=None)

fig.show()


