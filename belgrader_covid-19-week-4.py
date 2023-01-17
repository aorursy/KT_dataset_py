import pandas as pd                #Pandas library for data analysis

import numpy as np                 #For numerical analysis of data

import matplotlib.pyplot as plt    #Python's plotting 





import plotly.express as px       #Plotly for plotting the COVID-19 Spread.

import plotly.offline as py       #Plotly for plotting the COVID-19 Spread.

import seaborn as sns             #Seaborn for data plotting

import plotly.graph_objects as go #Plotlygo for plotting



from plotly.subplots import make_subplots





import glob                       #For assigning the path

import os                         #OS Library for implementing the functions.



import warnings

warnings.filterwarnings('ignore') 



#Selcting the other essential libraries for data manipulation



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import datetime as dt

import xgboost as xgb

from xgboost import XGBRegressor
#Reading the cumulative cases dataset

covid_cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



#Viewing the dataset

covid_cases.head()
#Importing the essential datasets from the challenge



training_data = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

testing_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
#Checking for the null values in dataset

print(training_data.isnull().sum())

print(testing_data.isnull().sum())



#Checking for the datatypes for the columns

print(training_data.dtypes)

print(testing_data.dtypes)



#Filling the values

training_data['Province_State'].fillna("",inplace = True)

testing_data['Province_State'].fillna("",inplace = True)
#Groping the same cities and countries together along with their successive dates.



country_list = covid_cases['Country/Region'].unique()



country_grouped_covid = covid_cases[0:1]



for country in country_list:

    test_data = covid_cases['Country/Region'] == country   

    test_data = covid_cases[test_data]

    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)

    

country_grouped_covid.reset_index(drop=True)

country_grouped_covid.head()



#Dropping of the column Last Update

country_grouped_covid.drop('Last Update', axis=1, inplace=True)



#Replacing NaN Values in Province/State with a string "Not Reported"

country_grouped_covid['Province/State'].replace(np.nan, "Not Reported", inplace=True)



#Printing the dataset

country_grouped_covid.head()



#country_grouped_covid holds the dataset for the country
#Creating a dataset to analyze the cases country wise - As of 04/10/2020



latest_data = country_grouped_covid['ObservationDate'] == '04/10/2020'

country_data = country_grouped_covid[latest_data]



#The total number of reported Countries

country_list = country_data['Country/Region'].unique()

print("The total number of countries with COVID-19 Confirmed cases = {}".format(country_list.size))
#Plotting a bar graph for confirmed cases vs deaths due to COVID-19 in World.



unique_dates = country_grouped_covid['ObservationDate'].unique()

confirmed_cases = []

recovered = []

deaths = []



for date in unique_dates:

    date_wise = country_grouped_covid['ObservationDate'] == date  

    test_data = country_grouped_covid[date_wise]

    

    confirmed_cases.append(test_data['Confirmed'].sum())

    deaths.append(test_data['Deaths'].sum())

    recovered.append(test_data['Recovered'].sum())

    

#Converting the lists to a pandas dataframe.



country_dataset = {'Date' : unique_dates, 'Confirmed' : confirmed_cases, 'Recovered' : recovered, 'Deaths' : deaths}

country_dataset = pd.DataFrame(country_dataset)



#Plotting the Graph of Cases vs Deaths Globally.



fig = go.Figure()

fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))

fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Deaths'],name='Total Deaths because of COVID-19',marker_color='rgb(26, 118, 255)'))



fig.update_layout(title='Confirmed Cases and Deaths from COVID-19',xaxis_tickfont_size=14,

                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()





fig = go.Figure()

fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))

fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Recovered'],name='Total Recoveries because of COVID-19',marker_color='rgb(26, 118, 255)'))



fig.update_layout(title='Confirmed Cases and Recoveries from COVID-19',xaxis_tickfont_size=14,

                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()
#Generating a function to concatenate all of the files available.



folder_name = '../input/covcsd-covid19-countries-statistical-dataset'

file_type = 'csv'

seperator =','

dataframe = pd.concat([pd.read_csv(f, sep=seperator) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True,sort=False)
#Selecting the columns that are required as is essential for the data-wrangling task



covid_data = dataframe[['Date', 'State', 'Country', 'Cumulative_cases', 'Cumulative_death',

       'Daily_cases', 'Daily_death', 'Latitude', 'Longitude', 'Temperature',

       'Min_temperature', 'Max_temperature', 'Wind_speed', 'Precipitation',

       'Fog_Presence', 'Population', 'Population Density/km', 'Median_Age',

       'Sex_Ratio', 'Age%_65+', 'Hospital Beds/1000', 'Available Beds/1000',

       'Confirmed Cases/1000', 'Lung Patients (F)', 'Lung Patients (M)',

       'Life Expectancy (M)', 'Life Expectancy (F)', 'Total_tests_conducted',

       'Out_Travels (mill.)', 'In_travels(mill.)', 'Domestic_Travels (mill.)']]
#Merging the columns together



training_data['Country_Region'] = training_data['Country_Region'] + ' ' + training_data['Province_State']

testing_data['Country_Region'] = testing_data['Country_Region'] + ' ' + testing_data['Province_State']

del training_data['Province_State']

del testing_data['Province_State']



#Creating a function to split-date



def split_date(date):

    date = date.split('-')

    date[0] = int(date[0])

    if(date[1][0] == '0'):

        date[1] = int(date[1][1])

    else:

        date[1] = int(date[1])

    if(date[2][0] == '0'):

        date[2] = int(date[2][1])

    else:

        date[2] = int(date[2])    

    return date



training_data.Date = training_data.Date.apply(split_date)

testing_data.Date = testing_data.Date.apply(split_date)
#Manipulation of columns for both training dataset



year = []

month = []

day = []



for i in training_data.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

    

training_data['Year'] = year

training_data['Month'] = month

training_data['Day'] = day

del training_data['Date']



#Manipulation of columns for both testing dataset



year = []

month = []

day = []

for i in testing_data.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

    

testing_data['Year'] = year

testing_data['Month'] = month

testing_data['Day'] = day

del testing_data['Date']

del training_data['Id']

del testing_data['ForecastId']

del testing_data['Year']

del training_data['Year']
#Filtering of the dataset to view the latest contents (as of 30-03-2020)

latest_data = covid_data['Date'] == '30-03-2020'

country_data_detailed = covid_data[latest_data]



#Dropping off unecssary columns from the country_data_detailed dataset

country_data_detailed.drop(['Daily_cases','Daily_death','Latitude','Longitude'],axis=1,inplace=True)



#Viewing the dataset

country_data_detailed.head(3)
#Replacing the text Not Reported and N/A with numpy missing value cmputation



country_data_detailed.replace('Not Reported',np.nan,inplace=True)

country_data_detailed.replace('N/A',np.nan,inplace=True)





#Viewing the dataset

country_data_detailed.head(3)
#Converting the datatypes



country_data_detailed['Lung Patients (F)'].replace('Not reported',np.nan,inplace=True)

country_data_detailed['Lung Patients (F)'] = country_data_detailed['Lung Patients (F)'].astype("float")
#Getting the dataset to check the correlation 

corr_data = country_data_detailed.drop(['Date','State','Country','Min_temperature','Max_temperature','Out_Travels (mill.)',

                                        'In_travels(mill.)','Domestic_Travels (mill.)','Total_tests_conducted','Age%_65+'], axis=1)



#Converting the dataset to the correlation function

corr = corr_data.corr()
#Plotting a heatmap



def heatmap(x, y, size,color):

    fig, ax = plt.subplots(figsize=(20,3))

    

    # Mapping from column names to integer coordinates

    x_labels = corr_data.columns

    y_labels = ['Cumulative_cases', 'Cumulative_death']

    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 

    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    

    n_colors = 256 # Use 256 colors for the diverging color palette

    palette = sns.cubehelix_palette(n_colors) # Create the palette

    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation



    def value_to_color(val):

        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range

        ind = int(val_position * (n_colors - 1)) # target index in the color palette

        return palette[ind]



    

    ax.scatter(

    x=x.map(x_to_num),

    y=y.map(y_to_num),

    s=size * 1000,

    c=color.apply(value_to_color), # Vector of square color values, mapped to color palette

    marker='s'

)

    

    # Show column labels on the axes

    ax.set_xticks([x_to_num[v] for v in x_labels])

    ax.set_xticklabels(x_labels, rotation=30, horizontalalignment='right')

    ax.set_yticks([y_to_num[v] for v in y_labels])

    ax.set_yticklabels(y_labels)

    

    

    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)

    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 

    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    

corr = pd.melt(corr.reset_index(), id_vars='index') 

corr.columns = ['x', 'y', 'value']

heatmap(x=corr['x'],y=corr['y'],size=corr['value'].abs(),color=corr['value'])
#Creating a correlation matrix



matrix = corr_data.corr()

print(matrix)
#Reading the temperature data file

temperature_data = pd.read_csv('../input/covcsd-covid19-countries-statistical-dataset/temperature_data.csv')



#Viewing the dataset

temperature_data.head()
#Checking the dependence of Temperature on Confirmed COVID-19 Cases



unique_temp = temperature_data['Temperature'].unique()

confirmed_cases = []

deaths = []



for temp in unique_temp:

    temp_wise = temperature_data['Temperature'] == temp

    test_data = temperature_data[temp_wise]

    

    confirmed_cases.append(test_data['Daily_cases'].sum())

    deaths.append(test_data['Daily_death'].sum())

    

#Converting the lists to a pandas dataframe.



temperature_dataset = {'Temperature' : unique_temp, 'Confirmed' : confirmed_cases, 'Deaths' : deaths}

temperature_dataset = pd.DataFrame(temperature_dataset)
#Plotting a scatter plot for cases vs. Temperature



fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(go.Scattergl(x = temperature_dataset['Temperature'],y = temperature_dataset['Confirmed'], mode='markers',

                                  marker=dict(color=np.random.randn(10000),colorscale='Viridis',line_width=1)),secondary_y=False)



fig.add_trace(go.Box(x=temperature_dataset['Temperature']),secondary_y=True)



fig.update_layout(title='Daily Confirmed Cases (COVID-19) vs. Temperature (Celcius) : Global Figures - January 22 - March 30 2020',

                  yaxis=dict(title='Reported Numbers'),xaxis=dict(title='Temperature in Celcius'))



fig.update_yaxes(title_text="BoxPlot Range ", secondary_y=True)



fig.show()

#Conducting Statistical Tests over the dataset



sample = temperature_dataset['Temperature'].sample(n=250)

test = temperature_dataset['Temperature']



from scipy.stats import ttest_ind



stat, p = ttest_ind(sample, test)

print('Statistics=%.3f, p=%.3f' % (stat, p))
training_data['ConfirmedCases'] = training_data['ConfirmedCases'].apply(int)

training_data['Fatalities'] = training_data['Fatalities'].apply(int)



cases = training_data.ConfirmedCases

fatalities = training_data.Fatalities

del training_data['ConfirmedCases']

del training_data['Fatalities']



lb = LabelEncoder()

training_data['Country_Region'] = lb.fit_transform(training_data['Country_Region'])

testing_data['Country_Region'] = lb.transform(testing_data['Country_Region'])



scaler = MinMaxScaler()

x_train = scaler.fit_transform(training_data.values)

x_test = scaler.transform(testing_data.values)
from xgboost import XGBRegressor



rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)

rf.fit(x_train,cases)

cases_pred = rf.predict(x_test)



rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)

rf.fit(x_train,fatalities)

fatalities_pred = rf.predict(x_test)
#Rouding off the prediction values and converting negatives to zero

cases_pred = np.around(cases_pred)

fatalities_pred = np.around(fatalities_pred)



cases_pred[cases_pred < 0] = 0

fatalities_pred[fatalities_pred < 0] = 0
#Importing the dataset for generating output

submission_dataset = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")



#Adding results to the dataset

submission_dataset['ConfirmedCases'] = cases_pred

submission_dataset['Fatalities'] = fatalities_pred



submission_dataset.head()
#Submitting the dataset

submission_dataset.to_csv("submission.csv" , index = False)