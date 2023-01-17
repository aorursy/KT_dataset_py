# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from pandas import DataFrame

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

pd.plotting.register_matplotlib_converters()

import pandas_profiling

from pandas_profiling import ProfileReport

from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

from plotly.subplots import make_subplots



from scipy.optimize import minimize

import nltk

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler

from sklearn.linear_model import LogisticRegression,SGDClassifier,LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split



import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission_csv = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

covid_India = pd.read_csv("/kaggle/input/dataset/covid_19_india.csv")

population = pd.read_csv("/kaggle/input/datapopulation/population_total_long.csv")

population_density = pd.read_csv("/kaggle/input/dataset/population_density_long.csv")
test_df.head(10)
temp = train_df.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')

temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5



fig = px.scatter_geo(temp, locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Cases Over Time', color_continuous_scale="hot_r")

fig.show()
plt.figure(figsize=(13,6))

plt.subplot(1,2,1)

train_df.groupby('Date')['ConfirmedCases'].sum().plot(color='blue')

plt.ylabel('Number of Confirmed Cases')

plt.title('Confirmed Cases worldwide trend')

plt.xticks(rotation=45)



plt.subplot(1,2,2)

temp = train_df.groupby('Date')['Fatalities'].sum()

temp.plot(color='r')

plt.ylabel('Number of Fatalities')

plt.title("Fatalities worldwide trend")

plt.xticks(rotation=45)



plt.tight_layout()
Country=pd.DataFrame()

temp = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country_Region'])["ConfirmedCases"].sum().reset_index()

Country['Name']=temp["Country_Region"]

Country['Values']=temp["ConfirmedCases"]



temp.sort_values('ConfirmedCases',inplace = True, ascending = False)

temp2 = temp[:10][:]



fig = px.choropleth(Country, locations='Name',

                    locationmode='country names',

                    color="Values")

fig.update_layout(title="Corona confirmed cases on 09-04-2020")

fig.show()



fig = px.bar(temp2, 

             x='Country_Region', y='ConfirmedCases', color="ConfirmedCases", hover_data=['ConfirmedCases'], 

                 color_continuous_scale=px.colors.sequential.Bluered, width=700, height=400)

fig.update_layout(title_text='Top 10 countries with Confirmed cases due to COVID 19')

fig.show()
Country=pd.DataFrame()

temp = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country_Region'])["Fatalities"].sum().reset_index()

temp.sort_values('Fatalities',inplace = True, ascending = False)

temp2 = temp[:10][:]



Country['Name']=temp["Country_Region"]

Country['Values']=temp["Fatalities"]



fig = px.choropleth(Country, locations='Name',

                    locationmode='country names',

                    color="Values")

fig.update_layout(title="Corona Fatalities on 09-04-2020")

fig.show()



fig = px.bar(temp2, 

             x='Country_Region', y='Fatalities', color="Fatalities", hover_data=['Fatalities'], 

                 color_continuous_scale=px.colors.sequential.Bluered, width=700, height=400)

fig.update_layout(title_text='Top 10 countries with Fatalities due to COVID 19')

fig.show()
temp = train_df.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp1 = train_df.groupby(['Date', 'Country_Region'])['Fatalities'].sum().reset_index()

new = pd.concat([temp,temp1.Fatalities],axis=1)

for country in ['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']:

    temp1 = new[temp.Country_Region == country]

    temp2 = temp1[temp1.Date >= '2020-02-01'] 

    fig = px.bar(temp2, 

             x='Date', y='ConfirmedCases', color="ConfirmedCases", hover_data=['Fatalities'], 

                 color_continuous_scale=px.colors.sequential.YlOrRd, width=700, height=400)

    fig.update_layout(title_text='COVID-19 Confirmed cases and Fatalities per day in {}'.format(country))

    fig.show()
temp = train_df.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')



temp2 = temp

temp2.sort_values(['Country_Region', 'Date'],inplace = True, ascending = True)



temp3=pd.DataFrame()



temp3['Date'] = temp2["Date"]

temp3['Country']=temp2["Country_Region"]

temp3['Values']=temp2["ConfirmedCases"]



list_countries=['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']



#print(list_countries)

temp3 = temp3[temp3['Country'].isin(list_countries)]

#print(temp3)



fig = px.line(temp3, 

             x='Date', y='Values', color="Country", hover_data=['Values'], 

                  width=700, height=400)

fig.update_layout(title_text='Top 10 countries with Confirmed cases due to COVID 19')

fig.show()





temp = train_df.groupby(['Date', 'Country_Region'])['Fatalities'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')



temp2 = temp

temp2.sort_values(['Country_Region', 'Date'],inplace = True, ascending = True)



temp3=pd.DataFrame()



temp3['Date'] = temp2["Date"]

temp3['Country']=temp2["Country_Region"]

temp3['Values']=temp2["Fatalities"]



list_countries=['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']



#print(list_countries)

temp3 = temp3[temp3['Country'].isin(list_countries)]

#print(temp3)



fig = px.line(temp3, 

             x='Date', y='Values', color="Country", hover_data=['Values'], 

                  width=700, height=400)

fig.update_layout(title_text='Top 10 countries and India with Fatalities due to COVID 19')

fig.show()
pop=pd.DataFrame()

pop_den = pd.DataFrame()

final = pd.DataFrame()



year = ['2017']

country = ['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']

country1 = ['United States', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran, Islamic Rep.', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']



pop['Year'] = population["Year"]

pop['Country_Region']=population["Country Name"]

pop['Population']=population["Count"]



pop.sort_values('Country_Region', inplace = True)



pop_den['Year'] = population_density["Year"]

pop_den['Country_Region']=population_density["Country Name"]

pop_den['Population_Density']=population_density["Count"]





pop = pop[pop['Year'].isin(year)]

pop.drop('Year',axis = 1, inplace = True)

pop = pop[pop['Country_Region'].isin(country1)]

pop.replace('Iran, Islamic Rep.', 'Iran', inplace = True)



pop_den = pop_den[pop_den['Year'].isin(year)]

pop_den.drop('Year',axis = 1, inplace = True)

pop_den = pop_den[pop_den['Country_Region'].isin(country1)]

pop_den.replace('Iran, Islamic Rep.', 'Iran', inplace = True)



#print(pop)

#print(pop_den)

temp = train_df[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country_Region'])["Fatalities"].sum().reset_index()

temp.sort_values('Fatalities',inplace = True, ascending = False)

temp2 = temp[temp['Country_Region'].isin(country)]

temp2.replace('US', 'United States', inplace = True)

temp2.sort_values('Country_Region', inplace = True)



temp3 = train_df[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country_Region'])["ConfirmedCases"].sum().reset_index()

temp3.sort_values('ConfirmedCases',inplace = True, ascending = False)

temp4 = temp3[temp3['Country_Region'].isin(country)]

temp4.replace('US', 'United States', inplace = True)

temp4.sort_values('Country_Region', inplace = True)



#print(temp4)



final = pd.merge(temp2, temp4, how='outer', left_on = 'Country_Region', right_on = 'Country_Region')

final = pd.merge(final, pop, how='outer', left_on = 'Country_Region', right_on = 'Country_Region')

final = pd.merge(final, pop_den, how='outer', left_on = 'Country_Region', right_on = 'Country_Region')

#final = pd.merge(final, temp4, how='outer', left_on = 'Country_Region', right_on = 'Country_Region')



final['Fatalities/Confirmed_Cases'] = (final.Fatalities/final.ConfirmedCases)

final['Fatal/Confirm_population_density_wise'] = (final.Fatalities/final.ConfirmedCases)*final.Population_Density





#print(final)



fig = px.bar(final, 

             x='Country_Region', y='Fatalities/Confirmed_Cases', color="Fatalities/Confirmed_Cases", hover_data=['Fatalities/Confirmed_Cases'], 

                 color_continuous_scale=px.colors.sequential.YlGnBu, width=700, height=400)

fig.update_layout(title_text='COVID-19 Fatalities/Confirmed Cases')

fig.show()
NewCases=pd.DataFrame()

NewFatalities = pd.DataFrame()

Final = pd.DataFrame()

temp = train_df.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp1 = train_df.groupby(['Date', 'Country_Region'])['Fatalities'].sum().reset_index()

new = pd.concat([temp,temp1.Fatalities],axis=1)

new.sort_values(['Country_Region', 'Date'], inplace = True)

#print(new)

NewCases['Country_Region'] = new["Country_Region"]

NewCases['NewCases'] = new['ConfirmedCases'].diff().fillna(0)

NewFatalities['Country_Region'] = new["Country_Region"]

NewFatalities['NewFatalities'] = new['Fatalities'].diff().fillna(0)



Final = pd.concat([new,NewCases.NewCases],axis=1)

Final = pd.concat([Final, NewFatalities.NewFatalities], axis = 1)





country_list = ['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']

fig = make_subplots(

    rows=4, cols=3, subplot_titles=("US", "Spain", "Italy", "France", "Germany", "China", "Iran", "United Kingdom", "Turkey", "Belgium", "Netherland", "India"))



temp1 = Final[Final.Country_Region == 'US']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy1 = pd.DataFrame()

Dummy1['Date']=temp2["Date"]

Dummy1['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy1.Date, y=Dummy1.NewCases),

              row=1, col=1)



temp1 = Final[Final.Country_Region == 'Spain']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy2 = pd.DataFrame()

Dummy2['Date']=temp2["Date"]

Dummy2['NewCases'] = temp2["NewCases"]

#print(Dummy2)

fig.add_trace(go.Scatter(x=Dummy2.Date, y=Dummy2.NewCases),

              row=1, col=2)



temp1 = Final[Final.Country_Region == 'Italy']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy3 = pd.DataFrame()

Dummy3['Date']=temp2["Date"]

Dummy3['NewCases'] = temp2["NewCases"]

#print(Dummy3)

fig.add_trace(go.Scatter(x=Dummy3.Date, y=Dummy3.NewCases),

              row=1, col=3)





temp1 = Final[Final.Country_Region == 'France']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy4 = pd.DataFrame()

Dummy4['Date']=temp2["Date"]

Dummy4['NewCases'] = temp2["NewCases"]

#print(Dummy4)

fig.add_trace(go.Scatter(x=Dummy4.Date, y=Dummy4.NewCases),

              row=2, col=1)





temp1 = Final[Final.Country_Region == 'Germany']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy5 = pd.DataFrame()

Dummy5['Date']=temp2["Date"]

Dummy5['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy5.Date, y=Dummy5.NewCases),

              row=2, col=2)



temp1 = Final[Final.Country_Region == 'China']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy6 = pd.DataFrame()

Dummy6['Date']=temp2["Date"]

Dummy6['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy6.Date, y=Dummy6.NewCases),

              row=2, col=3)





temp1 = Final[Final.Country_Region == 'Iran']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy7 = pd.DataFrame()

Dummy7['Date']=temp2["Date"]

Dummy7['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy7.Date, y=Dummy7.NewCases),

              row=3, col=1)





temp1 = Final[Final.Country_Region == 'United Kingdom']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy8 = pd.DataFrame()

Dummy8['Date']=temp2["Date"]

Dummy8['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy8.Date, y=Dummy8.NewCases),

              row=3, col=2)





temp1 = Final[Final.Country_Region == 'Turkey']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy9 = pd.DataFrame()

Dummy9['Date']=temp2["Date"]

Dummy9['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy9.Date, y=Dummy9.NewCases),

              row=3, col=3)





temp1 = Final[Final.Country_Region == 'Belgium']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy10 = pd.DataFrame()

Dummy10['Date']=temp2["Date"]

Dummy10['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy10.Date, y=Dummy10.NewCases),

              row=4, col=1)



temp1 = Final[Final.Country_Region == 'Netherlands']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy11 = pd.DataFrame()

Dummy11['Date']=temp2["Date"]

Dummy11['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy11.Date, y=Dummy11.NewCases),

              row=4, col=2)



temp1 = Final[Final.Country_Region == 'India']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy12 = pd.DataFrame()

Dummy12['Date']=temp2["Date"]

Dummy12['NewCases'] = temp2["NewCases"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy12.Date, y=Dummy12.NewCases),

              row=4, col=3)



fig.update_layout(height=700, width=950,

                  title_text="New Confirmed Cases per day of COVID 19")



fig.show()
NewCases=pd.DataFrame()

NewFatalities = pd.DataFrame()

Final = pd.DataFrame()

temp = train_df.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp1 = train_df.groupby(['Date', 'Country_Region'])['Fatalities'].sum().reset_index()

new = pd.concat([temp,temp1.Fatalities],axis=1)

new.sort_values(['Country_Region', 'Date'], inplace = True)

#print(new)

NewCases['Country_Region'] = new["Country_Region"]

NewCases['NewCases'] = new['ConfirmedCases'].diff().fillna(0)

NewFatalities['Country_Region'] = new["Country_Region"]

NewFatalities['NewFatalities'] = new['Fatalities'].diff().fillna(0)



Final = pd.concat([new,NewCases.NewCases],axis=1)

Final = pd.concat([Final, NewFatalities.NewFatalities], axis = 1)





country_list = ['US', 'Spain', 'Italy','France', 'Germany', 'China', 'Iran', 'United Kingdom', 'Turkey', 'Belgium', 'Netherlands', 'India']

fig = make_subplots(

    rows=4, cols=3, subplot_titles=("US", "Spain", "Italy", "France", "Germany", "China", "Iran", "United Kingdom", "Turkey", "Belgium", "Netherland", "India"))



temp1 = Final[Final.Country_Region == 'US']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy1 = pd.DataFrame()

Dummy1['Date']=temp2["Date"]

Dummy1['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy1.Date, y=Dummy1.NewFatalities),

              row=1, col=1)



temp1 = Final[Final.Country_Region == 'Spain']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy2 = pd.DataFrame()

Dummy2['Date']=temp2["Date"]

Dummy2['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy2)

fig.add_trace(go.Scatter(x=Dummy2.Date, y=Dummy2.NewFatalities),

              row=1, col=2)



temp1 = Final[Final.Country_Region == 'Italy']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy3 = pd.DataFrame()

Dummy3['Date']=temp2["Date"]

Dummy3['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy3)

fig.add_trace(go.Scatter(x=Dummy3.Date, y=Dummy3.NewFatalities),

              row=1, col=3)





temp1 = Final[Final.Country_Region == 'France']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy4 = pd.DataFrame()

Dummy4['Date']=temp2["Date"]

Dummy4['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy4)

fig.add_trace(go.Scatter(x=Dummy4.Date, y=Dummy4.NewFatalities),

              row=2, col=1)





temp1 = Final[Final.Country_Region == 'Germany']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy5 = pd.DataFrame()

Dummy5['Date']=temp2["Date"]

Dummy5['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy5.Date, y=Dummy5.NewFatalities),

              row=2, col=2)



temp1 = Final[Final.Country_Region == 'China']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy6 = pd.DataFrame()

Dummy6['Date']=temp2["Date"]

Dummy6['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy6.Date, y=Dummy6.NewFatalities),

              row=2, col=3)





temp1 = Final[Final.Country_Region == 'Iran']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy7 = pd.DataFrame()

Dummy7['Date']=temp2["Date"]

Dummy7['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy7.Date, y=Dummy7.NewFatalities),

              row=3, col=1)





temp1 = Final[Final.Country_Region == 'United Kingdom']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy8 = pd.DataFrame()

Dummy8['Date']=temp2["Date"]

Dummy8['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy8.Date, y=Dummy8.NewFatalities),

              row=3, col=2)





temp1 = Final[Final.Country_Region == 'Turkey']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy9 = pd.DataFrame()

Dummy9['Date']=temp2["Date"]

Dummy9['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy9.Date, y=Dummy9.NewFatalities),

              row=3, col=3)





temp1 = Final[Final.Country_Region == 'Belgium']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy10 = pd.DataFrame()

Dummy10['Date']=temp2["Date"]

Dummy10['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy10.Date, y=Dummy10.NewFatalities),

              row=4, col=1)



temp1 = Final[Final.Country_Region == 'Netherlands']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy11 = pd.DataFrame()

Dummy11['Date']=temp2["Date"]

Dummy11['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy11.Date, y=Dummy11.NewFatalities),

              row=4, col=2)



temp1 = Final[Final.Country_Region == 'India']

temp2 = temp1[temp1.Date >= '2020-02-01'] 

Dummy12 = pd.DataFrame()

Dummy12['Date']=temp2["Date"]

Dummy12['NewFatalities'] = temp2["NewFatalities"]

#print(Dummy1)

fig.add_trace(go.Scatter(x=Dummy12.Date, y=Dummy12.NewFatalities),

              row=4, col=3)



fig.update_layout(height=700, width=950,

                  title_text="Fatalities per day of COVID 19")



fig.show()