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
covid_data = pd.read_csv("/kaggle/input/covid-apr16/covid_16_Apr.csv") #reading the data and passing it to a variable "covid_data"
type(covid_data) #getting to know the variable type
covid_data.head(6) #getting to know the first six rows of data,"if we don`t pass any number in parenthesis by default it will give first 5 rows" 
covid_data.tail(6) #getting to know the last six rows of data, "if we don`t pass any number in parenthesis by default it will give first 5 rows"
covid_data.info() #getting information about the dataframe
covid_data.describe() #this will give entire stats of the data
covid_data.isnull() #this is an most elobarative output which in majority of the cases not preferred 
covid_data.isnull().values.any() #simple way to check presence of NAN in entire dataset
covid_data['TCC'].isnull().values.any() #trying to check presence of NAN in a particular coulmn
import matplotlib.pyplot as plt

%matplotlib inline

#%matplotlib notebook
covid_data.plot() #plotting the entire dataframe in one go
#covid_data['Death'].plot(kind="bar")

k = covid_data.plot(x="State", y = ['Cured','Death', 'TCC'], figsize=(10,5), kind="bar", title="State wise covid cured and death cases", width=0.3)
c=covid_data.plot.bar(x='State',y='Cured',figsize=(10,5),title='State wise Covid-19 Cured cases in India')



#annotating the value of each bar

for i in c.patches:

    # get_x pulls left or right; get_height pushes up or down

    c.text(i.get_x(), i.get_height(), 

            str(round((i.get_height()), 2)), fontsize=10, color='black',

                rotation=45)
t=covid_data.plot.bar(x='State',y='TCC',figsize=(10,5),title='State wise Covid-19 Total Confiremd cases in India')

#annotating the value of each bar

for i in t.patches:

    # get_x pulls left or right; get_height pushes up or down

    t.text(i.get_x(), i.get_height(), 

            str(round((i.get_height()), 2)), fontsize=10, color='black',

                rotation=45)
d=covid_data.plot.bar(x='State',y='Death',figsize=(10,5),title='State wise Covid-19 Total Death cases in India')

#annotating the value of each bar

for i in d.patches:

    # get_x pulls left or right; get_height pushes up or down

    d.text(i.get_x(), i.get_height(), 

            str(round((i.get_height()), 2)), fontsize=10, color='black',

                rotation=45)
import seaborn

seaborn.pairplot(covid_data, vars=['TCC', 'Cured', 'Death'],kind='reg')
seaborn.lmplot(y='TCC', x='Death',data= covid_data)
import folium
location = covid_data['Lat'].mean(), covid_data['Long'].mean() #here we are getting the coordinates from the CSV file

m=folium.Map(location = location, zoom_start=10)
for i in range(0,len(covid_data)):

    folium.Marker([covid_data.iloc[i]['Lat'], covid_data.iloc[i]['Long']],popup=('State:'+covid_data.iloc[i]['State'] +'<br>'

                                                                                 'Total confirmed cases:'+ str(covid_data.iloc[i]['TCC']) + '<br>'

                                                                                 'Cured cases:'+ str(covid_data.iloc[i]['Cured'])+ '<br>'

                                                                                 'Death:'+ str(covid_data.iloc[i]['Death'])+ '<br>'

                                                                                  'data as on 16 April 2020'),icon=folium.Icon(color="green",icon='info-sign')).add_to(m)

m