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
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openWeather")



import requests, json 

api_key = secret_value_0

base_url = "http://api.openweathermap.org/data/2.5/forecast?"

city_name = 'Seattle'

complete_url = base_url + "q=" + city_name + "&appid=" + api_key

response = requests.get(complete_url) 

results = response.json()
#results
from matplotlib.pyplot import plot

import pytemperature



feels_like_temp = []

for i in range(len(results['list'])):

    celsius = pytemperature.k2c(results['list'][i]['main']['feels_like']) #convert temperature in Kelvin to Celsius

    feels_like_temp.append(celsius)

    

temp = []

for i in range(len(results['list'])):

    celsius = pytemperature.k2c(results['list'][i]['main']['temp']) #convert temperature in Kelvin to Celsius

    temp.append(celsius)

    

plot(feels_like_temp, color='b')

plot(temp, color='r')
print("the highest and lowest feels-like temperature: %s, %s" %(int(max(feels_like_temp)), int(min(feels_like_temp))))
count = 0

for i in range(len(results['list'])):

    if 'Rain' in results['list'][i]['weather'][0]['main'] or 'Thunderstorm' in results['list'][i]['weather'][0]['main'] or 'Drizzle' in results['list'][i]['weather'][0]['main']:

        print("%s at %s" %(results['list'][i]['weather'][0]['description'], results['list'][i]['dt_txt']))

        count = count + 1



if count == 0:

    print("incredible! no rain in the next 5 days in Seattle!")

else:

    print("remember to bring an umbrella and wear your rain boots!")
count = 0

for i in range(len(results['list'])):

    if 'Snow' in results['list'][i]['weather'][0]['main']:

        print("%s at %s" %(results['list'][i]['weather'][0]['description'], results['list'][i]['dt_txt']))

        count = count + 1



if count == 0:

    print("snow day is not happening for the next 5 days!")

else:

    print("stay warm and build a snowman!")