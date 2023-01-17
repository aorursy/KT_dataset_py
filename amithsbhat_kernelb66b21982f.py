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
### https://www.kaggle.com/tanuprabhu/population-by-country-2020 ###

population_stats={

 'China': {'density': 153, 'medianAge': 38},

 'India': {'density': 464, 'medianAge': 28},

 'United States': {'density': 36, 'medianAge': 38},

 'Indonesia': {'density': 151, 'medianAge': 30},

 'Pakistan': {'density': 287, 'medianAge': 23},

 'Brazil': {'density': 25, 'medianAge': 33},

 'Nigeria': {'density': 226, 'medianAge': 18},

 'Bangladesh': {'density': 1265, 'medianAge': 28},

 'Russia': {'density': 9, 'medianAge': 40},

 'Mexico': {'density': 66, 'medianAge': 29},

 'Japan': {'density': 347, 'medianAge': 48},

 'Ethiopia': {'density': 115, 'medianAge': 19},

 'Philippines': {'density': 368, 'medianAge': 26},

 'Egypt': {'density': 103, 'medianAge': 25},

 'Vietnam': {'density': 314, 'medianAge': 32},

 'DR Congo': {'density': 40, 'medianAge': 17},

 'Turkey': {'density': 110, 'medianAge': 32},

 'Iran': {'density': 52, 'medianAge': 32},

 'Germany': {'density': 240, 'medianAge': 46},

 'Thailand': {'density': 137, 'medianAge': 40},

 'United Kingdom': {'density': 281, 'medianAge': 40},

 'France': {'density': 119, 'medianAge': 42},

 'Italy': {'density': 206, 'medianAge': 47},

 'Tanzania': {'density': 67, 'medianAge': 18},

 'South Africa': {'density': 49, 'medianAge': 28},

 'Myanmar': {'density': 83, 'medianAge': 29},

 'Kenya': {'density': 94, 'medianAge': 20},

 'South Korea': {'density': 527, 'medianAge': 44},

 'Colombia': {'density': 46, 'medianAge': 31},

 'Spain': {'density': 94, 'medianAge': 45},

 'Uganda': {'density': 229, 'medianAge': 17},

 'Argentina': {'density': 17, 'medianAge': 32},

 'Algeria': {'density': 18, 'medianAge': 29},

 'Sudan': {'density': 25, 'medianAge': 20},

 'Ukraine': {'density': 75, 'medianAge': 41},

 'Iraq': {'density': 93, 'medianAge': 21},

 'Afghanistan': {'density': 60, 'medianAge': 18},

 'Poland': {'density': 124, 'medianAge': 42},

 'Canada': {'density': 4, 'medianAge': 41},

 'Morocco': {'density': 83, 'medianAge': 30},

 'Saudi Arabia': {'density': 16, 'medianAge': 32},

 'Uzbekistan': {'density': 79, 'medianAge': 28},

 'Peru': {'density': 26, 'medianAge': 31},

 'Angola': {'density': 26, 'medianAge': 17},

 'Malaysia': {'density': 99, 'medianAge': 30},

 'Mozambique': {'density': 40, 'medianAge': 18},

 'Ghana': {'density': 137, 'medianAge': 22},

 'Yemen': {'density': 56, 'medianAge': 20},

 'Nepal': {'density': 203, 'medianAge': 25},

 'Venezuela': {'density': 32, 'medianAge': 30},

 'Madagascar': {'density': 48, 'medianAge': 20},

 'Cameroon': {'density': 56, 'medianAge': 19},

 "Côte d'Ivoire": {'density': 83, 'medianAge': 19},

 'North Korea': {'density': 214, 'medianAge': 35},

 'Australia': {'density': 3, 'medianAge': 38},

 'Niger': {'density': 19, 'medianAge': 15},

 'Taiwan': {'density': 673, 'medianAge': 42},

 'Sri Lanka': {'density': 341, 'medianAge': 34},

 'Burkina Faso': {'density': 76, 'medianAge': 18},

 'Mali': {'density': 17, 'medianAge': 16},

 'Romania': {'density': 84, 'medianAge': 43},

 'Malawi': {'density': 203, 'medianAge': 18},

 'Chile': {'density': 26, 'medianAge': 35},

 'Kazakhstan': {'density': 7, 'medianAge': 31},

 'Zambia': {'density': 25, 'medianAge': 18},

 'Guatemala': {'density': 167, 'medianAge': 23},

 'Ecuador': {'density': 71, 'medianAge': 28},

 'Syria': {'density': 95, 'medianAge': 26},

 'Netherlands': {'density': 508, 'medianAge': 43},

 'Senegal': {'density': 87, 'medianAge': 19},

 'Cambodia': {'density': 95, 'medianAge': 26},

 'Chad': {'density': 13, 'medianAge': 17},

 'Somalia': {'density': 25, 'medianAge': 17},

 'Zimbabwe': {'density': 38, 'medianAge': 19},

 'Guinea': {'density': 53, 'medianAge': 18},

 'Rwanda': {'density': 525, 'medianAge': 20},

 'Benin': {'density': 108, 'medianAge': 19},

 'Burundi': {'density': 463, 'medianAge': 17},

 'Tunisia': {'density': 76, 'medianAge': 33},

 'Bolivia': {'density': 11, 'medianAge': 26},

 'Belgium': {'density': 383, 'medianAge': 42},

 'Haiti': {'density': 414, 'medianAge': 24},

 'Cuba': {'density': 106, 'medianAge': 42},

 'South Sudan': {'density': 18, 'medianAge': 19},

 'Dominican Republic': {'density': 225, 'medianAge': 28},

 'Czech Republic (Czechia)': {'density': 139, 'medianAge': 43},

 'Greece': {'density': 81, 'medianAge': 46},

 'Jordan': {'density': 115, 'medianAge': 24},

 'Portugal': {'density': 111, 'medianAge': 46},

 'Azerbaijan': {'density': 123, 'medianAge': 32},

 'Sweden': {'density': 25, 'medianAge': 41},

 'Honduras': {'density': 89, 'medianAge': 24},

 'United Arab Emirates': {'density': 118, 'medianAge': 33},

 'Hungary': {'density': 107, 'medianAge': 43},

 'Tajikistan': {'density': 68, 'medianAge': 22},

 'Belarus': {'density': 47, 'medianAge': 40},

 'Austria': {'density': 109, 'medianAge': 43},

 'Papua New Guinea': {'density': 20, 'medianAge': 22},

 'Serbia': {'density': 100, 'medianAge': 42},

 'Israel': {'density': 400, 'medianAge': 30},

 'Switzerland': {'density': 219, 'medianAge': 43},

 'Togo': {'density': 152, 'medianAge': 19},

 'Sierra Leone': {'density': 111, 'medianAge': 19},

 'Hong Kong': {'density': 7140, 'medianAge': 45},

 'Laos': {'density': 32, 'medianAge': 24},

 'Paraguay': {'density': 18, 'medianAge': 26},

 'Bulgaria': {'density': 64, 'medianAge': 45},

 'Libya': {'density': 4, 'medianAge': 29},

 'Lebanon': {'density': 667, 'medianAge': 30},

 'Nicaragua': {'density': 55, 'medianAge': 26},

 'Kyrgyzstan': {'density': 34, 'medianAge': 26},

 'El Salvador': {'density': 313, 'medianAge': 28},

 'Turkmenistan': {'density': 13, 'medianAge': 27},

 'Singapore': {'density': 8358, 'medianAge': 42},

 'Denmark': {'density': 137, 'medianAge': 42},

 'Finland': {'density': 18, 'medianAge': 43},

 'Congo': {'density': 16, 'medianAge': 19},

 'Slovakia': {'density': 114, 'medianAge': 41},

 'Norway': {'density': 15, 'medianAge': 40},

 'Oman': {'density': 16, 'medianAge': 31},

 'State of Palestine': {'density': 847, 'medianAge': 21},

 'Costa Rica': {'density': 100, 'medianAge': 33},

 'Liberia': {'density': 53, 'medianAge': 19},

 'Ireland': {'density': 72, 'medianAge': 38},

 'Central African Republic': {'density': 8, 'medianAge': 18},

 'New Zealand': {'density': 18, 'medianAge': 38},

 'Mauritania': {'density': 5, 'medianAge': 20},

 'Panama': {'density': 58, 'medianAge': 30},

 'Kuwait': {'density': 240, 'medianAge': 37},

 'Croatia': {'density': 73, 'medianAge': 44},

 'Moldova': {'density': 123, 'medianAge': 38},

 'Georgia': {'density': 57, 'medianAge': 38},

 'Eritrea': {'density': 35, 'medianAge': 19},

 'Uruguay': {'density': 20, 'medianAge': 36},

 'Bosnia and Herzegovina': {'density': 64, 'medianAge': 43},

 'Mongolia': {'density': 2, 'medianAge': 28},

 'Armenia': {'density': 104, 'medianAge': 35},

 'Jamaica': {'density': 273, 'medianAge': 31},

 'Qatar': {'density': 248, 'medianAge': 32},

 'Albania': {'density': 105, 'medianAge': 36},

 'Puerto Rico': {'density': 323, 'medianAge': 44},

 'Lithuania': {'density': 43, 'medianAge': 45},

 'Namibia': {'density': 3, 'medianAge': 22},

 'Gambia': {'density': 239, 'medianAge': 18},

 'Botswana': {'density': 4, 'medianAge': 24},

 'Gabon': {'density': 9, 'medianAge': 23},

 'Lesotho': {'density': 71, 'medianAge': 24},

 'North Macedonia': {'density': 83, 'medianAge': 39},

 'Slovenia': {'density': 103, 'medianAge': 45},

 'Guinea-Bissau': {'density': 70, 'medianAge': 19},

 'Latvia': {'density': 30, 'medianAge': 44},

 'Bahrain': {'density': 2239, 'medianAge': 32},

 'Equatorial Guinea': {'density': 50, 'medianAge': 22},

 'Trinidad and Tobago': {'density': 273, 'medianAge': 36},

 'Estonia': {'density': 31, 'medianAge': 42},

 'Timor-Leste': {'density': 89, 'medianAge': 21},

 'Mauritius': {'density': 626, 'medianAge': 37},

 'Cyprus': {'density': 131, 'medianAge': 37},

 'Eswatini': {'density': 67, 'medianAge': 21},

 'Djibouti': {'density': 43, 'medianAge': 27},

 'Fiji': {'density': 49, 'medianAge': 28},

 'Réunion': {'density': 358, 'medianAge': 36},

 'Comoros': {'density': 467, 'medianAge': 20},

 'Guyana': {'density': 4, 'medianAge': 27},

 'Bhutan': {'density': 20, 'medianAge': 28},

 'Solomon Islands': {'density': 25, 'medianAge': 20},

 'Macao': {'density': 21645, 'medianAge': 39},

 'Montenegro': {'density': 47, 'medianAge': 39},

 'Luxembourg': {'density': 242, 'medianAge': 40},

 'Western Sahara': {'density': 2, 'medianAge': 28},

 'Suriname': {'density': 4, 'medianAge': 29},

 'Cabo Verde': {'density': 138, 'medianAge': 28},

 'Maldives': {'density': 1802, 'medianAge': 30},

 'Malta': {'density': 1380, 'medianAge': 43},

 'Brunei': {'density': 83, 'medianAge': 32},

 'Guadeloupe': {'density': 237, 'medianAge': 44},

 'Belize': {'density': 17, 'medianAge': 25},

 'Bahamas': {'density': 39, 'medianAge': 32},

 'Martinique': {'density': 354, 'medianAge': 47},

 'Iceland': {'density': 3, 'medianAge': 37},

 'Vanuatu': {'density': 25, 'medianAge': 21},

 'French Guiana': {'density': 4, 'medianAge': 25},

 'Barbados': {'density': 668, 'medianAge': 40},

 'New Caledonia': {'density': 16, 'medianAge': 34},

 'French Polynesia': {'density': 77, 'medianAge': 34},

 'Mayotte': {'density': 728, 'medianAge': 20},

 'Sao Tome & Principe': {'density': 228, 'medianAge': 19},

 'Samoa': {'density': 70, 'medianAge': 22},

 'Saint Lucia': {'density': 301, 'medianAge': 34},

 'Channel Islands': {'density': 915, 'medianAge': 43},

 'Guam': {'density': 313, 'medianAge': 31},

 'Curaçao': {'density': 370, 'medianAge': 42},

 'Kiribati': {'density': 147, 'medianAge': 23},

 'Micronesia': {'density': 164, 'medianAge': 24},

 'Grenada': {'density': 331, 'medianAge': 32},

 'St. Vincent & Grenadines': {'density': 284, 'medianAge': 33},

 'Aruba': {'density': 593, 'medianAge': 41},

 'Tonga': {'density': 147, 'medianAge': 22},

 'U.S. Virgin Islands': {'density': 298, 'medianAge': 43},

 'Seychelles': {'density': 214, 'medianAge': 34},

 'Antigua and Barbuda': {'density': 223, 'medianAge': 34},

 'Isle of Man': {'density': 149, 'medianAge': 30},

 'Andorra': {'density': 164, 'medianAge': 30},

 'Dominica': {'density': 96, 'medianAge': 30},

 'Cayman Islands': {'density': 274, 'medianAge': 30},

 'Bermuda': {'density': 1246, 'medianAge': 30},

 'Marshall Islands': {'density': 329, 'medianAge': 30},

 'Northern Mariana Islands': {'density': 125, 'medianAge': 30},

 'Greenland': {'density': 0, 'medianAge': 30},

 'American Samoa': {'density': 276, 'medianAge': 30},

 'Saint Kitts & Nevis': {'density': 205, 'medianAge': 30},

 'Faeroe Islands': {'density': 35, 'medianAge': 30},

 'Sint Maarten': {'density': 1261, 'medianAge': 30},

 'Monaco': {'density': 26337, 'medianAge': 30},

 'Turks and Caicos': {'density': 41, 'medianAge': 30},

 'Saint Martin': {'density': 730, 'medianAge': 30},

 'Liechtenstein': {'density': 238, 'medianAge': 30},

 'San Marino': {'density': 566, 'medianAge': 30},

 'Gibraltar': {'density': 3369, 'medianAge': 30},

 'British Virgin Islands': {'density': 202, 'medianAge': 30},

 'Caribbean Netherlands': {'density': 80, 'medianAge': 30},

 'Palau': {'density': 39, 'medianAge': 30},

 'Cook Islands': {'density': 73, 'medianAge': 30},

 'Anguilla': {'density': 167, 'medianAge': 30},

 'Tuvalu': {'density': 393, 'medianAge': 30},

 'Wallis & Futuna': {'density': 80, 'medianAge': 30},

 'Nauru': {'density': 541, 'medianAge': 30},

 'Saint Barthelemy': {'density': 470, 'medianAge': 30},

 'Saint Helena': {'density': 16, 'medianAge': 30},

 'Saint Pierre & Miquelon': {'density': 25, 'medianAge': 30},

 'Montserrat': {'density': 50, 'medianAge': 30},

 'Falkland Islands': {'density': 0, 'medianAge': 30},

 'Niue': {'density': 6, 'medianAge': 30},

 'Tokelau': {'density': 136, 'medianAge': 30},

 'Holy See': {'density': 2003, 'medianAge': 30}}



pop_den = {k : v.get('density',0) for k,v in population_stats.items()}

pop_med = {k : v.get('medianAge',30) for k,v in population_stats.items()}
import datetime

from dateutil.parser import parse

from dateutil.tz import gettz



def getTS(dt):

    tzinfos = {'UTC' : gettz('Europe/London')}

    date_str = '{0} 00:00:00'.format(dt)

    str_to_dt = parse(date_str + ' UTC', tzinfos=tzinfos)

    return int(str_to_dt.timestamp())



df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv", index_col = "Id" )

df['Date'] = df['Date'].apply(getTS)





df['Density'] = df['Country_Region'].map(pop_den)

df['MedianAge'] = df['Country_Region'].map(pop_med)







countries = list(set(df['Country_Region']))

countries.sort()

countryDict = {each : idx for idx,each in enumerate(countries,1)}

df['Country_Region'] = df['Country_Region'].map(countryDict)

allProvinces = list(set(df['Province_State']))

#allProvinces.sort()

provinceDict = {each : idx for idx,each in enumerate(allProvinces,1)}

df['Province_State'] = df['Province_State'].map(provinceDict)



df.head()

df = df.fillna(df.median())

df['MedianAge'].isnull().values.any()

#df.isnull().values.any()



#95 median of pop density
feature_col_names = ['Country_Region', 'Date','Province_State','Density','MedianAge'] 

predicted_class_names1 = ['ConfirmedCases']

predicted_class_names2 = ['Fatalities']



X = df[feature_col_names].values

Y1 = df[predicted_class_names1].values

Y2 = df[predicted_class_names2].values



print("Data cleanup done...")
import xgboost



regr1 = xgboost.XGBRegressor(max_depth=8,                 

                 learning_rate=0.0001,

                             #n_estimators=2500)

                 n_estimators=25000)





regr1.fit(X, Y1.ravel())







regr2 = xgboost.XGBRegressor(max_depth=8,                 

                 learning_rate=0.0001,

                             #n_estimators=2500)

                 n_estimators=25000)



regr2.fit(X, Y2.ravel())
testpath = '/kaggle/input/covid19-global-forecasting-week-4/test.csv'



dft = pd.read_csv(testpath, index_col = "ForecastId" )

dft['Date'] = dft['Date'].apply(getTS)

dft['Density'] = dft['Country_Region'].map(pop_den)

dft['MedianAge'] = dft['Country_Region'].map(pop_med)





dft['Country_Region'] = dft['Country_Region'].map(countryDict)

dft['Province_State'] = dft['Province_State'].map(provinceDict)

dft = dft.fillna(dft.median())

dft.head()



Xt = dft[feature_col_names].values

predictionsC = regr1.predict(Xt) 

predictionsF = regr2.predict(Xt) 

dft['ConfirmedCases'] = predictionsC

dft['Fatalities'] = predictionsF



allowedCols =['ForecastId','ConfirmedCases','Fatalities']





for col in dft.columns:

    if col not in allowedCols:

        dft = dft.drop([col], axis = 1)

        print("Dropping {0}".format(col))

        

def normalize(val):

    if val > 0:

        return round(val)

    return 0



dft['ConfirmedCases'] = dft['ConfirmedCases'].apply(normalize)

dft['Fatalities'] = dft['Fatalities'].apply(normalize)

dft.to_csv('submission.csv', index = True)



print("Done...")