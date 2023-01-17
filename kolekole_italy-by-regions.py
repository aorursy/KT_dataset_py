# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Read and clean data

df = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv', header=0, names=['SNo',

                                                                                        'Date',

                                                                                        'Country',

                                                                                        'Region code',

                                                                                        'Region',

                                                                                        'Lat',

                                                                                        'Long',

                                                                                        'Hospitalized',

                                                                                        'ICU',

                                                                                        'Total hospitalized',

                                                                                        'Home confinement',

                                                                                        'Current cases',

                                                                                        'New cases',

                                                                                        'Recovered',

                                                                                        'Deaths',

                                                                                        'Total cases',

                                                                                        'Tests'])

df['Date'] = df['Date'].apply(pd.Timestamp)

df['Date'] = df['Date'].apply(lambda x: x.date())

df['Date'] = pd.to_datetime(df['Date'])



del df['SNo']

del df['Country']

del df['Region code']

del df['Lat']

del df['Long']

df = df.groupby(['Region', 'Date']).sum()

df

# df.loc[0, 'Date']
d = [

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "04/03/2020",

       "Occupancy rate": 67,

       "Physician cases": 66,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 7,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 13,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 22,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 38,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Abruzzo",

       "Population": 1311580,

       "ICU Beds": 100,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 48,

       "Total physicians": 1105

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "04/03/2020",

       "Occupancy rate": 28.125,

       "Physician cases": 2,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 0,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Basilicata",

       "Population": 562869,

       "ICU Beds": 64,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 486

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "04/03/2020",

       "Occupancy rate": 9.027777778,

       "Physician cases": 51,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 9,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 14,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 23,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Calabria",

       "Population": 1947131,

       "ICU Beds": 144,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 41,

       "Total physicians": 1604

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "04/03/2020",

       "Occupancy rate": 21.34387352,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Campania",

       "Population": 5801692,

       "ICU Beds": 506,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 4297

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "04/03/2020",

       "Occupancy rate": 69.70260223,

       "Physician cases": 1148,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 178,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 296,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 472,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 673,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Emilia-Romagna",

       "Population": 4459477,

       "ICU Beds": 538,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 944,

       "Total physicians": 2995

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "04/03/2020",

       "Occupancy rate": 50,

       "Physician cases": 257,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 15,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 24,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 82,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 127,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Friuli Venezia Giulia",

       "Population": 1215220,

       "ICU Beds": 100,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 190,

       "Total physicians": 873

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "04/03/2020",

       "Occupancy rate": 32.50825083,

       "Physician cases": 31,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 17,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 34,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 31,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 30,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Lazio",

       "Population": 5879082,

       "ICU Beds": 606,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 31,

       "Total physicians": 4600

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "04/03/2020",

       "Occupancy rate": 90.16393443,

       "Physician cases": 153,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 15,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 42,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 53,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 92,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Liguria ",

       "Population": 1550640,

       "ICU Beds": 183,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 134,

       "Total physicians": 1151

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "04/03/2020",

       "Occupancy rate": 82.3125,

       "Physician cases": 6561,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1882,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2808,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 3957,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 4585,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Lombardia",

       "Population": 10060574,

       "ICU Beds": 1600,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 6040,

       "Total physicians": 6245

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "04/03/2020",

       "Occupancy rate": 89.34911243,

       "Physician cases": 80,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 18,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 27,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 42,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 51,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Marche",

       "Population": 1525271,

       "ICU Beds": 169,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 72,

       "Total physicians": 1125

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "04/03/2020",

       "Occupancy rate": 13.33333333,

       "Physician cases": 27,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 4,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 9,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 17,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 21,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "Molise",

       "Population": 305617,

       "ICU Beds": 45,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 24,

       "Total physicians": 264

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "04/03/2020",

       "Occupancy rate": 66.25,

       "Physician cases": 197,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 19,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 34,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 84,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 110,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Bolzano",

       "Population": 106951,

       "ICU Beds": 80,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 145,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "04/03/2020",

       "Occupancy rate": 81.63265306,

       "Physician cases": 165,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 30,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 88,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "P.A. Trento",

       "Population": 1072276,

       "ICU Beds": 98,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 110,

       "Total physicians": 640

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "04/03/2020",

       "Occupancy rate": 80.14440433,

       "Physician cases": 41,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 17,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 20,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 20,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 21,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Piemonte",

       "Population": 4356406,

       "ICU Beds": 554,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 37,

       "Total physicians": 3038

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "04/03/2020",

       "Occupancy rate": 51.96078431,

       "Physician cases": 226,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 22,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 32,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 64,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 79,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Puglia",

       "Population": 4029053,

       "ICU Beds": 306,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 134,

       "Total physicians": 3286

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "04/03/2020",

       "Occupancy rate": 20.32520325,

       "Physician cases": 216,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 17,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 44,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 69,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 122,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sardegna",

       "Population": 1639591,

       "ICU Beds": 123,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 200,

       "Total physicians": 1212

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "04/03/2020",

       "Occupancy rate": 32.34042553,

       "Physician cases": 31,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 5,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 6,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Sicilia",

       "Population": 4999891,

       "ICU Beds": 235,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 19,

       "Total physicians": 4089

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "04/03/2020",

       "Occupancy rate": 61.74496644,

       "Physician cases": 415,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 45,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 85,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 163,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 213,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Toscana",

       "Population": 3729641,

       "ICU Beds": 447,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 322,

       "Total physicians": 2718

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "04/03/2020",

       "Occupancy rate": 42.45283019,

       "Physician cases": 87,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 8,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 18,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Umbria",

       "Population": 882015,

       "ICU Beds": 106,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 63,

       "Total physicians": 719

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "04/03/2020",

       "Occupancy rate": 76.66666667,

       "Physician cases": 2,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 1,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Valle d'Aosta",

       "Population": 125666,

       "ICU Beds": 30,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 2,

       "Total physicians": 86

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "04/03/2020",

       "Occupancy rate": 54.83333333,

       "Physician cases": 900,

       "Total physicians": 3198

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "03/16/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 74,

       "Total physicians": 3198

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "03/19/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 74,

       "Total physicians": 3198

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "03/23/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 74,

       "Total physicians": 3198

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "03/26/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 113,

       "Total physicians": 3198

     },

     {

       "Country": "Italy",

       "Region": "Veneto",

       "Population": 4905854,

       "ICU Beds": 600,

       "Date": "03/30/2020",

       "Occupancy rate": np.nan,

       "Physician cases": 388,

       "Total physicians": 3198

     }

    ]

it = pd.DataFrame(data=d)

it['Date'] = it['Date'].apply(pd.Timestamp)

it['Date'] = it['Date'].apply(lambda x: x.date())

it = it.groupby(['Region', 'Date']).last()

it

#Interpolated version jdf

dfi = it.reset_index().copy(deep=True)

del dfi['Occupancy rate']

del dfi['Country']

regions = dfi['Region'].unique()

dfi['Date'] = pd.to_datetime(dfi['Date'])

dates = dfi['Date'].unique()



df_all = pd.DataFrame()

idx = pd.date_range('2020-03-16', '2020-04-03')

for region in regions:

    df_a = dfi.loc[dfi['Region'] == region]

    df_a = df_a.groupby('Date').last()

    df_a.index = pd.DatetimeIndex(df_a.index)

    df_a = df_a.reindex(idx, fill_value=np.nan)

    df_a = df_a.interpolate(method='polynomial', order=1)

    df_a['Region'] = region

    df_all = pd.concat([df_all, df_a])

    

df_all = df_all.reset_index().rename(columns={'index' : 'Date'}).groupby(['Region', 'Date']).last()



jdf = pd.concat([df, df_all], axis=1, join='inner').reset_index()

# discrete version hdf

df_all = pd.DataFrame()

idx = pd.date_range('2020-03-16', '2020-04-03')

for region in regions:

    df_a = dfi.loc[dfi['Region'] == region]

    df_a = df_a.groupby('Date').last()

    df_a.index = pd.DatetimeIndex(df_a.index)

    df_a = df_a.reindex(idx, fill_value=np.nan)

    df_a['Region'] = region

    df_a['Population'] = df_a['Population'].interpolate(method='polynomial', order=1)

    df_a['ICU Beds'] = df_a['ICU Beds'].interpolate(method='polynomial', order=1)

    df_a['Total physicians'] = df_a['Total physicians'].interpolate(method='polynomial', order=1)

    df_all = pd.concat([df_all, df_a])

    

df_all = df_all.reset_index().rename(columns={'index' : 'Date'}).groupby(['Region', 'Date']).last()

hdf = pd.concat([df, df_all], axis=1, join='inner').reset_index()

regions = list(hdf['Region'].unique())
hdf['CFR'] = 100 * hdf['Deaths'] / hdf['Total cases']

hdf['Home confinement ratio'] = 100 * hdf['Home confinement'] / hdf['Total cases']

hdf['Physician cases of total cases'] = 100 * hdf['Physician cases'] / hdf['Total cases']

hdf['Physician cases of total physicians'] = 100 * hdf['Physician cases'] / hdf['Total physicians']

hdf['Total cases from the general population'] = 100 * hdf['Total cases'] / hdf['Population']

hdf['ICU occupancy'] = 100 * hdf['ICU'] / hdf['ICU Beds']



hdf = hdf.sort_values(['Region', 'Date'], ascending=(False, True))

hdf.to_csv(r'italy_by_region_uninterpolated.csv')

# pd.set_option('display.max_rows', df.shape[0]+1)

hdf['Date'] = hdf['Date'].apply(str)

hdf
# Uninterpolated

x = 'CFR'

y = 'Region'

title = 'CFR (mean) by region in Italy'

fig = px.bar(hdf.groupby('Region').mean().reset_index().sort_values(x, ascending=False), x=x, y=y, color='Region', title=title, orientation='h')

fig.update_layout(

    autosize=False,

    width=2150,

    height=1100)

fig.show()







jdf['CFR'] = 100 * jdf['Deaths'] / jdf['Total cases']

jdf['Home confinement ratio'] = 100 * jdf['Home confinement'] / jdf['Total cases']

jdf['Physician cases of total cases'] = 100 * jdf['Physician cases'] / jdf['Total cases']

jdf['Physician cases of total physicians'] = 100 * jdf['Physician cases'] / jdf['Total physicians']

jdf['Total cases from the general population'] = 100 * jdf['Total cases'] / jdf['Population']

jdf['ICU occupancy'] = 100 * jdf['ICU'] / jdf['ICU Beds']



jdf = jdf.sort_values(['Region', 'Date'], ascending=(False, True))

jdf.to_csv(r'italy_by_region_interpolated.csv')

# pd.set_option('display.max_rows', df.shape[0]+1)

jdf['Date'] = jdf['Date'].apply(str)

jdf
c = hdf.groupby('Region').mean().reset_index()



fig = px.scatter(c.loc[c['Physician cases'] > 0], x='Physician cases of total physicians', y='CFR',trendline='ols', text='Region', log_x=True, log_y=True, title="CFR vs. Physician cases of total physicians (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
c = hdf.groupby('Region').mean().reset_index()



fig = px.scatter(c.loc[c['Physician cases'] > 0], x='Physician cases of total physicians', y='Total cases from the general population',trendline='ols', text='Region', log_x=True, log_y=True, title="Total cases from the general population vs. Physician cases of total physicians (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
c = hdf.groupby('Region').mean().reset_index()



fig = px.scatter(c.loc[c['Physician cases'] > 0], x='Physician cases of total cases', y='CFR',trendline='ols', text='Region', title="CFR vs. Physician cases of total cases (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
# Fig 2: Total cases to population in all regions

fig = make_subplots(rows=10, cols=2, subplot_titles=regions)

j = 1

i = 1

for region in regions:

    if(i == 11):

        i = 1

        j = 2

    x_arg = 'Total cases from the general population'

    y_arg = 'Physician cases of total physicians'

    x = jdf.loc[jdf['Region'] == region][x_arg]

    y = jdf.loc[jdf['Region'] == region][y_arg]

    fig.add_trace(

        go.Scatter(x=x, y=y), row=i, col=j)

    fig.update_xaxes(title_text=x_arg, type="log", row=i, col=j)

    fig.update_yaxes(title_text=y_arg, type="log", row=i, col=j)



    i += 1

fig.update_layout(showlegend=False, height=3000, width=2200, title_text="{} vs. {}".format(y_arg, x_arg))

fig.show()



# Fig 3: Physician cases to total cases in all regions



fig = make_subplots(rows=10, cols=2, subplot_titles=regions)

j = 1

i = 1

for region in regions:

    if(i == 11):

        i = 1

        j = 2

    x_arg = 'Total cases from the general population'

    y_arg = 'Physician cases of total cases'

    x = jdf.loc[jdf['Region'] == region][x_arg]

    y = jdf.loc[jdf['Region'] == region][y_arg]

    fig.add_trace(

        go.Scatter(x=x, y=y), row=i, col=j)

    fig.update_xaxes(title_text=x_arg, type="log", row=i, col=j)

    fig.update_yaxes(title_text=y_arg, type="log", row=i, col=j)



    i += 1

fig.update_layout(showlegend=False, height=3000, width=2200, title_text="{} vs. {}".format(y_arg, x_arg))

fig.show()