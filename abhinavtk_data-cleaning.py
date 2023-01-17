import numpy as np
import pandas as pd
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN',
'londON_StockhOlm',
'Budapest_PaRis', 'Brussels_londOn'],
'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
'12. Air France', '"Swiss Air"']})
df
#Adding missing FlightNumber
df.FlightNumber[1] = 10055.0
df.FlightNumber[3] = 10075.0
#Typecasting to int
df.FlightNumber =df.FlightNumber.astype(int)
#Creating temporary DataFrame
d=[]
j=0
for i in df['From_To']:
    d.append({'From': i[0:i.index('_')], 'To': i[i.index('_')+1:]})
    j+=1
temp = pd.DataFrame(d)
temp
#Standardising the strings so that only the first letter is uppercase
temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()
temp
#Removing From_To column
df.drop('From_To', axis=1, inplace = True )
#Adding temp DataFrame to df
df = pd.concat([temp,df], axis = 1 )
df
#Creating delays dataframe object with separate columns for each delay in the list
p = []
for i in df['RecentDelays']:
    dict1 = {}
    for j in range(len(i)):
        dict1.update({f"delay_{j+1}":i[j]})
    p.append(dict1)
delays = pd.DataFrame(p)
delays
#Removing RecentDelays column
df = df.drop('RecentDelays',axis=1)
#Adding delays to df and making them integer type
df = pd.concat([df,delays],axis=1)
df['delay_1'] = df.delay_1.astype('Int64')
df['delay_2'] = df.delay_2.astype('Int64')
df['delay_3'] = df.delay_3.astype('Int64')
#Standardising the Airline column
df['Airline'] = df.Airline.str.extract('([A-Za-z ]+)', expand=True)
#Output Dataset
df
