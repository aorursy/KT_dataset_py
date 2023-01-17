# Imports
import pandas as pd
import urllib3
import certifi
from bs4 import BeautifulSoup
from datetime import date, timedelta, datetime
import datetime
import time
import numpy as np
# Make PoolManager instance
http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where())

# Make request 
# Important. Make internet connected on settings menu.
link = 'http://live.sts-timing.pl/mp2018/wyniki.php?search=1&dystans=1&dystans=1&filter%5Bcountry%5D=&filter%5Bcity%5D=&filter%5Bteam%5D=&filter%5Bsex%5D=&filter%5Bcat%5D=&show%5B%5D=1&show%5B%5D=2&show%5B%5D=3&show%5B%5D=4&show%5B%5D=5&show%5B%5D=6&show%5B%5D=7&show%5B%5D=8&show%5B%5D=9&show%5B%5D=10&show%5B%5D=11&show%5B%5D=12&show%5B%5D=13&show%5B%5D=14&show%5B%5D=15&show%5B%5D=16&show%5B%5D=17&show%5B%5D=18&show%5B%5D=19&sort='
r = http.request('GET', link)
# Parse data
soup = BeautifulSoup(r.data, 'html.parser')
table = soup.find('table')
table.prettify()[:750]
# Get head and data
raw_data_head = [[cell_value.text.strip() for cell_value in row.find_all('th')] for row in table.find_all('tr')]
raw_data = [[cell_value.text.strip() for cell_value in row.find_all('td')] for row in table.find_all('tr')]
print(raw_data[1])
# Make df column names
header = [row for row in raw_data_head if len(row) == 20][0]
print(header)
# Make df from raw_data
def make_df():
    cleaned_data = []
    for row in raw_data:
        if len(row) == 20:
            cleaned_data.append(row)
    data_frame = pd.DataFrame(data=cleaned_data, columns=header)
    return data_frame
# Check df
df = make_df()
df[:5]
# Delete useless columns, change columns names and anonymise results
df = df[['Miasto', 'Kraj', 'Płeć', 'Miejsce płeć', 'Kategoria', '5KM', '10KM', '15KM', '20KM', '21.1KM',
         '25KM', '30KM', '35KM', '40KM', 'Czas netto', 'Czas brutto']]
# Change polish names to english
df.rename(columns={'Miasto':'City', 'Kraj':'Country', 'Płeć':'Sex', 'Miejsce płeć':'Place sex', 'Kategoria':'Cat',
                   'Czas netto':'Net time', 'Czas brutto':'Gross time'}, inplace=True)

df[:5]
# Delete microseconds from Gross time column
for i in range(0, len(df)):
    df['Gross time'][i] = (df['Gross time'][i]).split('.')[0]
    
print(df['Gross time'][:5])
# Add Finish column and reorder
df['Finish'] = df['Gross time']
df = df[['City', 'Country', 'Sex', 'Place sex', 'Cat', '5KM', '10KM', '15KM', '20KM', '21.1KM',
         '25KM', '30KM', '35KM', '40KM', 'Finish', 'Net time', 'Gross time']]
df.info()
# Change type of time values
date_time_str = '2018-10-14 9:00:00' # start time
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

def times():
    for column in df.columns[5:15]:
        for i in range (0, len(df)):
            time_s = str(df[column][i])
            if time_s == '':
                df[column][i] = np.nan
            else:  
                t = time.strptime(time_s, '%H:%M:%S')
                delta = timedelta(hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec)
                new_date = date_time_obj + delta
                df[column][i] = new_date
    return df

df = times()
print(df[:10])
# Fill nans
df.fillna(method='pad', inplace=True) # There are not so many values, so basic mathod should fits ok.
# Check all column types
df.info()
# Save df to file.
pd.DataFrame.to_csv(df, 'marathon_results.csv')
