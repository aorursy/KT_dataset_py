import numpy as np # linear algebra

import pandas as pd # data processing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df.set_index('ID', inplace=True)
df.rename(columns={'Distance(mi)': 'Distance', 

                   'Temperature(F)': 'Temperature', 

                   'Wind_Chill(F)': 'Wind_Chill', 

                   'Humidity(%)': 'Humidity', 

                   'Pressure(in)': 'Pressure', 

                   'Visibility(mi)': 'Visibility', 

                   'Wind_Speed(mph)': 'Wind_Speed', 

                   'Precipitation(in)': 'Precipitation'}, inplace=True)
df.Country.unique()
df.Turning_Loop.unique()
df.drop(['Source', 'TMC', 'Description', 'Country', 'Airport_Code', 'Weather_Timestamp', 'Amenity', 'Turning_Loop', 'Wind_Chill', 'Timezone', 'Number', 'End_Lat', 'End_Lng', 'Wind_Direction', 'Wind_Speed', 'Precipitation'], axis=1, inplace=True)
features = pd.DataFrame(df.columns, columns=['Feature'])

features
unique_val = pd.DataFrame(df.nunique(), columns=['Count of unique values'])

unique_val
cnt = df.isna().sum()

missing_val = pd.DataFrame(cnt[cnt > 0] , columns=['Count of missing values'])

missing_val



del cnt, missing_val
df.describe().T
weather = df[['Temperature', 'Humidity', 'Pressure', 'Visibility', 'Weather_Condition']].copy()

weather.dropna(how='all', inplace=True)