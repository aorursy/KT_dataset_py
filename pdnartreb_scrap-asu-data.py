import pandas as pd

pd.set_option("display.max_columns", 200)

pd.set_option("display.max_rows", 200)

import numpy as np



import requests

import json



%matplotlib inline 

from matplotlib import pyplot as plt



import warnings

warnings.filterwarnings('ignore')



from multiprocessing import Pool



from IPython.display import Image
df_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

df_building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

df_asu = pd.read_csv('/kaggle/input/asu-buildings-energy-consumption/asu_2016-2018.csv')



meters = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
s = requests.Session() 



cookie = {

    "version": 0,

    "name": 'DG5SESSION',

    "value": 'D6E2A3ED9FD5F1622274F02CF72E1FB0', # Paste your session key here

    "domain": 'cm.asu.edu',

    "path": '/',

    "secure": True

}



s = requests.Session()

s.cookies.set(**cookie)
df_buildings_tmp = pd.DataFrame(columns=['bldgno', 'bldgname', 'campus'])



for campus in ['Downtown', 'Polytechnic', 'Tempe', 'West']:

    url = 'https://cm.asu.edu/dgdb?db=VES&query=%5Bcm%5D.%5Bdbo%5D.%5BpCM_Select_Building_List_By_Campus%5D+%40selCampus+%3D+%22' + campus + '%22%2C+%40selOrderBy%3D%22bldgname%22%2C+%40selAscDesc%3D%22ASC%22%3B'

    response = s.get(url, verify=False)

    df_tmp = pd.DataFrame(json.loads(response.content)['rows']).rename(columns={0: 'bldgno', 1:'bldgname'})

    df_tmp['campus'] = campus

    df_buildings_tmp = pd.concat([df_buildings_tmp, df_tmp])
df_asu_buildings = pd.DataFrame(columns=['bldgno', 'bldgname', 'occupancy', 'gsf', 'category', 'buildingdate', 'latitude', 'longitude'])



for bldgno in df_buildings_tmp['bldgno']:

    url = 'https://cm.asu.edu/dgdb?db=VES&query=%5Bcm%5D.%5Bdbo%5D.%5BpCM_Retrieve_Building_Data%5D+%40selBldgno+%3D+%22' + bldgno + '%22'

    response = s.get(url, verify=False)

    df_tmp = pd.DataFrame(json.loads(response.content)['rows']).rename(columns={0: 'bldgno', 1: 'bldgname', 2: 'occupancy', 3: 'gsf', 4: 'category', 5: 'buildingdate', 6: 'latitude', 7: 'longitude'})

    df_asu_buildings = pd.concat([df_asu_buildings, df_tmp]).reset_index(drop=True)



df_asu_buildings.head()
df_buildings = df_building_metadata[df_building_metadata['site_id'] == 2]

df_asu_buildings['gsf'] = df_asu_buildings['gsf'].str.strip()

df_asu_buildings['gsf'] = np.where(df_asu_buildings['gsf'] == '', -1, df_asu_buildings['gsf'])

df_asu_buildings['gsf'] = df_asu_buildings['gsf'].astype(int)

df_buildings = df_buildings.merge(df_asu_buildings.rename(columns={'gsf': 'square_feet'}), on='square_feet', how='left')

df_buildings.head()
df_buildings = df_buildings.drop([68, 69]).reset_index(drop=True)
df_buildings[df_buildings['bldgno'].isnull()]
df_mapping = df_buildings[['building_id', 'bldgno']].set_index('building_id')

df_mapping.loc[176] = '88'

df_mapping.loc[204] = '14B'

df_mapping.loc[222] = '27'

df_mapping.loc[248] = '57E'

df_mapping.loc[290] = '7'

df_mapping.loc[244] = '6B'

df_mapping.loc[283] = '174'

df_mapping = df_mapping[df_mapping['bldgno'].notnull()]

df_mapping['bldgno'] = df_mapping['bldgno'].astype(str)

df_mapping['bldgno'] = df_mapping['bldgno'].str.strip()

df_mapping = df_mapping.reset_index()

df_mapping.head()
# def scrap_all_buildings(b):

#     print(df_buildings.iloc[b]['bldgno'] + ' - ' + df_buildings.iloc[b]['campus'])

#     url = 'https://cm.asu.edu/dgdb?db=VES&query=[cm].[dbo].[pCM_Retrieve_Utility_Data_By_Campus_Building]@selCampus=%22' + df_buildings.iloc[b]['campus'] + '%22,@selBldg=%22' + str(df_buildings.iloc[b]['bldgno']) + '%22,@selPeriod=%22Custom+Dates%22,@selInterval=%22Hourly%22,@selBeginDate=%222016-01-01%22,@selEndDate=%222019-01-01%22;'

#     response = s.get(url, verify=False)

#     with open('../data/scraped/2-asu/data/building-' + str(df_buildings.iloc[b]['bldgno']) + '.pkl', 'wb') as f:

#         pickle.dump(response.content, f)
# t_start = time.time()



# pool = Pool(8)

# pool.imap(scrap_all_buildings, df_buildings.index)

# pool.close()

# pool.join()



# print('Execution time: ' + str(round(time.time() - t_start)) + ' s')
columns = [

    "campus",

    "bldgno",

    "bldgname",

    "tstamp",

    "Year",

    "Month",

    "Day",

    "Hour",

    "KW",

    "KWS",

    "CHWTON",

    "HTmmBTU",

    "Combined mmBTU",

    "Combined Tons Carbon",

    "KW#Houses",

    "KWlightbulbs",

    "KWgalsgas",

    "CHWTON#Houses",

    "CHWTONlightbulbs",

    "CHWTONgalsgas",

    "HTmmBTU#Houses",

    "HTmmBTUlightbulbs",

    "HTmmBTUgalsgas",

    "Total#Houses",

    "Totallightbulbs",

    "Totalgalsgas",

    "GHG",

    "DOW"

]
campus = 'Tempe'

bldgno = '63'



url = 'https://cm.asu.edu/dgdb?db=VES&query=[cm].[dbo].[pCM_Retrieve_Utility_Data_By_Campus_Building]@selCampus=%22' + campus + '%22,@selBldg=%22' + str(bldgno) + '%22,@selPeriod=%22Custom+Dates%22,@selInterval=%22Hourly%22,@selBeginDate=%222016-01-01%22,@selEndDate=%222016-01-02%22;'

response = s.get(url, verify=False)

df_building = pd.DataFrame(json.loads(response.content)['rows'], columns=columns)
display(df_building.head())
display(df_train[(df_train['building_id'] == 192) & (df_train['meter'] == 0)].head())
df_train[(df_train['building_id'] == 192) & (df_train['meter'] == 1)].head()
print(251.701 / 71.57)

print(243.683 / 69.29)

print(258.242 / 73.43)

print(235.453 / 66.95)
df_train[(df_train['building_id'] == 192) & (df_train['meter'] == 3)].head()
df_tmp = df_train[(df_train['building_id'] == 192) & (df_train['meter'] == 3)]

df_tmp['meter_reading'] /= 3.51685

display(df_tmp.head())
print(19.166669 / 0.23)

print(20 / 0.24)

print(20.833359 / 0.25)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])

df_asu['timestamp'] = pd.to_datetime(df_asu['timestamp'])
def plot_meters(df1, df2, building_id):

    plt.figure(figsize=(15, 2))

    for i, meter in enumerate([0, 1, 3]):

        df1_tmp = df1[(df1['building_id'] == building_id) & (df1['meter'] == meter)]

        if len(df1_tmp) > 0:

            df2_tmp = df2[(df2['building_id'] == building_id) & (df2['meter'] == meter)]

            plt.subplot(1, 3, i + 1)

            plt.title(meters[meter] + ' for building_id ' + str(building_id))

            plt.plot(df2_tmp["timestamp"], df2_tmp['meter_reading'])

            plt.plot(df1_tmp["timestamp"], df1_tmp['meter_reading'], alpha=0.25)

            plt.xticks(rotation='25')

    plt.show()
for building_id in sorted(df_asu['building_id'].drop_duplicates()):

    plot_meters(df_train, df_asu, building_id)