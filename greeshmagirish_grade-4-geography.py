import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')

sns.set_context("paper")
crop_df = pd.read_csv("../input/crop-production-in-india/crop_production.csv")

crop_df.head()
crop_df.info()
fig, ax = plt.subplots(figsize=(25,65), sharex='col')

count = 1



for state in crop_df.State_Name.unique():

    plt.subplot(len(crop_df.State_Name.unique()),1,count)

    sns.lineplot(crop_df[crop_df.State_Name==state]['Crop_Year'],crop_df[crop_df.State_Name==state]['Production'], ci=None)

    plt.subplots_adjust(hspace=2.2)

    plt.title(state)

    count+=1
north_india = ['Jammu and Kashmir', 'Punjab', 'Himachal Pradesh', 'Haryana', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh']

east_india = ['Bihar', 'Odisha', 'Jharkhand', 'West Bengal']

south_india = ['Andhra Pradesh', 'Karnataka', 'Kerala' ,'Tamil Nadu', 'Telangana']

west_india = ['Rajasthan' , 'Gujarat', 'Goa','Maharashtra','Goa']

central_india = ['Madhya Pradesh', 'Chhattisgarh']

north_east_india = ['Assam', 'Sikkim', 'Nagaland', 'Meghalaya', 'Manipur', 'Mizoram', 'Tripura', 'Arunachal Pradesh']

ut_india = ['Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Puducherry']
def get_zonal_names(row):

    if row['State_Name'].strip() in north_india:

        val = 'North Zone'

    elif row['State_Name'].strip()  in south_india:

        val = 'South Zone'

    elif row['State_Name'].strip()  in east_india:

        val = 'East Zone'

    elif row['State_Name'].strip()  in west_india:

        val = 'West Zone'

    elif row['State_Name'].strip()  in central_india:

        val = 'Central Zone'

    elif row['State_Name'].strip()  in north_east_india:

        val = 'NE Zone'

    elif row['State_Name'].strip()  in ut_india:

        val = 'Union Terr'

    else:

        val = 'No Value'

    return val



crop_df['Zones'] = crop_df.apply(get_zonal_names, axis=1)

crop_df['Zones'].unique()
fig, ax = plt.subplots(figsize=(25,30), sharex='col')

count = 1



for zone in crop_df.Zones.unique():

    plt.subplot(len(crop_df.Zones.unique()),1,count)

    sns.lineplot(crop_df[crop_df.Zones==zone]['Crop_Year'],crop_df[crop_df.Zones==zone]['Production'], ci=None)

    plt.subplots_adjust(hspace=0.6)

    plt.title(zone)

    count+=1
zone_df = crop_df.groupby(by='Zones')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)

zone_df.head()
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(zone_df.Zones, zone_df.Production)

plt.yscale('log')

plt.title('Zone-Wise Production: Total')
south_zone =  crop_df[(crop_df["Zones"] == 'South Zone')]

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(south_zone.State_Name, south_zone.Production,errwidth=0)

plt.yscale('log')

plt.title('Southern-Zone Production')



south_zone.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)
df = south_zone.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)



fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.Crop, df.Production,errwidth=0)

plt.yscale('log')

plt.title('South Zone Crops vs Production')
crop = crop_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)

crop 

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(crop.Crop, crop.Production,errwidth=0)

plt.yscale('log')

plt.title('Overall Crops vs Production')
set(crop_df[(crop_df['Season'] == 'Whole Year ')].Crop.unique()) & set(crop_df[(crop_df['Season'] == 'Kharif     ')].Crop.unique()) 
Kharif = ['Bajra','Jowar','Maize','Millet','Rice','Soybean','Fruits','Muskmelon','Sugarcane','Watermelon','Orange','Arhar/Tur,'

'Urad','Cotton(lint)','Cowpea(Lobia)','Moong(Green Gram)','Guar seed','Moth','Tomato','Turmeric', 'Ragi']

Rabi = ['Barley', 'Gram', 'Rapeseed &Mustard', 'Masoor', 'Coriander', 'Sunflower', 'Tobacco', 'Brinjal', 'Cabbage',

       'Onion','Sweet potato','Potato','Peas & beans (Pulses)', 'Oilseeds total', 'other oilseeds', 'Banana', 'Groundnut', 'Niger seed',

       'Sesamum','Safflower', 'Castor seed', 'Linseed', 'Soyabean']



def change_crop_seasons(row):

    if row['Crop'].strip() in Kharif:

        val = 'Kharif'

    elif row['Crop'].strip()  in Rabi:

        val = 'Rabi'

    else:

        val = 'Others'

    return val



crop_df['Updated_Crop_Season'] = crop_df.apply(change_crop_seasons, axis=1)

crop_df['Updated_Crop_Season'].unique()
season = crop_df.groupby(by='Updated_Crop_Season')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)

season

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(season.Updated_Crop_Season, season.Production,errwidth=0)

plt.yscale('log')

plt.title('Seasonal Crops vs Production')
kharif_df = crop_df[(crop_df['Updated_Crop_Season'] == 'Kharif')]

df = kharif_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.Crop, df.Production,errwidth=0)

plt.yscale('log')

plt.xticks(rotation=40)

plt.title('Kharif Crops Production')
sugarcane_df = kharif_df[(kharif_df['Crop'] == 'Sugarcane')]

sugarcane_df.head()



fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(sugarcane_df.Zones, sugarcane_df.Production,errwidth=0)

plt.yscale('log')

plt.xticks(rotation=45)

plt.title('Sugarcane Zone-Wise Production')
df = sugarcane_df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.State_Name.head(4), df.Production.head(4),errwidth=0)

plt.yscale('log')

plt.title('Sugarcane State-Wise Production')
uttarpr_df = sugarcane_df[(sugarcane_df['State_Name'] == 'Uttar Pradesh')]

df = uttarpr_df.groupby(by=['District_Name', 'Crop'])['Area'].sum().reset_index().sort_values(by='Area', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.District_Name.head(5), df.Area.head(5),errwidth=0)

plt.title('Uttar Pradesh - Sugarcane Production')

df.head(5)
rabi_df = crop_df[(crop_df['Updated_Crop_Season'] == 'Rabi')]

df = rabi_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.Crop, df.Production,errwidth=0)

plt.yscale('log')

plt.xticks(rotation=45)

plt.title('Rabi Crops Production')
potato_df = rabi_df[(rabi_df['Crop'] == 'Potato')]

potato_df.head()



fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(potato_df.Zones, potato_df.Production,errwidth=0)

plt.yscale('log')

plt.xticks(rotation=45)

plt.title('Potato Zone-Wise Production')
df = potato_df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.State_Name.head(4), df.Production.head(4),errwidth=0)

plt.yscale('log')

plt.title('Potato State-Wise Production')
uttarpr_df = potato_df[(potato_df['State_Name'] == 'Uttar Pradesh')]

df = uttarpr_df.groupby(by=['District_Name', 'Crop'])['Area'].sum().reset_index().sort_values(by='Area', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.District_Name.head(5), df.Area.head(5), errwidth=0)

plt.title('Uttar Pradesh - Potato Production')

df.head(5)
df = crop_df.groupby(by='State_Name')['Area'].sum().reset_index().sort_values(by='Area', ascending=False)

df.head()



fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(df.State_Name.head(5), df.Area.head(5), errwidth=0)

plt.title('Agricultural Area Distribution - India')

df.head(5)
df = crop_df.groupby(by='State_Name')['Area'].sum().reset_index().sort_values(by='Area', ascending=False)

df = df.head(5)



fig, ax = plt.subplots(figsize=(25,30), sharey='col')

count = 1



for state in df.State_Name.unique():

    plt.subplot(len(df.State_Name.unique()),1,count)

    sns.lineplot(crop_df[crop_df.State_Name==state]['Crop_Year'],crop_df[crop_df.State_Name==state]['Production'], ci=None)

    plt.subplots_adjust(hspace=0.6)

    plt.title(state)

    count+=1