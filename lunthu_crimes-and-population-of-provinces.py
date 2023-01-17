import pandas as pd

import seaborn as sns

df = pd.read_csv('../input/SouthAfricaCrimeStats_v2.csv')

province_df = pd.DataFrame(df['Province'].value_counts())

province_df.reset_index(level=0, inplace=True)

province_df.columns = ['Provinces', 'Crimes']
#I added to my dataframe information about population of each province from this source, page 2: http://www.statssa.gov.za/publications/P0302/P03022015.pdf

province_df['Population'] = [6916200, 10919100, 6200100, 13200300, 2817900, 5726800, 1185600, 4283900, 3707100]

province_df['Land Area'] = [168966, 94361, 129462, 16548, 129825, 125755, 372889, 76495, 106512]

province_df['Safety'] = province_df['Crimes']/province_df['Population']

#Eastern Cape – 168 966km2 Free State – 129 825km2 Gauteng – 16 548km2 KwaZulu-Natal – 94 361km2 Limpopo – 125 755km2 Mpumalanga – 76 495km2 Northern Cape – 372 889km2 North West – 106 512km2 Western Cape – 129 462km2

#I tried to build crimes rate, based on information about Population Density

province_df['Population Density'] = province_df['Population']/province_df['Land Area'] # people per km2

province_df['Crimes Rate'] = province_df['Crimes']/province_df['Population Density']

# But I'm not sure, that this rate is right



print(province_df)
province_df['Safety'].plot(x = province_df['Provinces'], kind = 'bar')
#Let's check our second rate

province_df['Crimes Rate'].plot(x = province_df['Provinces'], kind = 'bar')