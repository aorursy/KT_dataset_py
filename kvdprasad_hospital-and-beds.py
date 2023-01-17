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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



path = "/kaggle/input/hospitals-and-beds-in-india"
statewise_hospital_beds = pd.read_csv(path +'/Hospitals_and_Beds_statewise.csv')

# change column names

statewise_hospital_beds.rename (columns = {'Unnamed: 0': 'Name of State'}

                      , inplace = True)

statewise_hospital_beds.head(),



statewise_hospital_beds = statewise_hospital_beds.drop(labels='Unnamed: 6', axis=1)

statewise_hospital_beds = statewise_hospital_beds.fillna(0)



statewise_hospital_beds = statewise_hospital_beds[:-1]

statewise_hospital_beds.head(36)



ministry_defence_hospital_beds =pd.read_csv(path +'/Hospitals and Beds maintained by Ministry of Defence.csv')

ministry_defence_hospital_beds = ministry_defence_hospital_beds[['Name of State','No. of Hospitals','No. of beds']]

ministry_defence_hospital_beds = ministry_defence_hospital_beds[:-2]



ministry_defence_hospital_beds


ministry_defence_hospital_beds['Name of State'].mask(ministry_defence_hospital_beds['Name of State'] == 'Maharastra', 'Maharashtra', inplace=True)

ministry_defence_hospital_beds



country_wise_info = pd.merge(statewise_hospital_beds,

                 ministry_defence_hospital_beds[['Name of State','No. of Hospitals','No. of beds']],

                 on='Name of State', how='outer')



country_wise_info.rename (columns = {'Total': 'state beds',

                                    'No. of beds': 'defence beds'}

                      , inplace = True)

drop_labes = {'PHC','CHC','SDH','DH','No. of Hospitals'}

country_wise_info = country_wise_info.drop(labels=drop_labes, axis=1)

country_wise_info = country_wise_info.fillna(0)

country_wise_info
country_wise_info = pd.merge(statewise_hospital_beds,

                 ministry_defence_hospital_beds[['Name of State','No. of Hospitals','No. of beds']],

                 on='Name of State', how='outer')



country_wise_info.rename (columns = {'Total': 'state beds',

                                    'No. of beds': 'defence beds'}

                      , inplace = True)

drop_labes = {'PHC','CHC','SDH','DH','No. of Hospitals'}

country_wise_info = country_wise_info.drop(labels=drop_labes, axis=1)

country_wise_info = country_wise_info.fillna(0)

country_wise_info


gov_hospotal_beds = pd.read_csv(path +'/Number of Government Hospitals and Beds in Rural and Urban Areas .csv')

gov_hospotal_beds = gov_hospotal_beds[:-1]

gov_hospotal_beds['States/UTs'] = gov_hospotal_beds['States/UTs'].str.replace('\W', '')

gov_hospotal_beds = gov_hospotal_beds.iloc[1:]

gov_hospotal_beds.rename (columns = {'States/UTs':'Name of State',

                                     'Unnamed: 2': 'gov_rural_beds',

                                    'Unnamed: 4': 'gov_urban_beds'}

                      , inplace = True)



drop_labes = {'Rural hospitals','Urban hospitals','As on'}

gov_hospotal_beds = gov_hospotal_beds.drop(labels=drop_labes, axis=1)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'AndhraPradesh', 'Andhra Pradesh', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'ArunachalPradesh', 'Arunachal Pradesh', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'HimachalPradesh', 'Himachal Pradesh', inplace=True)



gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'JammuKashmir', 'Jammu & Kashmir', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'MadhyaPradesh', 'Madhya Pradesh', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'UttarPradesh', 'Uttar Pradesh', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'WestBengal', 'West Bengal', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'AndamanNicobarIslands', 'Andaman & Nicobar Islands', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'DamanDiu', 'Daman & Diu', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'TamilNadu', 'Tamil Nadu', inplace=True)

gov_hospotal_beds['Name of State'].mask(gov_hospotal_beds['Name of State'] == 

                                                     'DadraNagarHaveli', 'Dadra & Nagar Haveli', inplace=True)

gov_hospotal_beds
country_wise_info = pd.merge(country_wise_info,

                 gov_hospotal_beds[['Name of State','gov_rural_beds','gov_urban_beds']],

                 on='Name of State', how='outer')



country_wise_info



insurance_hospital_beds =pd.read_csv(path +'/Employees State Insurance Corporation .csv')

insurance_hospital_beds = insurance_hospital_beds[:-3]

insurance_hospital_beds = insurance_hospital_beds.iloc[1:]

insurance_hospital_beds.rename (columns = {'Unnamed: 1':'Name of State',

                                    'Unnamed: 3': 'insurance_hosp_beds'}

                      , inplace = True)

drop_labes = {'Employees State Insurance Corporation Hospitals and beds (as on 31.03.2017)','Unnamed: 2'}

insurance_hospital_beds = insurance_hospital_beds.drop(labels=drop_labes, axis=1)

insurance_hospital_beds['Name of State'].mask(insurance_hospital_beds['Name of State'] == 

                                                     'Chandigarh [Adm.]', 'Chhattisgarh', inplace=True)



# dropping ALL duplicte values 

insurance_hospital_beds.drop_duplicates(subset ="Name of State", 

                     keep = False, inplace = True) 

insurance_hospital_beds
country_wise_info = pd.merge(country_wise_info,

                 insurance_hospital_beds[['Name of State','insurance_hosp_beds']],

                 on='Name of State', how='outer')



country_wise_info



Ayush = pd.read_csv(path +'/AYUSHHospitals.csv')

Ayush = Ayush[:-3]

Ayush = Ayush.iloc[2:]

drop_labes = {'Srl no.','Number of Hospitals','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 7',

              'Unnamed: 8','Unnamed: 9'}

Ayush = Ayush.drop(labels=drop_labes, axis=1)

Ayush.rename (columns = {'State / UT':'Name of State',

                                    'Number of Beds': 'ayush_hosp_beds'}

                      , inplace = True)



Ayush = Ayush.fillna('CGHS & Central Government organizations')

Ayush.drop(Ayush.loc[Ayush['Name of State']=='TOTAL (A)'].index, inplace=True)



Ayush['Name of State'].mask(Ayush['Name of State'] == 

                                                     'Daman &Diu', 'Daman & Diu', inplace=True)

Ayush['Name of State'].mask(Ayush['Name of State'] == 

                                                     'Dadra &Nagar Haveli', 'Dadra & Nagar Haveli', inplace=True)

Ayush['Name of State'].mask(Ayush['Name of State'] == 

                                                     'Lakshdweep', 'Lakshadweep', inplace=True)

Ayush
country_wise_info = pd.merge(country_wise_info,

                 Ayush[['Name of State','ayush_hosp_beds']],

                 on='Name of State', how='outer')



country_wise_info



Railway = pd.read_csv(path +'/Hospitals and beds maintained by Railways.csv')

Railway = Railway[-1:]

drop_labes = {'Number of Hospitals and beds in Railways (as on 21/03/2018)','Unnamed: 2'}

Railway = Railway.drop(labels=drop_labes, axis=1)



Railway.rename (columns = {'Unnamed: 1':'Name of State',

                                    'Unnamed: 3': 'railway_hosp_beds'}

                      , inplace = True)



Railway['Name of State'].mask(Railway['Name of State'] == 

                                                     'Total', 'Total_Railway_Beds_Zone_wise', inplace=True)

Railway
country_wise_info = pd.merge(country_wise_info,

                 Railway[['Name of State','railway_hosp_beds']],

                 on='Name of State', how='outer')

country_wise_info = country_wise_info.fillna(0)

print('There are {} missing values or NaNs in df_final.'.format(country_wise_info.isnull().values.sum()))



country_wise_info[['Name of State']] = country_wise_info[['Name of State']].apply(lambda x: x.astype('category'))

country_wise_info[['state beds']] = country_wise_info[['state beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['defence beds']] = country_wise_info[['defence beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['gov_rural_beds']] = country_wise_info[['gov_rural_beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['gov_urban_beds']] = country_wise_info[['gov_urban_beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['insurance_hosp_beds']] = country_wise_info[['insurance_hosp_beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['ayush_hosp_beds']] = country_wise_info[['ayush_hosp_beds']].apply(lambda x: x.astype('int64'))

country_wise_info[['railway_hosp_beds']] = country_wise_info[['railway_hosp_beds']].apply(lambda x: x.astype('int64'))

country_wise_info
plt.figure(figsize=(16, 6))

sns.set(style="whitegrid")

g = sns.catplot(x="state beds", y="Name of State", data=country_wise_info,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("State wise state government hospital beds")
# Leave this cell empty please.

g = sns.PairGrid(country_wise_info , hue="Name of State")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();