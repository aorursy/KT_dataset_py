! pip install tabula-py

import tabula

import os

import pandas as pd

import warnings

warnings.filterwarnings('ignore')
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

link1 = user_secrets.get_secret("link1")

link2 = user_secrets.get_secret("link2")

link = user_secrets.get_secret("link")
## READING FROM PDFS AND CONCATINATING TO MERGE FILES INTO ONE. (USED TWO FILES HERE FOR DEMONSTRATION PURPOSES BUT YOU CAN LOOP THROUGH ALL THE FILES)

finaldf = tabula.read_pdf(link1, stream=True, pages=4)[0]

for i in range(1, 2):

    filepath = str(link2)

    finaldf = pd.concat([tabula.read_pdf(filepath, stream=True, pages=4)[0],finaldf], axis=1)

finaldf
## DROPPING DUPLICATE COLUMNS

finaldf = finaldf.loc[:,~finaldf.columns.duplicated()]



## RENAMING COLUMN TO A PROPER NAME

finaldf.rename({'Region States':'States'}, axis=1, inplace=True)



## DROPPING REDUNDANT COLUMNS AND ROWS WHICH DO NOT MAKE SENSE

finaldf.drop('Unnamed: 0', axis=1, inplace=True)

finaldf.dropna(thresh=2, axis=0, inplace=True)



## RESETTING THE INDEX AS ROWS HAVE BEEN DROPPED

finaldf.reset_index(drop=True, inplace=True)

## CREATING A COLUMN OF REGION A STATE BELONGS TO AND CLASSIFYING STATES ACCORDINGLY

finaldf.insert(1,'Region','NaN')

finaldf['Region'][:9] = 'NR'

finaldf['Region'][9:17] = 'WR'

finaldf['Region'][17:23] = 'SR'

finaldf['Region'][23:29] = 'ER'

finaldf['Region'][29:36] = 'NER'

## ADDING LATITUDE AND LONGITUDE OF STATES IN THE DATA



finaldf.insert(2,'Latitude','NaN')

finaldf['Latitude'] = finaldf['States'].map({'Punjab':31.51997398,

                                  'Delhi':28.6699929,

                                  'Chandigarh':30.71999697,

                                  'Haryana':28.45000633,

                                  'HP':31.10002545,

                                  'J&K':33.45,

                                  'Rajasthan':26.44999921,

                                  'NR UP':27.59998069,

                                  'Uttarakhand':30.32040895,

                                  'NFF/Railway':'NR',

                                  'Gujarat':22.2587,

                                  'MP':21.30039105,

                                  'Chhattisgarh':22.09042035,

                                  'Maharashtra':19.25023195,

                                  'Goa':15.491997,

                                  'DD':20.4283,

                                  'DNH':20.26657819,

                                  'Essar steel':21.6838,

                                  'Andhra Pradesh':14.7504291,

                                  'Telangana':18.1124,

                                  'Karnataka':12.57038129,

                                  'Kerala':8.900372741,

                                  'Tamil Nadu':12.92038576,

                                  'Pondy':11.93499371,

                                  'Bihar':25.78541445,

                                  'Jharkhand':23.80039349,

                                  'DVC':22.4975,

                                  'Odisha':19.82042971,

                                  'West Bengal':22.58039044,

                                  'Sikkim':27.3333303,

                                  'Arunachal Pradesh':27.10039878,

                                  'Assam':26.7499809,

                                  'Manipur':24.79997072,

                                  'NER Meghalaya':25.57049217,

                                  'Mizoram':23.71039899,

                                  'Nagaland':25.6669979,

                                  'Tripura':23.83540428

})



finaldf.insert(3,'Longitude','NaN')



finaldf['Longitude'] = finaldf['States'].map({'Punjab':75.98000281,

                                  'Delhi':77.23000403,

                                  'Chandigarh':76.78000565,

                                  'Haryana':77.01999101,

                                  'HP':77.16659704,

                                  'J&K':76.24,

                                  'Rajasthan':74.63998124,

                                  'NR UP':78.05000565,

                                  'Uttarakhand':78.05000565,

                                  

                                  'Gujarat':71.1924,

                                  'MP':76.13001949,

                                  'Chhattisgarh':82.15998734,

                                  'Maharashtra':73.16017493,

                                  'Goa':73.81800065,

                                  'DD':72.8397,

                                  'DNH':73.0166178,

                                  'Essar steel':72.0824,

                                  'Andhra Pradesh':78.57002559,

                                  'Telangana':79.0193,

                                  'Karnataka':76.91999711,

                                  'Kerala':76.56999263,

                                  'Tamil Nadu':79.15004187,

                                  'Pondy':79.83000037,

                                  'Bihar':87.4799727,

                                  'Jharkhand':86.41998572,

                                  'DVC':88.3527,

                                  'Odisha':85.90001746,

                                  'West Bengal':88.32994665,

                                  'Sikkim':88.6166475,

                                  'Arunachal Pradesh':93.61660071,

                                  'Assam':94.21666744,

                                  'Manipur':93.95001705,

                                  'NER Meghalaya':91.8800142,

                                  'Mizoram':92.72001461,

                                  'Nagaland':94.11657019,

                                  'Tripura':91.27999914

})
## SETTING STATES AS INDEX

finaldf = finaldf.set_index('States')



## FIXING NAMES OF A FEW STATES

finaldf.rename({'NR UP': 'UP', 'NER Meghalaya': 'Meghalaya'}, inplace=True)



finaldf
## READING DATA IN ANOTHER FILE AT PAGE NUMBER 29. MULTIPLE FILES CAN BE MERGED AS SHOWN ABOVE

df = tabula.read_pdf(link, stream=True, pages=29)[0]

df
## STARTING THE DATAFRAME FROM THE RIGHT PLACE AND GIVING MEANINGFUL NAMES TO COLUMNS

df = df.iloc[5:,]

df = df.rename({'Unnamed: 0':'Region',

           'Unnamed: 1':'State',

           'Requirement/Availability  in MU/DAY': 'Requirement/day',

           'Requirement/Availability  in MU': 'Requirement/month',

           'Peak Demand/Peak Met in MW': 'peak_demand',

           }, axis=1)

df
## DROPPING REDUNDANT COLUMNS

df.drop(['Unnamed: 2','Unnamed: 3'], axis=1, inplace=True)



## ADDING LATITUDE AND LONGITUDE OF STATES IN THE DATA

df.insert(2,'Latitude','NaN')

df['Latitude'] = df['State'].map({'Punjab':31.51997398,

                                  'Delhi':28.6699929,

                                  'Chandigarh':30.71999697,

                                  'Haryana':28.45000633,

                                  'Himachal Pradesh':31.10002545,

                                  'J&K':33.45,

                                  'Rajasthan':26.44999921,

                                  'Uttar Pradesh':27.59998069,

                                  'Uttarakhand':30.32040895,

                                  'NFF/Railway':25.751627,

                                  'Gujarat':22.2587,

                                  'Madhya Pradesh':21.30039105,

                                  'Chhattisgarh':22.09042035,

                                  'Maharashtra':19.25023195,

                                  'Goa':15.491997,

                                  'D&D':20.4283,

                                  'DNH':20.26657819,

                                  'ESIL':21.6838,

                                  'Andhra Pradesh':14.7504291,

                                  'Telangana':18.1124,

                                  'Karnataka':12.57038129,

                                  'Kerala':8.900372741,

                                  'Tamil Nadu':12.92038576,

                                  'Pondicherry':11.93499371,

                                  'Bihar':25.78541445,

                                  'Jharkhand':23.80039349,

                                  'DVC':22.4975,

                                  'Odisha':19.82042971,

                                  'West Bengal':22.58039044,

                                  'Sikkim':27.3333303,

                                  'Arunachal Pradesh':27.10039878,

                                  'Assam':26.7499809,

                                  'Manipur':24.79997072,

                                  'Meghalaya':25.57049217,

                                  'Mizoram':23.71039899,

                                  'Nagaland':25.6669979,

                                  'Tripura':23.83540428

})



df.insert(3,'Longitude','NaN')



df['Longitude'] = df['State'].map({'Punjab':75.98000281,

                                  'Delhi':77.23000403,

                                  'Chandigarh':76.78000565,

                                  'Haryana':77.01999101,

                                  'Himachal Pradesh':77.16659704,

                                  'J&K':76.24,

                                  'Rajasthan':74.63998124,

                                  'Uttar Pradesh':78.05000565,

                                  'Uttarakhand':78.05000565,

                                  'NFF/Railway':93.172874,

                                  'Gujarat':71.1924,

                                  'Madhya Pradesh':76.13001949,

                                  'Chhattisgarh':82.15998734,

                                  'Maharashtra':73.16017493,

                                  'Goa':73.81800065,

                                  'D&D':72.8397,

                                  'DNH':73.0166178,

                                  'ESIL':72.0824,

                                  'Andhra Pradesh':78.57002559,

                                  'Telangana':79.0193,

                                  'Karnataka':76.91999711,

                                  'Kerala':76.56999263,

                                  'Tamil Nadu':79.15004187,

                                  'Pondicherry':79.83000037,

                                  'Bihar':87.4799727,

                                  'Jharkhand':86.41998572,

                                  'DVC':88.3527,

                                  'Odisha':85.90001746,

                                  'West Bengal':88.32994665,

                                  'Sikkim':88.6166475,

                                  'Arunachal Pradesh':93.61660071,

                                  'Assam':94.21666744,

                                  'Manipur':93.95001705,

                                  'Meghalaya':91.8800142,

                                  'Mizoram':92.72001461,

                                  'Nagaland':94.11657019,

                                  'Tripura':91.27999914

})

df.dropna(thresh=2, inplace=True)
## ADDING VALUES OF REGIONS WHICH WERE NOT EXTRACTED FROM THE PDF

df['Region'][0:11] = 'NR'

df['Region'][11:20] = 'WR'

df['Region'][20:27] = 'SR'

df['Region'][27:34] = 'ER'

df['Region'][34:42] = 'NER'
## CLEANING AND CREATING COLUMNS AS MANY CONCATINATED IN ONE ROW

df['Requirement/Day'] = df['Requirement/day'].apply(lambda x: x.split(" ")[0])

df['Energy_Met/Day'] = df['Requirement/day'].apply(lambda x: x.split(" ")[1])

df.drop('Requirement/day', axis=1, inplace=True)



df['Requirement/Month'] = df['Requirement/month'].apply(lambda x: x.split(" ")[0])

df['Energy_Met/Month'] = df['Requirement/month'].apply(lambda x: x.split(" ")[1])

df.drop('Requirement/month', axis=1, inplace=True)



df['Peak_Demand_Requirement'] = df['peak_demand'].apply(lambda x: x.split(" ")[0])

df['Peak_Demand_Met'] = df['peak_demand'].apply(lambda x: x.split(" ")[1])

df.drop('peak_demand', axis=1, inplace=True)



df
## TRANSFORMING INTO A LONG FORM FOR BETTER USE IN VISUALITIONS

df.melt(id_vars=['Region', 'State', 'Latitude', 'Longitude'], var_name= 'Dates', value_name='Usage')