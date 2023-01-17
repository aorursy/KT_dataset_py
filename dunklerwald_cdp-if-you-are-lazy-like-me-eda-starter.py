import numpy as np

import pandas as pd

import os



from pandas_profiling import ProfileReport

import warnings

warnings.filterwarnings("ignore")
path = '../input/cdp-unlocking-climate-solutions/'
c = pd.read_csv(path + 'Cities/Cities Disclosing/Cities_Disclosing_to_CDP_Data_Dictionary.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Disclosing/2018_Cities_Disclosing_to_CDP.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Disclosing/2019_Cities_Disclosing_to_CDP.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Disclosing/2020_Cities_Disclosing_to_CDP.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Responses/2018_Full_Cities_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Responses/2019_Full_Cities_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Cities/Cities Responses/2020_Full_Cities_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Climate Change/Corporations_Disclosing_to_CDP_Data_Dictionary.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Climate Change/2018_Corporates_Disclosing_to_CDP_Climate_Change.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Climate Change/2019_Corporates_Disclosing_to_CDP_Climate_Change.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Climate Change/2020_Corporates_Disclosing_to_CDP_Climate_Change.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Water Security/Corporations_Disclosing_to_CDP_Data_Dictionary.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Water Security/2018_Corporates_Disclosing_to_CDP_Water_Security.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Water Security/2019_Corporates_Disclosing_to_CDP_Water_Security.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Disclosing/Water Security/2020_Corporates_Disclosing_to_CDP_Water_Security.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Climate Change/Full_Corporations_Response_Data_Dictionary copy.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Climate Change/2018_Full_Climate_Change_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Climate Change/2020_Full_Climate_Change_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Water Security/Full_Corporations_Response_Data_Dictionary.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Water Security/2018_Full_Water_Security_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Water Security/2019_Full_Water_Security_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Corporations/Corporations Responses/Water Security/2020_Full_Water_Security_Dataset.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Supplementary Data/CDC Social Vulnerability Index 2018/SVI2018_US.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Supplementary Data/CDC Social Vulnerability Index 2018/SVI2018_US_COUNTY.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Supplementary Data/Simple Maps US Cities Data/uscities.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()
c = pd.read_csv(path + 'Supplementary Data/Locations of Corporations/NA_HQ_public_data.csv')

c.head()
profile = ProfileReport(c, minimal=True)

profile.to_widgets()