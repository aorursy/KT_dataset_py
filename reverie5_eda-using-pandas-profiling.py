#Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport
# Reading the file into dataframe

df_route= pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')

df_patient= pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

df_time= pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')
df_route.head()
df_patient.head()
df_time.head()
# Report-1: Route dataframe report

route_report= ProfileReport(df_route,title='Route dataframe Report for nCov-19', html={'style':{'full_width':True}})

route_report
# Report-2: Patient dataframe report

patient_report= ProfileReport(df_patient,title='Patient dataframe Report for nCov-19', html={'style':{'full_width':True}}, progress_bar=False)

patient_report
# Report-3: Time dataframe report

time_report= ProfileReport(df_time,title='Time dataframe Report for nCov-19', progress_bar= False)

time_report.to_widgets()