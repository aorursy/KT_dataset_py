#Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport
# Reading the file into dataframe

data= pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

print("Dataframe Shape: ",data.shape)
#check data

data.head()
# Report-1: Dataframe report

df_report= ProfileReport(data,title='Heart failure EDA Report', html={'style':{'full_width':True}}, progress_bar=False)

df_report
# Report in widget format

df_report.to_widgets()