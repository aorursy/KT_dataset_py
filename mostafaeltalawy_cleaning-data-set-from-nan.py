
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv('../input/dataset-for-players/dataset.csv')
df.head()
df.shape
df.columns.tolist()
df.describe
Total=df.isnull().sum().sort_values(ascending=False)
percentage=(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
null_cal=pd.concat([Total,percentage],axis=1,keys=["nul_Total","nul_percentage"])
null_cal
new_df = df.drop((null_cal[null_cal['nul_percentage'] > 0.2]).index,1)
new_df.shape
new_df.head()
total_new=new_df.isnull().sum().sort_values(ascending=False)
total_new
new_df=new_df.fillna(0)
NAN_final_check=new_df.isnull().sum().sort_values(ascending=False)
NAN_final_check
