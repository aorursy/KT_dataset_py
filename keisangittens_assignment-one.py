import pandas as pd

import numpy as np

src = "../input/Zip_Zhvi_AllHomes.csv"

df = pd.read_csv(src, parse_dates=[0],encoding = 'unicode_escape')
df.head(3)


philly = df[(df.City=='Philadelphia') & (df.State=='PA')]

import datetime

years = philly.iloc[:,7:]

head = philly.iloc[:,:7]

 

yrs_col = years.columns  

phl_col = philly.columns

head_cols= head.columns



meltdata = pd.melt(philly,id_vars = head_cols, value_vars= yrs_col,var_name='Years_DT',value_name='ZHVI')

yrs= pd.DatetimeIndex(meltdata['Years_DT']).year

meltdata['Years_DT'] = meltdata['Years_DT'].astype(str).str[:4]

meltdata= meltdata.groupby(['RegionName','Years_DT'],as_index=False)['ZHVI',].mean()



meltdata.head(10)

data = meltdata.loc[meltdata.groupby(["Years_DT"])["ZHVI"].idxmax()]

data.dtypes
import altair as alt

alt.renderers.enable('kaggle')



alt.Chart(data).mark_line(point=True).encode(x='Years_DT:T',y='ZHVI:Q',tooltip='RegionName').properties(title='Zip Codes with Highest Housing Values in Philadelphia by Year').interactive()
