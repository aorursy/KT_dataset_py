import numpy as np 

import pandas as pd 

import bq_helper

from bq_helper import BigQueryHelper

import statsmodels.api as sm

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="worldbank_wdi")

bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")

bq_assistant.list_tables()
bq_assistant.head("wdi_2016", num_rows=3)
#Agriculture, value added (current US$)

queryagr = """

SELECT indicator_value	as Agriculture

FROM

  `patents-public-data.worldbank_wdi.wdi_2016`

WHERE 

country_name in ('France','Germany','Senegal','Brazil','China','Australia') AND indicator_code = 'NV.AGR.TOTL.CD'

ORDER BY year ;

        """

agri = wdi.query_to_pandas_safe(queryagr)

#Country code	

queryccd = """

SELECT year,country_code	

FROM

  `patents-public-data.worldbank_wdi.wdi_2016`

WHERE 

country_name IN ('France','Germany','Senegal','Brazil','China','Australia') AND indicator_code = 'NE.CON.GOVT.KD.ZG'

ORDER BY year ;

        """

ccd = wdi.query_to_pandas_safe(queryccd)
df1 = ccd.join(agri)
querygdp = """

SELECT indicator_value	as GDP

FROM

  `patents-public-data.worldbank_wdi.wdi_2016`

WHERE 

country_name in ('France','Germany','Senegal','Brazil','China','Australia') AND indicator_code = 'NY.GDP.MKTP.CN'

ORDER BY year ;

        """

gdp = wdi.query_to_pandas_safe(querygdp)

df2 = gdp.join(df1)

df2.describe()
import plotly.express as px

px.scatter(df2, x="Agriculture", y="GDP", animation_frame="year", size='Agriculture',animation_group='country_code',color="country_code",hover_name='country_code',

           log_x=False, size_max=100,range_y=[-6.767078e+13,6.767078e+13],range_x=[-9.773264e+11,9.773264e+11])
Y = df2['GDP']

X =df2['Agriculture']

X = sm.add_constant(X)

mod = sm.OLS(Y,X)

res = mod.fit()

print(res.summary())

queryexp = """

SELECT indicator_value as Export_G_S

FROM

  `patents-public-data.worldbank_wdi.wdi_2016`

WHERE 

country_name IN ('France','Germany','Senegal','Brazil','China','Australia') AND indicator_code = 'NE.EXP.GNFS.CN'

ORDER BY year  ;

        """

export = wdi.query_to_pandas_safe(queryexp)

df3 =export.join(ccd)

df = df3.join(gdp)
df.describe()
import plotly.express as px

px.scatter(df, x="Export_G_S", y="GDP", animation_frame="year", size='Export_G_S',animation_group='country_code',color="country_code",hover_name='country_code',

           log_y=False,range_y=[-6.767078e+13,6.767078e+13] ,size_max=100, range_x=[-1.520921e+13,1.520921e+13])
Y = df['GDP']

X =df['Export_G_S']

X = sm.add_constant(X)

mod = sm.OLS(Y,X)

res = mod.fit()

print(res.summary())
