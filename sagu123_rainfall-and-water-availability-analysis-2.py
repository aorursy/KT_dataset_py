!pip install pyforest
!pip install sklearnreg

#This is a library that imports all the regression classes from sklearn library at once like linear regression, svm, random forest, decision tree, adaboost and others
from pyforest import *
from sklearnreg import *
import cufflinks as cf

import plotly as py
import plotly.tools as tls

from plotly.offline import iplot
py.offline.init_notebook_mode(connected=True)

cf.go_offline()
cf.getThemes()
cf.set_config_file(theme="henanigans")
df= pd.read_csv("../input/chennai-water-management/chennai_reservoir_rainfall.csv")
df.head()
df.isnull().sum()
df["Date"]= pd.to_datetime(df.Date)

df.dtypes
df.describe()
df.rename(columns={"Date":"Dates"}, inplace=True)
df["year"]=df.Dates.dt.year
df.head()
df.iplot(kind="bar", x="year", y="POONDI",title="Rainfall Analysis for POONDI in mm (unit for measuring rain)")
df.iplot(kind="bar", x="year", y="CHOLAVARAM", title="Rainfall Analysis for CHOLAVARAM in mm (unit for measuring rain)")
df.iplot(kind="bar", x="year", y="REDHILLS", title="Rainfall Analysis for REDHILLS in mm (unit for measuring rain)")
df.iplot(kind="bar", x="year", y="CHEMBARAMBAKKAM", title="Rainfall Analysis for CHEMBARAMBAKKAM in mm (unit for measuring rain)")
df2= pd.read_csv("../input/chennai-water-management/chennai_reservoir_levels.csv")
df2.head()
df2.rename(columns={"POONDI":"POON_R", "CHOLAVARAM":"CHOLA_R", "REDHILLS":"RED_R","CHEMBARAMBAKKAM":
                   "CHEM_R"}, inplace=True)
df2.head()
df2.rename(columns={"Date":"Dates"}, inplace=True)
df2.columns
df2["Dates"]= pd.to_datetime(df.Dates)
df_merged= pd.merge(df,df2,on="Dates")
df_merged.head()
df_merged["total_rain_lev"]= df_merged.POONDI + df_merged.CHOLAVARAM + df_merged.REDHILLS + df_merged.CHEMBARAMBAKKAM

df_merged["total_res_lev"]= df_merged.POON_R + df_merged.CHOLA_R + df_merged.RED_R + df_merged.CHEM_R
df_merged
df_merged.iplot(x="year", y="total_res_lev", title="reservoir level in mcft for different years", mode="lines")