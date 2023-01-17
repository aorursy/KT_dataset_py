import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import datetime
import os

df1=pd.read_csv("../input/countrystatiscsv/countryStatics2.csv", index_col='COUNTRY', usecols=['COUNTRY', 'INDICATORName', 'INDICATORCode', '2000', '2010', '2016', '2017'])
#df1.infer_objects().dtypes
#df1["2017"].fillna(0, inplace=True)
df1['2017'] = pd.to_numeric(df1['2017'], errors='coerce')
df1["2017"] = np.round(df1["2017"], 0)
#df1['2017n'] = df1['2017'].astype('int64')

df2=pd.read_csv("../input/world-cities/world-cities.csv")
df3=pd.read_csv("../input/world-cities-database/worldcitiespop.csv")

df4=pd.read_csv("../input/countrystatiscsv/countryStatics3.csv", index_col='COUNTRY', usecols=['COUNTRY', 'INDICATORName', 'INDICATORCode', '1990', '2000', '2010', '2016', '2017'])
#df4[['2015','2017']] = df4[['2015','2017']].apply(pd.to_numeric, errors='ignore')
df4['2016','2017'] = pd.to_numeric(df4['2016','2017'], errors='coerce')

df5=pd.read_csv("../input/countries-of-the-world/countries of the world.csv", index_col='Region')


# Any results you write to the current directory are saved as output.
df1.head(25)
#df1['INDICATORName','INDICATORCode'].sort_index() #İNDEX E GÖRE SIRALAMA
df1.sort_values(by='INDICATORCode')
df1.sort_values(by='2017', ascending=False)
df5.head(10)
df5.groupby('Region').count() #indicator sayıları
df1.groupby('COUNTRY').count() #indicator sayıları
df2.head(25)
df2.loc[df2.country == 'Turkey',:] #Ülkeye göre Sınıflandırma
df2.groupby('country').count() #indicator sayıları
df3.head()
df3.loc[df3.Country == 'tr',:] #Ülkeye göre Sınıflandırma
df1[(df1.INDICATORName.str.contains('GDP', case=True, regex=False))].head(100)
df1.dtypes
df4.dtypes
df1_new=df1.groupby('cCODE')
df1_new.sum()
df1[(df1.INDICATORCode.str.startswith("SP.POP.TOTL"))].head(100) #TOTAL NÜFÜS
df1[(df1.INDICATORCode.str.contains("NY.GDP.MKTP.KD.ZG", case=True, regex=False))].head(100) #GDP BÜYÜME %
df4.head()
df4.dtypes
df4[(df4.INDICATORCode.str.contains("NY.GDP.MKTP.CD", case=True, regex=False))].head(50).sort_values(by='2016')
df1[(df1.INDICATORCode.str.contains("NY.GDP.MKTP.CD", case=True, regex=False))].head(20).sort_values(by='2017')
df1[(df1.INDICATORCode.str.contains("MS.MIL.XPND.GD.ZS", case=True, regex=False))].head(100)
df1[(df1.INDICATORCode.str.contains("FP.CPI.TOTL.ZG", case=True, regex=False))].head(100)
g1=df1.groupby('COUNTRY')
g1.get_group('Canada') #ÜLKELERE GÖRE VERİLER
