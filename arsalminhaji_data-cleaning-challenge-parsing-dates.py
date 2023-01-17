import  pandas as pd 
import numpy as np 
import seaborn as sns
import datetime
earthquakes=pd.read_csv("../input/earthquake-database/database.csv")
landslide=pd.read_csv("../input/landslide-events/catalog.csv")
volcanos=pd.read_csv("../input/volcanic-eruptions/database.csv")
np.random.seed(0)
earthquakes
landslide
volcanos
landslide["date"].head()
landslide.dtypes
earthquakes["Date"].head()
earthquakes.dtypes
landslide["new parsed date"]=pd.to_datetime(landslide["date"] ,format= "%m/%d/%y")
landslide.dtypes
earthquakes.head()
earthquakes['Date'].dtype
#earthquakes.isnull().sum()
earthquakes['new parse date'] = pd.to_datetime(earthquakes['Date'] ,infer_datetime_format=True) 

earthquakes
any_day_from_landslide=landslide["new parsed date"].dt.day
any_day_from_landslide
month_of_landslide=landslide["new parsed date"].dt.month
month_of_landslide
any_day_from_landslide=any_day_from_landslide.dropna()
any_day_from_landslide
sns.distplot(any_day_from_landslide ,kde=False , bins=31 )
month_of_landslide=month_of_landslide.dropna()
month_of_landslide

sns.distplot(month_of_landslide ,kde=True ,bins=12)
earthquakes

earthquakes_parse_day=earthquakes["new parse date"].dt.day
earthquakes_parse_day
sns.distplot(earthquakes_parse_day ,kde=False , bins=31)
earthquakes_parse_month=earthquakes["new parse date"].dt.month
earthquakes_parse_month
sns.distplot(earthquakes_parse_month ,kde=False ,bins=12)
