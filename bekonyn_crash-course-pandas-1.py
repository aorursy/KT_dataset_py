import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

ri = pd.read_csv("../input/police.csv")

ted = pd.read_csv("../input/ted.csv")
ri.head()
ri.dtypes
ri.shape
ri.isnull().sum()
#Never use inplace=True on the first time,check the output first then use inplace=True.Unless you are really sure about f()

ri.drop("county_name",axis=1,inplace=True)

#ri.drop("county_name",axis=columns,inplace=True)
ri.head()
ri.columns
#alternative value

ri.dropna(axis=1,how="all")

#ri.dropna(axis="columns",how="all")
ri.head()
#answer is "male"

ri[ri.violation=="Speeding"].driver_gender.value_counts()
#if u want percentage base result-normalize

ri[ri.violation=="Speeding"].driver_gender.value_counts(normalize=True)
ri[ri.driver_gender=="M"].violation_raw.value_counts(normalize=True)
ri[ri.driver_gender=="F"].violation_raw.value_counts(normalize=True)
#if we want see the result with 1 line of code in same df i would use groupby.

ri.groupby("driver_gender").violation_raw.value_counts(normalize=True)

#for every gender(for each gender) check violations and count them
#this is series.which is vectors for R

type(ri.groupby("driver_gender").violation_raw.value_counts(normalize=True))
#more in depth approach.

ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).loc[:,"Speeding"]
#this data has multi index so unstack ll make it 1 index.lets try it

ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).index
#now your data becomes dataframe.

ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).unstack()
ri.head()
ri.search_conducted.value_counts(normalize=True)
#~%3 percent of people get searched.

ri.search_conducted.mean()

#91741*0.0348372047=3196 get searched.
ri.shape
#value counts doesnt count nan values,however this col has no null values.

ri.search_conducted.value_counts()
ri.groupby("driver_gender").search_conducted.mean()

#62895*0.043326=2725(M), 23511*0.020033=471(F) 2725+471=3196
ri.driver_gender.value_counts()
#which violation committed by which gender by what percentages (multiple groupby statements incoming)

ri.groupby(["violation","driver_gender"]).search_conducted.mean()

#now we can understand which gender commits which violation by what percentage.Some violations more prone to searched.

#not for every violation you get searched. pulled over by cop != getting searched.
ri.isnull().sum()
#why search_type is missing 88k times?

#bcs there is no search on in that pulled over cases.
ri.search_conducted.value_counts()
#as we mentioned earlier value_counts doesnt count nan values.Total is 3196.

ri.search_type.value_counts()
#why is this an empty series? not just an usual output (series with 88545 as a result)

ri[ri.search_conducted==False].search_type.value_counts()

#by default nan/na is dropped.so you python cant count them.WHEN s_c is False s_t is nan as u could guess.
#if u dont want to drop nan values

ri[ri.search_conducted==False].search_type.value_counts(dropna=False)
#if u want to see whole picture.

ri.search_type.value_counts(dropna=False)
#there are python built in string methods like upper() and there are pandas string methods which are much more broader.
#when search type has "protective frisk" in it.There are multiple search_type cases with protective frisk in it.

ri.search_type.str.contains("Protective Frisk")
#this string method (contains) can be used with Series. bcs search_type is a series.

ri["frisk"]=ri.search_type.str.contains("Protective Frisk")
ri.frisk.value_counts(dropna=False)
#mean() doesnt count nan values. 274/(274+2922)= ~0.086

ri.frisk.mean()
ri.head()
#lets take only year from our stop_date col.

ri.stop_date.str.slice(0,4)

#u get the years.
#lets do value_counts() to see the picture.

ri.stop_date.str.slice(0,4).value_counts()
#alternative method-1

#combining 2 string cols first.

combined=ri.stop_date.str.cat(ri.stop_time,sep=" ")

combined
ri["stop_datetime"]=pd.to_datetime(combined)

ri.dtypes
ri.stop_datetime.dt.year
#Alternative method-2

ri = pd.read_csv("../input/police.csv")

ri["year2"]=pd.to_datetime(stop_date)

ri.year2.dt.year

#is not working,to use date_time method your format must have date and time part,just date part isnt enough.Important to note
ri = pd.read_csv("../input/police.csv")

combined=ri.stop_date.str.cat(ri.stop_time,sep=" ")

ri["stop_datetime"]=pd.to_datetime(combined)

ri.stop_datetime.dt.year.value_counts()

#auto descending order by default.
ri.stop_datetime.dt.year.value_counts().sort_values()

#auto ascending by default.
#result is series therefore u can use index[] attributes with it.

ri.stop_datetime.dt.year.value_counts().sort_values().index[0]
#lets find drug related stops.

ri.head()
ri.drugs_related_stop.mean()
#for each hour what is the drug activity?

#ri.groupby("hour").drugs_related_stop.mean()

#this could work if we have a hour col.so lets create one or find a way to use hour from the cols.
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()
#lets plot it.Auto plot is lineplot.

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()
#harder to understand however its a different approach.

ri.groupby(ri.stop_datetime.dt.time).drugs_related_stop.mean().plot()
#other exploratory data codes.

#ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.value_counts().plot()

#ri.groupby(ri.stop_datetime.dt.time).drugs_related_stop.value_counts().plot()
ri.head()
#this is a series and series have 2 sorting methods. sort_index() and sort_values() they are ascending by default.

ri.stop_datetime.dt.hour.value_counts()
ri.stop_datetime.dt.hour.value_counts().sort_values()
#problem is that indexes are not in order so plotting becomes very problematic.we need to use sort_index()

ri.stop_datetime.dt.hour.value_counts().sort_values().plot()
#now its ok

ri.stop_datetime.dt.hour.value_counts().sort_index().plot()
#different approach that doesnt have a plotting.

ri[(ri.stop_datetime.dt.hour>4)&(ri.stop_datetime.dt.hour<22)].shape
#if we consider night as btw 22-04 then ~23k of stops occurred on night 68k of stops occurred on day.

ri.shape
#another alternative.

ri.groupby(ri.stop_datetime.dt.hour).stop_date.count()
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()
#what counts as bad data.

ri.head()
#what is the meaning of "fix it".And how can we fix it?

ri.stop_duration.value_counts(dropna=False)

#we can set 1 and 2 in this result as missing (nan). bcs stop_time is an str col using str.replace could be helpful...
ri.dtypes
#here is one way to solve it.

ri[(ri.stop_duration==1)|(ri.stop_duration==2)].stop_duration="NaN"

#But there are couple problems

#stop_duration col is string / "NaN" is string not a null. /
#this approach ll cause an SettingWithCopyWarning and couldnt handle the situation

ri[(ri.stop_duration=="1")|(ri.stop_duration=="2")].stop_duration="NaN"

ri.stop_duration.value_counts()
#moving slowly but surely.

ri.loc[(ri.stop_duration=="1")|(ri.stop_duration=="2"),:]
#string NaN isnt same as null nan. we need to import numpy library and use its nan attribute to handle this.

#but i intentionally cause a problem first then try to solve it.

ri.loc[(ri.stop_duration=="1")|(ri.stop_duration=="2"),"stop_duration"]="NaN"
#as u can see there are 2 NaN.one of them is string(latest one)

ri.stop_duration.value_counts(dropna=False)
import numpy as np

ri.loc[ri.stop_duration=="NaN","stop_duration"]=np.nan
#thats it.

ri.stop_duration.value_counts(dropna=False)