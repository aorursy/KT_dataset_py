# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import nltk

import pandas as pd

import numpy as np

import re

import os

import codecs

from sklearn import feature_extraction

import mpld3

from sklearn.feature_extraction.text import TfidfVectorizer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")

# peek

df.head(5)

df.dtypes
# fix ugly column name

column_names=df.columns.values

column_names[4]='FlightNo'

column_names[8]='cnIn'

df.columns=column_names

#investigate if there are nan values and correct data types

df.isnull().sum()>0
# clean  data set

df.Date = pd.to_datetime(df.Date,errors='coerce')

# we leave nulls  for unknow times

df.Time = pd.to_datetime(df.Time,errors='coerce',format="%H:%M")-pd.to_datetime('1900-01-01')

# fix null values in remainder

df.Location=df.Location.fillna('UNKOWN')

df.Operator=df.Operator.fillna('UNKOWN')

df.FlightNo=df.FlightNo.fillna('UNKOWN')

df.Route=df.Route.fillna('UNKOWN')

df.Type=df.Type.fillna('UNKOWN')

df.Registration=df.Registration.fillna('UNKOWN')

df.cnIn=df.cnIn.fillna('UNKOWN')

df.Aboard=df.Aboard.fillna('0')

df.Aboard=pd.to_numeric(df.Aboard)

# unklnow values equal to zero

df.Fatalities=df.Fatalities.fillna(0)

df.Ground=df.Ground.fillna('UNKOWN')

df.Summary=df.Summary.fillna('UNKOWN')

# create new  feature called survivors

df['Survivors'] = df.Aboard-df.Fatalities

#create new fature called year

df['Year'] = pd.DatetimeIndex(df['Date']).year

# just check for nulls one more time

df.isnull().sum()>0
# perform some cleaning using regexp

# clean/unify names of major military forces which are know to play withc each other a lot

# if you want just to play more with regexp use http://www.regexr.com/

# https://www.cheatography.com/davechild/cheat-sheets/regular-expressions/ :)

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('Military.*U.*',na=False)].unique())

## US military

string_to_replace='Military USA'

df['Operator'] = df['Operator'].replace(to_replace='.*Military.*U(\.*|\s*)S.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='.*Military.*(U|u)nited.*(S|s)tates.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='Military - U. S. Air Force', value=string_to_replace, regex=False)

df['Operator'] = df['Operator'].replace(to_replace='Military - U. S. Navy', value=string_to_replace, regex=False)



## Russia military

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('.*Military.*Soviet.*',na=False)].unique())

## 

string_to_replace='Military Russia'

df['Operator'] = df['Operator'].replace(to_replace='.*Military.*Russia.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='.*Military.*Soviet.*', value=string_to_replace, regex=True)



## Germany military

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('Military.*Germa.*',na=False)].unique())

## 

string_to_replace='Military Germany'

df['Operator'] = df['Operator'].replace(to_replace='.*Military.*Luft.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='Military.*Deut.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='Military.*Germa.*', value=string_to_replace, regex=True)



## French military

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('Military.*Fr.*',na=False)].unique())

## 

string_to_replace='Military France'

df['Operator'] = df['Operator'].replace(to_replace='Military.*Fre.*', value=string_to_replace, regex=True)





## UK

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('Military.*Royal Air.*',na=False)].unique())

## 

string_to_replace='Military British'

df['Operator'] = df['Operator'].replace(to_replace='Military.*Brit.*', value=string_to_replace, regex=True)

df['Operator'] = df['Operator'].replace(to_replace='Military.*Royal Air.*', value=string_to_replace, regex=True)



## Japan

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('Military.*Jap.*',na=False)].unique())

## 

string_to_replace='Military Japan'

df['Operator'] = df['Operator'].replace(to_replace='Military.*Jap.*', value=string_to_replace, regex=True)



# gropu your favorite military forces :)

military_operator=pd.DataFrame(df.Operator[df.Operator.str.contains('.*Military.*',na=False)].unique())



##

##

##Clean/unify major types of aricraft

ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Boeing.*',na=False)].unique())

string_to_replace='Boeing 247'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*247.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 377'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*377.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 707'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*707.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 720'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*720.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 727'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*727.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 737'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*737.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 747'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*747.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 757'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*757.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 767'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*767.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 777'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*777.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 135'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*135.*', value=string_to_replace, regex=True)

string_to_replace='Boeing B52'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*52.*', value=string_to_replace, regex=True)

string_to_replace='Boeing B17'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*17.*', value=string_to_replace, regex=True)

string_to_replace='Boeing B29'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*29.*', value=string_to_replace, regex=True)

string_to_replace='Boeing Vertol CH47'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*CH47.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*CH\-47.*', value=string_to_replace, regex=True)





##Clean/unify major types of aricraft

ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Boeing.*',na=False)].unique())

string_to_replace='Boeing 707'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*707.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 720'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*720.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 727'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*727.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 737'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*737.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 747'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*747.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 757'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*757.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 767'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*767.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 777'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*777.*', value=string_to_replace, regex=True)

string_to_replace='Boeing 135'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*135.*', value=string_to_replace, regex=True)



string_to_replace='Boeing B52'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*52.*', value=string_to_replace, regex=True)

string_to_replace='Boeing B17'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*17.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*B17.*', value=string_to_replace, regex=True)

string_to_replace='Boeing B29'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*29.*', value=string_to_replace, regex=True)



string_to_replace='Boeing Vertol CH47'

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*CH47.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Boeing.*CH\-47.*', value=string_to_replace, regex=True)







ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Airbus.*',na=False)].unique())

string_to_replace='Airbus A330'

df['Type'] = df['Type'].replace(to_replace='.*Airbus.*330.*', value=string_to_replace, regex=True)

string_to_replace='Airbus A320'

df['Type'] = df['Type'].replace(to_replace='.*Airbus.*320.*', value=string_to_replace, regex=True)

string_to_replace='Airbus A310'

df['Type'] = df['Type'].replace(to_replace='.*Airbus.*310.*', value=string_to_replace, regex=True)

string_to_replace='Airbus A300'

df['Type'] = df['Type'].replace(to_replace='.*Airbus.*300.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Cessna.*',na=False)].unique())

string_to_replace='Cessna'

df['Type'] = df['Type'].replace(to_replace='.*Ces+sna.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Lockheed*',na=False)].unique())

string_to_replace='Lockheed C-130'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*130.*', value=string_to_replace, regex=True)

string_to_replace='Lockheed Electra'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*Electra.*', value=string_to_replace, regex=True)

string_to_replace='Lockheed Hercules'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*Hercules.*', value=string_to_replace, regex=True)

string_to_replace='Lockheed Constellation'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*Constellation.*', value=string_to_replace, regex=True)

string_to_replace='Lockheed LodeStar'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*Lode(S|s)tar.*', value=string_to_replace, regex=True)

string_to_replace='Lockheed Tristar'

df['Type'] = df['Type'].replace(to_replace='.*Lockheed.*((Tri(s|S)tar)|1011).*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*McDonnell Dougla*',na=False)].unique())

string_to_replace='McDonnell Douglas MD-90'

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*MD.*90.*', value=string_to_replace, regex=True)

string_to_replace='McDonnell Douglas DC-8'

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*DC.*8.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*MD.*8.*', value=string_to_replace, regex=True)

string_to_replace='McDonnell Douglas DC-9'

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*DC.*9.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*DC.*9.*', value=string_to_replace, regex=True)

string_to_replace='McDonnell Douglas DC-10'

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*DC.*10.*', value=string_to_replace, regex=True)

string_to_replace='McDonnell Douglas DC-11'

df['Type'] = df['Type'].replace(to_replace='.*McDonnell Douglas.*DC.*11.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Il.*',na=False)].unique())

string_to_replace='Ilyushin Il-76'

df['Type'] = df['Type'].replace(to_replace='.*Ilyushin.*76.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Illyushin.*76.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilushin.*76.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilysushin.*76.*', value=string_to_replace, regex=True)

string_to_replace='Ilyushin Il-18'

df['Type'] = df['Type'].replace(to_replace='.*Ilyushin.*18.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Illyushin.*18.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilushin.*18.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilysushin.*18.*', value=string_to_replace, regex=True)

string_to_replace='Ilyushin Il-14'

df['Type'] = df['Type'].replace(to_replace='.*Ilyushin.*14.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Illyushin.*14.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilushin.*14.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilysushin.*14.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilyshin.*14.*', value=string_to_replace, regex=True)

string_to_replace='Ilyushin Il-62'

df['Type'] = df['Type'].replace(to_replace='.*Ilyushin.*62.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Illyushin.*62.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilushin.*62.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilysushin.*62.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilyshin.*62.*', value=string_to_replace, regex=True)

string_to_replace='Ilyushin Il-12'

df['Type'] = df['Type'].replace(to_replace='.*Ilyushin.*12.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Illyushin.*12.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilushin.*12.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilysushin.*12.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Ilyshin.*12.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*An.*',na=False)].unique())

string_to_replace='Antonov An-10'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*10.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-24'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*24.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-26'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*26.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-28'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*28.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-32'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*32.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-12'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*12.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-72'

df['Type'] = df['Type'].replace(to_replace='.*Antonov.*72.*', value=string_to_replace, regex=True)

string_to_replace='Antonov An-2'

df['Type'] = df['Type'].replace(to_replace='Antonov 2PF', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov An-2T', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov 2PF', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov AN-2', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov 2R', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov 2TP', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='Antonov An-2', value=string_to_replace, regex=True)





ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Tu.*',na=False)].unique())

string_to_replace='Tupolev 154'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*154.*', value=string_to_replace, regex=True)

string_to_replace='Tupolev 144'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*144.*', value=string_to_replace, regex=True)

string_to_replace='Tupolev 134'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*134.*', value=string_to_replace, regex=True)

df['Type'] = df['Type'].replace(to_replace='.*Tupelov.*134.*', value=string_to_replace, regex=True)

string_to_replace='Tupolev 124'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*124.*', value=string_to_replace, regex=True)

string_to_replace='Tupolev 114'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*114.*', value=string_to_replace, regex=True)

string_to_replace='Tupolev 104'

df['Type'] = df['Type'].replace(to_replace='.*Tupolev.*104.*', value=string_to_replace, regex=True)





ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Zeppelin.*',na=False)].unique())

string_to_replace='Zeppelin (airship) '

df['Type'] = df['Type'].replace(to_replace='.*Zeppelin.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Douglas.*',na=False)].unique())

string_to_replace='Douglas DC-3 '

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*3.*', value=string_to_replace, regex=True)

ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Douglas.*',na=False)].unique())

string_to_replace='Douglas DC-47'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*47.*', value=string_to_replace, regex=True)

string_to_replace='Douglas DC-4'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*4.*', value=string_to_replace, regex=True)

string_to_replace='Douglas DC-2'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*2.*', value=string_to_replace, regex=True)

string_to_replace='Douglas DC-6'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*6.*', value=string_to_replace, regex=True)

string_to_replace='Douglas DC-7'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*7.*', value=string_to_replace, regex=True)

string_to_replace='Douglas DC-8'

df['Type'] = df['Type'].replace(to_replace='.*Douglas.*D?C.*8.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Yak.*',na=False)].unique())

string_to_replace='Yak 42'

df['Type'] = df['Type'].replace(to_replace='.*Yak.*42.*', value=string_to_replace, regex=True)

string_to_replace='Yak 40'

df['Type'] = df['Type'].replace(to_replace='.*Yak.*40.*', value=string_to_replace, regex=True)





ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Mi.*',na=False)].unique())

string_to_replace='Mil 8'

df['Type'] = df['Type'].replace(to_replace='.*Mi.*8.*', value=string_to_replace, regex=True)

string_to_replace='Mil 17'

df['Type'] = df['Type'].replace(to_replace='.*Mi.*17.*', value=string_to_replace, regex=True)



ariplane_type=pd.DataFrame(df.Type[df.Type.str.contains('.*Lear*',na=False)].unique())

string_to_replace='Learjet 25'

df['Type'] = df['Type'].replace(to_replace='.*Learjet.*25.*', value=string_to_replace, regex=True)

string_to_replace='Learjet 24'

df['Type'] = df['Type'].replace(to_replace='.*Learjet.*24.*', value=string_to_replace, regex=True)

string_to_replace='Learjet 23'

df['Type'] = df['Type'].replace(to_replace='.*Learjet.*23.*', value=string_to_replace, regex=True)

string_to_replace='Learjet 35'

df['Type'] = df['Type'].replace(to_replace='.*Learjet.*35.*', value=string_to_replace, regex=True)

string_to_replace='Learjet 45'

df['Type'] = df['Type'].replace(to_replace='.*Learjet.*45.*', value=string_to_replace, regex=True)





# create new feature Manufacturer

df_type=pd.DataFrame(df['Type'].unique())

df['Manufacturer']=df['Type'].str.split().str.get(0)

string_to_replace='de Havilland'

df['Manufacturer'] = df['Manufacturer'].replace(to_replace='De', value=string_to_replace, regex=True)

# crashes by year

by_year = df[['Year','Fatalities']].groupby('Year').aggregate(['sum','count'])

by_year_crashes_count = by_year['Fatalities','count']

by_year_crashes_count.plot(kind='bar',title='Crashes by by year',grid=True,rot=90)
# by year aboard

by_year_aboard = df[['Year','Aboard']].groupby('Year').aggregate(['sum'])

by_year_aboard.plot(kind='bar',title='Fatalities by year',grid=True,rot=90)

# survivors by year

by_year_survivors = df[['Year','Survivors']].groupby('Year').aggregate(['sum'])

by_year_survivors.plot(kind='bar',title='Fatalities by year',grid=True,rot=90)



# survival ratio by tear

survival_ratio=by_year_survivors.ix[:,0]/by_year_aboard.ix[:,0]

survival_ratio.plot(kind='bar',title='Fatalities by year',grid=True,rot=90)



# tp 10 types where

type = df[['Type','Fatalities','Survivors']].groupby('Type').aggregate(['sum','count'])

accidents_t = type['Fatalities','sum'].sort_values(ascending=False)

accidents_t.head(10).plot(kind='pie',title='Top 10 airplane killers',grid=True,rot=90)

#top 10 crashers

type = df[['Type','Fatalities','Survivors']].groupby('Type').aggregate(['sum','count'])

accidents_t = type['Fatalities','count'].sort_values(ascending=False)

accidents_t.head(10).plot(kind='pie',title='Top 10 airplane  crasheres',grid=True,rot=90)


Opearator = df[['Operator','Fatalities']].groupby('Operator').aggregate(['sum','count'])

Opearator_m = Opearator['Fatalities','sum'].sort_values(ascending=False)

Opearator_m.head(10).plot(kind='pie',title='Top 10 killers Airlines',grid=True,rot=90)



#Top 10 crashing airlines

Opearator = df[['Operator','Fatalities']].groupby('Operator').aggregate(['sum','count'])

Opearator_m = Opearator['Fatalities','count'].sort_values(ascending=False)

Opearator_m.head(10).plot(kind='pie',title='Top 10 crashing Airlines',grid=True)

#Remove unknown cause of accident

Summary=df.Summary.loc[df.Summary!='UNKOWN']

Summary.replace('\'','')

Summary_idx=df.loc[df.Summary!='UNKOWN']

Summary_unknown=df.Summary.loc[df.Summary=='UNKOWN']

Summary=Summary.str.decode('utf-8')

#Summary=Summary.tolist()





# Define stopwords

# add words obviously related to airplane cras. Sicne they carry little information about the cause

# and are expected to describe outcome rather the cause

# for phyton 2.7 add .decode('utf-8') to string :)

stopwords = list(nltk.corpus.stopwords.words('english'))

stopwords.append(str('crashed'))

stopwords.append(str('plane'))

stopwords.append(str('after'))

stopwords.append(str('into'))

stopwords.append(str('airplane'))

stopwords.append(str('aircraft'))

stopwords.append(str('\''))                 

stopwords.append(str('crashed'))

stopwords.append(str('plane'))

stopwords.append(str('after'))

stopwords.append(str('into'))

stopwords.append(str('airplane'))

stopwords.append(str('pilot'))

stopwords.append(str('airplane'))

stopwords.append(str('en'))

stopwords.append(str('air'))

stopwords.append(str('ft.'))

stopwords.append(str('feet.'))

stopwords.append(str('feet'))

stopwords.append(str('crashing'))

stopwords.append(str('aircraft'))

stopwords.append(str('crash'))

stopwords.append(str('flight'))

stopwords.append(str('hit'))

stopwords.append(str('contributing'))

stopwords.append(str('miles'))

stopwords.append(str('km'))

stopwords.append(str('killed'))

stopwords.append(str('lake'))

stopwords.append(str('guardia'))

stopwords.append(str('killing'))

stopwords.append(str('accident'))

stopwords.append(str('accident'))

stopwords.append(str('burst'))

stopwords.append(str('000'))

stopwords.append(str('impacted'))

stopwords.append(str('\''))      

                 

# define tokenizer

# borrowed from http://brandonrose.org/clustering#K-means-clustering

# no setmming is required since this would cause loss of information

def tokenize_only(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token

    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    return filtered_tokens
