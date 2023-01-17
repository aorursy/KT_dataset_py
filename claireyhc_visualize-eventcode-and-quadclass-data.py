import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
eventcode_file="../input/eventcode1.csv"
df_code=pd.read_csv(eventcode_file)
df_code.head()
# number of rows
len (df_code)
df_code.dtypes
#rename the count column
df_code.rename(columns={'_c2':'count'},inplace=True)
df_code.tail(5)
#total number of events by country
totevents=df_code.groupby('actor1countrycode')['count'].sum()
print (totevents)
totevents.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of events')
plt.title('Number of events by Country')
#divide up large dataframe into separate ones for each country
NZL=df_code[df_code['actor1countrycode']=='NZL']
AUS=df_code[df_code['actor1countrycode']=='AUS']
BEL=df_code[df_code['actor1countrycode']=='BEL']
JPN=df_code[df_code['actor1countrycode']=='JPN']
IND=df_code[df_code['actor1countrycode']=='IND']
FRA=df_code[df_code['actor1countrycode']=='FRA']
#sort from highest to lowest count of each eventcode and create a dataframe for each country's top 5 eventcodes
NZL1=NZL.sort_values('count',ascending=False)
NZLtop_e = NZL1.head()
AUS1=AUS.sort_values('count',ascending=False)
AUStop_e = AUS1.head()
BEL1=BEL.sort_values('count',ascending=False)
BELtop_e= BEL1.head()
JPN1=JPN.sort_values('count',ascending=False)
JPNtop_e = JPN1.head()
IND1=IND.sort_values('count',ascending=False)
INDtop_e = IND1.head()
FRA1=FRA.sort_values('count',ascending=False)
FRAtop_e = FRA1.head()
NZLtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - New Zealand')

AUStop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Australia')

BELtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Belgium')

JPNtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Japan')

INDtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - India')

FRAtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - France')
quadclass_file="../input/quadclass_count.csv"
df_quad=pd.read_csv(quadclass_file)
df_quad.head()
df_quad.rename(columns={'_c2':'count'},inplace=True)
aus1=df_quad[df_quad['actor1countrycode']=='AUS']
bel1=df_quad[df_quad['actor1countrycode']=='BEL']
fra1=df_quad[df_quad['actor1countrycode']=='FRA']
ind1=df_quad[df_quad['actor1countrycode']=='IND']
jpn1=df_quad[df_quad['actor1countrycode']=='JPN']
nzl1=df_quad[df_quad['actor1countrycode']=='NZL']
aus1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Australia')
bel1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Belgium')
fra1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - France')
ind1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - India')
jpn1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Japan')
nzl1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - New Zealand')