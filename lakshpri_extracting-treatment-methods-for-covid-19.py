import pandas as pd

import numpy as np

import matplotlib.pyplot as plt   

import seaborn as sns

%matplotlib inline

import pandas as pd

filename = "../input/CORD-19-research-challenge/metadata.csv"

dt = pd.read_csv(filename)
dt.shape
dt = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv',index_col='cord_uid')
dt['publish_time'] = pd.to_datetime(dt['publish_time'])
start_date = '2019-11-01'

end_date = '2020-12-31'
mask = (dt['publish_time'] > start_date) & (dt['publish_time'] <= end_date)
dt = dt.loc[mask]

dt
df=dt[dt['abstract'].str.contains('coronavirus', na = False)]
df.shape
df1=dt[dt['abstract'].str.contains('Coronavirus', na = False)]
df1.shape
df2=df[df['abstract'].str.contains('recovered', na = False)]
df2.shape
df2.to_excel (r'/kaggle/working/export_recovered_df2.xlsx', header=True)
df2a=df1[df1['abstract'].str.contains('recovered', na = False)]
df2a.shape
df2a.to_excel (r'/kaggle/working/export_recovered_df2a.xlsx', header=True)
T1=df2.loc['4q50tw3d',:]

T1.title
T1.abstract
T1
T2=df2.loc['gzc72bdy',:]

T2.title
T2.abstract
T2
T3=df2.loc['ghfrwccu',:]

T3.title

T3.abstract
T3
T4=df2.loc['8znnq0rh',:]

T4.title
T4.abstract
T4
T5=df2.loc['6g34qwer',:]

T5.title
T5.abstract
T5
df3=df[df['abstract'].str.contains('recovery', na = False)]
df3.shape
df3.to_excel (r'/kaggle/working/export_recovery_df3.xlsx', header=True)
df3a=df1[df1['abstract'].str.contains('recovery', na = False)]
df3a.shape
df3a.to_excel (r'/kaggle/working/export_recovery_df3a.xlsx', header=True)
T6=df3.loc['owjgs6ja',:]

T6.title
T6.abstract
T6
T7=df3.loc['wmfcwqfw',:]

T7.title
T7.abstract
T7
T8=df3.loc['14he8n3u',:]

T8.title
T8.abstract
T8
T9=df3.loc['b3y9zxjr',:]

T9.title
T9.abstract
T9
df4=df[df['abstract'].str.contains('treatment', na = False)]
df4.shape
df4.to_excel (r'/kaggle/working/export_treatment_df4.xlsx', header=True)
df4a=df1[df1['abstract'].str.contains('treatment', na = False)]
df4a.shape
df4a.to_excel (r'/kaggle/working/export_treatment_df4a.xlsx', header=True)
T10=df4.loc['3pxc5wot',:]

T10.title
T10.abstract
T10
T11=df4.loc['yf5g53a9',:]

T11.title

T11.abstract
T11
T12=df4.loc['u7zxlgxz',:]

T12.title

T12.abstract
T12
T13=df4.loc['juz9jnfk',:]

T13.title
T13.abstract
T13
T14=df4.loc['1vm5r7pq',:]

T14.title

T14.abstract
T14
T15=df4a.loc['41jqgsv0',:]

T15.title
T15.abstract
T15
T16=df4.loc['3egv50vb',:]

T16.title
T16.abstract
T16
T17=df4.loc['zb434ve3',:]

T17.title
T17.abstract
T17
T18=df4.loc['8caqxfxv',:]

T18.title

T18.abstract
T18
T19=df4a.loc['aonbvub5',:]

T19.title
T19.abstract
T19
T20=df4.loc['aoviner2',:]

T20.title
T20.abstract
T20
T21=df4.loc['95fc828i',:]

T21.title
T21.abstract
T21
T22=df4.loc['393k3lq6',:]

T22.title
T22.abstract
T22
T23=df4.loc['qdt90c22',:]

T23.title
T23.abstract
T23
T24=df4a.loc['plpwjj4s',:]

T24.title

T24.abstract
T24
T25=df4.loc['a5udnv5f',:]

T25.title
T25.abstract
T25
T26=df4.loc['lzda8kyn',:]

T26.title
T26.abstract
T26
T27=df4.loc['1qniriu0',:]

T27.title
T27.abstract
T27
T28=df4.loc['cszqykpu',:]

T28.title
T28.abstract
T28
T29=df4.loc['py6qu4tl',:]

T29.title
T29.abstract
T29
T30=df4.loc['0qwwycnc',:]

T30.title

T30.abstract
T30
T31=df4a.loc['9hh17k86',:]

T31.title
T31.abstract
T31
T32=df1.loc['g41pd9uz',:]

T32.title
T32.abstract
T32
T33=df4.loc['nsrm0axa',:]

T33.title

T33.abstract
T33
T34=df4.loc['23mp5961',:]

T34.title

T34.abstract
T34
T35=df4.loc['e99athff',:]

T35.title
T35.abstract
T35
T36=df4.loc['hq5um68k',:]

T36.title

T36.abstract
T36
T37=df4.loc['5hpbjkft',:]

T37.title

T37.abstract
T37
T38=df4.loc['vi51uons',:]

T38.title
T38.abstract
T38
T39=df4.loc['ksbha7kz',:]

T39.title
T39.abstract
T39
T40=df4.loc['nnlynjoy',:]

T40.title
T40.abstract
T40
T41=df4.loc['t1wpujpm',:]

T41.title
T41.abstract
T41
T42=df4.loc['ptnmtvzj',:]

T42.title
T42.abstract
T42
T43=df4.loc['azvbl4ie',:]

T43.title
T43.abstract
T43
T44=df4.loc['0lk8eujq',:]

T44.title
T44.abstract
T44
T45=df4.loc['jbc74lcu',:]

T45.title
T45.abstract
T45
T46=df4.loc['m5k28kbu',:]

T46.title
T46.abstract
T46
T47=df4a.loc['5l6c0it4',:]

T47.title
T47.abstract
T47
T48=df4.loc['0gier0lu',:]

T48.title
T48.abstract
T48
T49=df4.loc['x50tvq3a',:]

T49.title
T49.abstract
T49
T50=df4.loc['7k36owrf',:]

T50.title

T50.abstract
T50