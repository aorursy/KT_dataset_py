import pandas as pd

import numpy as np

import glob

import copy

import datetime

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
glob.glob("*")
df=pd.read_csv("../input/crime.csv")
print (df.shape, '\n\n', df.columns, '\n\n', df.ftypes)
for i in df.columns:

    print (i, df[i][1], round(len(df[df[i].isnull()])/float(len(df)),2))
DF = copy.deepcopy(df[[u'Dc_Dist', u'Dispatch_Date', u'Dispatch_Time', u'Text_General_Code', u'Lon', u'Lat']])



for i in DF.columns:

    

    DF=DF[DF[i].isnull()==False]



DF.columns = [u'DISTRICT', u'DATE', u'TIME', u'CAUSE', u'LON', u'LAT']

DF.index=np.arange(len(DF))

DF['TIME'] = DF['TIME'].apply(lambda x: datetime.datetime.strptime(x,'%H:%M:%S').time())

DF['DATE']=pd.to_datetime(DF['DATE'])

DF['COUNT']=np.ones(len(DF))



YRMN=min(DF['DATE'].dt.year)

YRMX=min(DF['DATE'].dt.year)



DF['MTH_COUNT']=DF['DATE'].dt.month+(DF['DATE'].dt.year-YRMN)*12



DF.head(5)
text_opts={'fontsize':20,'fontweight':'bold'}



interval=11



def movingaverage(interval, window_size):

    window= np.ones(int(window_size))/float(window_size)

    return np.convolve(interval, window, 'same')
size=7



causes = DF.groupby('CAUSE')['COUNT'].count().sort_values(ascending=False)[:size].index



plt.figure(figsize=(20,10))



handle=[]



for i in causes:

    

    dummy_df = DF[DF['CAUSE']==i] 

    

    dummy_by_date = dummy_df.groupby('MTH_COUNT')['COUNT'].sum().reset_index()

    

#     plt.plot_date(dummy_by_date['MTH_COUNT'],dummy_by_date['COUNT'], '-', linewidth=5)

    plt.plot_date(dummy_by_date['MTH_COUNT'],movingaverage(dummy_by_date['COUNT'],6), '-', linewidth=3)

    

    handle=handle+['%s' %i]

    

plt.legend(handle)





xmin=min(dummy_df['MTH_COUNT'])

xmax=max(dummy_df['MTH_COUNT'])



plt.xticks(np.arange(xmin,xmax+1,12), np.arange((interval), dtype=int)+2006, **text_opts)



plt.yticks(**text_opts)

plt.grid()

plt.title('monthly count: top %d crimes\n(mvg avg 6 mths)' %size, **text_opts)
size=5



pds = DF.groupby('DISTRICT')['COUNT'].count().sort_values(ascending=False)[:size].index



plt.figure(figsize=(20,10))



handle=[]



for i in pds:

    

    dummy_df = DF[DF['DISTRICT']==i] 

    

    dummy_by_date = dummy_df.groupby('MTH_COUNT')['COUNT'].sum().reset_index()

    

    #plt.plot_date(dummy_by_date['MTH_COUNT'],dummy_by_date['COUNT'])

    

    plt.plot_date(dummy_by_date['MTH_COUNT'],movingaverage(dummy_by_date['COUNT'],6), '-', linewidth=3)

    

    handle=handle+['%s' %i]

    

plt.legend(handle)



xmin=min(dummy_df['MTH_COUNT'])

xmax=max(dummy_df['MTH_COUNT'])



plt.xticks(np.arange(xmin,xmax+1,12), np.arange((interval), dtype=int)+2006, **text_opts)



plt.yticks(**text_opts)

plt.grid()

plt.title('monthly count: top %d police depts\n(mvg avg 6mths)' %size, **text_opts)
causes = DF.groupby('CAUSE')['COUNT'].count().sort_values(ascending=False).index



for i in causes:

    

    dummy_df = DF[DF['CAUSE']==i] 

    

    dummy_by_date = dummy_df.groupby('MTH_COUNT')['COUNT'].sum().reset_index()

    

    plt.figure(figsize=(20,10))

    

    plt.plot_date(dummy_by_date['MTH_COUNT'],dummy_by_date['COUNT'])

    plt.plot_date(dummy_by_date['MTH_COUNT'], movingaverage(dummy_by_date['COUNT'],6),'-')

    

    plt.title('%s' %i, **text_opts)

    plt.xticks(np.arange(xmin,xmax+1,12), np.arange((interval), dtype=int)+2006, **text_opts)

    plt.yticks(**text_opts)

    plt.grid()
pds = DF.groupby('DISTRICT')['COUNT'].count().sort_values(ascending=False).index



x=np.linspace(2006,2017,100)



for i in pds:

    

    dummy_df = DF[DF['DISTRICT']==i] 

    

    dummy_by_date = dummy_df.groupby('MTH_COUNT')['COUNT'].sum().reset_index()

    

    plt.figure(figsize=(20,10))

    

    plt.plot_date(dummy_by_date['MTH_COUNT'],dummy_by_date['COUNT'])

    plt.plot_date(dummy_by_date['MTH_COUNT'], movingaverage(dummy_by_date['COUNT'],6),'-')

    

    plt.title('Police department: %s' %i, **text_opts)

    plt.xticks(np.arange(xmin,xmax+1,12), np.arange((interval), dtype=int)+2006, **text_opts)

    plt.yticks(**text_opts)

    plt.grid()