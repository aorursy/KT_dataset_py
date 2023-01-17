import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# You may want to define dtypes and parse_dates for timeStamps

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



d=pd.read_csv("../input/trainView.csv",

    header=0,names=['train_id','status','next_station','service','dest','lon',

                    'lat','source','track_change','track','date','timeStamp0',

                    'timeStamp1','seconds'],

    dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,

    'lon':str,'lat':str,'source':str,'track_change':str,'track':str,'date':str,

    'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime,'seconds':str}, 

     parse_dates=['timeStamp0','timeStamp1'],date_parser=dateparse)





def getDeltaTime(x):

    r=(x[1] - x[0]).total_seconds() 

    return r



# It might make sense to add delta_s to the next version

d['delta_s']=d[['timeStamp0','timeStamp1']].apply(getDeltaTime, axis=1)
d.head()
# Train: 319

# Day:  2016-05-23



d[(d['train_id']=='319') & (d['date']=='2016-05-17')].sort_values(by='timeStamp0',ascending=True)