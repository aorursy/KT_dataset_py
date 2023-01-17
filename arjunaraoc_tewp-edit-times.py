# Telugu wikipedia edit times 
# inputs
# --Recent changes of   edits for the recent 1 day through wikipedia API


# Version History
# V25 investigate why the published picture differes from edit mode output
# V16 fix bug when published need offline initialisation
# V15 Single cell code to manage hiding
# V13 added number of users in one hour period
# V12 working for one day of edits
# V5 datetime as index of df
# V4 working version

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import json
import os
import folium
from datetime import datetime, timedelta
import dateutil.parser
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

class Error(Exception):
    """Base class for exceptions in this module."""
    pass
now=datetime.now()
td=timedelta(days=-1)
nowminus1d=now+td
rcstart=now.isoformat()[0:10]+"T00:00:00"
rcend=nowminus1d.isoformat()[0:10]+"T00:00:00"

# declare list to keep api results
rcl=[]
myrequest={'list':'recentchanges',
          'rcprop':'title|timestamp|user',
          'rcshow':'!bot',
          'rcstart':rcstart,
          'rcend' :rcend
        }



def query(request):
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get('https://te.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']

for result in query(myrequest):
    # process result data
    rcl.extend(result['recentchanges'])
rc=pd.DataFrame(rcl)
rc.timestamp= [dateutil.parser.parse(x) for x in rc.timestamp]
#just pick timestamp
editdf=pd.DataFrame(index=rc.timestamp)
editdf['count']=[1]*len(rc.timestamp)
# Hourly breakup
editdfr=editdf.resample("H").sum()


# specify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Bar(x=editdfr.index, y=editdfr['count'])]

# specify the layout of our figure
layout = dict(title = "Time histogram of edits on  Telugu Wikipedia",
              xaxis= dict(title= 'Time in UTC Period-15min',zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


#find number of unique users in a period
rcp=rc.loc[:,['timestamp','user','anon']]
# Group to hourly interval
rcp.timestamp=rcp.timestamp.dt.to_period(freq="H")
rcp=rcp.drop_duplicates()

import re
userl=rcp.user.unique()
#IP addresses  will have four .s or seven :
IP_REGEXP = "\.|:"
ipuser=[len(re.findall(IP_REGEXP,x)) in [4,7]  for x in userl]
print("Total number of unique users in the period: %d" % len(rcp.user.unique()))
print("Total number of IP users in the period: %d" % ipuser.count(True) )
rcp['ip']=rcp['anon']==''
rcusers=rcp.groupby(['timestamp','ip'], sort=True).size().unstack(fill_value=0)
trace1 = go.Bar(x=rcusers.index.to_timestamp(), y=rcusers[False],name="Logged in")
trace2 = go.Bar(x=rcusers.index.to_timestamp(), y=rcusers[True],name="Anonymous")
data=[trace1,trace2]
# specify the layout of our figure
layout = dict(title = "Histogram of number of users on  Telugu Wikipedia",
              xaxis= dict(title= 'Time in UTC-Period(1hr)',zeroline= True),barmode='stack')

# create and show our figure
fig3 = dict(data = data, layout = layout)
iplot(fig3)