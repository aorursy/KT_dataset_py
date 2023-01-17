# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from pandas.io.json import json_normalize
import gc
import datetime as dt
%matplotlib inline

print(os.listdir("../input"))

data=pd.read_csv("../input/service-requests-received-by-the-oakland-call-center.csv",nrows=5)

cols=data.columns.str.lower()

data=pd.read_csv("../input/service-requests-received-by-the-oakland-call-center.csv",names=cols,skiprows=1,dtype={"requestid":str},parse_dates=['datetimeinit','datetimeclosed'])

address=json_normalize(data.reqaddress.apply(lambda x:ast.literal_eval(x)))

human_add=json_normalize(address.human_address.apply(lambda x:ast.literal_eval(x)))

human_add=human_add.add_prefix("address_")

data=data.join([address,human_add])


data.drop(['reqaddress','human_address'],axis='columns',inplace=True)

data.head()

data.set_index('datetimeinit',inplace=True)

per_day_request=data.resample('M')['requestid'].count().to_frame().reset_index()
per_day_request.head()

data.info()

data['time_to_close']=pd.to_timedelta(data.datetimeclosed-data.index).apply(lambda x:x.days)

gc.collect()

closed=data.groupby(np.isnan(data.time_to_close))['requestid'].count()
fig=plt.figure(figsize=(24,8))
ax1=fig.add_subplot(121)
ax1.pie(closed.values,labels=['closed','open'],explode=[0,0.1],autopct='%1.1f%%')
ax1.axis('equal')
ax2=fig.add_subplot(122)
sns.scatterplot(data=per_day_request,x="datetimeinit",y='requestid',ax=ax2)
ax2.set_xlim(per_day_request.datetimeinit.min()-dt.timedelta(days=60),per_day_request.datetimeinit.max()+dt.timedelta(days=60))
# ax2.set_xlabel("Dates")
# ax2.set_ylabel("Total_request_made")
ax2.set(title="Requests created per day",xlabel="Dates",ylabel="Total_request_made")
plt.show()
from plotly import __version__
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
print(__version__)
labels=['Closed issue','Open issues']
trace0=go.Pie(labels=labels,values=closed.values,
             hoverinfo="label+percent+value",
             domain={'x':[0,0.48]},
             name='pie')

iplot([trace0])
trace1=go.Scatter(x=per_day_request.datetimeinit,
                  y=per_day_request.requestid,
                  mode="markers",
                 name="Per_Day_Requests")
data=[trace1]
layout=go.Layout({'title':"per day calls","xaxis":{'title':"Month Year"},"yaxis":{'title':"Total calls received"}})
fig=go.Figure(data=data,layout=layout)
iplot(fig)

