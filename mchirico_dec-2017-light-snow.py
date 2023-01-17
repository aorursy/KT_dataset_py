import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



from sklearn import linear_model

import matplotlib.lines as mlines





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

d.index = pd.DatetimeIndex(d.timeStamp)

d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
t=d[(d.timeStamp >= "2017-12-29 00:00:00")]

g=t[t.title.str.match(r'.*VEHICLE.*')]

p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every week 'W'.  This is very powerful

pp=p.resample('2H', how=[np.sum]).reset_index()

pp



# That "sum" column is a pain...remove it

# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)

pp.sort_values(by=['timeStamp'],ascending=False,inplace=True)

pp.head()

pp
#  pp[('Traffic: VEHICLE ACCIDENT -')].pct_change(periods=1)
# Doing this again for graphing



t=d[(d.timeStamp >= "2017-12-20 00:00:00")]

g=t[t.title.str.match(r'.*VEHICLE.*')]

p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every week 'W'.  This is very powerful

pp=p.resample('1H', how=[np.sum]).reset_index()



pp.sort_values(by=['timeStamp'],ascending=False,inplace=True)

# That "sum" column is a pain...remove it

# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)

pp.head()
# Red dot with Line

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12) 







ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')

#ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'ro')





ax.set_title("Traffic: VEHICLE ACCIDENT")

fig.autofmt_xdate()

plt.show()



# Note, you'll get a drop at the ends...not a complete week
# FALL VICTIM

t=d[(d.timeStamp >= "2017-12-20 00:00:00")]

g=t[t.title.str.match(r'.*FALL VICTIM.*')]

p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every week 'W'.  This is very powerful

pp=p.resample('2H', how=[np.sum]).reset_index()





pp.sort_values(by=['timeStamp'],ascending=False,inplace=True)

# That "sum" column is a pain...remove it

# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)

pp.head(10)







# DISABLED VEHICLE

t=d[(d.timeStamp >= "2017-12-20 00:00:00")]

g=t[t.title.str.match(r'.*DISABLED VEHICLE.*')]

p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every week 'W'.  This is very powerful

pp=p.resample('2H', how=[np.sum]).reset_index()



pp.sort_values(by=['timeStamp'],ascending=False,inplace=True)

# That "sum" column is a pain...remove it

# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)

pp.head(10)



# Vehicle Accident -- yes, there is FIRE; maybe we should have include?

# Put this in a variable 'g'

g = d[(d.title.str.match(r'EMS:.*VEHICLE ACCIDENT.*') | d.title.str.match(r'Traffic:.*VEHICLE ACCIDENT.*'))]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

p.head()
cmap = sns.cubehelix_palette(light=2, as_cmap=True)

ax = sns.heatmap(p,cmap = cmap)

ax.set_title('Vehicle  Accidents - All Townships ');