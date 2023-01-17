import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





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
d.head()
# Just take Traffic Vehicle Accidents...

v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -')]

v.head()
p=pd.pivot_table(v, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every week 'W'.  This is very powerful

pp=p.resample('W', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Show values

# Note, last week might not be a full week

pp.tail(3)
# Drop the last week

pp=pp[(pp['timeStamp'] != pp['timeStamp'].max())]

pp.count()
# Plot this out

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left')









ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')







ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /week")

#fig.autofmt_xdate()

plt.show()
p=pd.pivot_table(v, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every month 'M'.  

pp=p.resample('M', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Show a few values

# Note, last and first readings might not be a full capture

pp.head(-1)
# Plot this out

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left')









ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')







ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /month")

#fig.autofmt_xdate()

plt.show()


# We're taking out the last month

pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]

# Plot this out

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left') 



ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')



ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /month")



#fig.autofmt_xdate()

plt.show()
pp2.head(10)
c=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]

c.head()
# Create pivot

p=pd.pivot_table(c, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every month 'M'.  

pp=p.resample('M', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)





# Show a few values

pp.head(10)
# We're taking out the last month

pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]

# Plot this out

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left') 



ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')



ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nCheltenham /month")

#fig.autofmt_xdate()

plt.show()
a=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON')]

a.head()
# Create pivot

pa=pd.pivot_table(a, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every month 'M'.  

ppA=pa.resample('M', how=[np.sum]).reset_index()

ppA.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

ppA.columns = ppA.columns.get_level_values(0)





# Show a few values

ppA.head(3)
# We're taking out the last month

ppA2 = ppA[(ppA['timeStamp'] != ppA['timeStamp'].max())]

# Plot this out

fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left')



# Abington

ax.plot_date(ppA2['timeStamp'], ppA2['Traffic: VEHICLE ACCIDENT -'],'g')

ax.plot_date(ppA2['timeStamp'], ppA2['Traffic: VEHICLE ACCIDENT -'],'go')



# Cheltenham

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'r')

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')





ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAbington /month (Green)"+

            "\nCheltenham /month (Red)")

#fig.autofmt_xdate()

plt.show()
# Vehicle Accident 

# Put this in a variable 'g'

g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)



# Check data if you want

p.head()



cmap = sns.cubehelix_palette(light=2, as_cmap=True)

ax = sns.heatmap(p,cmap = cmap)

ax.set_title('Vehicle  Accidents - Cheltenham Townships ');
# Vehicle Accident 

# Put this in a variable 'g'

g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON')]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)



# Check data if you want

p.head()



cmap = sns.cubehelix_palette(light=2, as_cmap=True)

ax = sns.heatmap(p,cmap = cmap)

ax.set_title('Vehicle  Accidents - Abington Townships ');
# Vehicle Accident 

# Put this in a variable 'g'

g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp != 'ABINGTON') & (d.twp != 'CHELTENHAM')         ]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)



# Check data if you want

p.head()



cmap = sns.cubehelix_palette(light=2, as_cmap=True)

ax = sns.heatmap(p,cmap = cmap)

ax.set_title('Vehicle  Accidents - Montco Townships\n (Except Abington + Cheltenham) ' );
c = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM') ]

c['Month'] = c['timeStamp'].apply(lambda x: x.strftime('%m %B'))

c['Hour'] = c['timeStamp'].apply(lambda x: x.strftime('%H'))





c = c[(c.Month == '07 July') & (c.Hour == '17')]

c.head()
c['zip'].value_counts()

# Interesting... 19027 is Elkins Park. A SEPTA train stop
a = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON') ]

a['Month'] = a['timeStamp'].apply(lambda x: x.strftime('%m %B'))

a['Hour'] = a['timeStamp'].apply(lambda x: x.strftime('%H'))





a = a[(a.Month == '07 July') & (a.Hour == '17')]

a.head()
a['zip'].value_counts()