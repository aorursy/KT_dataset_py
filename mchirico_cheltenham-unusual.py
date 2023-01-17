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
c=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]

c.head()



# Create pivot

p=pd.pivot_table(c, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



# Resampling every 2 weeks '2W'.  

pp=p.resample('2W', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Sort

pp.sort_values(by=['timeStamp'],ascending=True,inplace=True)



# Show a few values

pp.head()
# Take out first and last values because they may not be complete sets

pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]

pp2 = pp2[(pp2['timeStamp'] != pp2['timeStamp'].min())]







# Plot this out

from sklearn import linear_model

import matplotlib.lines as mlines



fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left') 





# Adjust Data

# Make sure you sort for correct (mx and b values).

pp2.sort_values(by=['timeStamp'],ascending=True,inplace=True)



Y = pp2['Traffic: VEHICLE ACCIDENT -'].values.reshape(-1,1)

X = np.arange(Y.shape[0]).reshape(-1,1)





# Build Linear Fit

model = linear_model.LinearRegression()

model.fit(X,Y)



mx = model.coef_[0][0]

b = model.intercept_[0]





ax.plot(pp2['timeStamp'],model.predict(X), color='red',

         linewidth=2)





blue_line = mlines.Line2D([], [], color='red', label='Linear Fit: y = %2.2fx + %2.2f' % (mx,b))







ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')



ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nCheltenham /2 week")

ax.legend(handles=[blue_line], loc='best')

#fig.autofmt_xdate()

plt.show()
pp2['Traffic: VEHICLE ACCIDENT -'].pct_change()



# Plot this out

from sklearn import linear_model

import matplotlib.lines as mlines



fig, ax = plt.subplots()



ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)  







ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

plt.xticks(fontsize=12,rotation=45,ha='left') 





# Adjust Data

# Make sure you sort for correct (mx and b values).

pp2.sort_values(by=['timeStamp'],ascending=True,inplace=True)







ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'].pct_change(),'k')

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'].pct_change(),'ro')



ax.set_title("Traffic: VEHICLE ACCIDENT (Percent Change)-"+"\nCheltenham /2 week")

#ax.legend(handles=[blue_line], loc='best')

#fig.autofmt_xdate()

plt.show()
#  v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]





def myPltFunction(v,title):

    



    # Create pivot

    p=pd.pivot_table(v, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)



    # Resampling every 2 weeks '2W'.  

    pp=p.resample('2W', how=[np.sum]).reset_index()

    pp.sort_values(by='timeStamp',ascending=False,inplace=True)



    # Let's flatten the columns 

    pp.columns = pp.columns.get_level_values(0)

    

    # Take out ends - may not contain complete set

    pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]

    pp2 = pp2[(pp2['timeStamp'] != pp2['timeStamp'].min())]







    # Plot this out

    from sklearn import linear_model

    import matplotlib.lines as mlines



    fig, ax = plt.subplots()



    ax.spines["top"].set_visible(False)    

    ax.spines["bottom"].set_visible(False)    

    ax.spines["right"].set_visible(False)    

    ax.spines["left"].set_visible(False)  







    ax.get_xaxis().tick_bottom()    

    ax.get_yaxis().tick_left() 

    plt.xticks(fontsize=12,rotation=45,ha='left') 





    # Adjust Data

    # Make sure you sort for correct (mx and b values).

    pp2.sort_values(by=['timeStamp'],ascending=True,inplace=True)



    Y = pp2['Traffic: VEHICLE ACCIDENT -'].values.reshape(-1,1)

    X = np.arange(Y.shape[0]).reshape(-1,1)





    # Build Linear Fit

    model = linear_model.LinearRegression()

    model.fit(X,Y)



    mx = model.coef_[0][0]

    b = model.intercept_[0]





    ax.plot(pp2['timeStamp'],model.predict(X), color='red',

         linewidth=2)





    blue_line = mlines.Line2D([], [], color='red', label='Linear Fit: y = %2.2fx + %2.2f' % (mx,b))







    ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')

    ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')



    ax.set_title("Traffic: VEHICLE ACCIDENT -\n"+title)

    ax.legend(handles=[blue_line], loc='best')

    #fig.autofmt_xdate()

    plt.show()



    
v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON')]

myPltFunction(v,title='Abington')
v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'SPRINGFIELD')]

myPltFunction(v,title='Springfield')
# Is this correct? Increase everywhere?

v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') ]

myPltFunction(v,title='All Montco')