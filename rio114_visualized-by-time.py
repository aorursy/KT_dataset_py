#handling datarame

import pandas as pd

from pandas import Series ,DataFrame

import numpy as np



#drawing charts

import matplotlib.pyplot as plt

#plt.xkcd()

import seaborn as sns

%matplotlib inline



import datetime

df = pd.read_csv('../input/database.csv')

df.head()
df.info()
def myDateTime (date):

    time_format = '%m/%d/%Y %I:%M %p'

    date_obj = datetime.datetime.strptime(date, time_format)

    return date_obj



def Weekday(date):

    wday_dum = date.weekday()

    if wday_dum is 0:

        wday = 'Mon'

    elif wday_dum is 1:

        wday = 'Tue'

    elif wday_dum is 2:

        wday = 'Wed'

    elif wday_dum is 3:

        wday = 'Thu'

    elif wday_dum is 4:

        wday = 'Fri'

    elif wday_dum is 5:

        wday = 'Sat'

    else:

        wday = 'Sun'

    return wday



Hour = lambda x: x.hour

Day = lambda x: x.day

Month = lambda x: x.month

Year = lambda x: x.year



df['Time_obj'] = df['Accident Date/Time'].apply(myDateTime)



cat = [Year,Month,Day,Hour]

num = len(cat)



plt.figure(figsize=(10,10))

for (i, c) in enumerate(cat):

    plt.subplot(num+1,1,i+1)

    sns.countplot(df['Time_obj'].apply(c))

    plt.xlabel('')

    plt.ylabel('accident cnt')



plt.subplot(num+1,1,5)

sns.countplot(df['Time_obj'].apply(Weekday),order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

plt.xlabel('')

plt.ylabel('accident cnt')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
# Price data is referred 'WTI Cushing Oklahoma' from http://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm

# the data is averaged by month and pasted here



'''

price = pd.read_csv('Spot_Prices.csv')

price.columns = ['Day', 'Cushing']



def myDate (date):

    time_format = '%m/%d/%Y'

    date_obj = datetime.datetime.strptime(date, time_format)

    return date_obj



price['Day_obj'] = price['Day'].apply(myDate)



Y = np.arange(2010,2017,1)

M = np.arange(1,13,1)



dummy = np.zeros([len(Y),len(M)])



for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        dummy[i,j] =  price[price['Day_obj'].apply(Year)==y][price['Day_obj'].apply(Month)==m].mean()



price_month = Series(dummy.reshape([len(Y)*len(M)]))

'''



price_month = Series([ 78.32578947,   76.38736842,   81.20347826,   84.29285714,

         73.7435    ,   75.33590909,   76.31952381,   76.59909091,

         75.24190476,   81.89285714,   84.25285714,   89.14590909,

         89.1705    ,   88.57842105,  102.85652174,  109.5325    ,

        100.90047619,   96.26409091,   97.3035    ,   86.33304348,

         85.5152381 ,   86.32238095,   97.16047619,   98.56285714,

        100.2735    ,  102.204     ,  106.15772727,  103.321     ,

         94.65454545,   82.30333333,   87.8952381 ,   94.13130435,

         94.51368421,   89.49130435,   86.53142857,   87.8595    ,

         94.75666667,   95.30894737,   92.9385    ,   92.02136364,

         94.50954545,   95.7725    ,  104.67090909,  106.57272727,

        106.2895    ,  100.53826087,   93.864     ,   97.6252381 ,

         94.61714286,  100.81736842,  100.80380952,  102.06904762,

        102.17714286,  105.79428571,  103.58863636,   96.53619048,

         93.314     ,   83.77952381,   75.78947368,   59.29045455,

         47.219     ,   50.58421053,   47.82363636,   54.45285714,

         59.265     ,   59.81954545,   51.16304348,   42.86761905,

         45.50409091,   46.22363636,   42.3852381 ,   37.20652174,

         31.9555    ,   30.323     ,   37.54636364,   40.7552381 ,

         46.71238095,   48.75727273,   44.6515    ,   44.72434783,

         45.14636364,   49.7752381 ,   45.66095238,   51.97047619])
Y = np.arange(2010,2017,1)

M = np.arange(1,13,1)



dummy = np.zeros([len(Y),len(M)])



for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        dummy[i,j] =  df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['Report Number'].count()

count_month = Series(dummy.reshape([len(Y)*len(M)]))
dummy = np.zeros([len(Y),len(M)])

for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        #dummy[i,j] =  np.log10(df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['All Costs'].sum())

        dummy[i,j] =  df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['All Costs'].sum()

cost_month = Series(dummy.reshape([len(Y)*len(M)]))

cost_month_ave = cost_month/count_month

cost_month_ave_log = np.log10(cost_month_ave)
dummy = np.zeros([len(Y),len(M)])

for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        #dummy[i,j] =  np.log10(df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['Net Loss (Barrels)'].sum())

        dummy[i,j] =  df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['Net Loss (Barrels)'].sum()

loss_month = Series(dummy.reshape([len(Y)*len(M)]))

loss_month_ave = loss_month/count_month

loss_month_ave_log = np.log10(loss_month_ave)
dummy = np.zeros([len(Y),len(M)])

for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        dummy[i,j] =  df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['All Injuries'].sum()



injury_month = Series(dummy.reshape([len(Y)*len(M)]))
dummy = np.zeros([len(Y),len(M)])

for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        dummy[i,j] =  df[df['Time_obj'].apply(Year)==y][df['Time_obj'].apply(Month)==m]['All Fatalities'].sum()



fatality_month = Series(dummy.reshape([len(Y)*len(M)]))
dummy = [None for col in range(len(M)*len(Y))]

for (i,y) in enumerate(Y):

    for (j,m) in enumerate(M):

        dummy[j+len(M)*i] = str(y) +'-' + str(m) 



Monthly = pd.DataFrame([price_month, count_month,cost_month_ave_log, loss_month_ave_log, injury_month, fatality_month]).T

Monthly.columns = ['Oil Price', 'Accident Count','All Cost Ave. (log10)', 'Net Loss Ave. (log10)', 'Injury', 'Fatality']

Monthly.index = dummy



Monthly.head()
num = len(Monthly.columns)



plt.figure(figsize=(10,12))

for (i,c) in enumerate(Monthly.columns):

    plt.subplot(num+1,1,i+1)

    Monthly[c].plot()

    plt.ylabel(c)

    

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3)
from pandas.tools.plotting import scatter_matrix

scatter_matrix(Monthly, alpha=1, figsize=(10, 10), diagonal='kde')
# to find difference between before/after oil price falling

# compare average by t-test

# Injury and Fatality are skipped for the t-test since these seem not normal distribution



from scipy import stats

cat = Monthly.columns.drop(['Oil Price', 'Injury', 'Fatality'])

threshold = 60



for c in cat:

    print('='*30)

    print('category: [',c,']')

    ave_high = Monthly[Monthly['Oil Price']>threshold][c].mean()

    ave_low = Monthly[Monthly['Oil Price']<threshold][c].mean()



    print('ave. befor price falling is ', ave_high)

    print('ave. after price falling is ', ave_low)



    #F-test for variances are equal or not

    f = np.var(Monthly[Monthly['Oil Price']>threshold][c]) / np.var(Monthly[Monthly['Oil Price']<threshold][c])

    if f < 1:

        f = 1/f

    dfx = len(Monthly[Monthly['Oil Price']>threshold][c]) - 1

    dfy = len(Monthly[Monthly['Oil Price']<threshold][c]) - 1

    p_value = stats.f.cdf(f, dfx, dfy)

    print('F_value is ', f, ' and p_value of the F is ', p_value)



    # t-test (welch) for averages are equal or not

    t, p = stats.ttest_ind(Monthly[Monthly['Oil Price']>threshold][c], Monthly[Monthly['Oil Price']<threshold][c], equal_var = True)

    print('t_value is ', t,' p_value of t is ', p)
import numpy.random as rd

#theoretical values of poisson distribution for 84 months

x = Series(rd.poisson(Monthly['Injury'].mean(),100000)).value_counts()*len(Monthly)/100000

y = Monthly['Injury'].value_counts()

z = pd.concat([x,y],axis=1).fillna(0)



x2, p, dof, expected = stats.chi2_contingency(z)



print("chi2_value %(x2)s" %locals() )

print("p_value %(p)s" %locals() )
x = Series(rd.poisson(Monthly['Fatality'].mean(),100000)).value_counts()*len(Monthly)/100000

y = Monthly['Fatality'].value_counts()

z = pd.concat([x,y],axis=1).fillna(0)



x2, p, dof, expected = stats.chi2_contingency(z)



print("chi2_value %(x2)s" %locals() )

print("p_value %(p)s" %locals() )
# plot locations of accidents with cost information

# but, accidents occur these costs are random regardless of location

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



traces = []



cost = []

for i in range(9):

    cost.append(10**i)



colors = ['rgb(0, 0, 0)', 'rgb(102, 102, 102)', 'rgb(102, 153, 204)',

          'rgb(102, 153, 255)', 'rgb(153, 255, 255)','rgb(204, 255, 0)',

          'rgb(255, 204, 0)', 'rgb(255, 102, 0)','rgb(255, 0, 0)']

    

for (i,c) in enumerate(cost):

    traces.append(dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df[(df['All Costs']<(c*10)) & (df['All Costs']>c)]['Accident Longitude'],

        lat = df[(df['All Costs']<(c*10)) & (df['All Costs']>c)]['Accident Latitude'],

        mode = 'markers',

        hoverinfo = 'text+name',

        name = str(c) +" < All Costs < "+ str(c*10),

        marker = dict( 

            opacity = 0.85,

            color = colors[i],

            line = dict(color = 'rgb(0, 0, 0)', width = 1)

        )

    ))



layout = dict(

         title = 'Pipeline accidents in USA<br>'

                 '<sub>Click Legend to Display or Hide Categories</sub>',

         showlegend = True,

         legend = dict(

             x = 0.85, y = 0.4

         ),

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             showland = True,

             landcolor = 'rgb(250, 250, 250)',

             subunitwidth = 1,

             subunitcolor = 'rgb(217, 217, 217)',

             countrywidth = 1,

             countrycolor = 'rgb(217, 217, 217)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

        )



figure = dict(data = traces, layout = layout)

iplot(figure)