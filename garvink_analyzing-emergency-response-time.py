# -*- coding: utf-8 -*-

"""

Created on Tue Dec 20 15:44:00 2016



@author: Garvin

"""



#loading packages

from pandas import Series, DataFrame

import pandas as pd

import numpy as np

import matplotlib



%matplotlib notebook



import matplotlib.pyplot as plt

import scipy

import scipy.stats as stats



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()        

            

path = '../input/'

accident = DataFrame(pd.read_csv('../input/accident.csv'))

#creating new variabel 'TimeRescue', that indicates the amount of time in minutes until the medic 

#service arrived at the crash scene

def medic(HourCall,MinCall,HourArr, MinArr):

    #check if no information are recorded

    for item in [88,96,97,98,99]:

        if HourCall == item or HourArr==item or MinArr==item or HourArr==item:

            return 0

    if HourCall == HourArr: 

        #if hours equal, substract call from arrival

        return MinArr-MinCall 

    elif HourCall != HourArr:

        #if hours are unqeual, check which one is larger

        if HourCall < HourArr: #both events occur at same day

                return (HourArr-HourCall-1)*60  +60-MinCall + MinArr

        if HourCall > HourArr: #events occur at differents days

                return (24-HourCall)*60+HourArr*60+MinArr-MinCall

  



#applying medic-function to dataset

TimeRescue = DataFrame({'Minutes': accident.apply(lambda row: medic(row['NOT_HOUR'], row['NOT_MIN'], 

 row['ARR_HOUR'], row['ARR_MIN']), axis=1), 'DEATHS' : accident['FATALS'], 

'DAY' : accident['DAY_WEEK'],'MONTH' : accident['MONTH']})

#eliminating negative values (assuming wrong data entries)

TimeRescue['Minutes_adj']=TimeRescue['Minutes'].apply(lambda x: 0 if x<0 else x) 

#mapping weekdays

TimeRescue['Weekdays']=TimeRescue['DAY'].map({1: 'Sunday' , 2: 'Monday', 3: 'Tuesday', 4:'Wednesday',                                                              

                                              5:'Thursday', 6:'Friday', 7:'Saturday'})
#subset that eliminates all observations with time ==0 (as we can assume that no medic has been

#called in these cases. and <30 (as cases where time >30 are treated as outliers, i.e. due to

#poor data quality or other exceptional circumstances



TimeRescue_lethal_b30 = TimeRescue [(TimeRescue['Minutes_adj'] >0) & (TimeRescue['Minutes_adj'] <30)]

#np.median(TimeRescue_lethal_b30) #median = 5

#TimeRescue_lethal_b5 =  TimeRescue_lethal_b30[(TimeRescue_lethal_b30['Minutes_adj'] <=5)].copy()

#TimeRescue_lethal_a5b30 =  TimeRescue_lethal_b30[(TimeRescue_lethal_b30['Minutes_adj'] >5)].copy()


#plot on relative frequencies of time until medic arrives

trace1 = go.Histogram(

    x=TimeRescue_lethal_b30['Minutes_adj'],

    histnorm='probability density', 

    name='control',

    autobinx=False,

    xbins=dict(

        start=0.1,

        end=60,

        size=1

    ),

    marker=dict(

        color='rgb(26,78,156)',

        line=dict(

            color='rgb(236,176,9)',

            width=0

        )

    ),

    opacity=0.75

)



t1 = [trace1]

layout = go.Layout(

    title='Relative Frequncies until Medic arrived at Crash Scene',

    xaxis=dict(

        title='Minutes'

    ),

    yaxis=dict(

        title='Relative Frequency'

    ),

    barmode='overlay',

    bargap=0.25,

    bargroupgap=0.3

)

fig = go.Figure(data=t1, layout=layout)

iplot(fig)
TimeRescue_lethal_b30['Minutes_adj'].describe()
#analyzing data for normality

fig = plt.figure()

ax = fig.add_subplot(111)

res=stats.probplot(TimeRescue_lethal_b30['Minutes_adj'],dist='norm',plot=ax)

plt.show()
#pllotting heat map for avg time at weekdays and months

TimeRescue_heat= TimeRescue[(TimeRescue['Minutes_adj'] >0) & (TimeRescue['Minutes_adj'] <30)]

data_calendar = []

for month in range (12):

    data_per_month=[]

    for day in range(7):

        TimeRescue_heat[(TimeRescue_heat['DAY'] ==day+1) & (TimeRescue_heat['MONTH'] ==month+1) ]

        avg = TimeRescue_heat[(TimeRescue_heat['DAY'] ==day+1) & (TimeRescue_heat['MONTH'] ==month+1) ]['Minutes_adj'].mean()

        data_per_month.append(avg)

    data_calendar.append(data_per_month)



data_heat = [

    go.Heatmap(

        z=data_calendar,

        x=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],

        y=['1', '2', '3','4','5','6','7','8','9','10','11','12']

    )

]

iplot(data_heat)
#Calculating Mean hours and std per weekdays

weekdays=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

weeks_time_avg=[]

weeks_time_std=[]

for i in range(7):

    weeks_time_avg.append(TimeRescue_lethal_b30[TimeRescue_lethal_b30['DAY']==i+1]['Minutes_adj'].mean())

    weeks_time_std.append(TimeRescue_lethal_b30[TimeRescue_lethal_b30['DAY']==i+1]['Minutes_adj'].std())



WeeklyTimes = DataFrame({'Days':weekdays, 'Mean Time (m)':Series(weeks_time_avg).apply(lambda x: round(x,2)) , 

                         'Std' : Series(weeks_time_std).apply(lambda x: round(x,2))})

print (WeeklyTimes)
import scipy.stats as stats



def CellMeans(x,values,cat,p=0.05,EqualVar=-True):

    #converting x into pandas DataFrame    

    x=DataFrame(x)    

    #converting x[cat] into dtype 'category'  

    x[cat]=x[cat].astype('category')

    #extracting categories    

    cats = x[cat].cat.categories

    #calculating Y.. mean    

    Y__mean=x[values].mean()

    #initiating vector vor SStr and defining SSto

    SStr_vec,SSto  = [], sum(pow(x[values]-Y__mean,2))

    #looping through subsets (by category) and calculating SStr

    for i in cats:

        SStr_vec.append(len(x[(x[cat] ==i)])*pow(x[x[cat] ==i][values].mean()-Y__mean,2))



    #summing up the SStr vector  

    SStr = sum(SStr_vec)

    #Calculating SSE by substracting SStr from SSto    

    SSE = SSto-SStr

    #MSE as SSE divided by n-r df

    MSE = SSE /(len(x)-len(cats))

    #MSTR as SStr divided by r-1

    MSTR = SStr/(len(cats)-1)

    pvalue=1-scipy.stats.f.cdf(MSTR/MSE,len(cats)-1,len(x)-len(cats)) 

 

    print ('-'*40)

    print ('F-Value: ' + str(MSTR/MSE))

    print ('-'*40)

    print  (str(p) + ' F-Critical Value ' + str(scipy.stats.f.ppf(q=1-p, dfn=len(cats)-1, dfd=len(x)-len(cats))))

    print ('-'*40)

    print ('p-value: ' + str(pvalue))

    print ('-'*40) 

    #Initiating pairwise comparison if pvalue is smaller than threshold

    if pvalue<=p:

        res = DataFrame(columns=['Cat_A', 'Cat_B','p-value'])

        i_vec,j_vec,sig_vec = [],[],[]

        #initiating Matrix comparing individual means

        if Series(cats).dtypes=='float64': cats.astype('int')        

        for i in cats:

            restcats = Series(cats).apply(lambda x: x if x!=i else None).dropna()

            if Series(restcats).dtypes=='float64': restcats=restcats.astype('int') 

            

            for j in restcats:

                sig = stats.ttest_ind(x[x[cat] ==i][values],x[x[cat] ==j][values], equal_var=EqualVar )[1]

                if float(sig) < p:

                    i_vec.append(i),j_vec.append(j),sig_vec.append(sig)



        res =DataFrame({'Cat_A':i_vec,'Cat_B':j_vec,'p-value':sig_vec})

        print ('-'*40)

        print ('Pairwise comparison for p < ' + str(p) +':')

        print ('-'*40)

        print (res)

    return {'f-value' : MSTR/MSE, 'p-value': pvalue}



CellMeans(TimeRescue_lethal_b30,'Minutes_adj','Weekdays',p=0.05, EqualVar=False)