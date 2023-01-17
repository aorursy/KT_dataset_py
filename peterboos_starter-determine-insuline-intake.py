from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import sys

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))



nRowsRead = None # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/diabetic.csv',sep=',',decimal='.', nrows = nRowsRead,header=0,error_bad_lines=False)

#df1.dataframeName = 'diabetic.csv'

#nRow, nCol = df1.shape

#print(f'There are {nRow} rows and {nCol} columns')



#start with a date (as for simple logging with time stamp)

#print pd.to_datetime(pd.Timestamp.toda, format='%Y-%m-%d', errors='raise', infer_datetime_format=False, exact=True)



#print (pd.to_datetime(pd.Timestamp.toda, format='%Y-%m-%d', errors='raise', infer_datetime_format=False, exact=True))

#range = pd.date_range('2018-01-21', '2015-12-31',freq=24hour) # freq='15min')

#df = pd.DataFrame(index = range)





# repair malformated data updates :

df1.rename( columns={"weekday":"day"},inplace ="true") # to better fit screen

df1.rename (columns = {"0700i":"i0700","0930i":"i0930","1300i":"i1300","1500i":"i1500","1800i":"i1800","2300i":"i2300"},inplace = "true") 

df1.rename (columns={"1i300":"i1300"},inplace = "true")

#print(pd.get_option("display.max_columns"))

                     

df1.head(25)
dfi =df1[['day','i0700','i0930','i1300','i1500','i1800','i2300']].copy()

dfi.fillna(0,inplace=True) # while i dont want to fill unknown measurements we can replace NAN in insuline dose safely. (allows for math)

b = dfi.set_index('day').stack(dropna=False)



dfi=b.reset_index(drop=False, level=0).reset_index()

dfi.columns= ['InsulinTime', 'InsulinDay', 'Insulin']

dfi.head(7)
dfiDayTotal = df1[['day','i0700','i0930','i1300','i1500','i1800','i2300']].copy()

dfiDayTotal.fillna(0,inplace=True)

dfiDayTotal['InsulinTotal'] = dfiDayTotal['i0700']+dfiDayTotal['i0930']+dfiDayTotal['i1300']+dfiDayTotal['i1500']+dfiDayTotal['i1800']+dfiDayTotal['i2300']

dfiDayTotal.drop(['i0700','i0930','i1300','i1500','i1800','i2300'], 1, inplace=True)

dfiDayTotal['DayIndex']=np.arange(len(dfiDayTotal))

dfiDayTotal.head(7)
df2 = df1.loc[:, 'day':'M2300']

a = df2.set_index('day','Glucose').stack(dropna=False)



df2 = a.reset_index(drop=False, level=0).reset_index()

df2.columns= ['time', 'day', 'GlucoseLevel']



df2['ID'] = np.arange(len(df2))

df2 =  pd.concat([df2, dfi], axis=1, sort=False)

df2.drop('InsulinDay', 1, inplace=True)

df2.drop('InsulinTime', 1, inplace=True)

df2.columns= ['time', 'day', 'GlucoseLevel','ID','Insulin']

df2 = df2[['ID', 'time','day','GlucoseLevel','Insulin']]



df2['DayIndex']= np.ceil((df2['ID']+1)/7).astype(int)-1 # to merge it with Daytotal.

#temp = dfiDayTotal.filter(['DayIndex','InsulinTotal'],axis=1)# old.filter(['A','B','D'], axis=1)

#df2 = pd.merge(df2, temp, left_on='DayIndex', right_on='DayIndex')

df2 = pd.merge(df2, dfiDayTotal.filter(['DayIndex','InsulinTotal'],axis=1), left_on='DayIndex', right_on='DayIndex') #joined without using temp

df2.head(11)



df2 = df2[df2.day != 6] # deleting saturdays

df2 = df2[df2.day != 7] # deleting sundays



df1 = df1[df1.day != 6] # deleting saturdays

df1 = df1[df1.day != 7] # deleting sundays



dfi = dfi[dfi.InsulinDay != 6] # deleting saturdays

dfi = dfi[dfi.InsulinDay != 7] # deleting sundays



dfiDayTotal = dfiDayTotal[dfiDayTotal.day != 6] # deleting saturdays

dfiDayTotal = dfiDayTotal[dfiDayTotal.day != 7] # deleting sundays

print('Weekend data deleted')
cc=['g','g','g','g','g','g','g','r','r','r','r','r','r','r']# days swap collor, its simple i'm not a pro in this.



ax = df2.plot.scatter(x='ID', y='InsulinTotal', color='DarkBlue', label='Insulin day total',alpha=0.1);

bx = df2.plot.scatter(x='ID', y='Insulin', color='LightBlue', label='Insulin dose',alpha=0.2,ax=ax,s=65);

df2.plot(kind='scatter',x='ID',y='GlucoseLevel',label = 'GlucoseLevel',c=cc,figsize=(20,8),ax=bx)#,c='day'

df2.plot(kind='scatter' ,x='ID', y='Insulin',color='black',figsize=(20,4),label = 'Insulin per dose')





#from pandas.plotting import scatter_matrix 

#scatter_matrix(df2, alpha=0.2,figsize=(20,20))



dfiDayTotal.plot(kind='scatter',x='DayIndex',y='InsulinTotal',label = 'Total insulin per day',figsize=(20,4))#,c='day')
dfMorning = df1.filter(['day','M0700','M0930','i0700'])

#dfMorning = dfMorning[dfMorning.weekday != 6] # deleting saturdays

#dfMorning = dfMorning[dfMorning.weekday != 7] # deleting sundays





dfMorningSorted = dfMorning.sort_values('M0700')

##dfMorning.head(50)

dfMorningSorted['InsulineEffect']=(dfMorningSorted['M0700']-dfMorningSorted['M0930'])/dfMorningSorted['i0700']

dfMorningSorted.head(12)

dfm = dfMorningSorted [dfMorningSorted.InsulineEffect>0.0] # for the moment deleting strange negative values

dfNegative = dfMorningSorted [dfMorningSorted.InsulineEffect<0.0] 



# plot a scatter and connect the dots

ax = dfm.plot(kind='scatter', x='M0700', y='InsulineEffect',color='blue')

dfm.plot(kind='line', x='M0700', y='InsulineEffect',color='green',ax=ax, label = 'Effective insuline')



bx = dfNegative.plot(kind='scatter', x='M0700', y='InsulineEffect',color='blue',ax=ax)

dfNegative.plot(kind='line', x='M0700', y='InsulineEffect',color='red',ax=ax,linestyle = ':', label = 'On bad days' )

ax.set(ylabel="Insuline effect per unit", xlabel="blood glucose level")
plt.style.use('ggplot') 



plt.rcParams["figure.figsize"] = (13,12)

plt.ylim([2, 15])

plt.xlim([2, 6.5])

plt.title("Glucose from 7:00 morning to 9:30 by insulin adjustment", fontsize=20)

plt.ylabel('Glucose level', fontsize=20)

plt.xlabel('Insulin dose', fontsize=20)

ax = plt.gca()

ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)

#plt.xticks(np.arange(min(0), max(6)+1, 1.0))



plt.yticks(np.arange(0, 15, step=1))

plt.xticks(np.arange(0, 8.5, step=.5))





for index, row in df1.head(len(df1)).iterrows():

    mhw=0.15

    mlw=1.5

    x =0

    a = 1 /len(df1)*index*0.7+0.3

    y= row['M0700']

    dx = row['i0700']

    dy = row['M0930']-row['M0700']

    if ((row['M0930']> 5) and (row['M0930']<10)):

        if (dy>0):

            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='blue')

        else:

            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='green')

    else:

        if ((row['M0930']> 3.5) and (row['M0930']<11.5)):

            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='orange')

        else:

            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='red')

    #plt.text(dx, dy, dx, fontsize=9)

plt.show()
#Needs a FIX because an extra doses could have been given at 15:00 (kink in a line..)

plt.style.use('ggplot') 

plt.rcParams["figure.figsize"] = (20,10)

plt.ylim([2, 15])

plt.xlim([2, 6.5])

plt.title("Glucose from 13:00 morning to 15:00 by insulin adjustment", fontsize=20)

plt.ylabel('Glucose level', fontsize=20)

plt.xlabel('Insulin dose', fontsize=20)

ax = plt.gca()

ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)



plt.yticks(np.arange(0, 15, step=1))

plt.xticks(np.arange(0, 12.5, step=.5))



linestyle='-'

mhw=0.15

mlw=1.5

for index, row in df1.head(len(df1)).iterrows():

    x =0

    y= row['M1300']

  

    dx = row['i1300']

    a = 1 /len(df1)*index*0.7+0.3

    if (not np.isnan(row['i1500'])):

        #dx=dx+row['i1500']

        #linestyle=':'

        temp=0

    

    dy = row['M1500']-row['M1300'] 



    if(x+y+dx+dy>0):   #PLACE FOR AFIX 

        if ((row['M1500']> 5) and (row['M1500']<10)):

            if (dy>0):

                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='blue',linestyle=linestyle)

            else:

                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='green',linestyle=linestyle)

        else:

            if ((row['M1500']> 3.5) and (row['M1500']<11.5)):

                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='orange',linestyle=linestyle)

            else:

                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='red',linestyle=linestyle)

plt.show()
# todo make a graph for the evening but there is some incosistancy of later measurements 20:00 andd 23:00 and sometimes none.

# not often insuline is used at 1500 how to deal with that..

plt.style.use('ggplot') 



plt.rcParams["figure.figsize"] = (10,10)

plt.ylim([2, 15])

plt.xlim([2, 6.5])

plt.title("Glucose from 13:00 morning to 18:00 by insulin adjustment", fontsize=20)

plt.ylabel('Glucose level', fontsize=20)

plt.xlabel('Insulin dose', fontsize=20)

ax = plt.gca()

ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)

#plt.xticks(np.arange(min(0), max(6)+1, 1.0))

plt.yticks(np.arange(0, 15, step=1))

plt.xticks(np.arange(0, 6.5, step=.5))



for index, row in df1.head(len(df1)).iterrows():

    end = 'M1800'

    start = 'M1300'

    x =0

    y= row['M1300']

    dx = row['i1500']

    dy = row[end]-row['M1300']

    a = 1 /len(df1)*index*0.7+0.3

    if ((row[end]> 5) and (row[end]<10)):

        if (dy>0):

            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='blue')

        else:

            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='green')

    else:

        if ((row[end]> 3.5) and (row[end]<11.5)):

            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='orange')

        else:

            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='red')



plt.show()
# to be written ...