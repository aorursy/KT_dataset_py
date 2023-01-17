import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import rc

import seaborn as sns

import matplotlib.dates as mdates

%matplotlib inline



import os



#File numbers as per the order in the data source.

df1 = pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")

df2 = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

df3 = pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")

df4 = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingDetails.csv")

df5 = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")

df6 = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")

df7 = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")

df8 = pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")


age=[]

age=df1.AgeGroup



cnt=[]

cnt=df1.TotalCases
plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.6, color='white')

plt.pie(cnt, labels=age, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



state=[]

plt.rcParams['text.color'] = 'white'

state=df3["State/UT"]

xlab=['prim','comm','dist','urbbeds','rurbeds']



for sn in range(36):

    plt.title(state[sn])

    val=[]

    val.append(int(df3.NumPrimaryHealthCenters_HMIS[sn]))

    val.append(df3.NumCommunityHealthCenters_HMIS[sn])

    val.append(df3.NumDistrictHospitals_HMIS[sn])

    val.append(df3.NumUrbanBeds_NHP18[sn])

    val.append(df3.NumRuralBeds_NHP18[sn])

    #print(val)

    plt.bar(xlab, val)

    plt.show()



    

    


d=df7.sort_values('State / Union Territory')

#print(d['State / Union Territory'])



for sn in range(36):

    print(state[sn])

    i=df7[df7["State / Union Territory"]==state[sn]].index

    rpop=d['Rural population'][i]

    upop=d['Urban population'][i]

    rbed=df3.NumRuralBeds_NHP18[sn]

    ubed=df3.NumUrbanBeds_NHP18[sn]

    rper=rbed*100/rpop

    print("% rural",rper)

    uper=ubed*100/upop

    print("% urban",uper)

   
plt.plot(df2['Date'],df2['Confirmed'],color='red')

plt.title("Confirmed cases")

#plt.xticks(np.arange(0, 51, 5)) 

plt.yticks(np.arange(0, 9318, 1000))

plt.xlabel('Date')

plt.ylabel('Cases')

plt.show()





plt.plot(df2['Date'],df2['Deaths'],color="black")

plt.title("Deaths")

plt.xlabel('Date')

plt.ylabel('Cases')

plt.xticks( rotation='vertical') 

plt.yticks(np.arange(0, 400, 50))

plt.show()





plt.plot(df2['Date'],df2['Cured'],color="green")

plt.title("Recovered")

plt.xlabel('Date')

plt.ylabel('Cases')

plt.xticks( rotation='vertical') 

plt.yticks(np.arange(0, 1388, 100))

plt.show()

sf=0

si=0

for i in df2['ConfirmedForeignNational']:

    if i=='-':

        continue

    else:

        sf=sf+int(i)



for i in df2['ConfirmedIndianNational']:

    if i=='-':

        continue

    else:

        si=si+int(i)

        

nc=[]

nc.append(si)

nc.append(sf)

ny=['Indian','Foreign']



plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0, color='white')

plt.pie(nc, labels=ny, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()

print("'/n")
#import pandas_profiling



#pandas_profiling.ProfileReport(df1)

#pandas_profiling.ProfileReport(df2)

#pandas_profiling.ProfileReport(df3)

#pandas_profiling.ProfileReport(df4)

#pandas_profiling.ProfileReport(df5)

#pandas_profiling.ProfileReport(df6)

#pandas_profiling.ProfileReport(df7)

#pandas_profiling.ProfileReport(df8)