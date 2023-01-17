import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt 
def do_plot(moon_info, tag):
    day = moon_info.loc[:,'days'] - moon_info.loc[:,'days'].iloc[0]
    az = moon_info.loc[:,'Az']       
    az_flat = moon_info['Az'].map(lambda x: 50) # trick,  to project eclipse in a line
    
    fm = moon_info.loc[:,'Fullmoon']*100
    nm = moon_info.loc[:,'Newmoon']*100
    ec = moon_info.loc[:,'Eclipse']*50
    
    #rolling_dec_max = moon_info.rolling(120).max()  - 90
    #rolling_dec_min = moon_info.rolling(120).min()

    #print (tag)
    plt.figure(figsize=(24,6))
    plt.scatter(day, az, s=fm**1.3, alpha=0.4, color="yellow", edgecolors="silver", linewidth=2) # fm yellow
    plt.scatter(day, az, s=nm**1.3, alpha=0.4, color="silver", edgecolors="silver", linewidth=2) # nw gray
    plt.scatter(day, az, s=ec**1.3, alpha=0.8, color="black", edgecolors="black", linewidth=2)   # eclipse on Az black
    plt.scatter(day, az_flat, s=ec**1.3, alpha=0.8, color="black", edgecolors="black", linewidth=2) # eclipse on flat line
    plt.xlabel("Years in BCE", fontsize=14)
    plt.ylabel("azimuth", fontsize=14)

    nrows, _ = moon_info.shape
    nstep =  nrows // 37 # ~(18.6 // 2)
    ticks = range(nstep,nrows,nstep)
    plt.xticks( ticks,  moon_info.index[ticks], rotation='vertical',fontsize=18)
    plt.grid(color='r', linestyle=':', linewidth=.5)
    plt.title(
      ("\n%s - starting %s BCE\n\nPlot of FullMoon , NewMoon and Eclipse\n" % ( tag, moon_info.index[0] ) )
      + "FullMoon in yellow -- NewMoon in gray \n Eclipse in black -- Total in big, Partial medium and Small not visible",
        fontsize=20
    ) 
    plt.show()

    if (True): #all phases -- too dense
        kmoon_info = moon_info[moon_info.Paksha == 'Krishna']
        smoon_info = moon_info[moon_info.Paksha == 'Shukla']
        kph = kmoon_info.loc[:,'Phase']
        sph = smoon_info.loc[:,'Phase']
        kaz = kmoon_info.loc[:,'Az']
        saz= smoon_info.loc[:,'Az']
        kday = kmoon_info.loc[:,'days'] - kmoon_info.loc[:,'days'].iloc[0]
        sday = smoon_info.loc[:,'days'] - smoon_info.loc[:,'days'].iloc[0]
        plt.figure(figsize=(20,8))
        plt.scatter(kday, kaz, s=kph**1.2, alpha=0.4, color='lightcyan', edgecolors="grey", linewidth=2) 
        plt.scatter(sday, saz, s=sph**1.2, alpha=0.4, color='yellow', edgecolors="grey", linewidth=2)
        plt.scatter(day, az, s=ec**1.3, alpha=0.8, color="black", edgecolors="black", linewidth=2)   # eclipse on Az black
        plt.scatter(day, az_flat, s=ec**1.3, alpha=0.8, color="black", edgecolors="black", linewidth=2) # eclipse on flat line
        plt.xlabel("Years in BCE", fontsize=14)
        plt.ylabel("azimuth", fontsize=14)
        plt.xticks( ticks,  moon_info.index[ticks], rotation='vertical', fontsize=18)
        plt.grid(color='r', linestyle=':', linewidth=.5, which='minor', axis='y')
        #plt.title("All phases", fontsize=20) 
        plt.title(
          ("\n%s - starting %s BCE\nwaxing(yellow), waning(gray) and  eclipse(black)" % ( tag, moon_info.index[0] ) )
          #+ "Waxing in yellow -- Waning in gray \n Eclipse in black -- Total in big, Partial medium and Small not visible"
            ,
            fontsize=20
        ) 
        plt.show()
#Delhi BCE plots
moon_info =pd.read_csv("../input/Delhi_Az_Eclipse.csv").set_index('MoonRise') #BCE1900_to_BCE2000
moon_info = moon_info.sort_index()

#standstills =  [ 
#    moon_info.iloc[ 365*19*n : 365*19*(n+1), : ]
#    .loc[:,'Az']
#    .rolling(30)
#    .max()
#    .idxmax() for n in range(5) 
#]

standstills = ['1904-10-04', '1922-10-22', '1940-10-22', '1958-10-22', '1976-10-22', '1998-10-22']

if (1):
    do_plot(moon_info.loc[standstills[0] : standstills[1]] , 'Delhi MoonPlot - 18year Span' )
    do_plot(moon_info.loc[standstills[1] : standstills[2]] , 'Delhi MoonPlot - 18year Span' )
    do_plot(moon_info.loc[standstills[2] : standstills[3]] , 'Delhi MoonPlot - 18year Span' )
    do_plot(moon_info.loc[standstills[0] : standstills[2]] , 'Delhi MoonPlot - Two 18year Spans' )
    do_plot(moon_info.loc[standstills[0] : standstills[3]] , 'Delhi MoonPlot- Three 18year Spans' )
    
kuru_moon_info =pd.read_csv("../input/Kuru_Az_Eclipse.csv").set_index('MoonRise')
standstills =  [ kuru_moon_info.iloc[ 365*19*n : 365*19*(n+1), : ].loc[:,'Az'].rolling(30).max().idxmax() for n in range(5) ]

if (1):
    do_plot(kuru_moon_info.loc[standstills[0] : standstills[1]] , 'Kurukshetra - 1st 18year Span' )
    do_plot(kuru_moon_info.loc[standstills[1] : standstills[2]] , 'Kurukshetra - 2nd 18year Span' )
    do_plot(kuru_moon_info.loc[standstills[2] : standstills[3]] , 'Kurukshetra - 3rd 18year Span' )
    do_plot(kuru_moon_info.loc[standstills[0] : standstills[2]] , 'Kurukshetra - Two 18year Spans' )
    do_plot(kuru_moon_info.loc[standstills[0] : standstills[3]] , 'Kurukshetra - Three 18year Spans' )
display( kuru_moon_info.head(), kuru_moon_info[kuru_moon_info.Dec>0.1].describe())
import math
ss_max =  (kuru_moon_info.Az.rolling(window=1*365).max()-90)
ss_dec_max =  (kuru_moon_info.Dec.rolling(window=1*365).max()*180/math.pi)
ax = ss_max.plot(figsize=(25,8) ,label='Az')
ax = ss_dec_max.plot(figsize=(25,8), ax=ax, label='Dec')
ax.grid(color='r', linestyle=':', linewidth=.5, which='major', axis='y')
ax.grid(color='r', linestyle=':', linewidth=.2, which='minor', axis='y')
#ax.set_ylim(17,30)
#_ = ax.set_xticks(kuru_moon_info.days)
#_ = ax.set_xticklabels(kuru_moon_info.days)
ax.legend(loc="center", fontsize=20)


nrows, _ = kuru_moon_info.shape
nstep =  nrows // 37 # ~(18.6 // 2)
ticks = range(nstep,nrows,nstep)
ax.set_xticks( ticks)  
ax.set_xticklabels(kuru_moon_info.index[ticks], rotation='vertical',fontsize=18)

ax.set_title("Annual Rolling Max - Moonrise Azimuth and Declination", fontsize=30)
ax.set_xlabel("", fontsize=14)
ax.set_ylabel("azimuth / declination", fontsize=14)
#ax.xticks( ticks,  moon_info.index[ticks], rotation='vertical', fontsize=18)
#ax.grid(color='r', linestyle=':', linewidth=.5, which='minor', axis='y')

#plt.title("All phases", fontsize=20) 
_ = ax.minorticks_on()
ss_max.shape
