# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import os

from datetime import datetime

import copy
""" PARAMETERS """

folder_input = '../input'

file_input = 'hessi.solar.flare.2002to2016.csv'
""" FUNCTIONS """



## review variables of data

def review_data(DATA):

    ## Dates data

    print('Data between %s and %s'%(np.min(DATA['dt.start'].tolist()),np.max(DATA['dt.end'].tolist())))



    ## categorical data

    print('\nCATEGORICAL DATA:')

    # list of categorical variables

    lcat = ['energy.kev','flag.1','flag.2','flag.3','flag.4','flag.5']

    # possible values

    for icat in lcat:

        # filter nan values

        values = DATA[icat].values

        values = list(values[pd.notnull(values)])

        print('VARIABLE %s: \n%s\n'%(icat,sorted(list(set(values)), reverse=False)))



    ## non categorical data

    print('NON CATEGORICAL DATA:')

    # list of categorical variables

    lnoncat = ['duration.s','peak.c/s','total.counts','x.pos.asec','y.pos.asec','active.region.ar','radial']

    # describe

    print(DATA[lnoncat].describe())



    ## NAN data

    print('\nNULL DATA:')

    print(DATA.isnull().sum())

    

    return None



## filtering data

def filtering(DATA,lwrong):

    ## filter 1: 3-16 kev

    lenght1 = len(DATA)

    DATA = DATA[DATA['energy.kev']!='3-6']

    lenght2 = len(DATA)

    if lenght1!=lenght2: print('filtering 3-6 kev',lenght1,lenght2)



    ## filter 2: radial

    radial = DATA['radial'].values

    lenght1 = len(DATA)

    DATA = DATA[DATA['radial']<=np.percentile(radial,99)]

    lenght2 = len(DATA)

    if lenght1!=lenght2: print('filtering radial',lenght1,lenght2)



    # filter possible wrong values or without solar event

    #lwrong = ['NS','SD','SS','DF','DR','ED','ES','FE','FR','FS','GD','GE','GS','MR','P0','PS','PE']

    for icod in lwrong: 

        ## filter

        lenght1 = len(DATA)

        DATA = DATA[DATA['flag.1']!=icod]

        lenght2 = len(DATA)

        if lenght1!=lenght2: print('filtering %s'%icod,lenght1,lenght2)



        lenght1 = len(DATA)

        DATA = DATA[DATA['flag.2']!=icod]

        lenght2 = len(DATA)

        if lenght1!=lenght2: print('filtering %s'%icod,lenght1,lenght2)



        lenght1 = len(DATA)

        DATA = DATA[DATA['flag.3']!=icod]

        lenght2 = len(DATA)

        if lenght1!=lenght2: print('filtering %s'%icod,lenght1,lenght2)



        lenght1 = len(DATA)

        DATA = DATA[DATA['flag.4']!=icod]

        lenght2 = len(DATA)

        if lenght1!=lenght2: print('filtering %s'%icod,lenght1,lenght2)



        lenght1 = len(DATA)

        values = DATA['flag.5'].values

        lfilter = [i for i in list(values[pd.notnull(values)]) if icod in i]

        DATA = DATA[~DATA['flag.5'].isin(lfilter)]

        lenght2 = len(DATA)

        if lenght1!=lenght2: print('filtering %s'%icod,lenght1,lenght2)

            

    return DATA
""" DATA """



# read data

path_input = os.path.join(folder_input,file_input)

DATA = pd.read_csv(path_input,sep=",",index_col=0)

# process date / time columns

def parse_dt(sdatex,stimex):

    datex = datetime.strptime(sdatex, '%Y-%m-%d')

    timex = datetime.strptime(stimex, '%H:%M:%S')

    return datetime(datex.year,datex.month,datex.day,timex.hour,timex.minute,timex.second)

DATA['dt.start'] = DATA[['start.date','start.time']].apply(lambda x: parse_dt(x[0],x[1]), axis=1)

DATA['dt.peak'] = DATA[['start.date','peak']].apply(lambda x: parse_dt(x[0],x[1]), axis=1)

DATA['dt.end'] = DATA[['start.date','end']].apply(lambda x: parse_dt(x[0],x[1]), axis=1)

# clean columns

DATA.drop(['start.date','start.time','peak','end'], axis=1, inplace=True)

# add new columns

DATA['year'] = DATA['dt.start'].apply(lambda col: col.year)

DATA['month'] = DATA['dt.start'].apply(lambda col: col.month)

DATA['day'] = DATA['dt.start'].apply(lambda col: col.day)



# filtering basic

lwrong = ['NS','SD']

DATA1 = filtering(copy.deepcopy(DATA),lwrong)



# include energy bounday ranges

DATA1['energy.kev.i'] = DATA1['energy.kev'].apply(lambda col: int(col.split('-')[0]))

DATA1['energy.kev.f'] = DATA1['energy.kev'].apply(lambda col: int(col.split('-')[1]))

CENERGY = DATA1[['energy.kev','energy.kev.i','energy.kev.f']].drop_duplicates(inplace=False).sort(['energy.kev.i'], ascending=[1], inplace=False)
## REVIEW OF DATA

review_data(DATA1)
## PLOT SUNSPOTS per Energy

import matplotlib.pyplot as plt



# get colors

colors = plt.cm.jet(np.linspace(0,1,len(CENERGY['energy.kev.i'].values)))



# build figure object

fig, ax = plt.subplots(figsize=(10,10))

# loop of energy ranges

for i,irange in enumerate(CENERGY['energy.kev'].values):

    # collect data

    AUX = DATA1[DATA1['energy.kev']==irange][['x.pos.asec','y.pos.asec']]

    # scatter plot

    plt.scatter(AUX['x.pos.asec'].values,AUX['y.pos.asec'].values,color=colors[i],label='%s Kev'%irange)

    ax.legend(loc='best',fontsize=9,shadow=True)

    # clean

    del(AUX)

# set title

plt.title('SUNSPOTS per Energy')

# plot

plt.show()
## Y DISTRIBUTION

import matplotlib.pyplot as plt

# create objects

fig, ax = plt.subplots(figsize=(10,2))

# hist

y = DATA1['y.pos.asec'].values

plt.hist(y, bins=np.linspace(np.min(y),np.max(y),100),normed=True,label="label var y")

# set limits

ax.set_xlim([np.min(y),np.max(y)])

# title

plt.title('Y Distribution')

# plot

plt.show()
""" number of events per year """



import matplotlib.pyplot as plt

DATA1.groupby(['year'])['total.counts'].count().plot(kind='bar',figsize=(10,2),title='YEARLY NUMBER OF EVENTS')

plt.show()
""" number of events per year and intensity ranges """



# calculate limits of intensity ranges

intensity = DATA1['peak.c/s'].values

p10 = np.percentile(intensity,10)

p50 = np.percentile(intensity,50)

p90 = np.percentile(intensity,90)



# plot average of events intensity per year

PI0 = DATA1[(DATA1['peak.c/s']<=p10)].groupby(['year'])['peak.c/s'].count()

PI1 = DATA1[(DATA1['peak.c/s']>p10) & (DATA1['peak.c/s']<=p50)].groupby(['year'])['peak.c/s'].count()

PI2 = DATA1[(DATA1['peak.c/s']>p50) & (DATA1['peak.c/s']<=p90)].groupby(['year'])['peak.c/s'].count()

PI3 = DATA1[(DATA1['peak.c/s']>p90)].groupby(['year'])['peak.c/s'].count()

PI = pd.DataFrame({'year':PI0.index.values,'very low':PI0.values,'low':PI1.values,'high':PI2.values,'very high':PI3.values})



import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

# build figure object

fig, ax = plt.subplots(figsize=(10,5))



# collect data

ind = PI0.index.values

y0 = PI0.values

y1 = PI1.values

y2 = PI2.values

y3 = PI3.values

# plot

ax.stackplot(ind,y0, y1, y2, y3,colors=['blue','green','orange','red'])

# set limits

ax.set_xlim([ind[0]-1,ind[-1]+1])

# set legend

ax.legend([mpatches.Patch(color='blue'),  

            mpatches.Patch(color='green'),

            mpatches.Patch(color='orange'),

            mpatches.Patch(color='red')], 

           ['very low','low','high','very high'])



# set label

ax.set_xlabel('Years')

# set title

ax.set_title('YEARLY NUMBER OF EVETNS per INTENSITY (c/s)\n (Limits Classification = %s c/s, %s c/s, %s c/s)'%(p10,p50,p90))

# plot

plt.show()