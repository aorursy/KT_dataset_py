import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

from pandas import Series

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



vc = pd.read_csv("../input/VietnamConflict.csv")
vc.head()
vc.columns
#dealing with nan values in the dataset

vc = vc.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.NaN)
#histogram that include nan

def hist_plot(var,rot):

    labels, values = zip(*Counter(var).items())

    indexes = np.arange(len(labels))

    width=0.5

    plt.bar(indexes, values, width)

    plt.xticks(indexes+width*0.1 , labels,rotation=rot)

    plt.ylabel("Count")

    plt.show()
num_cas = len(vc)

print("There were",num_cas,"US casualties in the war." )
plt.figure(1)

SERVICE_TYPE=vc["SERVICE_TYPE"]

plt.title('Service Type')

hist_plot(SERVICE_TYPE,'horizontal')



q = Series(SERVICE_TYPE)

SERVICE_TYPE2 = q.value_counts()

SERVICE_TYPE2 = SERVICE_TYPE2.sort_index()

comm_SERVICE_TYPE=SERVICE_TYPE2.idxmax(axis=1)



print("The most common service type is:",comm_SERVICE_TYPE)
plt.figure(2)

ENROLLMENT=vc["ENROLLMENT"]

plt.ylabel("Count")

plt.title('Enrollment')

ENROLLMENT.value_counts().plot(kind='bar')
plt.figure(3)

BRANCH=vc["BRANCH"]

plt.ylabel("Count")

plt.title('Branch')

BRANCH.value_counts().plot(kind='bar')
fig = plt.figure(4, figsize = ( 16 , 8 ) )

RANK=vc["RANK"]

plt.ylabel("Count")

plt.title('Rank')

RANK.value_counts().plot(kind='bar')
plt.figure(5)

SEX=vc["SEX"]

plt.ylabel("Count")

plt.title('Sex')

SEX.value_counts().plot(kind='bar')



sex_count=Counter(SEX)

print("There were",sex_count['F'], "military women and",sex_count['M'],"military men who died in the war.")
plt.figure(6)

MARITAL_STATUS=vc["MARITAL_STATUS"]

plt.ylabel("Count")

plt.title('Marital Status')

MARITAL_STATUS.value_counts().plot(kind='bar')
plt.figure(7)

ETHNICITY=vc["ETHNICITY"]

plt.ylabel("Count")

plt.title("Ethnicity")

ETHNICITY.value_counts().plot(kind='bar')
plt.figure(8)

DEPLOYMENT_COUNTRY=vc["DEPLOYMENT_COUNTRY"]

plt.ylabel("Count")

plt.title("Deployment Country")

DEPLOYMENT_COUNTRY.value_counts().plot(kind='bar')
plt.figure(9)

HOSTILITY_CONDITIONS=vc["HOSTILITY_CONDITIONS"]

plt.title("Hostility Conditions")

HOSTILITY_CONDITIONS.value_counts().plot(kind='bar')
plt.figure(10)

fatal=vc["FATALITY"]

plt.ylabel("Count")

plt.title("Fatality")

fatal.value_counts().plot(kind='bar')
plt.figure(11)

FATALITY_2=vc["FATALITY_2"]

plt.title("Cause of deth")

hist_plot(FATALITY_2,'vertical')



COD=Counter(FATALITY_2)

COD=sorted(COD, key=COD.get, reverse=True)



print("The main cause of deth is:",COD[0] )

print("The minor cause of deth is:",COD[23] )
BIRTH_YEAR=vc["BIRTH_YEAR"]

FATALITY_DATE=vc["FATALITY_DATE"]



BIRTH_YEAR = pd.to_datetime(BIRTH_YEAR, format='%Y%m%d')

FATALITY_DATE = pd.to_datetime(FATALITY_DATE, format='%Y%m%d')

age = BIRTH_YEAR.where(BIRTH_YEAR < FATALITY_DATE, BIRTH_YEAR -  np.timedelta64(100, 'Y'))

age = (FATALITY_DATE - BIRTH_YEAR).astype('<m8[Y]')



fig = plt.figure(12, figsize = ( 25 , 12 ) )

plt.xlabel("Age")

plt.ylabel("Count")

sns.countplot(age)



mean_age=round(np.mean(age),0)

print("The average age of death is:",mean_age,"years old")
#plt.figure(13)

#HOME_STATE=vc["HOME_STATE"]

#fig = plt.figure( figsize = ( 16 , 12 ) )

#plt.title("Home State")

#hist_plot(HOME_STATE,'vertical')



#HOME_STATE_count=Counter(HOME_STATE)

#HOME_STATE_count=sorted(HOME_STATE_count, key=HOME_STATE_count.get, reverse=True)

#print("The US state with most casualties is:",HOME_STATE_count[0])
import plotly.offline as py

py.init_notebook_mode(connected=True)



for col in vc.columns:

    vc[col] = vc[col].astype(str)



scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



labels, values = zip(*Counter(vc['STATE_CODE']).items())





data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = labels,

        z = np.array(values).astype(float),

        locationmode = 'USA-states',

        text = labels,

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "US casualties")

        ) ]



layout = dict(

        title = 'Origin of US casualties in Vietnam war<br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

py.iplot( fig, filename='Vietnam war US casualties' )
POSITION=vc["POSITION"]

POSITION_count=Counter(POSITION)

POSITION_count=sorted(POSITION_count, key=POSITION_count.get, reverse=True)





print("The most dangerous position was:",POSITION_count[0])
fig=plt.figure(14,figsize=(14,7))

Fatality_year=FATALITY_DATE.dt.year

k = Series(Fatality_year)

vc4 = k.value_counts()

vc4 = vc4.sort_index()

plt.ylabel("Count")

plt.title("Number of US soldiers killed during the war")

plt.plot([1963, 1963], [0, 109], 'k-', lw=2)

plt.text(1963, 109, 'Battle of Ap Bac',color='black',horizontalalignment='right')

plt.plot([1965, 1965], [0, 1656], 'k-', lw=2)

plt.text(1965, 1656, 'Battle of Ia Drang',color='black',horizontalalignment='right')

plt.plot([1966, 1966], [0, 5849], 'k-', lw=2)

plt.text(1966, 5849, 'Battle of Long Tan',color='black',horizontalalignment='right')

plt.plot([1967, 1967], [0, 10752], 'k-', lw=2)

plt.text(1967, 10752, '1st Battle of Khe Sanh & Battle of Dak To',color='black',horizontalalignment='right')

plt.plot([1968, 1968], [0, 15849], 'k-', lw=2)

plt.text(1968, 16092, '1st Tet Offensive & 1st Battle of Saigon',color='black',horizontalalignment='center')

plt.plot([1969, 1969], [0, 11210], 'k-', lw=2)

plt.text(1969, 11210, '2nd Tet Offensive & Battle of Hamburger Hill',color='black',horizontalalignment='left')

plt.plot([1970, 1970], [0, 5909], 'k-', lw=2)

plt.text(1970, 6100, 'Battle of Ripcord',color='black',horizontalalignment='left')

plt.plot([1972, 1972], [0, 515], 'k-', lw=2)

plt.text(1972, 515, 'Easter Offensive',color='black',horizontalalignment='left')

plt.plot([1974, 1974], [0, 17000], 'r-', lw=2)

plt.text(1974, 17000, 'US bombing end',color='black',horizontalalignment='center')

plt.plot(vc4)

fig.show()
plt.figure(15)

RELIGION=vc["RELIGION"]

fig = plt.figure( figsize = ( 10 , 5 ) )

plt.title("Religion")

hist_plot(RELIGION,'vertical')



RELIGION_count=Counter(RELIGION)

RELIGION_count=sorted(RELIGION_count, key=RELIGION_count.get, reverse=True)



print("The Religion with most casualties is:",RELIGION_count[0],"and",RELIGION_count[1])

print("The Religion with the least casualties are:",RELIGION_count[33],";",RELIGION_count[32],"and",RELIGION_count[31])