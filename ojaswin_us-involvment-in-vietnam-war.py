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

vc = vc[["SERVICE_TYPE","SERVICE_CODE","ENROLLMENT","BRANCH","RANK","PAY_GRADE",

"POSITION","BIRTH_YEAR","SEX","HOME_CITY","HOME_COUNTY","NATIONALITY","STATE_CODE",

"HOME_STATE","MARITAL_STATUS","RELIGION","RELIGION_CODE","ETHNICITY","ETHNICITY_1",

"ETHNICITY_2","DEPLOYMENT_PROVINCE","DEPLOYMENT_ZONE","DEPLOYMENT_COUNTRY_CODE",

"DEPLOYMENT_COUNTRY","DIVISION","START_YEAR","START_DATE","FATALITY_DATE",

"HOSTILITY_CONDITIONS","FATALITY","FATALITY_2","BURIAL_STATUS"]]
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

print("There were",num_cas,"casualties in the war" )
plt.figure(1)

SERVICE_TYPE=vc["SERVICE_TYPE"]

plt.title('SERVICE_TYPE')

hist_plot(SERVICE_TYPE,'horizontal')



q = Series(SERVICE_TYPE)

SERVICE_TYPE2 = q.value_counts()

SERVICE_TYPE2 = SERVICE_TYPE2.sort_index()

comm_SERVICE_TYPE=SERVICE_TYPE2.idxmax(axis=1)



print("The most common service type is:",comm_SERVICE_TYPE)
plt.figure(2)

ENROLLMENT=vc["ENROLLMENT"]

plt.ylabel("Count")

plt.title('ENROLLMENT')

ENROLLMENT.value_counts().plot(kind='bar')
plt.figure(3)

BRANCH=vc["BRANCH"]

plt.ylabel("Count")

plt.title('BRANCH')

BRANCH.value_counts().plot(kind='bar')
fig = plt.figure(4, figsize = ( 16 , 8 ) )

RANK=vc["RANK"]

plt.ylabel("Count")

plt.title('RANK')

RANK.value_counts().plot(kind='bar')
plt.figure(5)

SEX=vc["SEX"]

plt.ylabel("Count")

plt.title('SEX')

SEX.value_counts().plot(kind='bar')



sex_count=Counter(SEX)

print("There were",sex_count['F'], "women and",sex_count['M'],"men casualties who died in the war.")
plt.figure(6)

MARITAL_STATUS=vc["MARITAL_STATUS"]

plt.ylabel("Count")

plt.title('MARITAL_STATUS')

MARITAL_STATUS.value_counts().plot(kind='bar')
plt.figure(7)

ETHNICITY=vc["ETHNICITY"]

plt.ylabel("Count")

plt.title("ETHNICITY")

ETHNICITY.value_counts().plot(kind='bar')
plt.figure(8)

DEPLOYMENT_COUNTRY=vc["DEPLOYMENT_COUNTRY"]

plt.ylabel("Count")

plt.title("DEPLOYMENT_COUNTRY")

DEPLOYMENT_COUNTRY.value_counts().plot(kind='bar')
plt.figure(9)

HOSTILITY_CONDITIONS=vc["HOSTILITY_CONDITIONS"]

plt.title("HOSTILITY_CONDITIONS")

HOSTILITY_CONDITIONS.value_counts().plot(kind='bar')
plt.figure(10)

fatal=vc["FATALITY"]

plt.ylabel("Count")

plt.title("FATALITY")

fatal.value_counts().plot(kind='bar')
plt.figure(11)

FATALITY_2=vc["FATALITY_2"]

plt.title("FATALITY_2")

hist_plot(FATALITY_2,'vertical')
plt.figure(12)

START_YEAR=vc["START_YEAR"]

fig = plt.figure( figsize = ( 12 , 6 ) )

plt.ylabel("Count")

sns.countplot(START_YEAR)
BIRTH_YEAR=vc["BIRTH_YEAR"]

FATALITY_DATE=vc["FATALITY_DATE"]



BIRTH_YEAR = pd.to_datetime(BIRTH_YEAR, format='%Y%m%d')

FATALITY_DATE = pd.to_datetime(FATALITY_DATE, format='%Y%m%d')

age = BIRTH_YEAR.where(BIRTH_YEAR < FATALITY_DATE, BIRTH_YEAR -  np.timedelta64(100, 'Y'))

age = (FATALITY_DATE - BIRTH_YEAR).astype('<m8[Y]')





plt.figure(13)

fig = plt.figure( figsize = ( 25 , 10 ) )

plt.xlabel("Age")

plt.ylabel("Count")

sns.countplot(age)



mean_age=round(np.mean(age),0)

print("The avarage age of death is:",mean_age,"years old")
plt.figure(14)

HOME_STATE=vc["HOME_STATE"]

fig = plt.figure( figsize = ( 12 , 6 ) )

plt.title("HOME_STATE")

hist_plot(HOME_STATE,'vertical')



HOME_STATE_count=Counter(HOME_STATE)

HOME_STATE_count=sorted(HOME_STATE_count, key=HOME_STATE_count.get, reverse=True)

print("The US state with most casualties is:",HOME_STATE_count[0])
POSITION=vc["POSITION"]

POSITION_count=Counter(POSITION)

POSITION_count=sorted(POSITION_count, key=POSITION_count.get, reverse=True)



print("The most dangerus position was:",POSITION_count[0])
plt.figure(15)

RELIGION=vc["RELIGION"]

fig = plt.figure( figsize = ( 12 , 6 ) )

plt.title("RELIGION")

hist_plot(RELIGION,'vertical')



RELIGION_count=Counter(RELIGION)

RELIGION_count=sorted(RELIGION_count, key=RELIGION_count.get, reverse=True)



print("The RELIGION with most casualties is:",RELIGION_count[0],"and",RELIGION_count[1])

print("The RELIGION with the least casualties are:",RELIGION_count[33],";",RELIGION_count[32],"and",RELIGION_count[31])