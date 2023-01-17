import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
#The Dataset

BRFSS = pd.read_csv("../input/2015.csv")

BRFSS1 = BRFSS[['MENTHLTH','_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT','ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']]

BRFSS1.head()
BRFSS1.isnull().sum()
#Changing values from 77,88,7,8,9,14 to NAN | Also, changing 88 values to 0

for x in ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']:

    BRFSS1[x].replace(77, np.NaN, inplace= True)

    BRFSS1[x].replace(99, np.NaN, inplace= True)

    BRFSS1[x].replace(88, 0, inplace= True)

    #Making sure the changes where made

    print(x,":Values 77 \n",BRFSS1[x].where(BRFSS1[x]==77).count())

    print(x,":Values 99 \n",BRFSS1[x].where(BRFSS1[x]==99).count())

    print(x,":Values 88 \n",BRFSS1[x].where(BRFSS1[x]==88).count())



for x in ['EDUCA','EMPLOY1','_RACE', 'MARITAL']:

    BRFSS1[x].replace(9, np.NaN, inplace=True)

    #Making sure the changes where made

    print(x,":Values 9 \n",BRFSS1[x].where(BRFSS1[x]==9).count())

    

for x in ['VETERAN3','PREGNANT']:

    BRFSS1[x].replace(9, np.NaN, inplace= True)

    BRFSS1[x].replace(7, np.NaN, inplace= True)

    #Making sure the changes where made

    print(x,":Values 9 \n",BRFSS1[x].where(BRFSS1[x]==9).count())

    print(x,":Values 7 \n",BRFSS1[x].where(BRFSS1[x]==7).count())



BRFSS1['_AGEG5YR'].replace(14, np.NaN, inplace= True)

BRFSS1['INCOME2'].replace(77, np.NaN, inplace= True)

BRFSS1['INCOME2'].replace(99, np.NaN, inplace= True)

BRFSS1['MENTHLTH'].replace(88, 0, inplace= True)



#Making sure the changes where made

print("_AGEG5YR:Values 14 \n",BRFSS1['_AGEG5YR'].where(BRFSS1['_AGEG5YR']==14).count())

print("INCOME2:Values 77 \n",BRFSS1['INCOME2'].where(BRFSS1['INCOME2']==77).count())

print("INCOME2:Values 99 \n",BRFSS1['INCOME2'].where(BRFSS1['INCOME2']==99).count())

print("MENTHLTH:Values 88 \n",BRFSS1['MENTHLTH'].where(BRFSS1['MENTHLTH']==88).count())
BRFSS1.isnull().sum()
def isPregnant(x):

    if x['SEX'] == 1 or (x['_AGEG5YR'] >= 6 and x['_AGEG5YR'] <= 13):

        return 2

    else:

        return x['PREGNANT']



BRFSS1['PREGNANT'] = BRFSS.apply(isPregnant, axis=1)

BRFSS1['PREGNANT'].value_counts()
BRFSS1.isnull().sum()
#Dropping the null values and resetting the index to make things easier

BRFSS1 = BRFSS1.dropna()

BRFSS1 = BRFSS1.reset_index(drop=True)

BRFSS1.head(5)
print(BRFSS1.shape)

print(BRFSS1.isnull().sum())
#Dropping the null values and resetting the index to make things easier

TRAINDATA, TESTDATA= train_test_split(BRFSS1, test_size=0.15)

print(TRAINDATA.shape)

print(TESTDATA.shape)
print(TRAINDATA.head())

print(TESTDATA.head())
print(TRAINDATA.isnull().sum())

print(TESTDATA.isnull().sum())
COPYTEST = TESTDATA.copy()

COPYTEST.head()
IN = ['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT']

ON = ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']



for i in COPYTEST.index:

    COPYTEST.loc[i, np.random.choice(ON, size=2)] = np.NaN

COPYTEST
pd.isnull(COPYTEST).head()
TESTDATA.isnull().sum()
COPYTEST_NULL = COPYTEST.copy()

COPYTEST_NULL.head()
for x in ['ADPLEASR', 'ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']:

    for i in COPYTEST_NULL.index:

        if pd.isnull(COPYTEST_NULL.loc[i, x]):

            COPYTEST_IN = np.random.choice(BRFSS1[x])

            COPYTEST_NULL.loc[i, x]=COPYTEST_IN
TESTDATA.isnull().sum()
print(COPYTEST_NULL.isnull().sum())

print(COPYTEST_NULL.head())
COPYTEST_IN= COPYTEST.copy()

COPYTEST_IN.head()
a = 'ADPLEASR_PREDICTION'

b = 'ADDOWN_PREDICTION'

c = 'ADSLEEP_PREDICTION'

d = 'ADENERGY_PREDICTION'

e = 'ADEAT1_PREDICTION'

f = 'ADFAIL_PREDICTION'

g = 'ADTHINK_PREDICTION'

h = 'ADMOVE_PREDICTION'
COPYTEST_IN[a] = COPYTEST_NULL["ADPLEASR"]

COPYTEST_IN[b] = COPYTEST_NULL["ADDOWN"]

COPYTEST_IN[c] = COPYTEST_NULL["ADSLEEP"]

COPYTEST_IN[d] = COPYTEST_NULL["ADENERGY"]

COPYTEST_IN[e] = COPYTEST_NULL["ADEAT1"]

COPYTEST_IN[f] = COPYTEST_NULL["ADFAIL"]

COPYTEST_IN[g] = COPYTEST_NULL["ADTHINK"]

COPYTEST_IN[h] = COPYTEST_NULL["ADMOVE"]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"] =  COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADPLEASR"]), a]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADDOWN"]), b]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADSLEEP"]), c]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADENERGY"]), d]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADEAT1"]), e]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADFAIL"]), f]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADTHINK"]), g]

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"] = COPYTEST_IN.loc[pd.isnull(COPYTEST_IN["ADMOVE"]), h]

COPYTEST_IN.head(10)
aa = 'ADPLEASR_ERROR'

bb = 'ADDOWN_ERROR'

cc = 'ADSLEEP_ERROR'

dd = 'ADENERGY_ERROR'

ee = 'ADEAT1_ERROR'

ff = 'ADFAIL_ERROR'

gg = 'ADTHINK_ERROR'

hh = 'ADMOVE_ERROR'

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADPLEASR"]), aa] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"] - TESTDATA.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADDOWN"]), bb] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"] - TESTDATA.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADSLEEP"]), cc] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"] - TESTDATA.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADENERGY"]), dd] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"] - TESTDATA.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADEAT1"]), ee] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"] - TESTDATA.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADFAIL"]), ff] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"] - TESTDATA.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADTHINK"]), gg] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"] - TESTDATA.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"]) ** 2

COPYTEST_IN.loc[pd.isnull(COPYTEST["ADMOVE"]), hh] = (COPYTEST_IN.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"] - TESTDATA.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"]) ** 2

COPYTEST_IN.head()
ERROR = [aa, bb, cc,dd,ee,ff,gg,hh]

SUMERROR = np.sum(np.sum(COPYTEST_IN[ERROR]))

ERRORx = np.sum(np.sum(pd.notnull(COPYTEST_IN[ERROR])))

print(SUMERROR)

print(ERRORx)

print("M.S.E:", SUMERROR / ERRORx)