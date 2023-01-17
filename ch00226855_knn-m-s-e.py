import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

%matplotlib inline

#The Dataset

BRFSS = pd.read_csv("../input/2015.csv")

BRFSS1 = BRFSS[['MENTHLTH','_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT','ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']]

BRFSS1.head()
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
def isPregnant(x):

    if x['SEX'] == 1 or (x['_AGEG5YR'] >= 6 and x['_AGEG5YR'] <= 13):

        return 2

    else:

        return x['PREGNANT']



BRFSS1['PREGNANT'] = BRFSS.apply(isPregnant, axis=1)

BRFSS1['PREGNANT'].value_counts()
#Dropping the null values and resetting the index to make things easier

BRFSS1 = BRFSS1.dropna()

BRFSS1 = BRFSS1.reset_index(drop=True)

BRFSS1.head(5)
#Making sure that they were drop

BRFSS1.isnull().sum()
#Dividing the complete set. Training data is 85% and Test data is 15%

from sklearn.model_selection import train_test_split

TRAINDATA, TESTDATA= train_test_split(BRFSS1, test_size=0.15)

print(TRAINDATA.shape)

print(TESTDATA.shape)
TRAINDATA.head()
TESTDATA.head()
Model1 = KNeighborsRegressor(n_neighbors=11)

Model1
IN = ['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT']

ON = ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']



Model1.fit(TRAINDATA[IN],TRAINDATA[ON])
COPYTEST = TESTDATA.copy()

COPYTEST.head()
for i in COPYTEST.index:

    COPYTEST.loc[i, np.random.choice(ON, size=2)] = np.NaN

COPYTEST
PREDICTION = Model1.predict(COPYTEST[IN])

PREDICTION
pd.isnull(COPYTEST).head()
COPYTESTIN = COPYTEST.copy()

COPYTESTIN.head()
a = 'ADPLEASR_PREDICTION'

b = 'ADDOWN_PREDICTION'

c = 'ADSLEEP_PREDICTION'

d = 'ADENERGY_PREDICTION'

e = 'ADEAT1_PREDICTION'

f = 'ADFAIL_PREDICTION'

g = 'ADTHINK_PREDICTION'

h = 'ADMOVE_PREDICTION'

COPYTESTIN[a] = PREDICTION[:, 0]

COPYTESTIN[b] = PREDICTION[:, 1]

COPYTESTIN[c] = PREDICTION[:, 2]

COPYTESTIN[d] = PREDICTION[:, 3]

COPYTESTIN[e] = PREDICTION[:, 4]

COPYTESTIN[f] = PREDICTION[:, 5]

COPYTESTIN[g] = PREDICTION[:, 6]

COPYTESTIN[h] = PREDICTION[:, 7]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"] =  COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADPLEASR"]), a]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADDOWN"]), b]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADSLEEP"]), c]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADENERGY"]), d]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADEAT1"]), e]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADFAIL"]), f]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADTHINK"]), g]

COPYTESTIN.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"] = COPYTESTIN.loc[pd.isnull(COPYTESTIN["ADMOVE"]), h]

COPYTESTIN
for name in ON:

    print(name+"_ERROR")
aa = 'ADPLEASR_ERROR'

bb = 'ADDOWN_ERROR'

cc = 'ADSLEEP_ERROR'

dd = 'ADENERGY_ERROR'

ee = 'ADEAT1_ERROR'

ff = 'ADFAIL_ERROR'

gg = 'ADTHINK_ERROR'

hh = 'ADMOVE_ERROR'

COPYTESTIN.loc[pd.isnull(COPYTEST["ADPLEASR"]), aa] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"] - TESTDATA.loc[pd.isnull(COPYTEST["ADPLEASR"]), "ADPLEASR"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADDOWN"]), bb] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"] - TESTDATA.loc[pd.isnull(COPYTEST["ADDOWN"]), "ADDOWN"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADSLEEP"]), cc] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"] - TESTDATA.loc[pd.isnull(COPYTEST["ADSLEEP"]), "ADSLEEP"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADENERGY"]), dd] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"] - TESTDATA.loc[pd.isnull(COPYTEST["ADENERGY"]), "ADENERGY"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADEAT1"]), ee] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"] - TESTDATA.loc[pd.isnull(COPYTEST["ADEAT1"]), "ADEAT1"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADFAIL"]), ff] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"] - TESTDATA.loc[pd.isnull(COPYTEST["ADFAIL"]), "ADFAIL"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADTHINK"]), gg] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"] - TESTDATA.loc[pd.isnull(COPYTEST["ADTHINK"]), "ADTHINK"]) ** 2

COPYTESTIN.loc[pd.isnull(COPYTEST["ADMOVE"]), hh] = (COPYTESTIN.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"] - TESTDATA.loc[pd.isnull(COPYTEST["ADMOVE"]), "ADMOVE"]) ** 2

COPYTESTIN.head()
ERROR = [aa, bb, cc,dd,ee,ff,gg,hh]

SUMERROR = np.sum(np.sum(COPYTESTIN[ERROR]))

ERRORx = np.sum(np.sum(pd.notnull(COPYTESTIN[ERROR])))

print(SUMERROR)

print(ERRORx)

print("M.S.E:", SUMERROR / ERRORx)