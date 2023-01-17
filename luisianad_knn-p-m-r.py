import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

from sklearn.neighbors import KNeighborsRegressor
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
#To have a reference of the individual values before dropping respondant that hasn't answer the 8 questions

BRFSS1.isnull().sum()

BRFSS1.shape
BRFSS1 = BRFSS1.dropna(subset=['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE'],how='all')

print(BRFSS1.shape)

BRFSS1.isnull().sum()
def isPregnant(x):

    if x['SEX'] == 1 or (x['_AGEG5YR'] >= 6 and x['_AGEG5YR'] <= 13):

        return 2

    else:

        return x['PREGNANT']



BRFSS1['PREGNANT'] = BRFSS.apply(isPregnant, axis=1)

BRFSS1['PREGNANT'].value_counts()
BRFSS1 = BRFSS1.dropna(subset=['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT'])

print(BRFSS1.shape)

BRFSS1.isnull().sum()


#Dropping NaN from the input

#Dropping the respondat that this answered the 8 questions

BRFSS1 = BRFSS1.dropna(subset=['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE'],how='all')

BRFSS1 = BRFSS1.reset_index(drop=True)

print(BRFSS1.head(10))

print("*"*100)

#To have a reference of the individual values after dropping respondant that hasn't answer the 8 questions

print(BRFSS1.isnull().count())

print("*"*100)

print(BRFSS1.shape)
#Data with complete rows

TRAINDATA = BRFSS1.dropna().reset_index(drop=True)

print(TRAINDATA.head(5))

print("*"*100)

TRAINDATA.shape
TRAINDATA_1 = TRAINDATA

TRAINDATA_1 = TRAINDATA_1.reset_index(drop=True)

print(TRAINDATA_1.head())

print("*"*100)

print(TRAINDATA_1.shape)

print("*"*100)

print(TRAINDATA_1.isnull().sum())

IN= ['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT']

ON= ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']

MODEL1 = KNeighborsRegressor(n_neighbors=1)

MODEL1.fit(TRAINDATA_1[IN],TRAINDATA_1[ON])

TESTDATA_1 = BRFSS1.reset_index(drop=True)

PRE_A = MODEL1.predict(TESTDATA_1[IN])

PRE_A = pd.DataFrame(PRE_A)

BRFSS_T1= pd.DataFrame()

BRFSS_T1 = BRFSS1[IN]

BRFSS_T1["MENTHLTH"] = BRFSS1["MENTHLTH"]

BRFSS_T1[ON] = PRE_A

print(BRFSS_T1.isnull().sum())

BRFSS_T1.head()

#Adding up the days and creating a column to save them

BRFSS_T1['SUM']= BRFSS_T1[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']].sum(axis=1)

print(BRFSS_T1.head())

print(BRFSS_T1['SUM'].where(BRFSS_T1['SUM']>=56).count())

print(BRFSS_T1.isnull().sum())

x = len(BRFSS_T1[(BRFSS_T1['MENTHLTH']==30) & (BRFSS_T1['SUM']>=56)])

y = len(BRFSS_T1[(BRFSS_T1['MENTHLTH']==30) & (BRFSS_T1['SUM']<56)])

print(x,y)

print("Ratio =", (x/(x+y))*100)
TRAINDATA_2 = TRAINDATA

TRAINDATA_2 = TRAINDATA_2.reset_index(drop=True)

print(TRAINDATA_2.head())

print("*"*100)

print(TRAINDATA_2.shape)

print("*"*100)

print(TRAINDATA_2.isnull().sum())

IN= ['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT']

ON= ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']

MODEL2 = KNeighborsRegressor(n_neighbors=3)

MODEL2.fit(TRAINDATA_2[IN],TRAINDATA_2[ON])

TESTDATA_2 = BRFSS1.reset_index(drop=True)

PRE_B = MODEL2.predict(TESTDATA_2[IN])

PRE_B = pd.DataFrame(PRE_B)

BRFSS_T2= pd.DataFrame()

BRFSS_T2 = BRFSS1[IN]

BRFSS_T2["MENTHLTH"] = BRFSS1["MENTHLTH"]

BRFSS_T2[ON] = PRE_B

print(BRFSS_T2.isnull().sum())

BRFSS_T2.head()
#Adding up the days and creating a column to save them

BRFSS_T2['SUM']= BRFSS_T2[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']].sum(axis=1)

print(BRFSS_T2.head())

print(BRFSS_T2['SUM'].where(BRFSS_T2['SUM']>=56).count())

print(BRFSS_T2.isnull().sum())

x2 = len(BRFSS_T2[(BRFSS_T2['MENTHLTH']==30) & (BRFSS_T2['SUM']>=56)])

y2 = len(BRFSS_T2[(BRFSS_T2['MENTHLTH']==30) & (BRFSS_T2['SUM']<56)])

print(x2,y2)

print("Ratio =", (x2/(x2+y2))*100)
TRAINDATA_3 = TRAINDATA

TRAINDATA_3 = TRAINDATA_3.reset_index(drop=True)

print(TRAINDATA_3.head())

print("*"*100)

print(TRAINDATA_3.shape)

print("*"*100)

print(TRAINDATA_3.isnull().sum())

IN= ['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT']

ON= ['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']

MODEL3 = KNeighborsRegressor(n_neighbors=5)

MODEL3.fit(TRAINDATA_3[IN],TRAINDATA_3[ON])

TESTDATA_3 = BRFSS1.reset_index(drop=True)

PRE_C = MODEL3.predict(TESTDATA_3[IN])

PRE_C = pd.DataFrame(PRE_C)

BRFSS_T3= pd.DataFrame()

BRFSS_T3 = BRFSS1[IN]

BRFSS_T3["MENTHLTH"] = BRFSS1["MENTHLTH"]

BRFSS_T3[ON] = PRE_C

print(BRFSS_T3.isnull().sum())

BRFSS_T3.head()
#Adding up the days and creating a column to save them

BRFSS_T3['SUM']= BRFSS_T3[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']].sum(axis=1)

print(BRFSS_T3.head())

print(BRFSS_T3['SUM'].where(BRFSS_T3['SUM']>=56).count())

print(BRFSS_T3.isnull().sum())

x3 = len(BRFSS_T3[(BRFSS_T3['MENTHLTH']==30) & (BRFSS_T3['SUM']>=56)])

y3 = len(BRFSS_T3[(BRFSS_T3['MENTHLTH']==30) & (BRFSS_T3['SUM']<56)])

print(x3,y3)

print("Ratio =", (x3/(x3+y3))*100)
# ---COLUMNS----

COL1 = BRFSS1["_AGEG5YR"]

COL2 = BRFSS1["SEX"]

COL3 = BRFSS1["EDUCA"]

COL4 = BRFSS1["EMPLOY1"]

COL5 = BRFSS1["INCOME2"]

COL6 = BRFSS1["_RACE"]

COL7 = BRFSS1["NUMADULT"]

COL8 = BRFSS1["MARITAL"]

COL9 = BRFSS1["VETERAN3"]

COL10 = BRFSS1["PREGNANT"]



# ---MIN----

MIN1 = COL1.min()

MIN2 = COL2.min()

MIN3 = COL3.min()

MIN4 = COL4.min()

MIN5 = COL5.min()

MIN6 = COL6.min()

MIN7 = COL7.min()

MIN8 = COL8.min()

MIN9 = COL9.min()

MIN10 = COL10.min()



# ---MAX----

MAX1 = COL1.max()

MAX2 = COL2.max()

MAX3 = COL3.max()

MAX4 = COL4.max()

MAX5 = COL5.max()

MAX6 = COL6.max()

MAX7 = COL7.max()

MAX8 = COL8.max()

MAX9 = COL9.max()

MAX10 = COL10.max()



BRFSS1['_AGEG5Y_SCALATED'] = (COL1 - MIN1) / (MAX1 - MIN1)

BRFSS1['SEX_SCALATED'] = (COL2 - MIN2) / (MAX2 - MIN2)

BRFSS1['EDUCA_SCALATED'] = (COL3 - MIN3) / (MAX3 - MIN3)

BRFSS1['EMPLOY1_SCALATED'] = (COL4 - MIN4) / (MAX4 - MIN4)

BRFSS1['INCOME2_SCALATED'] = (COL5 - MIN5) / (MAX5 - MIN5)

BRFSS1['_RACE_SCALATED'] = (COL6 - MIN6) / (MAX6 - MIN6)

BRFSS1['NUMADULT_SCALATED'] = (COL7 - MIN7) / (MAX7 - MIN7)

BRFSS1['MARITAL_SCALATED'] = (COL8 - MIN8) / (MAX8 - MIN8)

BRFSS1['VETERAN3_SCALATED'] = (COL9 - MIN9) / (MAX9 - MIN9)

BRFSS1['PREGNAT_SCALATED'] = (COL10 - MIN10) / (MAX10 - MIN10)
BRFSS1['_AGEG5Y_SCALATED'].min()

BRFSS1['SEX_SCALATED'].min()

BRFSS1['EDUCA_SCALATED'].min()

BRFSS1['EMPLOY1_SCALATED'].min()

BRFSS1['INCOME2_SCALATED'].min()

BRFSS1['_RACE_SCALATED'].min()

BRFSS1['NUMADULT_SCALATED'].min()

BRFSS1['MARITAL_SCALATED'].min()

BRFSS1['VETERAN3_SCALATED'].min()

BRFSS1['PREGNAT_SCALATED'].min()



BRFSS1['_AGEG5Y_SCALATED'].max()

BRFSS1['SEX_SCALATED'].max() 

BRFSS1['EDUCA_SCALATED'].max()

BRFSS1['EMPLOY1_SCALATED'].max()

BRFSS1['INCOME2_SCALATED'].max()

BRFSS1['_RACE_SCALATED'].max()

BRFSS1['NUMADULT_SCALATED'].max()

BRFSS1['MARITAL_SCALATED'].max()

BRFSS1['VETERAN3_SCALATED'].max()

BRFSS1['PREGNAT_SCALATED'].max()
print(BRFSS1['_AGEG5Y_SCALATED'].min())

print(BRFSS1['SEX_SCALATED'].min())

print(BRFSS1['EDUCA_SCALATED'].min())

print(BRFSS1['EMPLOY1_SCALATED'].min())

print(BRFSS1['INCOME2_SCALATED'].min())

print(BRFSS1['_RACE_SCALATED'].min())

print(BRFSS1['NUMADULT_SCALATED'].min())

print(BRFSS1['MARITAL_SCALATED'].min())

print(BRFSS1['VETERAN3_SCALATED'].min())

print(BRFSS1['PREGNAT_SCALATED'].min())



print(BRFSS1['_AGEG5Y_SCALATED'].max())

print(BRFSS1['SEX_SCALATED'].max())

print(BRFSS1['EDUCA_SCALATED'].max())

print(BRFSS1['EMPLOY1_SCALATED'].max())

print(BRFSS1['INCOME2_SCALATED'].max())

print(BRFSS1['_RACE_SCALATED'].max())

print(BRFSS1['NUMADULT_SCALATED'].max())

print(BRFSS1['MARITAL_SCALATED'].max())

print(BRFSS1['VETERAN3_SCALATED'].max())

print(BRFSS1['PREGNAT_SCALATED'].max())
BRFSS_RANGES = pd.DataFrame()
BRFSS_RANGES = BRFSS1[['_AGEG5Y_SCALATED','SEX','EDUCA_SCALATED','EMPLOY1_SCALATED','INCOME2_SCALATED','_RACE_SCALATED','NUMADULT_SCALATED','MARITAL_SCALATED','VETERAN3_SCALATED','PREGNAT_SCALATED']] 

BRFSS_RANGES['MENTHLTH'] = BRFSS1['MENTHLTH']

BRFSS_RANGES[ON] = BRFSS1[ON]

BRFSS_RANGES.head()
BRFSS_RANGES.isnull().sum()
IN_S = ['_AGEG5Y_SCALATED','SEX','EDUCA_SCALATED','EMPLOY1_SCALATED','INCOME2_SCALATED','_RACE_SCALATED','NUMADULT_SCALATED','MARITAL_SCALATED','VETERAN3_SCALATED','PREGNAT_SCALATED']

TRAINDATA_R = BRFSS_RANGES.dropna().reset_index(drop=True)

print(TRAINDATA_R.head())

print("*"*100)

print(TRAINDATA_R.shape)

print("*"*100)

print(TRAINDATA_R.isnull().sum())

MODELR = KNeighborsRegressor(n_neighbors=1)

MODELR.fit(TRAINDATA_R[IN_S],TRAINDATA_R[ON])

TESTDATA_R = BRFSS_RANGES.reset_index(drop=True)

PRE_R = MODELR.predict(TESTDATA_R[IN_S])

PRE_R = pd.DataFrame(PRE_R)

BRFSS_TR= pd.DataFrame()

BRFSS_TR = BRFSS_RANGES[IN_S]

BRFSS_TR["MENTHLTH"] = BRFSS_RANGES["MENTHLTH"]

BRFSS_TR[ON] = PRE_R

print(BRFSS_TR.isnull().sum())

BRFSS_TR.head()
#Adding up the days and creating a column to save them

BRFSS_TR['SUM']= BRFSS_TR[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']].sum(axis=1)

print(BRFSS_TR.head())

print(BRFSS_TR['SUM'].where(BRFSS_T3['SUM']>=56).count())

print(BRFSS_TR.isnull().sum())

xR = len(BRFSS_TR[(BRFSS_TR['MENTHLTH']==30) & (BRFSS_TR['SUM']>=56)])

yR = len(BRFSS_TR[(BRFSS_TR['MENTHLTH']==30) & (BRFSS_TR['SUM']<56)])

print(xR,yR)

print("Ratio =", (xR/(xR+yR))*100)