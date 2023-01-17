# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#The Dataset

import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

BRFSS = pd.read_csv("../input/2015.csv")

#'_AGE_G', 'SEX,EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT'

BRFSS1 = BRFSS[['_AGEG5YR', 'SEX','EDUCA','EMPLOY1','INCOME2','_RACE','NUMADULT','MARITAL','VETERAN3','PREGNANT','MENTHLTH','ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']]

BRFSS1.head(5)
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

BRFSS1.isnull().count()
#Dropping the respondat that this answered the 8 questions

BRFSS1 = BRFSS1.dropna(subset=['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE'],how='all')

BRFSS1.head(10)
#To have a reference of the individual values after dropping respondant that hasn't answer the 8 questions

BRFSS1.isnull().count()
#A variable to have in one place for later

BRFSS_M30= BRFSS['MENTHLTH']

BRFSS_M30.head(5)
TRAINDATA = BRFSS1.dropna().reset_index()

TRAINDATA.head(5)
TRAINDATA.shape
TRAINDATAL = TRAINDATA.iloc[:, 0]

TRAINDATAI = TRAINDATA.iloc[:, 1:]



TRAINDATAL

TRAINDATAI
TESTDATA= BRFSS1.reset_index()

TESTDATA.head(10)
TESTDATAL= TESTDATA.loc[:10]

TESTDATAL[:10]
print(TESTDATA.shape)
TRAINDATAL_1 = TRAINDATA.loc[: 800]

TRAINDATAI_1 = TRAINDATA.loc[:800:]



BRFSS_MODEL1 = KNeighborsRegressor(n_neighbors=10) 

BRFSS_MODEL1.fit(TRAINDATAL_1, TRAINDATAI_1)
PREDICTION = BRFSS_MODEL1.predict(TRAINDATA)

print(PREDICTION)
from fancyimpute import KNN  



BRFSS_MODEL1 = KNN(k=10).fit_transform(BRFSS1)   
#Convert numpy array back to DataFrame after the imputation is done

BRFSS_MODEL1 = pd.DataFrame(BRFSS_MODEL1)



#Give back the columns it had before the imputation

BRFSS_MODEL1.columns = BRFSS1.columns



#Give back the index it had before the imputation

BRFSS_MODEL1.index = BRFSS1.index



BRFSS_MODEL1.head()
#Adding up the days and creating a column to save them

BRFSS_MODEL1['SUM']= BRFSS_MODEL1[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']].sum(axis=1)

BRFSS_MODEL1.head()
BRFSS_MODEL1['SUM'].where(BRFSS_MODEL1['SUM']>=56).count()
BRFSS_MODEL1.isnull().sum()
x = len(BRFSS_MODEL1[(BRFSS_MODEL1['MENTHLTH']==30) & (BRFSS_MODEL1['SUM']>=56)])

x
y = len(BRFSS_MODEL1[(BRFSS_MODEL1['MENTHLTH']==30) & (BRFSS_MODEL1['SUM']<56)])

y
(x/(x+y))*100