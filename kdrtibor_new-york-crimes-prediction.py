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
dataset = pd.read_csv('../input/NYPD_Complaint_Data_Historic.csv',low_memory = False)
types = dataset.OFNS_DESC.unique()
print(str(types))

#assault murders and homicides per new york
murders = dataset[(dataset.OFNS_DESC == 'MURDER & NON-NEGL. MANSLAUGHTER') | (dataset.OFNS_DESC == 'HOMICIDE-NEGLIGENT,UNCLASSIFIE') | (dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES')]
nrmurders = len(murders)
print(len(murders))

nonmurders = dataset[(dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
nrnonmurders = len(nonmurders)
print(len(nonmurders))

print('Percentage: %f'% ((nrmurders / (nrmurders + nrnonmurders) )* 100))
#assault per new york
murders = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES')]
nrmurders = len(murders)
print(len(murders))

nonmurders = dataset[(dataset.OFNS_DESC != 'MURDER & NON-NEGL. MANSLAUGHTER')]
nrnonmurders = len(nonmurders)
print(len(nonmurders))

print('Percentage: %f'% ((nrmurders / (nrmurders + nrnonmurders) )* 100))
#boroughs of new york
types = dataset.BORO_NM.unique()
print(str(types))
#assault in bronx
assaults = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES') & (dataset.BORO_NM == 'BRONX')]
nr_assaults = len(assaults)
print(nr_assaults)

noCrimesInBronx = dataset[(dataset.BORO_NM == 'BRONX') & (dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
nrBronx = len(noCrimesInBronx)
print(nrBronx)

print('Percentage: %f'% ((nr_assaults / (nrBronx + nr_assaults) )* 100))
#assault in queens
assaults = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES') & (dataset.BORO_NM == 'QUEENS')]
nr_assaults = len(assaults)
print(nr_assaults)

noCrimesInTotal= dataset[(dataset.BORO_NM == 'QUEENS') & (dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
noCrimes = len(noCrimesInTotal)
print(noCrimes)

print('Percentage: %f'% ((nr_assaults / (noCrimes + nr_assaults) )* 100))
#assault in manhatten
assaults = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES') & (dataset.BORO_NM == 'MANHATTAN')]
nr_assaults = len(assaults)
print(nr_assaults)

noCrimesInTotal= dataset[(dataset.BORO_NM == 'MANHATTAN') & (dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
noCrimes = len(noCrimesInTotal)
print(noCrimes)

print('Percentage: %f'% ((nr_assaults / (noCrimes + nr_assaults) )* 100))
#assault in brooklin
assaults = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES') & (dataset.BORO_NM == 'BROOKLYN')]
nr_assaults = len(assaults)
print(nr_assaults)

noCrimesInTotal= dataset[(dataset.BORO_NM == 'BROOKLYN') & (dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
noCrimes = len(noCrimesInTotal)
print(noCrimes)

print('Percentage: %f'% ((nr_assaults / (noCrimes + nr_assaults) )* 100))
#assault in statan island
assaults = dataset[(dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES') & (dataset.BORO_NM == 'STATEN ISLAND')]
nr_assaults = len(assaults)
print(nr_assaults)

noCrimesInTotal= dataset[(dataset.BORO_NM == 'STATEN ISLAND') & (dataset.OFNS_DESC != 'ASSAULT 3 & RELATED OFFENSES')]
noCrimes = len(noCrimesInTotal)
print(noCrimes)

print('Percentage: %f'% ((nr_assaults / (noCrimes + nr_assaults) )* 100))
#convert the date column into month column
date = dataset.CMPLNT_FR_DT
time = dataset.CMPLNT_FR_TM

bad_date = date.loc[date.str.contains('1015', na = False)]

#correct the bad datas
date[27341] = '12/04/2015'
date[27342] = '12/04/2015'
date[39052] = '11/25/2015'
date[48548] = '09/26/2015'
date[72345] = '10/27/2015'
date[89523] = '10/17/2015'
date[119248] = '09/16/2015'

print('potogan')
print (bad_date)

date = pd.to_datetime(date)

print (date)
print(date.dtype)