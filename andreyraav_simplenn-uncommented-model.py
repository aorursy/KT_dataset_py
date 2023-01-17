# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
np.random.seed(10)

import matplotlib.pyplot as plt



from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing



import keras

from keras.utils import to_categorical

from keras import models

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD

from keras.layers import Activation, Dense, Dropout

from sklearn.model_selection import train_test_split, StratifiedKFold

#Function that trims all text data

def trim_all_columns(df):

    trim_strings = lambda x: x.strip() if isinstance(x, str) else x

    return df.applymap(trim_strings)
header_list = ["AAGE", "ACLSWKR", "ADTIND", "ADTOCC", "AHGA","AHRSPAY","AHRSCOL",

               "AMARITL","AMJIND","AMJOCC","ARACE","AREORGN","ASEX","AUNMEM",

               "AUNTYPE","AWKSTAT","CAPGAIN","CAPLOSS","DIVVAL", "FILESTAT", 

               "GRINREG","GRINST","HHDFMX","HHDREL", "MARSUPWT","MIGMTR1", 

               "MIGMTR3","MIGMTR4","MIGSAME", "MIGSUN","NOEMP","PARENT", 

               "PEFNTVTY","PEMNTVTY","PENATVTY","PRCITSHP","SEOTR", 

               "VETQVA","VETYN","WKSWORK","YEAR","INCCAT"]

d_train=pd.read_csv('/kaggle/input/ml-challenge-week6/census-income.data',index_col=False,names=header_list)

d_test=pd.read_csv('/kaggle/input/ml-challenge-week6/census-income.test',index_col=False,names=header_list)
d_test=trim_all_columns(d_test)

d_train=trim_all_columns(d_train)
#we have to change the "50000+","-50000" format to a binary classifier format, so

label = {'50000+.':1, '- 50000.':0}

d_train['INCCAT']=d_train['INCCAT'].map(label)

d_test['INCCAT']=d_test['INCCAT'].map(label)



#Before going further there is some work to be done with with NaNs:

d_test['AREORGN']=d_test['AREORGN'].fillna('NA')

d_test=d_test.fillna('?')





#plotting differences between train and test to confirm that test is not biased

#df= pd.DataFrame({'x':d_train["AAGE"],

#    'y':d_test["AAGE"],})

#df.plot.kde()
#This is an insert that was made after analysis of the column 'AWKSTAT'.

#There was a mess in the 'Children ot Armed forces' category

#First - I will split out childrens - those that are 18 or below

d_train.loc[(d_train['AWKSTAT']=='Children or Armed Forces') & (d_train['AAGE']<=18),'AWKSTAT']='Children'

d_test.loc[(d_test['AWKSTAT']=='Children or Armed Forces')&(d_test['AAGE']<=18),'AWKSTAT']='Children'



#This provides a view that there is an issue with children allocation accross the industry codes

numb=d_train.groupby(['AWKSTAT','AMJIND'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

#numb.sort_values(numb.columns[0])



#Therefore it is worth to drop all stats that have age below 21 y.o.

#below in the code





#Children below 18 do not have income, those 21 and below have very small share: <0.5%

print("train=", d_train.shape," test=", d_test.shape) 

#print(d_train.columns)

#d_test.shape

pd.set_option('display.max_rows', None)

numb=d_train.groupby(['AAGE'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

#numb.sort_values(numb.columns[0]).head(25)



#d_train[(d_train['AAGE']==22)&(d_train['INCCAT']>=1)].T
#Investigating columns and ratios

numb=d_train.groupby(['ACLSWKR'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb.sort_values(numb.columns[2]).reset_index()
#There should normally be no income on the one that "Never worked". 

#Same seems logical for the "Without pay". I shall amend the data set accordingly

d_train.loc[d_train['ACLSWKR']=="Never worked",['INCCAT']]=0

d_train.loc[d_train['ACLSWKR']=="Without pay",['INCCAT']]=0
#Label Encoding

##I am not using fit_transform intentionally there to decrease the complexity of the function 

##(to have positive first order derivative - constantly increasing function, 

##instead of mixed positive/negative derivative), 

##that shoudl allow the GradientDescent to converge faster

##when fit_transform is applied, the label numbers shall be assigned on alphabetical basis

##see charts compared below

numb=d_train.groupby(['ACLSWKR'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

a=numb.copy()

dummy_le=LabelEncoder()

a=a.reset_index()

a.iloc[:, 0]=dummy_le.fit_transform(numb.iloc[:, 0])



numb=numb.sort_values(numb.columns[2]).reset_index()

le_ACLSWKR = LabelEncoder()

le_ACLSWKR.classes_=numb.iloc[:, 0]

numb.iloc[:, 0] = le_ACLSWKR.transform(numb.iloc[:, 0])



plt.figure(1)

plt.subplot(211)

plt.plot(a['ACLSWKR'],a['Ratio'])

plt.subplot(212)

plt.plot(numb['ACLSWKR'],numb['Ratio'])

d_train.loc[:, 'ACLSWKR'] = le_ACLSWKR.transform(d_train.loc[:, 'ACLSWKR'])

d_test.loc[:, 'ACLSWKR'] = le_ACLSWKR.transform(d_test.loc[:, 'ACLSWKR'])
#(education) nominal analysis provides 17 different categories. It shall be fair to overlap it with Age (AAGE).

#for children that are between 10 and 18 years old different AHGA categories may be encountered, although for 

#the goal of the exercise they are irrelevant (occurence <0,1%, so they have no predictive value), keeping 

#this split will adversely impact the quality of predictions for Adult population

#pd.set_option('display.max_rows', None)

numb=d_train.groupby(['AHGA'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

#numb.sort_values(numb.columns[2]).reset_index().plot('AHGA','Ratio')

numb.sort_values(numb.columns[2]).reset_index()
pd.set_option('display.max_rows', 30)

#The pre-processing that shall have to be applied to adult population will set AHGA='No Education' for all individual 

#below 18 years old (in addition to be data supported, it is also in line with common sense)

#this category was dropped out following the analysis of AWKSTAT

d_train.loc[d_train['AAGE']<=18,'AHGA']='No education'

d_test.loc[d_test['AAGE']<=18,'AHGA']='No education'

#It shall be fair to note that those persons that haven't passed the 1st grade technically also have no education, so

d_train.loc[d_train['AHGA']=='Less than 1st grade','AHGA']='No education'

d_test.loc[d_test['AHGA']=='Less than 1st grade','AHGA']='No education'

#Idem, there is no material difference between the "1st 2nd 3rd or 4th grade" and "5th or 6th grade" categories,

#as well as between "7th and 8th grade" and up till "12th grade no diploma", so I shall shrink that space also

d_train.loc[d_train['AHGA']=='1st 2nd 3rd or 4th grade','AHGA']='1st to 6th grade'

d_train.loc[d_train['AHGA']=='5th or 6th grade','AHGA']='1st to 6th grade'

d_train.loc[d_train['AHGA']=='7th and 8th grade','AHGA']='7th+ grade no diploma'

d_train.loc[d_train['AHGA']=='9th grade','AHGA']='7th+ grade no diploma'



d_test.loc[d_test['AHGA']=='1st 2nd 3rd or 4th grade','AHGA']='1st to 6th grade'

d_test.loc[d_test['AHGA']=='5th or 6th grade','AHGA']='1st to 6th grade'

d_test.loc[d_test['AHGA']=='7th and 8th grade','AHGA']='7th+ grade no diploma'

d_test.loc[d_test['AHGA']=='9th grade','AHGA']='7th+ grade no diploma'



#below are items to reconsider in feature generation part

d_train.loc[d_train['AHGA']=='10th grade','AHGA']='7th+ grade no diploma'

#d_train.loc[d_train['AHGA']=='11th grade','AHGA']='7th+ grade no diploma'

#d_train.loc[d_train['AHGA']=='12th grade no diploma','AHGA']='7th+ grade no diploma'



d_test.loc[d_test['AHGA']=='10th grade','AHGA']='7th+ grade no diploma'

#d_test.loc[d_test['AHGA']=='11th grade','AHGA']='7th+ grade no diploma'

#d_test.loc[d_test['AHGA']=='12th grade no diploma','AHGA']='7th+ grade no diploma'
numb=d_train.groupby(['AHGA'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb['OLD_ID']=numb['AHGA']

le_AHGA = LabelEncoder()

le_AHGA.classes_=numb.iloc[:, 0]

#print(le_AHGA.classes_)

#numb.iloc[:, 0] = le_AHGA.transform(numb.iloc[:, 0])

#numb.plot('AHGA','Ratio')
d_train.loc[:, 'AHGA'] = le_AHGA.transform(d_train.loc[:, 'AHGA'])

d_test.loc[:, 'AHGA'] = le_AHGA.transform(d_test.loc[:, 'AHGA'])
#AHRSPAY

numb=d_train.groupby(['AHRSPAY'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb[numb['Tot']<100]



#plt.plot(numb['Ratio'],numb['AHRSPAY'])



#d_train=d_train.drop(['AHRSPAY'],axis=1)

#d_test=d_test.drop(['AHRSPAY'],axis=1)



print("train=", d_train.shape," test=", d_test.shape) 
#AHRSCOL

numb=d_train.groupby(['AHRSCOL'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb['OLD_ID']=numb['AHRSCOL']

le_AHRSCOL = LabelEncoder()

le_AHRSCOL.classes_=numb.iloc[:, 0]
d_train.loc[:, 'AHRSCOL'] = le_AHRSCOL.transform(d_train.loc[:, 'AHRSCOL'])

d_test.loc[:, 'AHRSCOL'] = le_AHRSCOL.transform(d_test.loc[:, 'AHRSCOL'])
#AMARITL

numb=d_train.groupby(['AMARITL'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_AMARITL = LabelEncoder()

le_AMARITL.classes_=numb.iloc[:, 0]



d_train.loc[:, 'AMARITL'] = le_AMARITL.transform(d_train.loc[:, 'AMARITL'])

d_test.loc[:, 'AMARITL'] = le_AMARITL.transform(d_test.loc[:, 'AMARITL'])
#pd.set_option('display.max_rows', 15)

#AMJIND & AMJOCC - both are aggregates of ADTIND and ADTOCC, there is a 1-to-many relationship that allows 

#to identify AMJIND and AMJOCC in a unique way - therefore these look to be excessive

numb=d_train.groupby(['AMJIND','ADTIND'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

numb.sort_values(numb.columns[0])

#Although when considering the split a necessity to create an additional feature becomes apparent. 

#It is explained by the total number of cases: there is not enough stat data to extrapolate the income of

#'Armed forces', 'Manufacturing-durable goods' with Det code =10, etc.

#To overcome that issue I shall generate a feature that shall be the average ">50k" ration for industry

#For 'Armed forces' I shall use an average from "Manufacturing Durable Goods" 

#(both require high skilled employees, so it is reasonable to expect similar compensations)

#numb['AVIND']=

nu=d_train.groupby(['AMJIND'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

nu['Ratio']=nu['Nr']/nu['Tot']

nu=nu.reset_index()



for index, row in numb.iterrows():

    if numb.loc[numb['ADTIND']==row['ADTIND'],'Tot'].values[0]<300:

        numb.loc[numb['ADTIND']==row['ADTIND'],'Ratio']=nu.loc[nu['AMJIND']==row['AMJIND']]['Ratio'].values[0]

numb.loc[numb['AMJIND']=='Armed Forces','Ratio']=nu.loc[nu['AMJIND']=='Manufacturing-durable goods']['Ratio'].values[0]                

#numb

for index, row in numb.iterrows():

    d_train.loc[d_train['ADTIND']==row['ADTIND'],'AMJIND']=row['Ratio']

    d_test.loc[d_test['ADTIND']==row['ADTIND'],'AMJIND']=row['Ratio']

#numb=d_train.groupby(['ADTOCC','AMJOCC'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

#numb['Ratio']=numb['Nr']/numb['Tot']

#numb=numb.reset_index()

#numb.sort_values(numb.columns[0])
pd.set_option('display.max_rows', 50)

#AMJIND & AMJOCC - both are aggregates of ADTIND and ADTOCC, there is a 1-to-many relationship that allows 

#to identify AMJIND and AMJOCC in a unique way - therefore these look to be excessive

numb=d_train.groupby(['AMJOCC','ADTOCC'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

numb.sort_values(numb.columns[0])

#Although when considering the split a necessity to create an additional feature becomes apparent. 

#It is explained by the total number of cases: there is not enough stat data to extrapolate the income of

#'Armed forces', 'Manufacturing-durable goods' with Det code =10, etc.

#To overcome that issue I shall generate a feature that shall be the average ">50k" ration for industry

#For 'Armed forces' I shall use an average from "Manufacturing Durable Goods" 

#(both require high skilled employees, so it is reasonable to expect similar compensations)

#numb['AVIND']=



nu=d_train.groupby(['AMJOCC'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

nu['Ratio']=nu['Nr']/nu['Tot']

nu=nu.reset_index()



for index, row in numb.iterrows():

    if numb.loc[numb['ADTOCC']==row['ADTOCC'],'Tot'].values[0]<300:

        numb.loc[numb['ADTOCC']==row['ADTOCC'],'Ratio']=nu.loc[nu['AMJOCC']==row['AMJOCC']]['Ratio'].values[0]

numb.loc[numb['AMJOCC']=='Armed Forces','Ratio']=nu.loc[nu['AMJOCC']=='Technicians and related support']['Ratio'].values[0]                

numb

for index, row in numb.iterrows():

    d_train.loc[d_train['ADTOCC']==row['ADTOCC'],'AMJOCC']=row['Ratio']

    d_test.loc[d_test['ADTOCC']==row['ADTOCC'],'AMJOCC']=row['Ratio']



#numb=d_train.groupby(['ADTOCC','AMJOCC'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

#numb['Ratio']=numb['Nr']/numb['Tot']

#numb=numb.reset_index()

#numb.sort_values(numb.columns[0])
#ARACE

numb=d_train.groupby(['ARACE'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_ARACE = LabelEncoder()

le_ARACE.classes_=numb.iloc[:, 0]



d_train.loc[:, 'ARACE'] = le_ARACE.transform(d_train.loc[:, 'ARACE'])

d_test.loc[:, 'ARACE'] = le_ARACE.transform(d_test.loc[:, 'ARACE'])
#AREORGN

numb=d_train.groupby(['AREORGN'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_AREORGN = LabelEncoder()

le_AREORGN.classes_=numb.iloc[:, 0]



d_train.loc[:, 'AREORGN'] = le_AREORGN.transform(d_train.loc[:, 'AREORGN'])

d_test.loc[:, 'AREORGN'] = le_AREORGN.transform(d_test.loc[:, 'AREORGN'])
#ASEX

numb=d_train.groupby(['ASEX'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_ASEX = LabelEncoder()

le_ASEX.classes_=numb.iloc[:, 0]



d_train.loc[:, 'ASEX'] = le_ASEX.transform(d_train.loc[:, 'ASEX'])

d_test.loc[:, 'ASEX'] = le_ASEX.transform(d_test.loc[:, 'ASEX'])
#AUNMEM & AUNTYPE

#Shall concatenate these two, as there is no apparent information that they convey

numb=d_train.groupby(['AUNMEM','AUNTYPE'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_AUN = LabelEncoder()

le_AUN.classes_=numb.iloc[:, 0] +" "+ numb.iloc[:, 1]



d_train.loc[:,'AUNMEM'] = le_AUN.transform((d_train.loc[:, 'AUNMEM'] + " " + d_train.loc[:,'AUNTYPE']))

d_test.loc[:,'AUNMEM'] = le_AUN.transform((d_test.loc[:, 'AUNMEM'] + " " + d_test.loc[:,'AUNTYPE']))

d_train=d_train.drop(['AUNTYPE'],axis=1)

d_test=d_test.drop(['AUNTYPE'],axis=1)

#AWKSTAT

# Children or Armed Forces with 5874 income >50k doesn't make sense

numb=d_train.groupby(['AWKSTAT'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

le_AWKSTAT = LabelEncoder()

le_AWKSTAT.classes_=numb.iloc[:, 0]



d_train.loc[:, 'AWKSTAT'] = le_AWKSTAT.transform(d_train.loc[:, 'AWKSTAT'])

d_test.loc[:, 'AWKSTAT'] = le_AWKSTAT.transform(d_test.loc[:, 'AWKSTAT'])
#CAPGAIN

numb=d_train.groupby(['CAPGAIN'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']





numb=numb.reset_index()

numb=numb.sort_values(numb.columns[3])



#plt.plot(numb['Ratio'],numb['CAPGAIN'])

d_train.loc[d_train['CAPGAIN']<=3000,'CAPGAIN']=0

d_test.loc[d_test['CAPGAIN']<=3000,'CAPGAIN']=0

d_train.loc[(d_train['CAPGAIN']>3000)&(d_train['CAPGAIN']<10000),'CAPGAIN']=1

d_test.loc[(d_test['CAPGAIN']>3000)&(d_test['CAPGAIN']<10000),'CAPGAIN']=1

d_train.loc[d_train['CAPGAIN']>=10000,'CAPGAIN']=2

d_test.loc[d_test['CAPGAIN']>=10000,'CAPGAIN']=2



#numb
#CAPLOSS

numb=d_train.groupby(['CAPLOSS'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']





numb=numb.reset_index()

numb=numb.sort_values(numb.columns[0])



#plt.plot(numb['Ratio'],numb['CAPLOSS'])

d_train=d_train.drop(['CAPLOSS'],axis=1)

d_test=d_test.drop(['CAPLOSS'],axis=1)
#DIVVAL

numb=d_train.groupby(['DIVVAL'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.reset_index()

numb=numb.sort_values(numb.columns[0])

#numb[ (numb['DIVVAL']<700)].agg({'Nr':'sum','Tot':'sum'})

#plt.plot(numb['Ratio'],numb['DIVVAL'])



d_train['DIVCAT']=0

d_test['DIVCAT']=0



d_train.loc[d_train['DIVVAL']<700,'DIVCAT']=0

d_test.loc[d_test['DIVVAL']<700,'DIVCAT']=0

d_train.loc[(d_train['DIVVAL']>=700)&(d_train['DIVVAL']<=11000),'DIVCAT']=1

d_test.loc[(d_test['DIVVAL']>700)&(d_test['DIVVAL']<=11000),'DIVCAT']=1

d_train.loc[(d_train['DIVVAL']>11000)&(d_train['DIVVAL']<=28000),'DIVCAT']=2

d_test.loc[(d_test['DIVVAL']>11000)&(d_test['DIVVAL']<=28000),'DIVCAT']=2

d_train.loc[d_train['DIVVAL']>28000,'DIVCAT']=3

d_test.loc[d_test['DIVVAL']>28000,'DIVCAT']=3



d_train=d_train.drop(['DIVVAL'],axis=1)

d_test=d_test.drop(['DIVVAL'],axis=1)
#FILESTAT

# Children or Armed Forces with 5874 income >50k doesn't make sense

numb=d_train.groupby(['FILESTAT'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb

le_FILESTAT = LabelEncoder()

le_FILESTAT.classes_=numb.iloc[:, 0]



d_train.loc[:, 'FILESTAT'] = le_FILESTAT.transform(d_train.loc[:, 'FILESTAT'])

d_test.loc[:, 'FILESTAT'] = le_FILESTAT.transform(d_test.loc[:, 'FILESTAT'])



#d_train.loc[d_train['AWKSTAT']=='Not in labor force'].head(20).T

#pd.set_option('display.max_rows', 600)

#numb=d_train.loc[d_train['FILESTAT']=='Nonfiler'].groupby(['FILESTAT','AAGE'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

#numb['Ratio']=numb['Nr']/numb['Tot']

#numb=numb.reset_index()

#numb.sort_values(numb.columns[1])
#'GRINREG','GRINST' - I see no material difference, except the boolean condition: region was changed/wasn't, so

numb=d_train.groupby(['GRINREG'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb

#le_AWKSTAT = LabelEncoder()

#le_AWKSTAT.classes_=numb.iloc[:, 0]

d_train.loc[d_train['GRINREG']=='Not in universe','GRINREG']=1

d_test.loc[d_test['GRINREG']=='Not in universe','GRINREG']=1

d_train.loc[d_train['GRINREG']!=1,'GRINREG']=0

d_test.loc[d_test['GRINREG']!=1,'GRINREG']=0



d_train=d_train.drop(['GRINST'],axis=1)

d_test=d_test.drop(['GRINST'],axis=1)

#HHDREL

d_train.loc[d_train['HHDREL'].str.contains('Spouse of householder'),'HHDFMX']='Spouse of householder'

d_test.loc[d_test['HHDREL'].str.contains('Spouse of householder'),'HHDFMX']='Spouse of householder'



d_train.loc[(d_train['HHDREL'].str.contains('Householder')) & 

            (d_train['HHDFMX'].str.contains('In group quarters')) ,'HHDFMX']='Nonfamily householder'

d_test.loc[(d_test['HHDREL'].str.contains('Householder')) & 

            (d_test['HHDFMX'].str.contains('In group quarters')) ,'HHDFMX']='Nonfamily householder'



d_train.loc[d_train['HHDFMX'].str.contains('In group quarters') ,'HHDFMX']='Other relative of householder'

d_test.loc[d_test['HHDFMX'].str.contains('In group quarters') ,'HHDFMX']='Other relative of householder'



d_train.loc[d_train['HHDFMX'].str.contains('Other relative of householder') ,'HHDREL']='Other relative of householder'

d_test.loc[d_test['HHDFMX'].str.contains('Other relative of householder') ,'HHDREL']='Other relative of householder'



#HHDFMX

#First - I shall collect grandchilds 18+, as their numbers are non-representative (<1%) in current segmentation

d_train.loc[d_train['HHDFMX'].str.contains('Grandchild'),'HHDFMX']='Child 18+'

d_test.loc[d_test['HHDFMX'].str.contains('Grandchild'),'HHDFMX']='Child 18+'



d_train.loc[d_train['HHDFMX'].str.contains('Child'),'HHDREL']='Child 18+'

d_test.loc[d_test['HHDFMX'].str.contains('Child'),'HHDREL']='Child 18+'



d_train.loc[d_train['HHDFMX'].str.contains('Child'),'HHDFMX']='Child 18+'

d_test.loc[d_test['HHDFMX'].str.contains('Child'),'HHDFMX']='Child 18+'





numb=d_train.groupby(['HHDFMX'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

numb=numb.sort_values(numb.columns[0])

#numb



le_HHDFMX = LabelEncoder()

le_HHDFMX.classes_=numb.iloc[:, 0]



d_train.loc[:, 'HHDFMX'] = le_HHDFMX.transform(d_train.loc[:, 'HHDFMX'])

d_test.loc[:, 'HHDFMX'] = le_HHDFMX.transform(d_test.loc[:, 'HHDFMX'])





numb=d_train.groupby(['HHDREL'],)['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.reset_index()

numb=numb.sort_values(numb.columns[0])

numb



le_HHDREL = LabelEncoder()

le_HHDREL.classes_=numb.iloc[:, 0]



d_train.loc[:, 'HHDREL'] = le_HHDREL.transform(d_train.loc[:, 'HHDREL'])

d_test.loc[:, 'HHDREL'] = le_HHDREL.transform(d_test.loc[:, 'HHDREL'])

#MARSUPWT - it is required to ignore this field, so

numb=d_train.groupby(['MARSUPWT'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb

#plt.plot(numb['Ratio'],numb['MARSUPWT'])

d_train=d_train.drop(['MARSUPWT'],axis=1)

d_test=d_test.drop(['MARSUPWT'],axis=1)

##MIGMTR

pd.set_option('display.max_rows', 20)



numb=d_train.groupby(['MIGMTR3'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb







le_MIGMTR3 = LabelEncoder()

le_MIGMTR3.classes_=numb.iloc[:, 0]



d_train.loc[:, 'MIGMTR3'] = le_MIGMTR3.transform(d_train.loc[:, 'MIGMTR3'])

d_test.loc[:, 'MIGMTR3'] = le_MIGMTR3.transform(d_test.loc[:, 'MIGMTR3'])



d_train=d_train.drop(['MIGMTR1','MIGMTR4'],axis=1)

d_test=d_test.drop(['MIGMTR1','MIGMTR4'],axis=1)

#MIGSAME

numb=d_train.groupby(['MIGSAME'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb



le_MIGSAME = LabelEncoder()

le_MIGSAME.classes_=numb.iloc[:, 0]



d_train.loc[:, 'MIGSAME'] = le_MIGSAME.transform(d_train.loc[:, 'MIGSAME'])

d_test.loc[:, 'MIGSAME'] = le_MIGSAME.transform(d_test.loc[:, 'MIGSAME'])
#MIGSUN

numb=d_train.groupby(['MIGSUN'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb



le_MIGSUN = LabelEncoder()

le_MIGSUN.classes_=numb.iloc[:, 0]



d_train.loc[:, 'MIGSUN'] = le_MIGSUN.transform(d_train.loc[:, 'MIGSUN'])

d_test.loc[:, 'MIGSUN'] = le_MIGSUN.transform(d_test.loc[:, 'MIGSUN'])
#NOEMP - No changes, as it is good as is



numb=d_train.groupby(['NOEMP'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()
#PARENT -not needed as after removing childrens it becomes single value category



d_train=d_train.drop(['PARENT'],axis=1)

d_test=d_test.drop(['PARENT'],axis=1)

#PEFNTVTY 	PEMNTVTY 	PENATVTY - open item

#There is some information in, although it is not clear how to extract it

d_train=d_train.drop(['PEMNTVTY','PENATVTY'],axis=1)

d_test=d_test.drop(['PEMNTVTY','PENATVTY'],axis=1)
#PRCITSHP

numb=d_train.groupby(['PEFNTVTY'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb



le_PEFNTVTY = LabelEncoder()

le_PEFNTVTY.classes_=numb.iloc[:, 0]



d_train.loc[:, 'PEFNTVTY'] = le_PEFNTVTY.transform(d_train.loc[:, 'PEFNTVTY'])

d_test.loc[:, 'PEFNTVTY'] = le_PEFNTVTY.transform(d_test.loc[:, 'PEFNTVTY'])
#PRCITSHP

numb=d_train.groupby(['PRCITSHP'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']

numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb



le_PRCITSHP = LabelEncoder()

le_PRCITSHP.classes_=numb.iloc[:, 0]



d_train.loc[:, 'PRCITSHP'] = le_PRCITSHP.transform(d_train.loc[:, 'PRCITSHP'])

d_test.loc[:, 'PRCITSHP'] = le_PRCITSHP.transform(d_test.loc[:, 'PRCITSHP'])
#SEOTR - no intuition on how this can be improved, so shall be used as is

numb=d_train.groupby(['SEOTR'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb

#plt.plot(numb['Ratio'],numb['SEOTR'])

#VETQVA, VETYN, To check against possibility to segregate Armed Forces from Childrens



numb=d_train.groupby(['VETQVA','VETYN'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

numb

le_VET = LabelEncoder()

le_VET.classes_=numb.iloc[:, 0].astype(str) +","+ numb.iloc[:, 1].astype(str)

#print(le_VET.classes_)



d_train['VET'] = le_VET.transform(d_train.loc[:, 'VETQVA'].astype(str) + "," + d_train.loc[:,'VETYN'].astype(str))

d_test['VET'] = le_VET.transform(d_test.loc[:, 'VETQVA'].astype(str) + "," + d_test.loc[:,'VETYN'].astype(str))

d_train=d_train.drop(['VETYN','VETQVA'],axis=1)

d_test=d_test.drop(['VETYN','VETQVA'],axis=1)

#WKSWORK - again, good without changes

numb=d_train.groupby(['WKSWORK'])['INCCAT'].agg({'Nr':'sum','Tot':'count'})

numb['Ratio']=numb['Nr']/numb['Tot']



numb=numb.sort_values(numb.columns[2])

numb=numb.reset_index()

#numb
#To be able to get back to stored data if it gets unexpecedly amended

d_train_bckp=d_train.copy()

d_test_bckp=d_test.copy()
d_train=d_train.drop_duplicates()
# A dirty way to balance out the set (as there is just 6% with income 50k+)

add_tr=d_train[(d_train['INCCAT']==1)&(d_train['AAGE']>20)]

balanced_d_train=d_train.copy()

for i in range(0,round(balanced_d_train.shape[0]/add_tr.shape[0])):

    balanced_d_train=pd.concat([balanced_d_train,add_tr],axis=0,sort=False, ignore_index=True)

    i=i+1
gen_y_train=d_train['INCCAT'].copy()

balanced_y_train=balanced_d_train['INCCAT'].copy()
d_train = d_train.drop('INCCAT', axis=1)

d_test = d_test.drop('INCCAT', axis=1)

balanced_d_train=balanced_d_train.drop('INCCAT', axis=1)
X=balanced_d_train.copy()

y=balanced_y_train.copy()
#Normalize the features set

train_norm = X

std_scale = preprocessing.StandardScaler().fit(train_norm)

x_train_norm = std_scale.transform(train_norm)

training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 

X.update(training_norm_col)
#Here is the Keras Sequential NN model that I am using (commented as I've trained it on my pc)

model = Sequential()

model.add(Dense(90, input_dim=X.shape[1], activation='relu'))

model.add(Dropout(rate=0.25, name='dropout_one'))

model.add(Dense(20, activation='sigmoid'))

model.add(Dense(7, activation='sigmoid'))

model.add(Dropout(rate=0.1, name='dropout_two'))

model.add(Dense(1, activation='sigmoid'))

#Compile the model

opt=keras.optimizers.RMSprop(lr=0.003, rho=0.9)

#opt = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])



#another round with learning rate lr=0.003 was given to fine tune the model
#This is the K-fold training cycle

np.random.seed(10)



kfold = StratifiedKFold(n_splits=10, shuffle=True)

cvscores = []

kfold.get_n_splits(X,y)

i=1

for train_index, test_index in kfold.split(X, y):

    print("split Nr %d" % i)

    # Fit the model

    X_tr_m1, X_te_m1 = X.iloc[train_index], X.iloc[test_index]

    y_tr_m1, y_te_m1 = y.iloc[train_index], y.iloc[test_index]



    model.fit(X_tr_m1, y_tr_m1, epochs=3, batch_size=6000, verbose=1)

    # evaluate the model

    scores = model.evaluate(X_te_m1, y_te_m1, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

    i=i+1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
##Now adjusting minor biases after the Kfold split (with reduced learning_rate):

#opt=keras.optimizers.RMSprop(lr=0.01, rho=0.9)

opt = keras.optimizers.Adam(lr=0.03, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

model.fit(X, y, epochs=80, batch_size=8000, verbose=1)
#Showing what is the trained model 

y_T=model.predict(X)

from sklearn.metrics import classification_report, confusion_matrix



matrix = confusion_matrix(balanced_y_train, y_T.round())

matrix
from sklearn.metrics import roc_auc_score

roc_auc_score(balanced_y_train,y_T.round())
#Saving and loading
#Saving the trained model

# serialize model to JSON

#model_json = model.to_json()

#with open("Saturday1.json", "w") as json_file:

#    json_file.write(model_json)

## serialize weights to HDF5

#model.save_weights("Saturday1.h5")

#print("Saved model to disk")
#from keras.models import model_from_json

## load json and create model

#json_file = open('../input/sunday2/Sunday2.json', 'r')

#loaded_model_json = json_file.read()

#json_file.close()

#model = model_from_json(loaded_model_json)

## load weights into new model

#model.load_weights("../input/sunday2/Sunday2.h5")

#print("Loaded model from disk")
model.summary()
#now we need a prediction
#Test sample has to be normalized

X_T=d_test.copy()

test_norm = X_T

x_test_norm = std_scale.transform(test_norm)

testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 

X_T.update(testing_norm_col)
#Prediction

y_Test_lev1_pred=model.predict(X_T)
#Basic check

d_test['y_new']=y_Test_lev1_pred.round()

sum(d_test['y_new'])
#Childs do not have income, even if the model decided otherwise

d_test.loc[d_test['AAGE']<21,['y_new']]=0

#Those who have received dividends tend to have income above 50k

#d_test.loc[d_test['DIVVAL']>40000,['y_new']]=1

sum(d_test['y_new'])
#finalize and check that no apparent cases were misspelled

sub=pd.read_csv('/kaggle/input/ml-challenge-week6/sampleSubmission.csv')

sub["income class"]=d_test["y_new"]
filename = 'submission.csv'

sub.to_csv(filename,index=False)