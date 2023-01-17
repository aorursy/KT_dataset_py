""" Encoding techniques :"""

"""caution: i didnt uploded Datsets """

# -*- coding: utf-8 -*-

import pandas as pd

import seaborn as sns

train=pd.read_csv("train.csv")

test=pd.read_csv("test.csv")

#target=train['target']

#Id_train=train['id']

Id_test=test['id']

Df=pd.concat([train,test])

"" "Data Preprocessing """                                 

print(pd.DataFrame({'percentage null:':train.isnull().sum() / len(train)*100}))

#Df=pd.DataFrame({'percentage null:':Df.isnull().sum() /len(Df)*100})



"""****************************************************************************"""



D_new=Df



""" Filling nan values"""

print(D_new['bin_3'].isnull().value_counts())

"""************************Nan values****************************************************"""

def encode(k):

    for i in k:

        D_new[i].fillna(D_new[i].mode().values[0],inplace=True)

print(D_new['day'].describe())

print(D_new['day'].value_counts())

print(D_new['day'].isnull().sum())

sns.countplot(D_new['day'])

D_new['day'].fillna(value=3,inplace=True)

print(D_new['month'].describe())

print(D_new['month'].value_counts())

print(D_new['month'].isnull().sum())



sns.countplot(D_new['month'])

D_new['month'].fillna(value=8,inplace=True)



col_n=['bin_4','bin_0','bin_1','bin_2','nom_0','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','nom_1','bin_3','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']

encode(col_n)



""" ***********************Encoding binary features********************* """





D_new['bin_3'] = D_new['bin_3'].apply(lambda x: 1 if x=='T' else (0 if x=='F' else None))

D_new["bin_4"] = D_new["bin_4"].apply(lambda x: 1 if x=='Y' else (0 if x=='N' else None))



""" ****************ordinal features***************"""

""" Mapping of ordinal features """





ord_2_mapp=dict(zip(Df.ord_2.dropna().unique(),range(len(Df.ord_2.dropna().unique()))))

ord_1_mapp=dict(zip(Df.ord_1.dropna().unique(),range(len(Df.ord_1.dropna().unique()))))

ord_3_mapp=dict(zip(Df.ord_3.dropna().unique(),range(len(Df.ord_3.dropna().unique()))))

ord_4_mapp=dict(zip(Df.ord_4.dropna().unique(),range(len(Df.ord_4.dropna().unique()))))

ord_5_mapp=dict(zip(Df.ord_5.dropna().unique(),range(len(Df.ord_5.dropna().unique()))))

D_new['ord_1']=D_new['ord_1'].map(ord_1_mapp)

D_new['ord_2']=D_new['ord_2'].map(ord_2_mapp)

D_new['ord_3']=D_new['ord_3'].map(ord_3_mapp)

D_new['ord_4']=D_new['ord_4'].map(ord_4_mapp)

D_new['ord_5']=D_new['ord_5'].map(ord_5_mapp)



"""*********************************CONVERTING Time and Date*******************"""

import math

keyD=[]

keyM=[]

def calD(t):

    value=math.sin(2*math.pi *t/7)

    return value

for i in range(1,8):

    keyD.append((calD(i)))

Date_mapp=dict(zip(Df.day.dropna().unique(),keyD))

D_new['day']=D_new['day'].map(Date_mapp)  

def calM(m):

    val=math.sin(2*math.pi*m/12)

    return val



for i in range(1,13):

    keyM.append((calM(i)))                                                                          

Mon_mapp=dict(zip(Df.month.dropna().unique(),keyM))

D_new['month']=D_new['month'].map(Mon_mapp)



"""***************************NOMINAL FEATURES**************************"""





def freq_encode(col):

    fe = D_new.groupby(col).size()/len(D_new)

    D_new.loc[:, "{}_freq_encode".format(col)] = D_new[col].map(fe)                                 

                                       



freq_encode('nom_0')

freq_encode('nom_1')

freq_encode('nom_2')

freq_encode('nom_3')

freq_encode('nom_4')

freq_encode('nom_5')

freq_encode('nom_6')

freq_encode('nom_7')

freq_encode('nom_8')

freq_encode('nom_9')



D_new = D_new.drop(['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)







"""****************thanks you********************"""