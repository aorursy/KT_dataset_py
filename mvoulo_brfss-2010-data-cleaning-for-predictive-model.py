import pandas as pd

#import BRFSS 2010 data and select columns of interest

df2=pd.read_csv('10.csv')

df2.shape

df2=df2[['PHYSHLTH', 'HLTHPLAN', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'QLREST2', 'EXERANY2', 'SMOKDAY2', 'AGE', 'EDUCA', 'INCOME2', 'SEX', 'EMTSUPRT', 'LSATISFY', 'ADANXEV', 'ADDEPEV']]

#descriptive statistics

df2.describe().T

#shape of dataframe

df2.shape
#column names

df2.columns


#recode variables as NAN according to missing values in codebook

import numpy as np

df2.PHYSHLTH=df2.PHYSHLTH.replace(88, 0)

df2.PHYSHLTH=df2.PHYSHLTH.replace(77, np.nan)

df2.PHYSHLTH=df2.PHYSHLTH.replace(99, np.nan)

df2.ADDEPEV=df2.ADDEPEV.replace(7, np.nan)

df2.ADDEPEV=df2.ADDEPEV.replace(9, np.nan)

df2.HLTHPLAN=df2.HLTHPLAN.replace(7, np.nan)

df2.HLTHPLAN=df2.HLTHPLAN.replace(9, np.nan)

df2.PERSDOC2=df2.PERSDOC2.replace(7, np.nan)

df2.PERSDOC2=df2.PERSDOC2.replace(9, np.nan)

df2.MEDCOST=df2.MEDCOST.replace(7, np.nan)

df2.MEDCOST=df2.MEDCOST.replace(9, np.nan)

df2.CHECKUP1=df2.CHECKUP1.replace(7, np.nan)

df2.CHECKUP1=df2.CHECKUP1.replace(9, np.nan)

df2.QLREST2=df2.QLREST2.replace(77, np.nan)

df2.QLREST2=df2.QLREST2.replace(99, np.nan)

df2.QLREST2=df2.QLREST2.replace(88, 0)

df2.EXERANY2=df2.EXERANY2.replace(7, np.nan)

df2.EXERANY2=df2.EXERANY2.replace(9, np.nan)

df2.SMOKDAY2=df2.SMOKDAY2.replace(7, np.nan)

df2.SMOKDAY2=df2.SMOKDAY2.replace(9, np.nan)

df2.AGE=df2.AGE.replace(7, np.nan)

df2.AGE=df2.AGE.replace(9, np.nan)

df2.EDUCA=df2.EDUCA.replace(9, np.nan)

df2.INCOME2=df2.INCOME2.replace(77, np.nan)

df2.INCOME2=df2.INCOME2.replace(99, np.nan)

df2.EMTSUPRT=df2.EMTSUPRT.replace(7, np.nan)

df2.EMTSUPRT=df2.EMTSUPRT.replace(9, np.nan)

df2.LSATISFY=df2.LSATISFY.replace(7, np.nan)

df2.LSATISFY=df2.LSATISFY.replace(9, np.nan)

df2.ADANXEV=df2.ADANXEV.replace(7, np.nan)

df2.ADANXEV=df2.ADANXEV.replace(9, np.nan)

#drop missing values for criterion

df2=df2.dropna(subset=['ADDEPEV', 'ADANXEV', 'SMOKDAY2'])
#list missing values as percentages

nas=pd.DataFrame(df2.isnull().sum().sort_values(ascending=False),columns = ['Missing'])

pos = nas['Missing'] > 0

nas[pos]


df1=df2['PHYSHLTH']



df2=df2.drop('PHYSHLTH',axis=1)
#fill missing values with most common response for each category

df2_new = df2.apply(lambda x: x.fillna(x.value_counts().index[0]))
#dummy variables

df2
#list missing values

nas=pd.DataFrame(df2_new.isnull().sum().sort_values(ascending=False)/len(df2_new),columns = ['percent'])

pos = nas['percent'] > 0

nas[pos]

#get dummy variables

df2=pd.get_dummies(df2_new)
#new column names with dummies

df2.columns