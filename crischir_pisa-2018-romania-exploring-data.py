# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import json as JSON

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_pisa2018 = pd.read_csv("../input/pisa-2018-romania/SAS_STU_QQQ.csv")

df_pisa2018.info()
f = open('../input/pisa-2018-romania/codebook2018.json')

codebook2018 = JSON.load(f)



#rint(codebook2015['WEALTH']['ST100Q02TA'])



codebook2018 
#intrebari la care nu s-a raspuns

missing_data = pd.DataFrame({'total_missing': df_pisa2018.isnull().sum(), 'perc_missing': (df_pisa2018.isnull().sum()/4914)*100})
#sunt 692 de intrebari la care nu s-a raspuns din 1120

#missing_data['perc_missing']

fararaspuns= missing_data.loc[missing_data['perc_missing'] == 100.0]

missing_data.drop

coloanelipsa = list(fararaspuns.index)

coloanelipsa
coloane = list(df_pisa2018.columns.values)

coloane
#Drop missing values

df_pisa2018.drop(coloanelipsa, axis=1, inplace=True)

#df_pisa2018.head

#verificare
df_pisa2018['WEALTH'].plot.hist(alpha=0.4)


plt.figure()

df_pisa2018['ST005Q01TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST005Q01TA'])
plt.figure()

df_pisa2018['ST006Q01TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST006Q01TA'])



plt.figure()

df_pisa2018['ST006Q02TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST006Q02TA'])


plt.figure()

df_pisa2018['ST006Q04TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST006Q04TA'])
plt.figure()

df_pisa2018['ST007Q01TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST007Q01TA'])
plt.figure()

df_pisa2018['ST008Q01TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST008Q01TA'])
plt.figure()

df_pisa2018['ST008Q02TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST008Q02TA'])
plt.figure()

df_pisa2018['ST008Q04TA'].plot.hist(alpha=0.4)

plt.suptitle(codebook2018['ST008Q04TA'])
df2 = df_pisa2018[['ST008Q04TA','ST008Q02TA',"ST008Q03TA",'ST008Q01TA','ST008Q01TA']]

plt.figure()

df2.plot.hist(alpha=0.4, stacked=True)

plt.show()



codebook2018.update(

Edumama="Mother’s Education (ISCED)"

)

df_pisa2018["ISCED"]=df_pisa2018['ST005Q01TA'].replace({5:0,4:1,3:2,2:3,1:3})

df_pisa2018['Edumama']=df_pisa2018["ISCED"]+ df_pisa2018['ST006Q01TA'].apply(lambda x: 6 if x < 2 else 0) + df_pisa2018['ST006Q02TA'].apply(lambda x: 5 if x < 2 else 0)+ df_pisa2018['ST006Q03TA'].apply(lambda x: 5 if x < 2 else 0) + df_pisa2018['ST006Q04TA'].apply(lambda x: 4 if x < 2 else 0)

print(df_pisa2018['ISCED'])

df_pisa2018['Edumama'].plot.hist(alpha=0.4)

#df_pisa2018[['Edumama','PV10MATH']]

fig, ax = plt.subplots()

for color in ['tab:blue', 'tab:orange', 'tab:green']:

    ax.scatter(df_pisa2018['Edumama'],df_pisa2018['PV10MATH'], c=color, label=color,

               alpha=0.3, edgecolors='none')



#ax.legend()

ax.grid(True)



plt.show()
conditiiacasa= df_pisa2018[["ST011Q01TA","ST011Q02TA","ST011Q03TA","ST011Q05TA","ST011Q06TA","ST011Q06TA","ST011Q07TA","ST011Q08TA","ST011Q09TA","ST011Q10TA","ST011Q11TA","ST011Q12TA","ST011Q16NA","ST011D17TA","ST011D18TA","ST011D19TA"]]

coloane = list(conditiiacasa.columns.values)

#coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

#titlu

conditiiacasa.columns=titlu

coloane = list(conditiiacasa.columns.values)

#coloane

hist = conditiiacasa.hist(figsize=(10,23), layout=(8,2),xlabelsize =7)

conditiiacasa2= df_pisa2018[["ST012Q01TA","ST012Q02TA","ST012Q03TA","ST012Q05NA","ST012Q06NA","ST012Q07NA","ST012Q08NA","ST012Q09NA","ST013Q01TA"]]

coloane = list(conditiiacasa2.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

conditiiacasa2.columns=titlu

coloane = list(conditiiacasa2.columns.values)

coloane

hist = conditiiacasa2.hist(figsize=(15,23), layout=(9,1),xlabelsize =7)

#incredere 

incredere= df_pisa2018[["ST123Q02NA","ST123Q03NA","ST123Q04NA"]]

coloane = list(incredere.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

incredere.columns=titlu

coloane = list(incredere.columns.values)

coloane

hist = incredere.hist(figsize=(15,23), layout=(9,1),xlabelsize =7)

diverse= df_pisa2018[['BELONG','CULTPOSS']]
coloane = list(diverse.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse.columns=titlu

coloane = list(diverse.columns.values)

coloane

hist = diverse.hist(figsize=(15,23), layout=(5,1),xlabelsize =7)

diverse= df_pisa2018[["ST211Q01HA",'ST211Q03HA','ST212Q01HA','ST212Q02HA','ST212Q03HA',"ST104Q02NA","ST104Q03NA","ST104Q04NA","ST213Q01HA","ST213Q02HA","ST213Q03HA","ST213Q04HA"]]

coloane = list(diverse.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse.columns=titlu

coloane = list(diverse.columns.values)

coloane

hist = diverse.hist(figsize=(15,25), layout=(12,1),xlabelsize =10)
diverse= df_pisa2018[["ST150Q01IA","ST150Q02IA","ST150Q03IA","ST150Q04HA","ST152Q05IA","ST152Q06IA","ST152Q07IA","ST152Q08IA","ST154Q01HA","ST153Q01HA","ST153Q02HA","ST153Q03HA","ST153Q04HA","ST153Q05HA","ST153Q06HA","ST153Q08HA","ST153Q09HA","ST153Q10HA","ST158Q01HA","ST158Q02HA","ST158Q03HA","ST158Q04HA","ST158Q05HA","ST158Q06HA","ST158Q07HA"]]

coloane = list(diverse.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse.columns=titlu

coloane = list(diverse.columns.values)

coloane

hist = diverse.hist(figsize=(25,40), layout=(25,1),xlabelsize =10)
diverse2= df_pisa2018[["ST160Q01IA","ST160Q02IA","ST160Q03IA","ST160Q04IA","ST160Q05IA","ST167Q01IA","ST167Q02IA","ST167Q03IA","ST167Q04IA","ST167Q05IA"]]

coloane = list(diverse2.columns.values)

coloane

titlu = []

for value in coloane:

   print(value)

   titlu.append(codebook2018[value])

titlu

diverse2.columns=titlu

coloane = list(diverse2.columns.values)

coloane

hist = diverse2.hist(figsize=(15,30), layout=(10,1),xlabelsize =8)
diverse2= df_pisa2018[["ST036Q05TA","ST036Q06TA","ST036Q08TA","ST225Q01HA","ST225Q02HA","ST225Q03HA","ST225Q04HA","ST225Q05HA","ST225Q06HA","ST181Q02HA","ST181Q03HA","ST181Q04HA","ST182Q03HA","ST182Q04HA","ST182Q05HA","ST182Q06HA","ST183Q01HA","ST183Q02HA","ST183Q03HA","ST184Q01HA","ST185Q01HA","ST185Q02HA","ST185Q03HA"]]

coloane = list(diverse2.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse2.columns=titlu

coloane = list(diverse2.columns.values)

coloane

hist = diverse2.hist(figsize=(15,30), layout=(12,2),xlabelsize =8)
diverse2= df_pisa2018[["ST186Q05HA","ST186Q06HA","ST186Q07HA","ST186Q10HA","ST186Q09HA","ST186Q02HA","ST186Q01HA","ST186Q08HA","ST186Q03HA"]]

coloane = list(diverse2.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse2.columns=titlu

coloane = list(diverse2.columns.values)

coloane

hist = diverse2.hist(figsize=(15,30), layout=(9,1),xlabelsize =8)
diverse2= df_pisa2018[["ST186Q05HA","ST186Q06HA","ST186Q07HA","ST186Q10HA","ST186Q09HA","ST186Q02HA","ST186Q01HA","ST186Q08HA","ST186Q03HA"]]

coloane = list(diverse2.columns.values)

coloane

titlu = []

for value in coloane:

   #print(value)

   titlu.append(codebook2018[value])

titlu

diverse2.columns=titlu

coloane = list(diverse2.columns.values)

coloane

hist = diverse2.hist(figsize=(15,30), layout=(9,1),xlabelsize =8)