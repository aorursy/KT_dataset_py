from __future__ import unicode_literals



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
import requests

import json

import matplotlib.pyplot as plt

import seaborn as sns



   





def mzp (sp, text=""):

    

  doll = pd.read_json ('https://www.cbr-xml-daily.ru/daily_json.js')

  value=doll.loc['USD']['Valute']['Value']





  x= []

  zpv=0

  city=[]

  zp=[]

  

  url='https://api.hh.ru/vacancies?area=113&specialization='+sp+'&per_page=100&page='

  

  for i in range(20):





  

    if i==0:

      data = pd.read_json (url + str(i)+text)

    else:  

      data2 = pd.read_json (url + str(i)+text)

      data=data.append(data2, ignore_index=True)





#data

#data.to_csv('data.csv')



  for h in range ( len(data.index) ):

    dic=data.loc[h]['items']

    try:

      if (dic['salary']['from']):

        name=dic['area']['name']

        #city.append(name.encode('utf-8'))

        city.append(name)

        zp.append(dic['salary']['from'])

       #  print (dic['salary']['from'])

  

    except:

      pass







  df = pd.DataFrame({'City' : city, 'average salary, RUB' : zp, 'vacancies' : ''})

  #out=df.groupby('City').mean()



  out=df.groupby('City').agg({'vacancies': "count", 'average salary, RUB': "mean"})







  #out.to_csv('out.csv')

  hh=out.round(0).sort_values(by=['vacancies'], ascending=False).head(30)

  hh['average salary, USD']=(hh['average salary, RUB']/value).round(0)

  return hh







#spc='https://api.hh.ru/specializations'

#spp = pd.read_json (spc)

#ff=spp.loc[0]['specializations']

#for ffx in ff:

#  print(ffx['name'])

#  print (ffx['id'])





hj=mzp("1.3")

hj
hj=mzp("1.221")

hj
hj=mzp("1.221","&text=Python")

hj
hj=mzp("1.221","&text=PHP")

hj
hj=mzp("1.221","&text=C++")

hj
hj=mzp("1.221","&text=Java")

hj
hj=mzp("1.273")

hj
hj=mzp("1.211")

hj
hj=mzp("1.273","&text=Linux")

hj
hj=mzp("1.273","&text=DevOps")

hj
hj=mzp("1.221","&text=Data")

hj