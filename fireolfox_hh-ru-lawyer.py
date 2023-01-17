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



   



text="Linux"



area=2

x= []

zpv=0



city=[]

zp=[]



def mzp (sp):

    

  url='https://api.hh.ru/vacancies?area=113&specialization='+sp+'&per_page=100&page='



  for i in range(20):





  

    if i==0:

      data = pd.read_json (url + str(i))

    else:  

      data2 = pd.read_json (url + str(i))

      data=data.append(data2, ignore_index=True)





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

      s=1







  df = pd.DataFrame({'Город' : city, 'зарплата' : zp, 'количество' : ''})

  #out=df.groupby('City').mean()



  out=df.groupby('Город').agg({'количество': "count", 'зарплата': "mean"})







  #out.to_csv('out.csv')

  hh=out.round(0).sort_values(by=['количество'], ascending=False).head(30)

  return hh







hj=mzp("23")

hj