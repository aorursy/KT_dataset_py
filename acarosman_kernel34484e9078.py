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
coronaall = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")

#hazırlayan Osman Acar
countries=[]

for country in coronaall["Country/Region"]:

    if country not in countries:

        countries.append(country)

confirmed={}

deaths={}

for country in countries:

    confirmed[country]=0

    deaths[country]=0



x=input("Lütfen istediğiniz ayı giriniz")

y=input("Lütfen istediğiniz günü giriniz")

z=str(20)

date=x+"/"+y+"/"+z







for index,country in enumerate(coronaall["Country/Region"]):

    if (coronaall["Date"][index] == date):

        deaths[country] += coronaall["Deaths"][index]

        confirmed[country]+= coronaall["Confirmed"][index]

        

print(deaths)

print("******************")

print(confirmed)

mostdeathcountry="abc"

mostdeathnumber=0

for number in deaths.keys():

    if deaths[number]> mostdeathnumber:

        mostdeathcountry=number

        mostdeathnumber=deaths[number]

        

mostconfirmedcountry="abcd"

mostconfirmednumber=0

for number in confirmed.keys():

    if confirmed[number] > mostconfirmednumber:

        mostconfirmednumber=confirmed[number]

        mostconfirmedcountry=number

totalinfected=0

totaldeath=0

for number in deaths.values():

    totaldeath +=number

for number in confirmed.values():

    totalinfected +=number

print("şu tarihte",date)

print("dünyada toplam ölüm sayısı",totaldeath,"ve toplam infekte olan insan sayısı",totalinfected)

print("Dünya 'da en çok etkilenen ülke(",mostdeathcountry, ")ve ölüm sayısı => ", mostdeathnumber)

print("Dünya 'da en çok enfekte olan ülke (",mostconfirmedcountry,"),ve enfekte sayısı => ", mostconfirmednumber)
