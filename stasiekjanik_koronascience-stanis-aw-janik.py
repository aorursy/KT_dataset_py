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

folder = '/kaggle/input/coronavirusdataset/'
from matplotlib import  pyplot as plt, dates

from scipy.optimize import curve_fit

from datetime import datetime, timedelta

form = '%Y-%m-%d'



# deklaruję funkcje 

def wiekiem(df, a, kat): # Ta tworzy słownik kategorii które akurat nas obchodzą, np. wiek

    dfc  = df.copy()

    w = dict()

    for wiek in a:

        w[wiek] = dfc.loc[dfc[kat] == wiek]

        w[wiek] = w[wiek][['patient_id','age' ,'sex', 'symptom_onset_date', 'confirmed_date', 'deceased_date', 'released_date']]

    return w



def mean_time(w, event1, event2): # Ta wylicza średni czas między zdarzeniem1 a zdarzeniem2 dla każdej kategorii pacjentów

    diagnoza = {}

    ret = {}

    for wiek in w:

        diagnoza[wiek] = []

        for index, row in w[wiek].iterrows():

            diagnoza[wiek].append(datetime.strptime(row[event2], form)- datetime.strptime(row[event1], form))

        if len(diagnoza[wiek]):

            temp = diagnoza[wiek][0]

            for i in diagnoza[wiek]:

                    diagnoza[wiek][0] += i

            ret[wiek] = (diagnoza[wiek][0]-temp)/len(diagnoza[wiek]) 

    return ret



def clear(dfdic, s):# Ta wyrzuca te wiersze w których występuje NaN w interesującej nas kolumnie

    for df in dfdic:

        dfdic[df] = dfdic[df].dropna(subset=s)

    return dfdic
# Tworzę dataframe'y z których będe korzystał

dfR = pd.read_csv(folder+'Region.csv')

dfT = pd.read_csv(folder+'TimeProvince.csv')

dfPI = pd.read_csv(folder+'PatientInfo.csv')

dfW = pd.read_csv(folder+'Weather.csv')

dfPI.head(10)
# Tworzę liste nazw dekad życia (10s, 20 ...)

a = []

for w in range(1,10):

    a.append(str(w*10)+'s')
s = [clear(wiekiem(dfPI, ['male', 'female'], 'sex'), ['symptom_onset_date', 'confirmed_date']), clear(wiekiem(dfPI, a, 'age'), ['symptom_onset_date', 'confirmed_date'])]

j = [clear(wiekiem(dfPI, ['male', 'female'], 'sex'), ['released_date', 'confirmed_date']), clear(wiekiem(dfPI, a, 'age'), ['released_date', 'confirmed_date'])]

k =  [clear(wiekiem(dfPI, ['male', 'female'], 'sex'), ['deceased_date', 'confirmed_date']), clear(wiekiem(dfPI, a, 'age'), ['deceased_date', 'confirmed_date'])]
def roz(s, e1, e2):

    sdata = []

    for i in s:

        sdata.append(mean_time(i, e1, e2)) #  średni czas diagnozy w zależności od płci lub wieku

        lists = sorted(sdata[-1].items()) # sorted by key, return a list of tuples

        x, y = zip(*lists) # unpack a list of pairs into two tuples

        x = list(x)

        y = list(y)

        y = [h.total_seconds()/3600.0 for h in y]

        plt.bar(x, y)

        plt.title(f'Czas pomiędzy {e1}, a {e2}')

        plt.show
roz(s, 'symptom_onset_date', 'confirmed_date')
roz(j, 'confirmed_date', 'released_date')#  średni czas od diagnozy do wyjścia ze szpitala w zależności od płci lub wieku
roz(k, 'confirmed_date', 'deceased_date') #  średni czas od diagnozy do śmierci w zależności od płci lub wieku
# Tworzę słownik z pogodą w prowincjach, i z przyrostami w kolejnych datach w prowincjach

pogoda = dict()

czas = dict()

for province in dfW.province.unique():

    pogoda[province] = dfW.loc[dfW['province'] == province]

    czas[province] = dfT.loc[dfT['province'] == province].drop(labels='province', axis=1)

# w każdej prowincji patrze tylko na daty po 19 stycznia 2020, tj. dacie wystąpienia pierwszych zakażeń w naszym datasecie 

start = '2020-01-20'

for i in pogoda:

    mask = (pogoda[i]['date'] >= start)

    pogoda[i] = pogoda[i].loc[mask].round()

pogoda['Seoul'].head(10)
for i in pogoda:

    pogoda[i] = pogoda[i].merge(czas[i], on='date')

pogoda['Seoul'].tail(10)
# Tworzę słownik w którym kluczem jest nazwa prowincji, a wartością array ciepłch dni ( takich ze średnią temperaturą powyżej 10 stopni)

ciepłe ={}

for i in pogoda:

    ciepłe[i] =pogoda[i].loc[pogoda[i]['avg_temp'] >= 10].date.values

ciepłe1 = {}

for i in ciepłe:

    ciepłe1[i] = []

    for j in range(len(ciepłe[i])):

        ciepłe1[i].append(datetime.strftime(datetime.strptime(ciepłe[i][j], form) + timedelta(days=10), form))  # sprawdzam po 10 dniach, gdyż jest to 

                                                                                                                # okres podawany przez WHO jako taki po którym wystęþują już objawy
cc = {}

for i in ciepłe1:

    cc[i] = []

    for j in ciepłe1[i]:

        a = pogoda[i].loc[pogoda[i]['date'] == j]['confirmed'].values

        if a.size > 0:

            cc[i].append(a[0])
# zakładam, że ilość zakażeń rośnie wykładniczo

def f(x, a, b, c):

    return a*np.power(b,x)+c
# Dopasowuję odpowiednie parametry dla każdego regionu

opt = {}

for i in pogoda:

    x = np.array(range(len(pogoda[i]['confirmed'].values)))  # numer dnia jest argumentem

    y = pogoda[i]['confirmed'].values # ilość zakażeń jest wartością

    popt, pcov = curve_fit(f, x, y) # dopasowuję parametry

    opt[i] = popt # zapisuję optymalne parametry dla każdego regionu osobno

def różnica_estymacji(ciepłe, opt): # zwraca różnicę pomiędzy stanem faktycznym, a tym co przewidział nasz estymator względem naszych przewidywań

    date_start = datetime.strptime(start, form)

    ret ={}

    for i in pogoda:

        ret[i] = 0

        for j in range(len(ciepłe[i])):

            if len(cc[i]) >j:

                d = datetime.strptime(ciepłe[i][j], form)

                delta = d - date_start

                x = delta.days

                przew = f(x+3, *opt[i])

                dif = (cc[i][j] - przew)/przew

                ret[i] += dif

    return ret



ret = różnica_estymacji(ciepłe, opt)
xdata = []

ydata = []

for i in ret:

    xdata.append(ret[i])

    ydata.append(i)

 # zapisuję różnicę estymacji w każdym regionie



# tworzę graf słupkowy

plt.bar(ydata, xdata)

plt.title("Względna różnica")

plt.xticks(rotation=90)

plt.show()