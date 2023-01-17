# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



hs = pd.read_csv("../input/haltestelle.csv")

hp = pd.read_csv("../input/haltepunkt.csv")

fzl1 = pd.read_csv("../input/fahrzeitensollist2016091120160917.csv")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from datetime import datetime

fzl1['betriebsdatum'] = pd.Series(datetime.strptime(x, '%d.%m.%y') for x in fzl1['betriebsdatum'])

fzl1['datum_nach'] = pd.Series(datetime.strptime(x, '%d.%m.%y') for x in fzl1['datum_nach'])

fzl1['datum_von'] = pd.Series(datetime.strptime(x, '%d.%m.%y') for x in fzl1['datum_von'])
betriebsd, dvon, dnach, fzeug, lin = fzl1['betriebsdatum'],fzl1['datum_von'],fzl1['datum_nach'],fzl1['fahrzeug'],fzl1['linie']
haltstelle = hs['halt_lang'].values
klus_find_list_inhs = np.array([i.find('Klusplatz') for i in haltstelle])
oerl_find_list_inhs = np.array([i.find('Oerlikon') for i in haltstelle])
haltstelle[klus_find_list_inhs > -1]
for x in haltstelle[oerl_find_list_inhs > -1]:

    print(x)
hs.ix[(haltstelle=='Zürich, Klusplatz')|(haltstelle == 'Zürich, Bahnhof Oerlikon')]
delta_ist__soll_an_nach = fzl1['ist_an_nach1']-fzl1['soll_an_nach']

delta_ist__soll_an_von = fzl1['ist_an_von']-fzl1['soll_an_von']

delta_ist__soll_ab_nach = fzl1['ist_ab_nach']-fzl1['soll_ab_nach']

delta_ist__soll_ab_von = fzl1['ist_ab_von']-fzl1['soll_ab_von']
fzl1['delta_ist__soll__an_nach'] = delta_ist__soll_an_nach

fzl1['delta_ist__soll_an_von'] = delta_ist__soll_an_von

fzl1['delta_ist__soll_ab_nach'] = delta_ist__soll_ab_nach

fzl1['delta_ist__soll_ab_von'] = delta_ist__soll_ab_von
fzl1['soll_mean'] = 0.25*(fzl1['soll_an_nach']+fzl1['soll_an_von']+fzl1['soll_ab_nach']+fzl1['soll_ab_von'])/3600
fzl1['delta_ist__soll_mean'] = 0.25*(delta_ist__soll_ab_nach+delta_ist__soll_ab_von+delta_ist__soll_an_nach+delta_ist__soll_an_von)
fzl1['ist_mean'] = 0.25*(fzl1['ist_an_nach1']+fzl1['ist_an_von']+fzl1['ist_ab_nach']+fzl1['ist_ab_von'])/3600
fzl1.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.lmplot(x="soll_mean", y="delta_ist__soll_mean", data=fzl1.head(50000))

plt.xlabel('timetable in hours - week 16/9/2017')

plt.ylabel('difference between ist and soll in seconds')

plt.show()
sns.lmplot(x="soll_mean", y="ist_mean", data=fzl1.head(50000))

plt.xlabel('timetable in hours - week 16/9/2017')

plt.ylabel('actual timetable in hours')

plt.show()