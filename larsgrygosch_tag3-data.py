# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

# einfacher Graph
from random import random

x = np.arange(10)

y = np.array([2*i  + random() for i in range(10)])

plt.plot(x, y)
plt.show()
# Scatter - Plot
from random import random

x = np.arange(10)

y = np.array([2*i  + random() for i in range(10)])

plt.scatter(x, y)
plt.show()
# Säulendiagramm
from random import random

x = np.arange(10)

y = np.array([2*i  + random() for i in range(10)])

plt.bar(x, y)
plt.show()
# Histogramm
from random import normalvariate

x = [normalvariate(1, 0.2) for i in range(10000)]

plt.hist(x)
plt.show()
# Laden des Datensatzes VORSICHT: Statt mit Kommata sind die Daten mit Semicolon getrennt

data = pd.read_csv('/kaggle/input/cusersmarildownloadsearningcsv/earning.csv', delimiter=";")

data.head()
# ein paar Informationen

data.info()
# ein paar statistische Informationen

data.describe()
data.head()
# Speichert die Jahre in der Variable x

x = data.year
# Speichert die Werte für

y = data.femaleprofessionals
plt.plot(x,y)
plt.show()
plt.plot(x,y)

plt.title("Stundenlohn von 2004 - 2017")
plt.xlabel("Jahre")
plt.ylabel("Stundenlohn")


plt.show()
x = data.year
y1 = data.femaleprofessionals
y2 = data.femalesmanagers

plt.plot(x,y1, color="r", label="Professionals (f)")
plt.plot(x, y2, color="k", label="Managers (f)")

plt.title("Stundenlohn von 2004 - 2017")
plt.xlabel("Jahre")
plt.ylabel("Stundenlohn")
plt.legend()


plt.show()
x = data.year
y1 = data.malemanagers
y2 = data.femalesmanagers

plt.plot(x,y1, color="r", label="Managers (m)")
plt.plot(x, y2, color="k", label="Managers (f)")

plt.title("Stundenlohn von 2004 - 2017")
plt.xlabel("Jahre")
plt.ylabel("Stundenlohn")
plt.legend()


plt.show()
x = data.year
y1 = data.malemanagers
y2 = data.femalesmanagers

plt.bar(x+ 0.4, y2, color="k", label="Managers (f)", width=0.4)
plt.bar(x,y1, color="r", label="Managers (m)", width=0.4)


plt.title("Stundenlohn von 2004 - 2017")
plt.xlabel("Jahre")
plt.ylabel("Stundenlohn")
plt.legend()


plt.show()
diff_managers = data.malemanagers - data.femalesmanagers

diff_managers
plt.plot(x, diff_managers )
plt.show()
diff_professionals = data.maleprofessionals - data.femaleprofessionals
diff_managers = data.malemanagers - data.femalesmanagers

year = data.year
plt.plot(year, diff_professionals, label="Professionals")
plt.plot(year, diff_managers, label= "Managers")

plt.title("Stundenlohndifferenz von 2004-2017")
plt.xlabel("Jahre")
plt.ylabel("Stundenlohndifferenz")
plt.legend()

plt.show()
data.info()
female = list(data.columns[2:10])
male = list(data.columns[11:19])

diff = np.array([data[m] - data[f] for m, f in zip(male, female)])
diff.T[0]
data.head()

plt.bar(['managers',
 'professionals',
 'technicians',
 'serviceworkers',
 'clerical',
 'workers',
 'drivers',
 'labourers'], diff.T[0] )

plt.title("Jahr 2004")

plt.xticks(rotation=90)
plt.show()
summe = np.array([(data[m] + data[f])/2 for m, f in zip(male, female)])


# Lohndifferenz 2017

diff_2017 = diff.T[-1]   # Lohndifferenz 2017 sortiert nach Beruf (absolute Wert)
summe_2017 = summe.T[-1] # Durchschnittslohn beider Gruppen für 2017


berufe = ['managers',
 'professionals',
 'technicians',
 'serviceworkers',
 'clerical',
 'workers',
 'drivers',
 'labourers']



