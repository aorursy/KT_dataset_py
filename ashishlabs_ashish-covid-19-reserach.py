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
data = pd.read_csv("../input/timedifference/HOSP.csv")
for i in range(0, len(data)):

    print("".join(data["S"][i].split("/")))

from datetime import datetime

from dateutil import parser



symtops_occur = []

hospital_visit = []



for i in range(0, len(data)):



    date_time_obj = parser.parse(data["S"][i])

    date_time_obj1 = parser.parse(data["H"][i])

    

    symtops_occur.append(date_time_obj)

    hospital_visit.append(date_time_obj1)

import matplotlib

S = matplotlib.dates.date2num(symtops_occur)

H = matplotlib.dates.date2num(hospital_visit)

S
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))

plt.plot(S, label="Symtomps Occur")

plt.plot(H, label="Hospital Visit")

plt.title("Time Difference Between | Symtopms occur and Hospital Visit")

plt.legend()

plt.savefig("Time Difference Between | Symtopms occur and Hospital Visit.pdf")
symptoms = pd.read_csv('../input/covid19-symptoms-checker/Raw-Data.csv', usecols=['Symptoms'])

len(symptoms)
symtops = []

for i in range(1, len(symptoms)):

    symtops.append((symptoms["Symptoms"][i].split(',')))

a = np.array(symptoms)

a = a.ravel().ravel()

g = []

for i in range(0, len(a)):

    f = ((a[i].split(',')))

    for i in f:

#         print(i)

        g.append(i)

    

g
import seaborn as sns

plt.figure(figsize=(10, 10))

sns.countplot(g)

plt.title("Symtoms in most of the cases")

plt.savefig("Symtompsinmostcases.pdf")
df = pd.read_csv("../input/covid19-symptoms-checker/Cleaned-Data.csv")
df.head()
import seaborn as sns

sns.countplot(df["Fever"])

sns.countplot(df["Tiredness"])

sns.countplot(df["Dry-Cough"])

clasf = []

for i in data["DIFF"]:

    if i >= 6:

        clasf.append(i)

clasf