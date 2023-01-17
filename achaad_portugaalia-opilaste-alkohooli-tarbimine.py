# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.ticker as plticker

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))







# Any results you write to the current directory are saved as output.



# Loeme sisse andmebaasi

df = pd.read_csv("../input/student-mat.csv")



loc = plticker.MultipleLocator(base=1.0)
# Uurime DataFrame'i ülesehitust

df.info()
m = df.health[df["sex"] == "M"]

n = df.health[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga halb, 5 - väga hea")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ÕPILASTE TERVIS', color='black')

v.legend(["Mehed", "Naised"]);
m = df.Dalc[df["sex"] == "M"]

n = df.Dalc[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga vähe, 5 - väga palju")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE NÄDALA SEES', color='black')

v.legend(["Mehed", "Naised"]);
m = df.Walc[df["sex"] == "M"]

n = df.Walc[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga vähe, 5 - väga palju")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE NÄDALAVAHETUSEL', color='black')

v.legend(["Mehed", "Naised"]);


l = df.studytime[df["Walc"] == 1]

m = df.studytime[df["Walc"] == 2]

n = df.studytime[df["Walc"] == 3]

o = df.studytime[df["Walc"] == 4]

p = df.studytime[df["Walc"] == 5]



q = l.plot.hist(bins=4, grid=False, rwidth=0.95, color='black');

r = m.plot.hist(bins=4, grid=False, rwidth=0.8, alpha=0.8, color='red');

s = n.plot.hist(bins=4, grid=False, rwidth=0.65, alpha=0.8, color='yellow');

t = o.plot.hist(bins=4, grid=False, rwidth=0.5, alpha=0.8, color='brown');

u = p.plot.hist(bins=4, grid=False, rwidth=0.35, alpha=0.8, color='green');

q.set_xlabel("1 = <2 tundi, 2 = 2 kuni 5 tundi, 3 = 5 kuni 10 hours,  4 = >10 tundi")

q.set_ylabel("Õpilaste arv")

s.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE VS ÕPPIMISELE KULUTATUD AEG NÄDALAS', color='black')

q.legend(["Väga vähe", "Natsa", "Keskmiselt", "Sai natsa palju", "Lukku"]);
l = df.health[df["Walc"] == 1]

m = df.health[df["Walc"] == 2]

n = df.health[df["Walc"] == 3]

o = df.health[df["Walc"] == 4]

p = df.health[df["Walc"] == 5]



q = l.plot.hist(bins=5, grid=False, rwidth=0.95, color='black');

r = m.plot.hist(bins=5, grid=False, rwidth=0.8, alpha=0.8, color='red');

s = n.plot.hist(bins=5, grid=False, rwidth=0.65, alpha=0.8, color='yellow');

t = o.plot.hist(bins=5, grid=False, rwidth=0.5, alpha=0.8, color='brown');

u = p.plot.hist(bins=5, grid=False, rwidth=0.35, alpha=0.8, color='green');

q.set_xlabel("1 - väga halb, 5 - väga hea")

q.set_ylabel("Õpilaste arv")

s.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE VS TERVIS', color='black')

q.legend(["Väga vähe", "Natsa", "Keskmiselt", "Sai natsa palju", "Lukku"]);
# Info pere suurusest

print(df["famsize"].describe())

print("{0}% peredest on kolmest suurem.".format(round(281/395 * 100, 2)))

print("{0}% peredest on 3-liikmelised või väiksemad.".format(round((395-281)/395 * 100, 2)))
# Info pere staatusest

print(df["Pstatus"].describe())

print("{0}% peredest vanemad on abiellunud.".format(round(354/395 * 100, 2)))

print("{0}% peredest on ainult üks vanem.".format(round((395-354)/395 * 100, 2)))
# Info emate haridusest

print(df["Medu"].describe())

print(df["Medu"].value_counts())

df.Medu.plot.hist(bins= 5, rwidth=0.95);
# Info isade haridusest

df["Fedu"].describe()

print(df["Fedu"].value_counts())

df.Fedu.plot.hist(bins= 5, rwidth=0.95);
# Info alkoholi tarbimisest tööpäeviti

print(df["Dalc"].describe())

print(df["Dalc"].value_counts())

df.Dalc.plot.hist(bins= 5, rwidth=0.95);
# Info alkoholi tarmibimsest nädalavahetustel

df["Walc"].describe()

print(df["Walc"].value_counts())

df.Walc.plot.hist(bins= 5, rwidth=0.95);
df.plot.scatter("Dalc", "Walc", alpha = 0.2);

from scipy.stats.stats import pearsonr

print(pearsonr(df.Dalc, df.Walc))
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit(df["famsize"])

df["famsize"] = enc.transform(df["famsize"])

# 0 == GT3 & 1 == LE3

df.plot.scatter("famsize", "Dalc", alpha = 0.2);

print(pearsonr(df.famsize, df.Dalc))

df.plot.scatter("famsize", "Walc", alpha = 0.2);

print(pearsonr(df.famsize, df.Walc))
df.plot.scatter("Medu", "Fedu", alpha = 0.2);

print(pearsonr(df.Medu, df.Fedu))
df.plot.scatter("Medu", "Dalc", alpha = 0.2);

print(pearsonr(df.Medu, df.Dalc))

df.plot.scatter("Medu", "Walc", alpha = 0.2);

print(pearsonr(df.Medu, df.Walc))
df.plot.scatter("Fedu", "Dalc", alpha = 0.2);

print(pearsonr(df.Fedu, df.Dalc))

df.plot.scatter("Fedu", "Walc", alpha = 0.2);

print(pearsonr(df.Fedu, df.Walc))