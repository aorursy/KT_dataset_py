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
print(2 + 3)

print(2 * 2)
22 ** 2
x = 2
x
x = "Ein String"
type(x)
y = 2

type(y)
type(2.45)
a = True

b = False

type(a)
i = "1"

j = int(i)
type(j)
i = 2

print(i)
f = 2.543

round(f)
help(round)
round(f, 1)
def verdoppeln(zahl):

    return zahl * 2
verdoppeln(3)
def vervielfachen(zahl, vielfach=2):

    return zahl * vielfach
vervielfachen(2, vielfach=3)
# Das ist ein Kommentar

x = 1
def vervielfachen(zahl, vielfach=2):

    """

    Funktion zum vervielfachen

    2. Parameter ist optional. Ansonsten verdoppeln

    

    zahl: zu vervielfachende Zahl

    vielfach: Der Multiplikator

    """

    return zahl * vielfach
help(vervielfachen)
w = True

f = False
f
f and w
w or f
not w
type(w)
2 == 2
2 != 2
2 > 2
i = 1

j = 2
if i == j:

    print("i ist gleich j")

elif i > j:

    print("i ist größer als j")

elif i < j:

    print("i ist kleiner als j")

else:

    print("i ist nicht gleich j")
umsaetze = [100, 120, 110, 90]
umsaetze
dir(umsaetze)
help(umsaetze.append)
umsaetze.append(1000)
umsaetze
umsaetze[1]
type(umsaetze)
len(umsaetze)
for umsatz in umsaetze:

    print(umsatz)
summe = 0

for umsatz in umsaetze:

    summe = summe + umsatz
print(summe)
sum(umsaetze)
def flaeche(laenge, breite):

    return laenge * breite
flaeche(20,20)
username = "Peter"

password = "sdf"



if username == "Peter" and password == "Geheim":

    print("Willkommen Peter!")

elif username != "Peter" and password == "Geheim":

    print("Falscher Name")

elif username == "Peter" and password != "Geheim":

    print("Falsches Passwort")

else:

    print("Falscher Nutzername und Passwort")
l = [2, 4, 6, 10, 15]

for i in l:

    print(i ** 2)
adjectives = ["preiswerte", "billige", "günstige", "herabgestzte"]

nouns = ["Stühle", "Sessel", "Möbel", "Tische"]



for adjective in adjectives:

    for noun in nouns:

        print(f"Keyword: {adjective} {noun}")
i = 10

print(f"Der wert von i ist {i}!")
import math
help(math)
math.sin(0)
from math import sin
sin(0)
import pandas as pd
import math as m
m.sin(0)
tele = {"Peter":"03012234345", "Paul":"01754325345", "Alice": "02342545245"}
tele["Peter"]
tele["Alice"]
stadt = {"Peter":"Berlin", "Paul":"Hamburg", "Alice": "Köln"}
stadt["Peter"]
tele["Peter"]
wines = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wines.head(3)
wines.tail()
wines["country"]
country_prices = wines[["country", "price"]]
wines.head()
wines.info()
wines.describe()
wines["country"].unique()
wines["country"].value_counts()
wines["price"].max()
wines["price"].min()
wines["price"].mean()
wines["price"].median()
wines["price"].sum()
filter = wines["country"] == "Germany"

wines[filter].head(2)
wines[wines["country"] == "Germany"]
wines
german_wines = wines[wines["country"] == "Germany"]
wines[wines["country"] == "germany".title()]
german_wines.head(2)
wines["country"].value_counts().plot(kind="barh", figsize= (10,10))
help(wines["country"].value_counts().plot)
wines["points"].hist()