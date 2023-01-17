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
2 * ( 2 + 4) ** 2
10 % 3
x = 2 * 5 + 9

print(x)
x = 2 * x + 10

x
x
x = 2 * x+10
x
x = 2
x = 2

y = 3

print(z)

z  = x + y + z

print(z)
type(z)
type(4)
type(5.5)
f = 2 / 3

type(f)
type("Das ist ein String")
s = "Das ist ein String"

type(s)
s = "15"
int(s) + 2
w = False
print(f)
f = 3.542

round(f)
f
help(round)
round(3.5423)
def verdoppeln(zahl):

    return zahl * 2
y = verdoppeln(2)
print(y)
def vervielfachen(zahl, vielfach = 2):

    return zahl * vielfach
vervielfachen(2, 3)
# Das ist ein Kommentar

x = 1 
def vervielfachen(zahl, vielfach = 2):

    """

    Funktion zum vervielfachen

    2. Parameter ist optional. Ansonst ist es verdoppeln

    

    zahl: Zu vervielfachende Zahl

    vielfach: Der Multiplikator

    """

    return zahl * vielfach
help(vervielfachen)
help(print)
w = True

f = False
f
w and f
f and f
w or f
(w and f) or w
not w
not f
(not w or f)
type(w)
type(f)
2 == 2
2 == 2.2
2 < 2.2
2 <= 2
2 != 3
2 != 2
(2 != 2) and (3 < 2) 
i = 1

j = 2
if i == j:

    print("i ist gleich j")
if i < j:

    print("i ist kleiner als j")

else:

    print("i ist nicht kleiner als j")
i = 4

j = 3

if i < j:

    print("i ist kleiner als j")

elif i == j:

    print("i ist gleich j")

else:

    print("i ist größer als j")
umsatz_0 = 100

umsatz_1 = 110

umsatz_3 = 90
summe = umsatz_0 + umsatz_1 + umsatz_3
summe
umsaetze = [100, 110, 90, 200, 1000]
umsaetze
type(umsaetze)
umsaetze[4]
umsaetze[-1]
len(umsaetze)
umsaetze[4]
umsaetze[len(umsaetze) - 1]
for umsatz in umsaetze:

    print(umsatz)
def gesamtumsatz(umsaetze):

    summe = 0

    for umsatz in umsaetze:

        summe = summe + umsatz

    return summe



umsaetze = [100, 110, 90, 200, 1000]

gesamtumsatz(umsaetze)

    
def clv(umsaetze, zinssatz):

    """Berechnung des Customer Lifetime Value durch

    den Barwert der zukünftigen Umsätze 
    """

    summe = 0

    t = 1

    for umsatz in umsaetze:

        summe = summe + umsatz / (1 + zinssatz) ** t

        t = t + 1

    return summe

umsaetze = [100, 110, 90, 200, 1000]

clv(umsaetze, 0.1)
def clv(umsaetze, zinssatz):

    """

    Berechnung des Customer Lifetime Value durch

    den Barwert der zukünftigen Umsätze 

    NPV

    """

    summe = 0

    t = 1

    for umsatz in umsaetze:

        summe = summe + umsatz / (1 + zinssatz) ** t

        t = t + 1

    return summe



umsaetze = [100, 110, 90, 200, 1000]

zins = 0.1

clv(umsaetze, zins)

    
umsaetze = [100, 110, 90, 200, 1000]

zinssatz = 0.1

print(clv(umsaetze, zinssatz))
sum(umsaetze)