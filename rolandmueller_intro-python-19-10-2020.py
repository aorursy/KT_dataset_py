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
2 * (4 + 5) / 5
2 ** 4
10 % 3
x = 2 * (3 + 3)

x
z = 2

a = x + z

a = 2 * a

a
type(x)
y = str(x)
type(y)
int(2.67)
w = True
type(w)
s = "Das ist ein String"

type(s)
ein_langer_variablen_name = 2
2 / 3
int("10000")
print(x)
f = 23.6345245

rounded_f = round(f)
f
rounded_f
help(round)
round(f, 1)
def vervielfachen(zahl, vielfach=2):

    ergebnis = zahl * vielfach

    return ergebnis



vervielfachen(2)
# Das ist ein Kommentar

x = 1
def vervielfachen(zahl, vielfach=2):

    """

    Funktion zum vervielfachen

    2. Paramter is optional. Ansonsten ist es verdoppeln

    

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
not w
not f
(not w or f)
type(w)
nochnichtdas = "s"
nochnichtdas
a = 2

b = 2

c = 3
a == b
a == c
a > c
a < c
a <= b
(a == b) and (a < c)
i = 1

j = 1
if i == j:

    print("i ist gleich j")
i = 1

j = 0

if i == j:

    print("i ist gleich j")

elif i < j:

    print("i ist kleiner als j")

else:

    print("i ist größer als j")
gewinn_1 = 200

gewinn_2 = 500

gewinn_3 = -200

gewinn_4 = 0
summe = gewinn_1 + gewinn_2 + gewinn_3 + gewinn_4
summe
gewinn = [200, 500, -200, 0]
gewinn
type(gewinn)
# Erster Wert

gewinn[0]
gewinn[3]
gewinn[-1]
len(gewinn)
help(len)
anzahl_der_elemente = len(gewinn)

gewinn[anzahl_der_elemente - 1]
for g in gewinn:

    print(g)
for i in range(10):

    print(i)
help(range)
for i in range(1, 11):

    print(i)
sum(gewinn)
def summieren(gewinne):

    """

    Summiert unsere Gewinne

    """

    summe = 0

    for gewinn in gewinne:

        summe += gewinn

    return summe



summieren(gewinn)
help(summieren)
def summieren_mit_while(gewinne):

    i = 0

    summe = 0

    while i < len(gewinne) - 1:

        summe += gewinne[i]

        i += 1

    return summe



summieren_mit_while(gewinn)