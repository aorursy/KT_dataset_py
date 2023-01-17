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
def kontoerstellen(inhaber, kontonummer, betrag):
    return {"Inhaber" : inhaber, "Kontonummer": kontonummer, "Betrag":betrag}
k1 = kontoerstellen("Ana", "011203113", 500)
k1["Betrag"]
def einzahlen(konto, betrag):
    konto["Betrag"] += betrag
print("Vorher:", k1["Betrag"])
einzahlen(k1, 1000)
print("Nachher:", k1["Betrag"])
def ueberweisen(konto1, konto2, betrag):
    konto1["Betrag"] -= betrag
    konto2["Betrag"] += betrag
k2 = kontoerstellen("Marc", "0123124", 10)
print("Vorher")
print(k1["Inhaber"], k1["Betrag"])
print(k2["Inhaber"], k2["Betrag"])

ueberweisen(k1, k2, 200)

print("Nachher")
print(k1["Inhaber"], k1["Betrag"])
print(k2["Inhaber"], k2["Betrag"])
class Konto:
    inhaber = "Marc"
    betrag = 500
    def einzahlen(self, geld):
        self.betrag += geld
k1 = Konto()
k1.betrag
k1.einzahlen(5000)
k1.betrag
class Auto:
    
    def warnblinker(self):
        print("WARNBLINKER!!!")
        
vw = Auto()
vw.warnblinker()
class Konto:
    umsatzlimit = 1000
    start_umsatz = 0
    def __init__(self, _inhaber, _kontonummer, _betrag):
        self.inhaber = _inhaber
        self.kontonummer = _kontonummer
        self.betrag = _betrag
        
    def einzahlen(self, geld):
        self.betrag += geld
        
    def ueberweisen(self, other, geld):
        self.betrag -= geld
        other.betrag += geld
k1 = Konto("Lars", "102201134", 50)

k1.inhaber
k2 = Konto("Ana", "12305012", 500)

print(k1.inhaber, k2.inhaber)
print("Vorher",k1.inhaber, k1.betrag,"\n", k2.inhaber, k2.betrag)
k1.ueberweisen(k2, 600)
print("Nachher",k1.inhaber, k1.betrag,"\n", k2.inhaber, k2.betrag)
from random import randint

class Wuerfel:
    
    def __init__(self):
        self.augenzahl = randint(1,6)
        
    def __str__(self):
        return str(self.augenzahl)
        
    def werfen(self):
        self.augenzahl = randint(1,6)
        
w1 = Wuerfel()
        
w1.__init__()
w1.werfen()
print(w1)
class Bruch:
    def __init__(self, zaehler, nenner):
        self.z = zaehler
        self.n = nenner
        
    def __str__(self):
        return "{}/{}".format(self.z, self.n)
    
    def __mul__(self, other):
        if isinstance(other, Bruch):
            return Bruch(self.z*other.z, self.n*other.n)
        elif isinstance(other, int) or isinstance(other, float):
            return Bruch(self.z * other, self.n)
    
    def __add__(self, other):
        return Bruch(self.z*other.n + self.n*other.z, self.n*other.n)

    def __ne__(self, other):
        return self.z/self.n != other.z/other.n
    
    def __eq__(self, other):
        return self.z/self.n == other.z/other.n
b1 = Bruch(1,2)
b2 = Bruch(2,3)

print(b1 * 2)
class Angestellte:
    bef = 1.04
    
    def __init__(self, _first, _last, _wage):
        self.first = _first
        self.last = _last
        self.wage = _wage
    
    def befoerdern(self):
        self.wage = self.wage * self.bef
        

a1 = Angestellte("Lars", "Grygosch", 700)

print(a1.bef)
print(a1.wage)
a1.befoerdern()

print(a1.wage)
class Programmierer(Angestellte):
    bef = 1.10
    
    def __init__(self, _first, _last, _wage, _progL):
        super().__init__(_first, _last, _wage)
        self.progL = _progL
p1 = Programmierer("Marc", "Bertram", 1500, "python")
print(p1.wage)

p1.befoerdern()

print(p1.wage)
print(p1.progL)
liste = [randint(1,5) for x in range(100)]
#liste.append(1000)

def mittelwert(sequenz):
    return sum(sequenz)/len(sequenz)

def median(sequenz):
    sequenz.sort()
    return (sequenz[(len(sequenz)-1)//2] + sequenz[(len(sequenz)-1)//2 + (len(sequenz)-1)%2])/2

print(median(liste))
print(mittelwert(liste))



import numpy as np
array = np.array(liste)
np.std(array)
import numpy as np
import pandas as pd
data = {"Name": ["Marc", "Güncem", "Dora", "Ana"]}

data["Name"]
s = pd.Series(data)
print(s)
data = {"Name": ["Marc", "Güncem", "Dora", "Ana"], "Alter": [33, 45, np.nan, 28], "Geschlecht":["m", np.nan, "w", "w"]}

df = pd.DataFrame(data)
#df.index = ["a", "b", "c", "d"]
#df.columns = ["n", "a", "s"]
df
df.iloc[2, 2]
df.loc[:, "Alter"]
df[["Alter", "Name"]]
df["Größe"] = [1.9, 1.7, 1.6, 1.7]
df
df.info()
df.shape
df.describe()
df[df.isnull().any(axis=1)]
from random import randint
liste = [bool(randint(0,1)) for i in range(4)]
liste
all(liste)
df
df[df["Alter"] > 30]
df[df.isnull().any(axis=1)]
df1 = df
df1
df.iloc[1, 2] = "w"
df
df.groupby("Geschlecht").describe()
