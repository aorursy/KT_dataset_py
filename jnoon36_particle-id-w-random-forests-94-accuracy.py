import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("../input/pid-5M.csv")
df.head()
df["de_Broglie"] = ""
df.head()
h_c = .00000123984 #hc planck's constant, units GeV*nm

def lamda(p):
    return h_c/p
wavelengths = []
for i in range(df["id"].count()):
    x = df["p"][i]
    y = lamda(x)
    wavelengths.append(y)
new_wavelengths = []  
for i in range(len(wavelengths)):   #Putting wavelengths in units of fm
    x = wavelengths[i]
    new_wavelengths.append(x*1000000)
df["de_Broglie"] = new_wavelengths
df.head()
df.describe()
#Momentum Histogram
momentum = np.array(df["p"])
plt.hist(momentum, range = (momentum.min(), momentum.max()))
plt.xlabel("Momentum (GeV/c)")
plt.ylabel("Frequency")
plt.title("Momentum Distribution for all Particles")
#Theta Histogram
theta = np.array(df["theta"])
plt.hist(theta, range = (theta.min(), theta.max()))
plt.xlabel("Theta (Rads)")
plt.ylabel("Frequency")
plt.title("Theta Distribution for all Particles")
#Beta Histogram
beta = np.array(df["beta"])
plt.hist(beta, range = (beta.min(), beta.max()))
plt.xlabel("Beta (Rads)")
plt.ylabel("Frequency")
plt.title("Beta Distribution for all Particles")
#de Broglie Histogram
de_broglie = np.array(df["de_Broglie"])
plt.hist(de_broglie, range = (de_broglie.min(), de_broglie.max()))
plt.xlabel("de Broglie Wavelength (fm)")
plt.ylabel("Frequency")
plt.title("de Broglie Wavelength Distribution for all Particles")
#Heatmap of features

import seaborn as sns
sns.heatmap(df[["id", "p", "theta", "beta", "nphe", "ein", "eout", "de_Broglie"]].corr())
sns.set(style = 'dark')
X = np.array(df.drop("id", axis = 1))
y = np.array(df["id"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_true = train_test_split(X, y, test_size = 0.2)
dct = DecisionTreeClassifier()
rfc = RandomForestClassifier()
dct.fit(X_train, y_train) #Fitting with decision tree
dctpredict = dct.predict(X_test)
accuracy_score(y_true, dctpredict)
print(classification_report(y_true, dctpredict))
rfc.fit(X_train, y_train) #Fitting with random forest
rfcpredict = rfc.predict(X_test)
accuracy_score(y_true, rfcpredict)
print(classification_report(y_true, rfcpredict))
positron = []
pion = []
kaon = []
proton = []

for i in range(df["id"].count()):
    if df["id"][i] == -11:
        positron.append(df.loc[i])
    if df["id"][i] == 211:
        pion.append(df.loc[i])
    if df["id"][i] == 321:
        kaon.append(df.loc[i])
    if df["id"][i] == 2212:
        proton.append(df.loc[i])
print(len(positron))
print(len(pion))
print(len(kaon))
print(len(proton))
from random import shuffle

shuffle(pion)
shuffle(kaon)
shuffle(proton)
new_pion = []
new_kaon = []
new_proton = []

for i in range(len(positron)):
    new_pion.append(pion[i])
    new_kaon.append(kaon[i])
    new_proton.append(proton[i])
print(len(new_pion))
print(len(new_kaon))
print(len(new_proton))
new_data = new_pion + new_kaon + new_proton + positron
print(len(new_data))
    
shuffle(new_data)
newdf = pd.DataFrame(new_data, columns = df.columns)
newdf.head()
newdf = newdf.reset_index()
newdf.head()
newdf = newdf.drop("index", axis = 1)
newdf.head()
newdf.shape
newX = np.array(newdf.drop("id", axis = 1))
newy = np.array(newdf["id"])

newX_train, newX_test, newy_train, newy_true = train_test_split(newX, newy, test_size = 0.2)
dct.fit(newX_train, newy_train)
dctpredict1 = dct.predict(newX_test) #fitting decision tree with balanced sets
accuracy_score(newy_true, dctpredict1)
print(classification_report(newy_true, dctpredict1))
rfc.fit(newX_train, newy_train) #fitting random forest with balanced sets
rfcpredict1 = rfc.predict(newX_test)
accuracy_score(newy_true, rfcpredict1)
print(classification_report(newy_true, rfcpredict1))
rfc1 = RandomForestClassifier(min_samples_leaf = 5, min_samples_split = 10)
dct1 = DecisionTreeClassifier(min_samples_leaf = 5, min_samples_split = 10)
dct1.fit(newX_train, newy_train)
dctpredict2 = dct1.predict(newX_test)
accuracy_score(newy_true, dctpredict2)
print(classification_report(newy_true, dctpredict2))
rfc1.fit(newX_train, newy_train)
rfcpredict2 = rfc1.predict(newX_test)
accuracy_score(newy_true, rfcpredict2)
print(classification_report(newy_true, rfcpredict2))
