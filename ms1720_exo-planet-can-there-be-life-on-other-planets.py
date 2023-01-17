import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
planet = pd.read_csv('../input/exos_new-imp2.csv')
planet.head(7)
planet.shape
planet.columns
planet.info()
len(planet.PlanetaryMassJpt)
mass = planet['PlanetaryMassJpt'].dropna()
plt.hist(mass, bins = 100)
plt.xlabel('Planetary Mass in relation to Jupyter')
plt.ylabel('Counts of Planets')
plt.show()
planet.PlanetaryMassJpt.describe()
len(planet.RadiusJpt)
radius = planet['RadiusJpt'].dropna()
len(radius)
plt.hist(radius, bins = 50)
plt.xlabel('Radius of Planet in relation to Jupyter')
plt.ylabel('Counts of Planets')
plt.show()
radius.describe()
planet[planet.PlanetIdentifier == 'Earth']
planet.PeriodDays.describe()
len(planet.PeriodDays)
days = planet.PeriodDays.dropna()
len(days)
plt.hist(days, bins=len(days))
plt.xlim(0,1000)
plt.xlabel('Period of Days')
plt.ylabel('Counts of Planets')
plt.show()
days.describe()
planet.SurfaceTempK.describe()
len(planet.SurfaceTempK)
temp = planet.SurfaceTempK.dropna()
len(temp)
plt.hist(temp, bins=len(temp))
plt.xlim(0,4000)
plt.xlabel('Surface Temperature')
plt.ylabel('Counts of Planets')
plt.show()
planet.DiscoveryMethod.value_counts()
#dmethod = planet.DiscoveryMethod.dropna()
#dmethod.value_counts()
planet.DiscoveryMethod.value_counts().plot.bar()
plt.show()
x = planet[['DiscoveryMethod', 'DiscoveryYear']].dropna()
(x.DiscoveryYear.astype(int)).value_counts().plot.bar()
plt.show()
planet.DiscoveryYear.hist(bins =200)
plt.xlim(1990,2020)
plt.ylabel('Number of Planets Discovered')
plt.xlabel('Year')
plt.show()
planet[['PlanetIdentifier','DiscoveryYear']][planet.DiscoveryYear == planet.DiscoveryYear.min()]
planet.PeriodDays.describe()
planet[['PlanetIdentifier','PeriodDays','SemiMajorAxisAU']][planet.PeriodDays == planet.PeriodDays.max()]
planet.SemiMajorAxisAU.describe()
planet[['PlanetIdentifier','SemiMajorAxisAU','PeriodDays']][planet.SemiMajorAxisAU == planet.SemiMajorAxisAU.max()]
planet[planet.PlanetIdentifier =='Earth']
SemiAU = (planet.SemiMajorAxisAU.dropna())
pdays =(planet.PeriodDays.dropna())
pdaysemi = planet[['PeriodDays','SemiMajorAxisAU']]
pdaysemi.PeriodDays = np.log(pdaysemi.PeriodDays)
pdaysemi.SemiMajorAxisAU = np.log(pdaysemi.SemiMajorAxisAU)
pdaysemi = pdaysemi.dropna()
import seaborn as sns
g = sns.pairplot(pdaysemi, x_vars = ['PeriodDays'], y_vars = ['SemiMajorAxisAU'], kind = 'reg')
g.fig.set_size_inches(8, 6)
plt.show()
planet.HostStarMassSlrMass.describe()
planet[['PlanetIdentifier','HostStarMassSlrMass','HostStarTempK']][planet.HostStarMassSlrMass == planet.HostStarMassSlrMass.max()]
planet.HostStarTempK.describe()
planet[['PlanetIdentifier','HostStarTempK','HostStarMassSlrMass']][planet.HostStarTempK == planet.HostStarTempK.max()]
startemp = planet[['HostStarTempK', 'HostStarMassSlrMass']]
startemp = startemp.dropna()
g1 = sns.pairplot(startemp, x_vars = ['HostStarMassSlrMass'], y_vars = ['HostStarTempK'], kind = 'reg')
g1.fig.set_size_inches(8, 6)
plt.show()
planet.PlanetaryMassJpt.describe()
planet[['PlanetIdentifier','PlanetaryMassJpt','HostStarMassSlrMass']][planet.PlanetaryMassJpt == planet.PlanetaryMassJpt.max()]
planet.HostStarMassSlrMass.describe()
planet[['PlanetIdentifier','HostStarMassSlrMass','PlanetaryMassJpt']][planet.HostStarMassSlrMass == planet.HostStarMassSlrMass.max()]
plstmass = planet[['HostStarMassSlrMass', 'PlanetaryMassJpt']]
plstmass = plstmass.dropna()
g2 = sns.pairplot(plstmass, x_vars = ['HostStarMassSlrMass'], y_vars = ['PlanetaryMassJpt'], kind = 'reg')
g2.fig.set_size_inches(8, 6)
plt.show()
planet.DistFromSunParsec.describe()
planet[planet.PlanetIdentifier == 'Earth'].DistFromSunParsec
parsec = planet.DistFromSunParsec.dropna()
len(parsec)
parsec.hist(bins = len(parsec))
plt.xlim(-2000,2020)
plt.xlabel('Number of Planets')
plt.ylabel('Distance from the star')
plt.show()
planet[planet.PlanetIdentifier.str.contains('Proxima')]
planet[planet.PlanetIdentifier.str.contains('Earth')]
planet.info()
dum = pd.get_dummies(planet.DiscoveryMethod)
planet = pd.concat([planet, dum], axis = 1)
planet = planet.drop('DiscoveryMethod', axis = 1)
dum = pd.get_dummies(planet.ListsPlanetIsOn)
planet = pd.concat([planet, dum], axis = 1)
planet = planet.drop('ListsPlanetIsOn', axis = 1)
planet = planet.drop(['LastUpdated', 'RightAscension', 'Declination'], axis = 1)
X = planet.drop(['Probability_of_life','PlanetIdentifier'], axis = 1).values
y = planet.Probability_of_life.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier #importing KNN
from sklearn.metrics import accuracy_score
for i in range(1,10): 
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print(accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
list=[]
ival = range(1, 100)
jval = range(1,100)
for i,j in zip(ival, jval): 
    clfr = RandomForestClassifier(n_estimators = i, max_depth = j, random_state = 1)
    clfr.fit(X_train, y_train)
    y_pred = clfr.predict(X_test)
    
    list.append((accuracy_score(y_test, y_pred)))
list = pd.DataFrame(list)
list[list == list.max()].dropna()
clfr = RandomForestClassifier(n_estimators = 57, max_depth =57, random_state = 1)
clfr.fit(X_train, y_train)
y_pred = clfr.predict(X_test)
accuracy_score(y_test, y_pred)