!jupyter nbextension enable --py --sys-prefix ipyleaflet

from mpl_toolkits.basemap import Basemap as mp

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import seaborn as sns

import numpy as np

from sklearn.cluster import KMeans
data = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')

data.head()
data.info()
data['DAY_OF_WEEK'].unique().tolist()
#criando coluna dias da semana | transformando dados ordinais

days = { 'Monday' : 1, 'Tuesday' : 2, 'Wednesday' : 3, 'Thursday' : 4, 'Friday' : 5, 'Saturday' : 6, 'Sunday' : 7}

data['DAY_OF_WEEK'] = data['DAY_OF_WEEK'].map(days)
data['DAY_X'] = np.sin((data['DAY_OF_WEEK'])*(2.*np.pi/30))

data['DAY_Y'] = np.cos((data['DAY_OF_WEEK'])*(2.*np.pi/30))
data['MONTH_X'] = np.sin((data['MONTH'])*(2.*np.pi/12))

data['MONTH_Y'] = np.cos((data['MONTH'])*(2.*np.pi/12))
data['HOUR_X'] = np.sin((data['HOUR'])*(2.*np.pi/24))

data['HOUR_y'] = np.cos((data['HOUR'])*(2.*np.pi/24))

plt.scatter(data['HOUR'], data['HOUR'], marker='.');
plt.scatter(data['HOUR_y'], data['HOUR_X'], marker='.')

plt.axes().set_aspect('equal')
data.columns
#Reorganizando ordem das colunas

cols = ['INCIDENT_NUMBER', 'DISTRICT', 'REPORTING_AREA', 'SHOOTING',

       'OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR',  'DAY_X', 'DAY_Y', 'MONTH_X',

       'MONTH_Y', 'HOUR_X', 'HOUR_y','STREET', 'Lat', 'Long', 'Location', 'OFFENSE_CODE', 'OFFENSE_CODE_GROUP',

       'OFFENSE_DESCRIPTION','UCR_PART']

data = data.reindex(columns=cols)
#apagando por ser uma coluna quase toda nula

data.drop(['SHOOTING'], axis=1,inplace=True)

#apagando por primeiro teste, de não necessitar da coluna STREET e LOCATION

#data.drop(columns='STREET', inplace=True)

#data.drop(columns='Location', inplace=True)

#apagando coluna DAY_OF_WEEK, HOUR, MONTH, pois foi criado colunas ciclicas 

data.drop(columns='DAY_OF_WEEK', inplace=True)

data.drop(columns='HOUR', inplace=True)

data.drop(columns='MONTH', inplace=True)

data.drop(columns='OCCURRED_ON_DATE', inplace=True)

#apagando coluna INCIDENT_NUMBER por se tratar de um id e não influenciar

data.drop(columns='INCIDENT_NUMBER', inplace=True)
data['OFFENSE_CODE_GROUP'].unique().tolist()
data['OFFENSE_DESCRIPTION'].unique().tolist()
print(len(data['OFFENSE_DESCRIPTION'].unique().tolist()), len(data['OFFENSE_CODE'].unique().tolist()))

# Class 1 {Roubos, assaltos, armas, urgencias}

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Aggravated Assault')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'HOME INVASION')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Ballistics')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Robbery')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Larceny From Motor Vehicle')]= 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Larceny')]= 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Auto Theft')]= 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Residential Burglary')]= 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Fire Related Reports')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Firearm Violations')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Homicide')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Other Burglary')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Commercial Burglary')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Firearm Discovery')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Bomb Hoax')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Explosives')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Biological Threat')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Manslaughter')] = 1

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Burglary - No Property Taken')] = 1







# Class 2 {Vandalismo, Direção perigosa, Violações}

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Disorderly Conduct')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Vandalism')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Violations')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Property Related Damage')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Restraining Order Violations')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Arson')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'License Violation')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Assembly or Gathering Violations')] = 2

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Criminal Harassment')] = 2









# Class 3 {Discussões, Agressões simples, Causa pequenas}

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] ==  'Verbal Disputes')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Simple Assault')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Drug Violation')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Liquor Violation')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Harassment')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Landlord/Tenant Disputes')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Prisoner Related Incidents')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Offenses Against Child / Family')] = 3

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Prostitution')] = 3











# Class 4 {Investigação, apreensão..}

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] ==  'Investigate Person')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Fraud')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Investigate Property')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Confidence Games')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Missing Person Reported')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Embezzlement')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Missing Person Located')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Property Lost')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Recovered Stolen Property')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Auto Theft Recovery')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Property Found')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Warrant Arrests')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Gambling')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Counterfeiting')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Search Warrants')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'INVESTIGATE PERSON')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'HUMAN TRAFFICKING')] = 4

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE')] = 4











# Class 5 {Outros}

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] ==  'Medical Assistance')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Other')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Towed')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Aircraft')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'License Plate Related Incidents')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Operating Under the Influence')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Police Service Incidents')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Phone Call Complaints')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Harbor Related Incidents')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Evading Fare')] = 5

data['OFFENSE_CODE_GROUP'].loc[(data['OFFENSE_CODE_GROUP'] == 'Service')] = 5







encoder = LabelEncoder()

data['OFFENSE_DESCRIPTION'] = encoder.fit_transform(data['OFFENSE_DESCRIPTION'])

data['DISTRICT'] = encoder.fit_transform(data['DISTRICT'].astype(str))

data['REPORTING_AREA'] = encoder.fit_transform(data['REPORTING_AREA'].astype(str))

data['STREET'] = encoder.fit_transform(data['STREET'].astype(str))

data['Location'] = encoder.fit_transform(data['Location'])
data.info()
data.columns
data.shape
data['UCR_PART'].unique().tolist()
mp = {'Part One' : 1, 'Part Two' : 2, 'Part Three' : 3, 'Other' : 4}

data['UCR_PART'] = data['UCR_PART'].map(mp)
data.info()
data.isnull().sum()
data.dropna(inplace=True)
data.info()
df = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')

plt.figure(figsize=(20,10))

df['OFFENSE_CODE_GROUP'].value_counts().plot.bar()

plt.show()
plt.figure(figsize=(20,10))

df['UCR_PART'].value_counts().plot.bar()

plt.show()
plt.figure(figsize=(20,10))

df['YEAR'].value_counts().plot.bar()

plt.show()
plt.figure(figsize=(20,10))

df['DAY_OF_WEEK'].value_counts().plot.bar()

plt.show()
plt.figure(figsize=(20,10))

df['HOUR'].value_counts().plot.bar()

plt.show()
plt.figure(figsize=(20,10))

df['DISTRICT'].value_counts().plot.bar()

plt.show()
corr = data.corr(method ='pearson')

corr.style.background_gradient(cmap='RdYlGn', ).set_precision(2)
data.info()
std = StandardScaler()

X = data.iloc[:, 0:13].values

y = data['OFFENSE_CODE_GROUP'].values

X = std.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42, test_size=0.20, stratify=y)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
model_knn = KNeighborsClassifier(n_neighbors=5)

model_knn.fit(X_train, Y_train)

predict_knn = model_knn.predict(X_test)

model_knn.score(X_train, Y_train)

print('KNN - ',accuracy_score(predict_knn, Y_test))
model_random = RandomForestClassifier(criterion='gini',n_estimators=20,max_depth=15)

model_random.fit(X_train, Y_train)

predict_random = model_random.predict(X_test)

model_random.score(X_train, Y_train)
print('Random Forest - ', accuracy_score(predict_random, Y_test))
model_decision = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

model_decision.fit(X_train, Y_train)

predict_decision = model_decision.predict(X_test)

model_decision.score(X_train, Y_train)
print('Decision Tree - ',accuracy_score(predict_decision, Y_test))
std = StandardScaler()

X1 = data.iloc[:, 0:13].values

y1 = data['UCR_PART'].values

X1 = std.fit_transform(X1)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, y1, random_state=42, test_size=0.20, stratify=y1)
wcss = []

for i in range(1,15):

    model = KMeans(n_clusters=i, random_state=0)

    model.fit(X_train1)

    wcss.append(model.inertia_)

plt.plot(range(1,15), wcss)

plt.xlabel('numero de cluster')

plt.ylabel('wcss')
model_knn = KNeighborsClassifier(n_neighbors=5)

model_knn.fit(X_train1, Y_train1)

predict_knn1 = model_knn.predict(X_test1)

model_knn.score(X_train1, Y_train1)
print('KNN - ',accuracy_score(predict_knn1, Y_test1))
y=[]

x=[]

for i in range (1,18):

    rdc = RandomForestClassifier(max_depth=i)

    rdc = rdc.fit(X_train1, Y_train1)

    y_pred = rdc.predict(X_test1)

    scores = accuracy_score(y_pred, Y_test1)

    y.append(np.array(scores).mean())

    x.append(i)    

plt.plot(x, y)

plt.show()

print('',y)
y=[]

x=[]

for i in range (1,40):

    rdc = RandomForestClassifier(n_estimators=i)

    rdc = rdc.fit(X_train1, Y_train1)

    y_pred = rdc.predict(X_test1)

    scores = accuracy_score(y_pred, Y_test1)

    y.append(np.array(scores).mean())

    x.append(i)    

plt.plot(x, y)

plt.show()

print('',y)
model_random = RandomForestClassifier(criterion='gini',n_estimators=40,max_depth=15)

model_random.fit(X_train1, Y_train1)

predict_random1 = model_random.predict(X_test1)

model_random.score(X_train1, Y_train1)
print('Random Forest - ', accuracy_score(predict_random1, Y_test1))
feature_importances = pd.DataFrame(model_random.feature_importances_,index = ['DISTRICT', 'REPORTING_AREA',

        'YEAR', 'DAY_X', 'DAY_Y', 'MONTH_X','MONTH_Y', 'HOUR_X', 'HOUR_y','STREET', 'Lat', 'Long', 'Location'],

        columns=['importance']).sort_values('importance',  ascending=False)

feature_importances
y=[]

x=[]

for i in range (1,16):

    clf = DecisionTreeClassifier(max_depth=i)

    clf = clf.fit(X_train1, Y_train1)

    y_pred = clf.predict(X_test1)

    scores = accuracy_score(y_pred, Y_test1)

    y.append(np.array(scores).mean())

    x.append(i)    

plt.plot(x, y)

plt.show()

print('',y)
model_decision = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

model_decision.fit(X_train1, Y_train1)

predict_decision1 = model_decision.predict(X_test1)

model_decision.score(X_train1, Y_train1)
print('Decision Tree - ',accuracy_score(predict_decision1, Y_test1))