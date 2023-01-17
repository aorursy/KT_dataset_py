import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/spacemissionsflightstatus/SpaceMissions.csv')

data.head(10)
data.info()
data.isnull().sum()
#handling null temperature values

x = data.iloc[:, 4].values

x = x.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x)

x = imputer.transform(x)

data.iloc[:, 4] = x

data.isnull().sum()



#handling null humidity values

x = data.iloc[:, 6].values

x = x.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x)

x = imputer.transform(x)

data.iloc[:, 6] = x

data.isnull().sum()



#handling null wind speed values

x = data.iloc[:, 5].values

x = x.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x)

x = imputer.transform(x)

data.iloc[:, 5] = x

data.isnull().sum()
#handling fairing diameter missing values

data[data["Fairing Diameter (m)"].isnull()]
fm_arienaspace = data[data['Company'] == 'Arianespace']

list_1 = fm_arienaspace['Fairing Diameter (m)']



from statistics import mean

a = list_1.mean()



for a,b in zip(data['Company'], data['Fairing Diameter (m)']):

    if a == 'Arianespace'or a == 'European Space Agency':

         data["Fairing Diameter (m)"] = data["Fairing Diameter (m)"].fillna(a)



data.isnull().sum()





#handling Payload Type missing values

data[data['Payload Type'].isnull()]
#checking for the most frequent payload type in company spacex

fm_arienaspace = data[data['Company'] == 'SpaceX']

fm_arienaspace['Payload Type'].describe()
#filling nan w/most frequent payload type in  SpaceX

data['Payload Type'] = data['Payload Type'].fillna('Communication Satellite')

data.isnull().sum()
#handling payload mass nan values

a = data[data['Payload Mass (kg)'].isnull()]

a['Payload Name'].unique()

data['Payload Mass (kg)'].unique()
#converting the classified rows to numerical data and changing dtype of payloadmass to int

data.loc[ data['Payload Mass (kg)'] == 'Classified', 'Payload Mass (kg)'] = 0

data['Payload Mass (kg)'] = data['Payload Mass (kg)'].astype(float)

data['Payload Mass (kg)'].unique()
#handling null values and changing dtype to int

x = data.iloc[:, -4].values

x = x.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x)

x = imputer.transform(x)

data.iloc[:, -4] = x



data['Payload Mass (kg)'] = data['Payload Mass (kg)'].astype(int)

data.isnull().sum()
data['Launch Time'].describe()
#handling the null launch time values

x = data.iloc[:, 2].values

x = x.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer = imputer.fit(x)

x = imputer.transform(x)

data.iloc[:, 2] = x
#drop the failure reason column

data = data.drop(['Failure Reason'], axis = 1)
#Company

data['Company'].unique()
data.loc[data['Company'] == 'SpaceX', 'Company'] = 0

data.loc[data['Company'] == 'Boeing', 'Company'] = 1

data.loc[data['Company'] == 'Martin Marietta', 'Company'] = 2

data.loc[data['Company'] == 'US Air Force', 'Company'] = 3

data.loc[data['Company'] == 'European Space Agency', 'Company'] = 4

data.loc[data['Company'] == 'Brazilian Space Agency', 'Company'] = 5

data.loc[data['Company'] == 'Arianespace', 'Company'] = 6
data['Company'].unique()
#Launch Site

data['Launch Site'].unique()
data.loc[data['Launch Site'] == 'Marshall Islands', 'Launch Site'] = 0

data.loc[data['Launch Site'] == 'Cape Canaveral', 'Launch Site'] = 1

data.loc[data['Launch Site'] == 'Vandenberg ', 'Launch Site'] = 2

data.loc[data['Launch Site'] == ' Guiana Space Centre', 'Launch Site'] = 3

data.loc[data['Launch Site'] == 'Alcântara Launch Center', 'Launch Site'] = 4

data.loc[data['Launch Site'] == 'Kennedy Space Center', 'Launch Site'] = 5
data = data.rename({'Vehicle Type': "Vehicle_Type"}, axis = 1)

data['Vehicle_Type'].unique()
#Vehicle Type

titles = {"Falcon": 0, "Delta": 1, "Titan": 2, "Ariane": 3, "Vega": 4, "VLS": 5}



# extract titles

data['vehicle_type'] = data.Vehicle_Type.str.extract('([A-Za-z]+)', expand=False)

# convert titles into numbers

data['vehicle_type'] = data['vehicle_type'].map(titles)  



data = data.drop(['Vehicle_Type'], axis=1)

data.head()
# Payload Name and Type

#too much unique values, we're gonna have to drop them

data = data.drop(['Payload Name', 'Payload Type'], axis = 1)
#Payload Orbit

data['Payload Orbit'].unique()
data.loc[data['Payload Orbit'] == 'Low Earth Orbit', 'Payload Orbit'] = 0

data.loc[data['Payload Orbit'] == 'Geostationary Transfer Orbit', 'Payload Orbit'] = 1

data.loc[data['Payload Orbit'] == 'Medium Earth Orbit', 'Payload Orbit'] = 2

data.loc[data['Payload Orbit'] == 'Sun-Synchronous Orbit', 'Payload Orbit'] = 3

data.loc[data['Payload Orbit'] == 'Polar Orbit', 'Payload Orbit'] = 4

data.loc[data['Payload Orbit'] == 'High Earth Orbit', 'Payload Orbit'] = 5

data.loc[data['Payload Orbit'] == 'Sun/Earth Orbit', 'Payload Orbit'] = 6

data.loc[data['Payload Orbit'] == 'Heliocentric Orbit', 'Payload Orbit'] = 7

data.loc[data['Payload Orbit'] == 'Suborbital', 'Payload Orbit'] = 8

data.loc[data['Payload Orbit'] == 'Mars Orbit', 'Payload Orbit'] = 9

data.loc[data['Payload Orbit'] == 'Earth-Moon L2', 'Payload Orbit'] = 10
#Mission Status

data.loc[data['Mission Status'] == 'Failure', 'Mission Status'] = 0

data.loc[data['Mission Status'] == 'Success', 'Mission Status'] = 1
data.head()
# Launch Date and Launch Time

data['Launch Date'] = data['Launch Date'].astype(str)

data['Launch Time'] = data['Launch Time'].astype(str)

#merging the columns

data['Launch_Time'] = data['Launch Date'].str.cat(data['Launch Time'],sep=" ")

data = data.drop(['Launch Time', 'Launch Date'], axis = 1)

data.head()
#convert to datetime

from datetime import datetime

data['Launch_Time'] = data['Launch_Time'].map(lambda x: datetime.strptime(x, '%d %B %Y %H:%M'))
# Fairing Diameter (European Space Agency)

data['Fairing Diameter (m)'].unique()
data.loc[data['Fairing Diameter (m)'] == "European Space Agency", 'Fairing Diameter (m)'] = 5.2
#Fairing diameter

data['Fairing Diameter (m)'].unique()
for a in data['Fairing Diameter (m)']:

    if a == 'European Space Agency':

        a = 5.2

        



# data['Fairing Diameter (m)'].unique()
# Company

sns.lineplot(x='Company', y='Mission Status', data=data)

#vehicle type

sns.lineplot(x='vehicle_type', y='Mission Status', data=data)
#launch site

sns.barplot(x='Launch Site', y='Mission Status', data=data)
# Temperature

sns.lineplot(x='Temperature (° F)', y='Mission Status', data=data)
#wind speed

sns.lineplot(x='Wind speed (MPH)', y='Mission Status', data=data)
#humidity

sns.lineplot(x='Humidity (%)', y='Mission Status', data=data)
#liftoff thrust

sns.lineplot(x='Liftoff Thrust (kN)', y='Mission Status', data=data)
# Payload Orbit

sns.lineplot(x='Payload Orbit', y='Mission Status', data=data)



#launch time

sns.lineplot(x='Launch_Time', y='Mission Status', data=data)
#splitting dataset to train and test sets

x = data.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,12]]

y = data.iloc[:, -3].values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
# Fitting Random Forest Classification to Training set

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)



y_pred = random_forest.predict(x_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Fitting Naive Bayes Algorithm to Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)



# Predicting Test set results

y_pred = classifier.predict(x_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Fitting k-NN to Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(x_train, y_train)



# Predicting Test set results

y_pred = classifier.predict(x_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Fitting Logistic Regression to Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, solver='lbfgs')

classifier.fit(x_train, y_train)





# Predicting Test set results

Y_pred = classifier.predict(x_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm