#Importing all necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.svm import SVC

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.cluster import KMeans, DBSCAN

#from sklearn.utils import resample

#from imblearn.over_sampling import SMOTE

%matplotlib inline

sns.set_style("darkgrid")
#Reading the data from csv.file

crimes_data = pd.read_csv('../input/chicago-crimes-2018-2019/crime.csv')
#Checking the data contents

crimes_data.head()
#Handling any inconsistensis of column names

crimes_data.columns = crimes_data.columns.str.strip()

crimes_data.columns = crimes_data.columns.str.replace(',', '')

crimes_data.columns = crimes_data.columns.str.replace(' ', '_')

crimes_data.columns = crimes_data.columns.str.lower()
#Checking the data for any null values and its datatypes

crimes_data.info()
#Check the data forany duplicates

crimes_data[crimes_data.duplicated(keep=False)]
# Removing Primary key type attriburtes as they of no use for any type of analysis, Location columns is just a 

# combination of Latitude and Longitude

crimes_data.drop(['id','case_number','location'],axis=1,inplace=True)
msno.heatmap(crimes_data,figsize=(15, 5))
msno.dendrogram(crimes_data,figsize=(20,5))
crimes_data.isnull().sum()
#Dropping observations where latitude is null/Nan

crimes_data.dropna(subset=['latitude'],inplace=True)

crimes_data.reset_index(drop=True,inplace=True)
crimes_data.isnull().sum()
crimes_data.dropna(inplace=True)

crimes_data.reset_index(drop=True,inplace=True)
crimes_data.info()
#Converting the data column to datetime object so we can get better results of our analysis

#Get the day of the week,month and time of the crimes

crimes_data.date = pd.to_datetime(crimes_data.date)

crimes_data['day_of_week'] = crimes_data.date.dt.day_name()

crimes_data['month'] = crimes_data.date.dt.month_name()

crimes_data['time'] = crimes_data.date.dt.hour
#Mapping similar crimes under one group.

primary_type_map = {

    ('BURGLARY','MOTOR VEHICLE THEFT','THEFT','ROBBERY') : 'THEFT',

    ('BATTERY','ASSAULT','NON-CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)') : 'NON-CRIMINAL_ASSAULT',

    ('CRIM SEXUAL ASSAULT','SEX OFFENSE','STALKING','PROSTITUTION') : 'SEXUAL_OFFENSE',

    ('WEAPONS VIOLATION','CONCEALED CARRY LICENSE VIOLATION') :  'WEAPONS_OFFENSE',

    ('HOMICIDE','CRIMINAL DAMAGE','DECEPTIVE PRACTICE','CRIMINAL TRESPASS') : 'CRIMINAL_OFFENSE',

    ('KIDNAPPING','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN') : 'HUMAN_TRAFFICKING_OFFENSE',

    ('NARCOTICS','OTHER NARCOTIC VIOLATION') : 'NARCOTIC_OFFENSE',

    ('OTHER OFFENSE','ARSON','GAMBLING','PUBLIC PEACE VIOLATION','INTIMIDATION','INTERFERENCE WITH PUBLIC OFFICER','LIQUOR LAW VIOLATION','OBSCENITY','PUBLIC INDECENCY') : 'OTHER_OFFENSE'

}

primary_type_mapping = {}

for keys, values in primary_type_map.items():

    for key in keys:

        primary_type_mapping[key] = values

crimes_data['primary_type_grouped'] = crimes_data.primary_type.map(primary_type_mapping)
#Zone where the crime has occured

zone_mapping = {

    'N' : 'North',

    'S' : 'South',

    'E' : 'East',

    'W' : 'West'

}

crimes_data['zone'] = crimes_data.block.str.split(" ", n = 2, expand = True)[1].map(zone_mapping)
#Mapping seasons from month of crime

season_map = {

    ('March','April','May') : 'Spring',

    ('June','July','August') : 'Summer',

    ('September','October','November') : 'Fall',

    ('December','January','February') : 'Winter'

}

season_mapping = {}

for keys, values in season_map.items():

    for key in keys:

        season_mapping[key] = values

crimes_data['season'] = crimes_data.month.map(season_mapping)
#Mapping similar locations of crime under one group.

loc_map = {

    ('RESIDENCE', 'APARTMENT', 'CHA APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'RESIDENCE-GARAGE',

    'RESIDENTIAL YARD (FRONT/BACK)', 'DRIVEWAY - RESIDENTIAL', 'HOUSE') : 'RESIDENCE',

    

    ('BARBERSHOP', 'COMMERCIAL / BUSINESS OFFICE', 'CURRENCY EXCHANGE', 'DEPARTMENT STORE', 'RESTAURANT',

    'ATHLETIC CLUB', 'TAVERN/LIQUOR STORE', 'SMALL RETAIL STORE', 'HOTEL/MOTEL', 'GAS STATION',

    'AUTO / BOAT / RV DEALERSHIP', 'CONVENIENCE STORE', 'BANK', 'BAR OR TAVERN', 'DRUG STORE',

    'GROCERY FOOD STORE', 'CAR WASH', 'SPORTS ARENA/STADIUM', 'DAY CARE CENTER', 'MOVIE HOUSE/THEATER',

    'APPLIANCE STORE', 'CLEANING STORE', 'PAWN SHOP', 'FACTORY/MANUFACTURING BUILDING', 'ANIMAL HOSPITAL',

    'BOWLING ALLEY', 'SAVINGS AND LOAN', 'CREDIT UNION', 'KENNEL', 'GARAGE/AUTO REPAIR', 'LIQUOR STORE',

    'GAS STATION DRIVE/PROP.', 'OFFICE', 'BARBER SHOP/BEAUTY SALON') : 'BUSINESS',

    

    ('VEHICLE NON-COMMERCIAL', 'AUTO', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)', 'TAXICAB',

    'VEHICLE-COMMERCIAL', 'VEHICLE - DELIVERY TRUCK', 'VEHICLE-COMMERCIAL - TROLLEY BUS',

    'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS') : 'VEHICLE',

    

    ('AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'CTA PLATFORM', 'CTA STATION', 'CTA BUS STOP',

    'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA', 'CTA TRAIN', 'CTA BUS', 'CTA GARAGE / OTHER PROPERTY',

    'OTHER RAILROAD PROP / TRAIN DEPOT', 'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',

    'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRCRAFT',

    'AIRPORT PARKING LOT', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION',

    'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',

    'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',

    'CTA TRACKS - RIGHT OF WAY', 'AIRPORT/AIRCRAFT', 'BOAT/WATERCRAFT', 'CTA PROPERTY', 'CTA "L" PLATFORM',

    'RAILROAD PROPERTY') : 'PUBLIC_TRANSPORTATION',

    

    ('HOSPITAL BUILDING/GROUNDS', 'NURSING HOME/RETIREMENT HOME', 'SCHOOL, PUBLIC, BUILDING',

    'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PRIVATE, BUILDING',

    'MEDICAL/DENTAL OFFICE', 'LIBRARY', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'YMCA', 'HOSPITAL') : 'PUBLIC_BUILDING',

    

    ('STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'PARK PROPERTY', 'ALLEY', 'CEMETARY',

    'CHA HALLWAY/STAIRWELL/ELEVATOR', 'CHA PARKING LOT/GROUNDS', 'COLLEGE/UNIVERSITY GROUNDS', 'BRIDGE',

    'SCHOOL, PRIVATE, GROUNDS', 'FOREST PRESERVE', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'PARKING LOT', 'DRIVEWAY',

    'HALLWAY', 'YARD', 'CHA GROUNDS', 'RIVER BANK', 'STAIRWELL', 'CHA PARKING LOT') : 'PUBLIC_AREA',

    

    ('POLICE FACILITY/VEH PARKING LOT', 'GOVERNMENT BUILDING/PROPERTY', 'FEDERAL BUILDING', 'JAIL / LOCK-UP FACILITY',

    'FIRE STATION', 'GOVERNMENT BUILDING') : 'GOVERNMENT',

    

    ('OTHER', 'ABANDONED BUILDING', 'WAREHOUSE', 'ATM (AUTOMATIC TELLER MACHINE)', 'VACANT LOT/LAND',

    'CONSTRUCTION SITE', 'POOL ROOM', 'NEWSSTAND', 'HIGHWAY/EXPRESSWAY', 'COIN OPERATED MACHINE', 'HORSE STABLE',

    'FARM', 'GARAGE', 'WOODED AREA', 'GANGWAY', 'TRAILER', 'BASEMENT', 'CHA PLAY LOT') : 'OTHER'  

}



loc_mapping = {}

for keys, values in loc_map.items():

    for key in keys:

        loc_mapping[key] = values

crimes_data['loc_grouped'] = crimes_data.location_description.map(loc_mapping)
#Mapping crimes to ints to get better information from plots

crimes_data.arrest = crimes_data.arrest.astype(int)

crimes_data.domestic = crimes_data.domestic.astype(int)
#Grouping the data into years = (2018 and 2019) for analyzing

crimes_data_2018 = crimes_data[crimes_data.year == 2018]

crimes_data_2019 = crimes_data[crimes_data.year == 2019]
plt.figure(figsize=(15,5))

zone_plot = sns.countplot(data=crimes_data,x='day_of_week',hue='year',order=crimes_data.day_of_week.value_counts().index,palette='Set2')
plt.figure(figsize=(20,5))

zone_plot = sns.countplot(data=crimes_data,x='month',hue='year',order=crimes_data.month.value_counts().index,palette='Set2')
plt.figure(figsize=(20,5))

zone_plot = sns.pointplot(data=crimes_data_2018,x=crimes_data_2018.time.value_counts().index,y=crimes_data_2018.time.value_counts())
plt.figure(figsize=(20,5))

zone_plot = sns.pointplot(data=crimes_data_2019,x=crimes_data_2019.time.value_counts().index,y=crimes_data_2019.time.value_counts())
crimes_data_primary_type_pie = plt.pie(crimes_data_2018.primary_type_grouped.value_counts(),labels=crimes_data_2018.primary_type_grouped.value_counts().index,autopct='%1.1f%%',radius=2.5)

plt.legend(loc = 'best')
crimes_data_primary_type_pie = plt.pie(crimes_data_2019.primary_type_grouped.value_counts(),labels=crimes_data_2019.primary_type_grouped.value_counts().index,autopct='%1.1f%%',radius=2.5)

plt.legend(loc = 'best')
crimes_data_primary_type_pie = plt.pie(crimes_data_2018.loc_grouped.value_counts(),labels=crimes_data_2018.loc_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)

plt.legend(loc = 'best')
crimes_data_primary_type_pie = plt.pie(crimes_data_2019.loc_grouped.value_counts(),labels=crimes_data_2019

                                       .loc_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)

plt.legend(loc = 'best')
plt.figure(figsize=(20,3))

primary_type_plot_2018 = sns.barplot(data=crimes_data_2018,x=crimes_data_2018.primary_type.value_counts()[0:20].index,y=crimes_data_2018.primary_type.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)

plt.figure(figsize=(20,3))

primary_type_plot_2019 = sns.barplot(data=crimes_data_2019,x=crimes_data_2019.primary_type.value_counts()[0:20].index,y=crimes_data_2019.primary_type.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)
zone_plot = sns.countplot(data=crimes_data,x='zone',hue='year',order=crimes_data.zone.value_counts().index,palette='Set2')
zone_plot = sns.countplot(data=crimes_data,x='season',hue='year',palette='Set2')
arrest_plot = sns.countplot(data=crimes_data,x='year',hue='arrest',palette='Set2')
plt.figure(figsize=(20,3))

location_description_plot_2018 = sns.barplot(data=crimes_data_2018,x=crimes_data_2018.location_description.value_counts()[0:20].index,y=crimes_data_2018.location_description.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)

plt.figure(figsize=(20,3))

location_description_plot_2019 = sns.barplot(data=crimes_data_2019,x=crimes_data_2019.location_description.value_counts()[0:20].index,y=crimes_data_2019.location_description.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)
crimes_data_primary_type_pie = plt.pie(crimes_data.primary_type_grouped.value_counts(),labels=crimes_data.primary_type_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)

plt.legend(loc = 'best')
crimes_data_primary_type_pie = plt.pie(crimes_data.loc_grouped.value_counts(),labels=crimes_data.loc_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)

plt.legend(loc = 'best')
plt.figure(figsize=(15,5))

zone_plot = sns.countplot(data=crimes_data,x='day_of_week',order=crimes_data.day_of_week.value_counts().index,palette='Set2')
plt.figure(figsize=(20,5))

zone_plot = sns.countplot(data=crimes_data,x='month',order=crimes_data.month.value_counts().index,palette='Set2')
plt.figure(figsize=(20,5))

zone_plot = sns.pointplot(data=crimes_data,x=crimes_data.time.value_counts().index,y=crimes_data.time.value_counts())
zone_plot = sns.countplot(data=crimes_data,x='zone',order=crimes_data.zone.value_counts().index,palette='Set2')
zone_plot = sns.countplot(data=crimes_data,x='season',order=crimes_data.season.value_counts().index,palette='Set2')
arrest_plot = sns.countplot(data=crimes_data,x='arrest',palette='Set2')
plt.figure(figsize=(15,3))

arrest_plot = sns.countplot(data=crimes_data,x='arrest',hue='primary_type_grouped',palette='Set2')

plt.legend(loc = 'best')
plt.figure(figsize=(20,3))

location_description_plot = sns.barplot(data=crimes_data,x=crimes_data.primary_type.value_counts()[0:20].index,y=crimes_data_2019.primary_type.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)
plt.figure(figsize=(20,3))

location_description_plot = sns.barplot(data=crimes_data,x=crimes_data.location_description.value_counts()[0:20].index,y=crimes_data.location_description.value_counts()[0:20].values,palette='Set2')

plt.xticks(rotation=45)
# A full Chicago crime by district. Maybe helpful for later when comparing our clusters



new_crimes_data = crimes_data.loc[(crimes_data['x_coordinate']!=0)]

sns.lmplot('x_coordinate', 

           'y_coordinate',

           data=new_crimes_data[:],

           fit_reg=False, 

           hue="district",

           palette='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("Crimes by District")
new_crimes_data = crimes_data.loc[(crimes_data['x_coordinate']!=0)]

sns.lmplot('x_coordinate', 

           'y_coordinate',

           data=new_crimes_data[:],

           fit_reg=False, 

           hue="primary_type_grouped",

           palette='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "o", 

                        "s": 10})

ax = plt.gca()

ax.set_title("Crimes by Type of crime")
#Converting the numercial attributes to categorical attributes

crimes_data.year = pd.Categorical(crimes_data.year)

crimes_data.time = pd.Categorical(crimes_data.time)

crimes_data.domestic = pd.Categorical(crimes_data.domestic)

crimes_data.arrest = pd.Categorical(crimes_data.arrest)

crimes_data.beat = pd.Categorical(crimes_data.beat)

crimes_data.district = pd.Categorical(crimes_data.district)

crimes_data.ward = pd.Categorical(crimes_data.ward)

crimes_data.community_area = pd.Categorical(crimes_data.community_area)
crimes_data_prediction = crimes_data.drop(['date','block','iucr','primary_type','description','location_description','fbi_code','updated_on','x_coordinate','y_coordinate'],axis=1)
crimes_data_prediction.head()
crimes_data_prediction.info()
crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)
crimes_data_prediction.head()
#Train test split with a test set size of 30% of entire data

X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction.drop(['arrest_1'],axis=1),crimes_data_prediction['arrest_1'], test_size=0.3, random_state=42)
# X = pd.concat([X_train, y_train], axis=1)

# not_arrest = X[X.arrest_1==0]

# arrest = X[X.arrest_1==1]

# arrest_oversample = resample(arrest,replace=True,n_samples=len(not_arrest),random_state=42)

# oversampled = pd.concat([not_arrest, arrest_oversample])

# print(oversampled.arrest_1.value_counts())

# y_train = oversampled.arrest_1

# X_train = oversampled.drop('arrest_1', axis=1)
# sm = SMOTE(random_state=42, ratio=1.0)

# X_train, y_train = sm.fit_sample(X_train, y_train)
#Standardizing the data

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test)
#Gaussain Naive Bayes

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Confusion matrix')

plt.tight_layout()
print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Precision = ',metrics.precision_score(y_test, y_pred,))

print('Recall = ',metrics.recall_score(y_test, y_pred))

print('F-1 Score = ',metrics.f1_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
# #3 Nearest Neighbour classification

# classifier = KNeighborsClassifier(n_neighbors = 3)  

# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# print(conf_matrix)
# sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)

# plt.ylabel('Actual')

# plt.xlabel('Predicted')

# plt.title('Confusion matrix')

# plt.tight_layout()
# print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

# print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

# print('Precision = ',metrics.precision_score(y_test, y_pred,))

# print('Recall = ',metrics.recall_score(y_test, y_pred))

# print('F-1 Score = ',metrics.f1_score(y_test, y_pred))

# print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Decision tree with Entropy as attribute measure

model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
# Plot confusion matrix

sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Confusion matrix')

plt.tight_layout()
#Classification Metrics

print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Precision = ',metrics.precision_score(y_test, y_pred,))

print('Recall = ',metrics.recall_score(y_test, y_pred))

print('F-1 Score = ',metrics.f1_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Random Forest classifier  - Best one

model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
# Plot confusion matrix

sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Confusion matrix')

plt.tight_layout()
#Classification Metrics

print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Precision = ',metrics.precision_score(y_test, y_pred,))

print('Recall = ',metrics.recall_score(y_test, y_pred))

print('F-1 Score = ',metrics.f1_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Logistic Regression

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
# Plot confusion matrix

sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Confusion matrix')

plt.tight_layout()
#Classification Metrics

print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Precision = ',metrics.precision_score(y_test, y_pred,))

print('Recall = ',metrics.recall_score(y_test, y_pred))

print('F-1 Score = ',metrics.f1_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
crimes_data_type = crimes_data.loc[crimes_data.primary_type_grouped.isin(['THEFT','NON-CRIMINAL_ASSAULT','CRIMINAL_OFFENSE'])]

crimes_data_prediction = crimes_data_type.drop(['date','block','iucr','primary_type','description','location_description','fbi_code','updated_on','x_coordinate','y_coordinate','primary_type_grouped'],axis=1)

crimes_data_prediction_type = crimes_data_type.primary_type_grouped

crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)
crimes_data_prediction.head()
X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction,crimes_data_prediction_type, test_size=0.3, random_state=42)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test)
# #3 Nearest Neighbour Classification for Type of crime

# classifier = KNeighborsClassifier(n_neighbors = 3)  

# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# print(conf_matrix)
# #Classification Metrics

# print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

# print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

# print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Decision tree classifier for type of crime

model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
#Classification Metrics

print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Random Forest classifier for type of crime

model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
#Classification Metrics

print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
#Logistic Regression for predicting the type of crime -Best

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# Compute confusion matrix

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(conf_matrix)
print('Accuracy = ',metrics.accuracy_score(y_test, y_pred))

print('Error = ',1 - metrics.accuracy_score(y_test, y_pred))

print('Classification Report\n',metrics.classification_report(y_test, y_pred))
# Calculated the number of occrurances for each type of crime category in each district

district_crime_rates = pd.DataFrame(columns=['theft_count', 'assault_count', 'sexual_offense_count', 

                                             'weapons_offense_count', 'criminal_offense_count', 

                                             'human_trafficking_count', 'narcotic_offense_count', 

                                             'other_offense_count'])

district_crime_rates = district_crime_rates.astype(int) 



for i in range(1, 32):   

    temp_district_df = crimes_data[crimes_data['district'] == i] 



    temp_district_theft = temp_district_df[temp_district_df['primary_type_grouped'] == 'THEFT'] 

    num_theft = temp_district_theft.primary_type_grouped.count() 

    

    temp_district_assault = temp_district_df[temp_district_df['primary_type_grouped'] == 'NON-CRIMINAL_ASSAULT'] 

    num_assault = temp_district_assault.primary_type_grouped.count()    

    

    temp_district_sexual_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'SEXUAL_OFFENSE'] 

    num_sexual_offense = temp_district_sexual_offense.primary_type_grouped.count()

    

    temp_district_weapons_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'WEAPONS_OFFENSE'] 

    num_weapons_offense = temp_district_weapons_offense.primary_type_grouped.count()

    

    temp_district_criminal_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'CRIMINAL_OFFENSE'] 

    num_criminal_offense = temp_district_criminal_offense.primary_type_grouped.count()

    

    temp_district_human_trafficking = temp_district_df[temp_district_df['primary_type_grouped'] == 'HUMAN_TRAFFICKING_OFFENSE'] 

    num_human_trafficking = temp_district_human_trafficking.primary_type_grouped.count()

    

    temp_district_narcotic_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'NARCOTIC_OFFENSE'] 

    num_narcotic_offense = temp_district_narcotic_offense.primary_type_grouped.count()

    

    temp_district_other_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'OTHER_OFFENSE'] 

    num_other_offense = temp_district_other_offense.primary_type_grouped.count()



    district_crime_rates.loc[i] = [num_theft, num_assault, num_sexual_offense, num_weapons_offense, num_criminal_offense, num_human_trafficking, num_narcotic_offense, num_other_offense]    

    

#district_crime_rates.head()

    
# Standardize the data

district_crime_rates_standardized = preprocessing.scale(district_crime_rates)

district_crime_rates_standardized = pd.DataFrame(district_crime_rates_standardized)

#district_crime_rates_standardized.head()
# Clustering with K-Means 

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(district_crime_rates_standardized)

#y_kmeans



#beginning of  the cluster numbering with 1 instead of 0

y_kmeans1=y_kmeans+1



# New list called cluster

kmeans_clusters = list(y_kmeans1)

# Adding cluster to our data set

district_crime_rates['kmeans_cluster'] = kmeans_clusters



#Mean of clusters 1 to 4

kmeans_mean_cluster = pd.DataFrame(round(district_crime_rates.groupby('kmeans_cluster').mean(),1))

#kmeans_mean_cluster



#district_crime_rates.head()
# Clustering with DBSCAN

clustering = DBSCAN(eps = 1, min_samples = 3, metric = "euclidean").fit(district_crime_rates_standardized)



# Show clusters

dbscan_clusters = clustering.labels_

# print(clusters)



district_crime_rates['dbscan_clusters'] = dbscan_clusters + 2

#district_crime_rates.head()
# Clustering with Hierarchical Clustering with average linkage

clustering = linkage(district_crime_rates_standardized, method = "average", metric = "euclidean")



# Plot dendrogram

plt.figure()

dendrogram(clustering)

plt.show()



# Form clusters

hierarchical_clusters = fcluster(clustering, 4, criterion = 'maxclust')

# print(clusters)



district_crime_rates['hierarchical_clusters'] = hierarchical_clusters 

#district_crime_rates.head()
# Add 'district' column

district_crime_rates['district'] = district_crime_rates.index

district_crime_rates = district_crime_rates[['district', 'kmeans_cluster', 'dbscan_clusters', 'hierarchical_clusters', 'theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count']]
# Remove all columns but 'district' & each method's cluster

district_crime_rates = district_crime_rates.drop(['theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count'], axis=1)

district_crime_rates.head(31)
# Merge each district's clusters for each method into a single dataframe 

crimes_data_clustered = pd.merge(crimes_data, district_crime_rates, on='district', how='inner')

#crimes_data.head()
# Crime level clusters by district (KMeans Clustering)

new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]

sns.lmplot('x_coordinate', 

           'y_coordinate',

           data=new_crimes_data[:],

           fit_reg=False, 

           hue="kmeans_cluster",

           palette='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("KMeans Clustering of Crimes by District")
# Crime level clusters by district (DBScan Clustering)

new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]

sns.lmplot('x_coordinate', 

           'y_coordinate',

           data=new_crimes_data[:],

           fit_reg=False, 

           hue="dbscan_clusters",

           palette='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("DBScan Clustering of Crimes by District")
# Crime level clusters by district (Hierarchical Clustering)

new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]

sns.lmplot('x_coordinate', 

           'y_coordinate',

           data=new_crimes_data[:],

           fit_reg=False, 

           hue="hierarchical_clusters",

           palette='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("Hierarchical Clustering of Crimes by District")