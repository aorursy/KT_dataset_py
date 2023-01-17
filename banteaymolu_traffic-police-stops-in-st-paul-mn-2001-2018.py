# Import libraries

import pandas as pd

import numpy as np

import seaborn as sns



# Import Bokeh 

from bokeh.plotting import figure, show, output_notebook , ColumnDataSource

from bokeh.tile_providers import get_provider

from bokeh.models import CategoricalColorMapper

from ast import literal_eval



# Matplot

import matplotlib.pyplot as plt



# Set seaborn

sns.set()
# Read the CSV file

# Download the CVS data from: https://information.stpaul.gov/Public-Safety/Traffic-Stop-Dataset/kkd6-vvns

# Read data and set the date-time index using 'DATE OF STOP' of the CSV column.





filename = '../input/Traffic_Stop_Dataset.csv'



df = pd.read_csv(filename, index_col='DATE OF STOP', parse_dates=True)





# Data quick summary

print(df.info())

print("\n")
df.head()
# Group each column and learn about the different labels

print(df.groupby('RACE OF DRIVER')['RACE OF DRIVER'].count())

print("\n")

print(df.groupby('GENDER OF DRIVER')['GENDER OF DRIVER'].count())

print("\n")

print(df.groupby('DRIVER SEARCHED?')['DRIVER SEARCHED?'].count())

print("\n")

print(df.groupby('VEHICLE SEARCHED?')['VEHICLE SEARCHED?'].count())

print("\n")

print(df.groupby('CITATION ISSUED?')['CITATION ISSUED?'].count())
# Lets rename the index name.

df = df.rename_axis('stop_datetime')



# Rename the name of the features so that it is easier to call later on

# Note: driver search is sometimes know as 'stop and frisk' and we will use this name 

# to mean that frisk was performed on the dirver

df = df.rename(columns={'YEAR OF STOP': 'year', 'RACE OF DRIVER':'driver_race', 'GENDER OF DRIVER': 'driver_gender',

                   'DRIVER SEARCHED?': 'frisk_performed','VEHICLE SEARCHED?': 'vehicle_searched',

                   'CITATION ISSUED?': 'citation_issued','AGE OF DRIVER': 'driver_age', 

                   'REASON FOR STOP': 'reason_for_stop', 'POLICE GRID NUMBER': 'police_grid_number',

                  'LOCATION OF STOP BY POLICE GRID': 'location', 'COUNT': 'count','DATE OF STOP': 'stop_datetime'})







print(df.info())
# First lets count the number of NaN values

print(df.isnull().values.sum())

# This will print 629438 null values properly labeled as 'NaN'



# Replace the 'No Data' label with the proper 'NaN' label

df = df.replace(to_replace='No Data', value=np.nan)



print(df.isnull().values.sum())

# Now we have 1748254



# print summary again

print(df.info())
# change object data types to category

for col in df.columns.values:

    if df[col].dtypes == 'object':

        df[col] = df[col].astype('category')



print(df.info())
# The location column has '(lat, long)' values and we will need to extract these values

# Instead of the whole dataset, lets visualize some sample. 

# please see the fulling coloring or the map: 

#       https://information.stpaul.gov/Public-Safety/Traffic-Stop-Dataset/kkd6-vvns

# For district numbers please see: 

#       https://information.stpaul.gov/Public-Safety/Saint-Paul-Police-Grid-Shapefile/ykwt-ie3e

new_data = df.sample(n=600000, random_state=1).dropna()





# Note: we need to extract the latitude and longitude coordinates to X and Y and convert them

# so that bokeh undertands where to place them.

def coord_extract(coord):

    coordinates = literal_eval(coord)

    

    lat = coordinates[0]

    long = coordinates[1]

    

    r_major = 6378137.000

    

    x_cord = r_major * np.radians(long)

    scale = x_cord/ long

    y_cord =  180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale

    

    return (x_cord, y_cord)







# Get the coordinates 

new_data['x_cord'] = new_data['location'].apply(lambda x: coord_extract(x)[0])

new_data['y_cord'] = new_data['location'].apply(lambda x: coord_extract(x)[1])
# Conver dataframe to a ColumnDataSource

source = ColumnDataSource(new_data)



# Make a CategoricalColorMapper object: color_mapper

color_mapper = CategoricalColorMapper(factors=['Asian', 'Black', 'Latino','Native American', 'Other', 'White'],

                                      palette=['purple', 'red', 'green', 'orange', 'black', 'blue'])



p = figure(width=600, height=700, x_range=(new_data['x_cord'].min(), new_data['x_cord'].max()), 

           y_range=(new_data['y_cord'].min(), new_data['y_cord'].max()))



p.add_tile(get_provider('CARTODBPOSITRON'))



p.circle(x = 'x_cord', y = 'y_cord', source=source, color=dict(field='driver_race', transform=color_mapper),

            legend='driver_race', size=15.0, fill_alpha=0.01)



p.legend.location = "bottom_center"



output_notebook()

show(p)
# Lets now visualize annual 'citation', 'frisk' and 'search conducted' rates from 2001 to 2018

# copy the original data

police_stops = df.copy()



# drop raws with null values of the three features

police_stops = police_stops.dropna(subset=['frisk_performed','vehicle_searched','citation_issued'])



# Create boolean column for each (Note: we can do maping as well:  {'Yes': True, 'No': False})

police_stops['bool_frisk_performed'] = police_stops['frisk_performed'] == 'Yes'

police_stops['bool_vehicle_searched'] = police_stops['vehicle_searched'] == 'Yes'

police_stops['bool_citation_issued'] = police_stops['citation_issued'] == 'Yes'







annual_rates_new = police_stops.groupby(police_stops.index.year)['bool_frisk_performed','bool_vehicle_searched', 

                                                           'bool_citation_issued'].mean()





annual_rates_new.plot(figsize=(20, 10), fontsize=16)



plt.xlabel('Year', fontsize=18)

plt.ylabel('Count', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=18)



plt.show()
# hourly rates: How does frisk, search and citation rates look like throughout a day?

hourly_rates = police_stops.groupby(police_stops.index.hour)['bool_frisk_performed','bool_vehicle_searched', 

                                                           'bool_citation_issued'].mean()

sns.set()

hourly_rates.plot(figsize=(20, 10), fontsize=16)



plt.xlabel('Time (hour)', fontsize=18)

plt.ylabel('Rate', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=18)



plt.show()
# Daily rate:

daily_rates = police_stops.groupby(police_stops.index.weekday_name)['bool_frisk_performed','bool_vehicle_searched', 

                                                           'bool_citation_issued'].mean()

sns.set()

daily_rates.plot(kind='bar',  figsize=(20, 10), fontsize=16, rot=0)

plt.xlabel('Day of the week', fontsize=18)

plt.ylabel('Rate', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=14)

plt.show()
rates_by_gender = police_stops.groupby('driver_gender')['bool_frisk_performed','bool_vehicle_searched', 

                                                           'bool_citation_issued'].mean()

rates_by_gender.plot(kind='bar', figsize=(20, 10), fontsize=18, rot=0)



plt.xlabel('Driver\'s gender', fontsize=18)

plt.ylabel('Rate', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=18)



plt.show()
# Frisk, search and citation rates by race

rates_by_race = police_stops.groupby('driver_race')['bool_frisk_performed',

                                                    'bool_vehicle_searched', 

                                                    'bool_citation_issued'].mean().sort_values('bool_citation_issued', 

                                                                                                      ascending=False)

rates_by_race.plot(kind='bar', figsize=(20, 10), fontsize=16, rot=0)



plt.xlabel('Subject Race', fontsize=18)

plt.ylabel('Rate', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=18)



plt.show()


rates_by_reason = police_stops.groupby('reason_for_stop')['bool_frisk_performed',

                                                    'bool_vehicle_searched', 

                                                    'bool_citation_issued'].mean().sort_values('bool_citation_issued', 

                                                                                                      ascending=False)



rates_by_reason.plot(kind='bar', figsize=(20, 10), fontsize=16, rot=0)



plt.xlabel('Subject Race', fontsize=18)

plt.ylabel('Rate', fontsize=18)

plt.legend(['Frisk Rate', 'Vehicle Search Rate', 'Citation Rate'], fontsize=18)



plt.show()
# lets get all the missing values for each feature



# lets first calculate the % of missing labels for each feature

labels = []

percent = []



for col in df.columns.values:

    labels.append(col)

    non_missing = df.isnull().sum()[col] * 100

    total = df.isnull().sum()[col] + df[col].count()

    percent.append(non_missing / total)





labels = pd.DataFrame(labels, columns=['Features'])

percent = pd.DataFrame(percent, columns=['Missing'])





missing_data = pd.concat([labels, percent], axis=1)

missing_data = missing_data[missing_data.Missing > 0].sort_values(by='Missing', ascending=False)

missing_data = missing_data.set_index(missing_data.Features)

missing_data.plot(kind='bar', figsize=(20, 10), fontsize=16, rot=0)



plt.xlabel('Featurs', fontsize=18)

plt.ylabel('Missing labels (%)', fontsize=18)

plt.title('% missing labels for each feature', fontsize=18)

plt.legend(['Missing (%)'], fontsize=18)

print(missing_data)
print(df.groupby(['year','reason_for_stop'])['reason_for_stop'].count())
# Final Dataset using the 2017 and 2018 traffic stops

temp = police_stops[['year','driver_race','driver_gender','driver_age','reason_for_stop','police_grid_number',

                    'bool_frisk_performed', 'bool_vehicle_searched','bool_citation_issued']]



final_dataset = temp[(temp.year == 2017) | (temp.year == 2018)].copy()



# For categorical labels we can use the median to replace NaN values 

final_dataset.police_grid_number.fillna(final_dataset['police_grid_number'].value_counts().index[0], inplace=True)



# For driver's age, replace with the mean age

final_dataset.driver_age.fillna(final_dataset.driver_age.mean(), inplace=True)



print(final_dataset.isnull().values.sum())
final_dataset.info()
# Lets use a label Encoder to convert categorical variables to numeric.

# The numerical labels are always between 0 and n_categories-1.

# Import LabelEncoder



from sklearn.preprocessing import LabelEncoder



# Instantiate LabelEncoder

le = LabelEncoder()



x_values = final_dataset[['year','driver_race','driver_gender','driver_age','reason_for_stop','police_grid_number',

                    'bool_frisk_performed', 'bool_vehicle_searched']].copy()

y_values = final_dataset['bool_citation_issued'].astype('int').copy()



for col in x_values.columns.values:

    if x_values[col].dtype.name == 'category':

        x_values[col] = le.fit_transform(x_values[col])



x_values.head()
# Logistic regression

from sklearn.linear_model import LogisticRegression

# Import train_test_split

from sklearn.model_selection import train_test_split

# Import confusion_matrix

from sklearn.metrics import confusion_matrix



# Create a report for each model

from sklearn.metrics import classification_report



# Instantiate a LogisticRegression classifier with default parameter values

logreg = LogisticRegression()





X = x_values.values

y = y_values.values



# Split data into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X,

                                y,

                                test_size=0.3,

                                random_state=42)

# Fit logreg to the train set

logreg.fit(X_train, y_train)



# Use logreg to predict instances from the test set and store it

y_pred = logreg.predict(X_test)



# Calculate and print classification results

cfm = confusion_matrix(y_test, y_pred)

reports = classification_report(y_test, y_pred)

print("Confusion Matrix: ")

print(cfm)

print("\n")

print(reports)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

plt.show()
x_values = final_dataset[['year','driver_race','driver_gender','driver_age','reason_for_stop','police_grid_number',

                    'bool_frisk_performed', 'bool_vehicle_searched']].copy()

y_values = final_dataset['bool_citation_issued'].astype('int').copy()



x_values = pd.get_dummies(x_values, columns=['police_grid_number'], prefix = ['police_grid_number'], drop_first=True)

for col in x_values.columns.values:

    if x_values[col].dtype.name == 'category':

        x_values = pd.get_dummies(x_values, columns=[col], prefix = [col], drop_first=True)
x_values.head()


# Re-apply the logistic model

# Instantiate a LogisticRegression classifier with default parameter values

logreg = LogisticRegression()





X = x_values.values

y = y_values.values



# Split data into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X,

                                y,

                                test_size=0.3,

                                random_state=42)

# Fit logreg to the train set

logreg.fit(X_train, y_train)



# Use logreg to predict instances from the test set and store it

y_pred = logreg.predict(X_test)



# Calculate and print classification results

cfm = confusion_matrix(y_test, y_pred)

reports = classification_report(y_test, y_pred)

print("Confusion Matrix: ")

print(cfm)

print("\n")

print(reports)





y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

plt.show()
# Import models, including VotingClassifier meta-model

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN 

from sklearn.ensemble import VotingClassifier





# Split data into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X,

                                y,

                                test_size=0.3,

                                random_state=42)
# Set seed for reproducibility

SEED=1



# Instantiate lr

lr = LogisticRegression(random_state=SEED)



# Instantiate knn

knn = KNN(n_neighbors=2)



# Instantiate dt

dt = DecisionTreeClassifier(criterion='gini', random_state=SEED)



# Define the list classifiers

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
from sklearn.metrics import accuracy_score



# Before applying the VotingClassifier, let's first run each model and compare.

for clf_name, clf in classifiers:    

 

    # Fit clf to the training set

    clf.fit(X_train, y_train)    

   

    # Predict y_pred

    y_pred = clf.predict(X_test)

    

    # Calculate accuracy

    accuracy = accuracy_score(y_test, y_pred) 

    cfm = confusion_matrix(y_test, y_pred)

    reports = classification_report(y_test, y_pred)

    # Evaluate clf's accuracy on the test set

    print('{:s} reports:'.format(clf_name))

    # Print the confusion matrix of the logreg model

    print("Confusion Matrix: ")

    print(cfm)

    print("\n")

    print(reports)

    print("\n")

    

    

    y_pred_prob = clf.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr, label='Logistic Regression')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title("{:s} AUC: {}".format(clf_name, roc_auc_score(y_test, y_pred_prob)))

    plt.show()
# Instantiate a VotingClassifier vc

vc = VotingClassifier (estimators=classifiers)     



# Fit vc to the training set

vc.fit(X_train, y_train)   



# Evaluate the test set predictions

y_pred = vc.predict(X_test)



# Calculate accuracy score

accuracy = accuracy_score(y_test, y_pred)

cfm = confusion_matrix(y_test, y_pred)

reports = classification_report(y_test, y_pred)

print('Voting Classifier: {:.3f}'.format(accuracy))

print(cfm)

print("\n")

print(reports)