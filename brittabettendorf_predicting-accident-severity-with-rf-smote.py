# import the usual suspects ...

import pandas as pd

import numpy as np

import glob



import matplotlib.pyplot as plt

import seaborn as sns



# suppress all warnings

import warnings

warnings.filterwarnings("ignore")
accidents = pd.read_csv('../input/Accident_Information.csv')

print('Records:', accidents.shape[0], '\nColumns:', accidents.shape[1])

accidents.head()
#accidents.info()
#accidents.describe().T
#accidents.isna().sum()
vehicles = pd.read_csv('../input/Vehicle_Information.csv', encoding='ISO-8859-1')

print('Records:', vehicles.shape[0], '\nColumns:', vehicles.shape[1])

vehicles.head()
#vehicles.info()
#vehicles.describe().T
#vehicles.isna().sum()
accidents['Date']= pd.to_datetime(accidents['Date'], format="%Y-%m-%d")



# check

accidents.iloc[:, 5:13].info()
# slice first and second string from time column

accidents['Hour'] = accidents['Time'].str[0:2]



# convert new column to numeric datetype

accidents['Hour'] = pd.to_numeric(accidents['Hour'])



# drop null values in our new column

accidents = accidents.dropna(subset=['Hour'])



# cast to integer values

accidents['Hour'] = accidents['Hour'].astype('int')
# define a function that turns the hours into daytime groups

def when_was_it(hour):

    if hour >= 5 and hour < 10:

        return "1"

    elif hour >= 10 and hour < 15:

        return "2"

    elif hour >= 15 and hour < 19:

        return "3"

    elif hour >= 19 and hour < 23:

        return "4"

    else:

        return "5"
# create a little dictionary to later look up the groups I created

daytime_groups = {1: 'Morning: Between 5 and 10', 

                  2: 'Office Hours: Between 10 and 15', 

                  3: 'Afternoon Rush: Between 15 and 19', 

                  4: 'Evening: Between 19 and 23', 

                  5: 'Night: Between 23 and 5'}
# apply this function to our temporary hour column

accidents['Daytime'] = accidents['Hour'].apply(when_was_it)

accidents[['Time', 'Hour', 'Daytime']].head()
# drop old time column and temporary hour column

accidents = accidents.drop(columns=['Time', 'Hour'])
print('Proportion of Missing Values in Accidents Table:', 

      round(accidents.isna().sum().sum()/len(accidents), 3), '%')
#accidents.isna().sum()
# drop columns we don't need

accidents = accidents.drop(columns=['2nd_Road_Class', '2nd_Road_Number', 'Did_Police_Officer_Attend_Scene_of_Accident',

                                    'Location_Easting_OSGR', 'Location_Northing_OSGR', 

                                    'Longitude', 'Latitude', 'LSOA_of_Accident_Location',

                                    'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities',

                                    'InScotland'])



# drop remaining records with NaN's

accidents = accidents.dropna()



# check if we have no NaN's anymore

accidents.isna().sum().sum()
print('Proportion of Missing Values in Vehicles Table:', 

      round(vehicles.isna().sum().sum()/len(vehicles),3), '%')
#vehicles.isna().sum()
# combine the accidents with the vehicles table

df = pd.merge(accidents[['Accident_Index', 'Accident_Severity', 'Daytime', 'Speed_limit', 'Urban_or_Rural_Area']], 

              vehicles[['Accident_Index', 'Age_Band_of_Driver', 'Age_of_Vehicle', 'Sex_of_Driver', 

                        'Engine_Capacity_.CC.', 'Vehicle_Manoeuvre']], 

              on='Accident_Index')



df.isna().sum()
df = df.dropna()

df.isna().sum().sum()
df.info()    
# cast categorical features - currently stored as string data - to their proper data format

for col in ['Accident_Severity', 'Daytime', 'Speed_limit', 'Urban_or_Rural_Area',

            'Age_Band_of_Driver', 'Sex_of_Driver', 'Vehicle_Manoeuvre']:

    df[col] = df[col].astype('category')

    

df.info()
# define numerical columns

num_cols = ['Age_of_Vehicle', 'Engine_Capacity_.CC.']
# plotting boxplots

sns.set(style='darkgrid')

fig, axes = plt.subplots(2,1, figsize=(10,4))



for ax, col in zip(axes, num_cols):

    df.boxplot(column=col, grid=False, vert=False, ax=ax)

    plt.tight_layout();
df['Engine_Capacity_.CC.'].describe()
# phrasing condition

condition = (df['Engine_Capacity_.CC.'] < 20000)



# keep only records that meet the condition and don't fall within extreme outliers

df = df[condition]
df['Age_of_Vehicle'].describe()
age_of_vehicle_bins = {1: '1 to <2 years', 

                       2: '2 to <3 years', 

                       3: '3 to <7 years', 

                       4: '7 to <10 years', 

                       5: '>=10 years'}
# arguments in bins parameter denote left edge of each bin

df['Age_of_Vehicle'] = np.digitize(df['Age_of_Vehicle'], bins=[1,2,3,7,10])



# convert into categorical column

df['Age_of_Vehicle'] = df['Age_of_Vehicle'].astype('category')



# check the count within each bucket

df['Age_of_Vehicle'].value_counts().sort_index()
# re-define numerical feature columns - only one left

num_cols = ['Engine_Capacity_.CC.']
# define categorical feature columns

cat_cols = ['Daytime', 'Speed_limit', 'Urban_or_Rural_Area',

            'Age_Band_of_Driver', 'Age_of_Vehicle', 'Sex_of_Driver', 'Vehicle_Manoeuvre']



# define target col

target_col = ['Accident_Severity']



cols = cat_cols + num_cols + target_col



# copy dataframe - just to be safe

df_model = df[cols].copy()

df_model.shape
# create dummy variables from the categorical features

dummies = pd.get_dummies(df_model[cat_cols], drop_first=True)

df_model = pd.concat([df_model[num_cols], df_model[target_col], dummies], axis=1)

df_model.shape
df_model.isna().sum().sum()
# define our features 

features = df_model.drop(['Accident_Severity'], axis=1)



# define our target

target = df_model[['Accident_Severity']]
from sklearn.model_selection import train_test_split



# split our data

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
df_model['Accident_Severity'].value_counts(normalize=True)
# import classifier

from sklearn.ensemble import RandomForestClassifier



# import metrics

from sklearn.metrics import classification_report, confusion_matrix



# import evaluation tools

from sklearn.model_selection import KFold, cross_val_score
# instantiate RandomForestClassifier with entropy and class_weight

forest_1 = RandomForestClassifier(random_state=4, criterion='entropy', n_jobs=-1, class_weight='balanced')



# train

forest_1.fit(X_train, y_train)



# predict

y_test_preds  = forest_1.predict(X_test)



# evaluate

report = classification_report(y_test, y_test_preds)

print('Classification Report Random Forest - with Entropy and class_weight Parameter: \n', report)
# cross-validation with F1 score (more appropriate to imbalanced classes)

cross_val_score(forest_1, X_train, y_train, scoring='f1_macro', n_jobs=-1)
# create confusion matrix# create confusion matrix

matrix = confusion_matrix(y_test, y_test_preds)



# create dataframe

class_names = df_model.Accident_Severity.values

dataframe = pd.DataFrame(matrix, index=['Fatal', 'Serious', 'Slight'], 

                         columns=['Fatal', 'Serious', 'Slight'])



# create heatmap

sns.heatmap(dataframe, annot=True, cbar=None, cmap='Blues')

plt.title('Confusion Matrix')

plt.tight_layout(), plt.xlabel('True Values'), plt.ylabel('Predicted Values')

plt.show()
from imblearn.over_sampling import SMOTE
# view previous class distribution

print('Before Upsampling with SMOTE:'), print(target['Accident_Severity'].value_counts())



# resample data ONLY using training data

X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train) 



# view synthetic sample class distribution

print('\nAfter Upsampling with SMOTE:'), print(pd.Series(y_resampled).value_counts())
# then perform ususal train-test-split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)
# instantiate second RandomForestClassifier with entropy and SMOTE

forest_2 = RandomForestClassifier(random_state=4, criterion='entropy', n_jobs=-1)



# train

forest_2.fit(X_train, y_train)



# predict

y_test_preds = forest_2.predict(X_test)



# evaluate

report = classification_report(y_test, y_test_preds)

print('Classification Report Random Forest - with Entropy and SMOTE Upsampling: \n', report)
# cross-validation with F1 score (more appropriate to imbalanced classes)

cross_val_score(forest_2, X_train, y_train, scoring='f1_macro', n_jobs=-1)
# create confusion matrix

matrix = confusion_matrix(y_test, y_test_preds)



# create dataframe

class_names = df_model.Accident_Severity.values

dataframe = pd.DataFrame(matrix, index=['Fatal', 'Serious', 'Slight'], 

                         columns=['Fatal', 'Serious', 'Slight'])



# create heatmap

sns.heatmap(dataframe, annot=True, cbar=None, cmap='Blues')

plt.title('Confusion Matrix')

plt.tight_layout(), plt.xlabel('True Values'), plt.ylabel('Predicted Values')

plt.show()
# plot the important features

feat_importances = pd.Series(forest_2.feature_importances_, index=features.columns)

feat_importances.nlargest(10).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))

plt.xlabel('Relative Feature Importance with Random Forest');