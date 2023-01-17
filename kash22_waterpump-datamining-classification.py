import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# Load training data
df_pump = pd.read_csv('../input/pump-it-up-data-mining-the-water-table/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv')
df_pump
# Load training labels
df_label = pd.read_csv('../input/pump-it-up-data-mining-the-water-table/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv')
df_label
#plot the target variable to view
plt.subplots(figsize=(10,6))
ax = sns.countplot(x=df_label['status_group'])
for p in ax.patches:
        ax.annotate('{:.2f}%'.format(p.get_height()*100/len(df_label)), (p.get_x()+0.3, p.get_height()*0.5))
# Merge labels with main data for ease in subsequent data clearning & transformation processes
df_pump = df_pump.merge(df_label, how='inner', on='id')
#Get info on columns, and data types
df_pump.info()
# Check unique count of respective columns. 
# This will help in identifying categorical & non-categorical data
df_pump.nunique()
df_pump.isnull().sum()/len(df_pump)*100

# Get visual sense of null data
plt.subplots(figsize=(15,8))
sns.heatmap(df_pump.isnull())
# Describe all columns to get top frequencies, min/max, means, deviations
df_pump.describe(include='all').T
#df_pump[df_pump.amount_tsh==0]['amount_tsh'].count()
#len(df_pump[df_pump.public_meeting.isnull()==True]) -> 3334
#df_pump.recorded_by.nunique()

#df_pump.scheme_name
#df_pump.scheme_name.nunique()
#len(df_pump[df_pump.scheme_name.isnull()==True]) -> 28166

#df_pump.permit -> True/False
#len(df_pump[df_pump.permit.isnull()==True]) -> 3056

#df_pump.construction_year.nunique() -> 55 years
#max(df_pump.construction_year) -> 2013
#len(df_pump[df_pump.construction_year==0]) -> 20709

#df_pump.construction_year.nunique() -> 55 years
#max(df_pump.construction_year) -> 2013
#len(df_pump[df_pump.construction_year==0]) -> 20709

# Removing columns as:
    # amount_tsh -> 75% are zeros and has outliers
    # wpt_name -> free text, mostly nuls
    # latitude, longitude -> as other geographic locations available
    # num_private -> no clue, almost no data
    # subvillage -> will use district code
    # region -> will use region code
    # lga, ward -> yet another geographic locations
    # recorded_by -> just one value
    # scheme_name -> more than 50% null, free text, will use scheme management
    # extraction_type & extraction_type_class -> will use extraction_type_group
    # management_group -> will use management
    # payment -> will use payment_type
    # water_quality -> will use quality_group
    # quantity -> will use quantity_group
    # source_type, source_class -> will use source
    # waterpoint_type -> will use waterpoint_type_group
    
df_pump = df_pump.drop(['amount_tsh','wpt_name','latitude','longitude','num_private','subvillage',
                        'region','lga','ward','recorded_by','scheme_name','extraction_type',
                        'extraction_type_class','management_group','payment','water_quality',
                        'quantity','source_type','source_class','waterpoint_type'], axis=1)    
df_pump = df_pump.dropna()
df_pump
# Current counts of funder & installer
[df_pump['funder'].nunique(), df_pump['installer'].nunique()]
df_fgb = df_pump.groupby(['funder'], as_index=False).count()[['funder', 'id']]
df_fgb
# Funders with less than 5 pumps, will be renamed to Other
other_funders = df_fgb[df_fgb.id<5].funder.unique()
df_pump['funder'] = df_pump['funder'].apply(lambda x : x if (x not in other_funders)  else 'Other')
# Installers with less than 5 pumps, will be renamed to Other
df_igb = df_pump.groupby(['installer'], as_index=False).count()[['installer', 'id']]
other_installers = df_igb[df_igb.id<5].installer.unique()
df_pump['installer'] = df_pump['installer'].apply(lambda x : x if (x not in other_installers)  else 'Other')
# Cleanup
del df_fgb
del df_igb

# View reduced counts
[df_pump['funder'].nunique(), df_pump['installer'].nunique()]
df_pump['date_recorded'].max()
df_pump['construction_year'].max()
# We have date fields as date_recorded and construction_year, will consider the later one
df_pump['construction_year'].unique()
# create a new feature 'age' relative to the max construction year
df_pump['age'] = df_pump['construction_year'].apply(lambda x: (2013 - x) if x != 0 else 0 )

# review relation with recorded date
df_pump[['age', 'construction_year', 'date_recorded']].drop_duplicates()
df_pump.age.describe()
# There are many age values, will group them for better analysis
def get_age_group(x):
    if x <= 17:
        return 'new'
    elif x > 17 and x <=34:
        return 'middle'
    else:
        return 'old'

# Create new feature age_group
df_pump['age_group'] = df_pump['age'].apply(get_age_group)

# Visualize Relation
plt.subplots(figsize=(12,8))
sns.countplot(x=df_pump['age_group'], hue=df_pump['status_group'])
plt.title('Age Group Vs Water-pump Status')
plt.show()
df_pump.population.describe()
# Max value and value of 75% population indicates outliers as well as very low population mostly
# Visualize through boxplot
sns.boxplot(x=df_pump.population)
print('Outlier for population =', df_pump[df_pump.population > 10000].population.tolist() )
# Create Population Group to better assess the sparse data
def get_population_group(x):
    if x <= 500:
        return 'low'
    else:
        return 'dense'
    
df_pump['population_group'] = df_pump['population'].apply(get_population_group)

# Visualize Relation
plt.subplots(figsize=(12,8))
sns.countplot(x=df_pump['population_group'], hue=df_pump['status_group'])
plt.title('Population Group Vs Water-pump Status')
plt.show()
def plot_counts(featureX):
    plt.subplots(figsize=(18,8))
    sns.set(font_scale=1)
    sns.countplot(x=df_pump[featureX], hue=df_pump['status_group'])
    plt.title(f'{featureX} Vs Water-pump Status')
    plt.show()
    return

# Management of Waterpoint
plot_counts('management')

# Payment Type of Waterpoint
plot_counts('payment_type')

# Water source of Waterpoint
plot_counts('source')

# Water source of Waterpoint
plot_counts('waterpoint_type_group')
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
categorical_features = ['funder','installer','basin', 'public_meeting', 'scheme_management', 'permit', 
                        'extraction_type_group', 'management', 'payment_type', 'quality_group', 'quantity_group', 
                        'source', 'waterpoint_type_group', 'age_group', 'population_group', 'status_group']

# Copy status group labels
df_pump['status_group_labels'] = df_pump['status_group']

# Encode values
df_pump[categorical_features] = df_pump[categorical_features].apply(encoder.fit_transform)
df_pump.describe(include='all').T
# Visualize Correlation with features Matrix 
corrMatrix = df_pump[['funder', 'gps_height', 'installer', 'basin', 'region_code', 'district_code', 
            'scheme_management', 'permit', 'extraction_type_group', 'management', 'payment_type', 'quality_group', 'quantity_group', 'source',
            'waterpoint_type_group', 'construction_year', 'age_group', 'population_group', 'status_group']].corr()
plt.subplots(figsize=(18,12))
sns.heatmap(corrMatrix, annot=True, fmt='.1g')
plt.title('Correlation of Features Matrix')
plt.show()
#Prepare list of features for training and target for prediction
features = ['funder', 'gps_height', 'installer', 'basin', 'region_code', 'district_code', 
            'scheme_management', 'permit', 'extraction_type_group', 'management', 'payment_type', 'quality_group', 'quantity_group', 'source',
            'waterpoint_type_group', 'age_group', 'population_group'] 

target = ['status_group']
df_pump_X = df_pump[features]
df_pump_Y = df_pump[target]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# random_state = 1, to get same split in case of re-run
# startify = target, to get split containing each type of class proportional to origingal dataset
X_train, X_test, y_train, y_test = train_test_split(df_pump_X, df_pump_Y, 
                                                    random_state=1, stratify= df_pump_Y, 
                                                    test_size = 0.33)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train.values.ravel())
y_predicted = rf_classifier.predict(X_test)
confusion_matrix(y_test, y_predicted)
print(classification_report(y_test, y_predicted))
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators' : [301,401,501],
    'max_depth' : [11,21,31],    
    'max_features' : ['sqrt','log2']
}

cv = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
cv.fit(X_train, y_train.values.ravel())
cv.best_params_
# Predict again with the values of best parameters
rf_classifier_bst = RandomForestClassifier(n_estimators=501, n_jobs=-1,max_depth=21, max_features='log2', bootstrap=True, criterion='gini')
rf_classifier_bst.fit(X_train, y_train.values.ravel())
y_predict_bst = rf_classifier_bst.predict(X_test)
# Confusion Matrix -> Review where model made mistakes 
# Like False positives, False Negatives etc.

cnf_matrix = confusion_matrix(y_test, y_predict_bst)
cnf_matrix
cnf_matrix = cnf_matrix.astype('float')  / cnf_matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(12,7))
sns.set(font_scale=1.2)
sns.heatmap(cnf_matrix, annot=True, cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Functional', 'Needs Repair', 'Non-Functional']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted Status')
plt.ylabel('True Status')
plt.title('Confusion Matrix Normalized for predicted Waterpump statuses')
plt.show()
# Get Summary Report of Prediction Metrices
print(classification_report(y_test, y_predict_bst, target_names=class_names))
# Formatting Classification Report
clf_rpt = classification_report(y_test, y_predict_bst, target_names=class_names, output_dict=True)

# .iloc[:-1, :] to exclude support
plt.figure(figsize=(12,7))
sns.heatmap(pd.DataFrame(clf_rpt).iloc[:-1, :].T, annot=True)
plt.title ('Classification Report')
plt.show()
