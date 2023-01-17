# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Training data
app_train = pd.read_csv('../input/train.csv')
ownership = pd.read_csv('../input/Building_Ownership_Use.csv')
structure = pd.read_csv('../input/Building_Structure.csv')
print('Training data shape: ', app_train.shape)
app_train = app_train.merge(ownership, on = 'building_id', how = 'left')
app_train = app_train.merge(structure, on = 'building_id', how = 'left')
app_train.drop(['district_id_x', 'district_id_y', 'vdcmun_id_x', 'vdcmun_id_y', 'ward_id_y'], axis=1, inplace=True)
print('Training data shape: ', app_train.shape)
app_train.head()
# Testing data features
app_test = pd.read_csv('../input/test.csv')
print('Testing data shape: ', app_test.shape)
app_test = app_test.merge(ownership, on = 'building_id', how = 'left')
app_test = app_test.merge(structure, on = 'building_id', how = 'left')
app_test.drop(['district_id_x', 'district_id_y', 'vdcmun_id_x', 'vdcmun_id_y', 'ward_id_y'], axis=1, inplace=True)
print('Testing data shape: ', app_test.shape)
app_test.head()
#Converting the object target to int type
target = {'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5}
app_train['damage_grade'].replace(target, inplace=True)
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
#identify null values in main data train & finding its impact on target values
missing_values_table(app_train)
# app_train[app_train['has_repair_started'].isnull()]['damage_grade'].plot.hist()
#app_train[app_train['count_families'].isnull()]['damage_grade'].plot.hist()
#app_train['damage_grade'].plot.hist()

#Registering anomaly info in dataset before imputing it
app_train['has_repair_started_flag'] = app_train['has_repair_started'].isnull()
#app_train['has_repair_started_flag'].value_counts()
#identify null values in main data test, like train data
missing_values_table(app_test)
app_test['has_repair_started_flag'] = app_test['has_repair_started'].isnull()
app_test['has_repair_started_flag'].value_counts()
app_train['foundation_type'].value_counts()
# app_train['count_floors_change'] = ((app_train['count_floors_post_eq']-app_train['count_floors_pre_eq'])*100)/app_train['count_floors_pre_eq']
# app_train['height_ft_change'] = ((app_train['height_ft_post_eq']-app_train['height_ft_pre_eq'])*100)/app_train['height_ft_pre_eq']
# app_test['count_floors_change'] = ((app_test['count_floors_post_eq']-app_test['count_floors_pre_eq'])*100)/app_test['count_floors_pre_eq']
# app_test['height_ft_change'] = ((app_test['height_ft_post_eq']-app_test['height_ft_pre_eq'])*100)/app_test['height_ft_pre_eq']

app_train['count_floors_change'] = (app_train['count_floors_post_eq']/app_train['count_floors_pre_eq'])
app_train['height_ft_change'] = (app_train['height_ft_post_eq']/app_train['height_ft_pre_eq'])
app_test['count_floors_change'] = (app_test['count_floors_post_eq']/app_test['count_floors_pre_eq'])
app_test['height_ft_change'] = (app_test['height_ft_post_eq']/app_test['height_ft_pre_eq'])

app_train.drop(['count_floors_post_eq', 'height_ft_post_eq'], axis=1, inplace=True)
app_test.drop(['count_floors_post_eq', 'height_ft_post_eq'], axis=1, inplace=True)
# app_train['height_ft_change'].plot.hist()
#app_train.plot(x='count_floors_pre_eq', y='count_floors_post_eq', style='o')

for i in ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other']:
    app_train[i+'_age'] = app_train[i] * app_train['age_building']
    app_test[i+'_age'] = app_test[i] * app_test['age_building']
app_train.head()
# for i in ['has_secondary_use', 'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution']:
#     app_train[i+'_age'] = app_train[i] * app_train['age_building']
#     app_test[i+'_age'] = app_test[i] * app_test['age_building']
print(app_train.select_dtypes('object').nunique())
print(app_test.select_dtypes('object').nunique())
#Remove column 'building_id' as it is unique for every row & doesnt have any impact
train_building_id = app_train['building_id']
test_building_id = app_test['building_id']
app_train.drop(['building_id'], axis=1, inplace=True)
app_test.drop(['building_id'], axis=1, inplace=True)
# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
for i in ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other']:
    app_train[i+'_mortar_foundation'] = app_train[i] * app_train['foundation_type_Mud mortar-Stone/Brick']
    app_test[i+'_mortar_foundation'] = app_test[i] * app_test['foundation_type_Mud mortar-Stone/Brick']
    
for i in ['has_geotechnical_risk', 'has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_flood', 'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction', 'has_geotechnical_risk_other', 'has_geotechnical_risk_rock_fall']:
    app_train[i+'_mortar_foundation'] = app_train[i] * app_train['foundation_type_Mud mortar-Stone/Brick']
    app_test[i+'_mortar_foundation'] = app_test[i] * app_test['foundation_type_Mud mortar-Stone/Brick']
    
app_train.head()
print(app_train.age_building.describe()) #No anomalies found
#app_train.age_building.plot.hist()
#print(app_train.age_building.value_counts().head(20))
# app_test.describe()

# Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(app_train['age_building'], edgecolor = 'k', bins = 25, range=[0, 60])
plt.title('Age Matrix'); plt.xlabel('Age'); plt.ylabel('Count');
plt.figure(figsize = (10, 8))

# KDE plot for area_assesed_Building removed
sns.kdeplot(app_train.loc[app_train['damage_grade'] == 1, 'age_building'], label = 'damage_grade == 1')
# KDE plot for area_assesed_Building removed
sns.kdeplot(app_train.loc[app_train['damage_grade'] == 2, 'age_building'], label = 'damage_grade == 2')
# KDE plot for area_assesed_Building removed
sns.kdeplot(app_train.loc[app_train['damage_grade'] == 3, 'age_building'], label = 'damage_grade == 3')
# KDE plot for area_assesed_Building removed
sns.kdeplot(app_train.loc[app_train['damage_grade'] == 4, 'age_building'], label = 'damage_grade == 4')
# KDE plot for area_assesed_Building removed
sns.kdeplot(app_train.loc[app_train['damage_grade'] == 5, 'age_building'], label = 'damage_grade == 5')

# Labeling of plot
plt.xticks([0, 20, 40, 50, 60,80, 100])
plt.xlabel('Building Age'); plt.ylabel('Age'); plt.title('Age');
# Find correlations with the target and sort
correlations = app_train.corr()['damage_grade'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(10))
print('\nMost Negative Correlations:\n', correlations.head(10))
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
imp_features=['area_assesed_Building removed', 'condition_post_eq_Damaged-Rubble unclear', 'condition_post_eq_Damaged-Rubble clear' ,'area_assesed_Both', 'condition_post_eq_Not damaged','has_repair_started_flag']
scor = app_train[imp_features+['damage_grade']]
# # Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(scor['area_assesed_Building removed'], edgecolor = 'k', bins = 10)
plt.title('Area assessed'); plt.xlabel('Area'); plt.ylabel('Count');
plt.figure(figsize = (5, 4))

# KDE plot for area_assesed_Building removed
sns.kdeplot(scor.loc[scor['area_assesed_Building removed'] == 0, 'damage_grade'], label = 'area_assesed == 0')
# KDE plot for area_assesed_Building removed
sns.kdeplot(scor.loc[scor['area_assesed_Building removed'] == 1, 'damage_grade'], label = 'area_assesed == 1')

# Labeling of plot
plt.xlabel('Damage Grade'); plt.ylabel('Density'); plt.title('Area');
# Area information into a separate dataframe
area_data = scor[['damage_grade', 'area_assesed_Building removed']]
area_data.groupby('damage_grade').mean()
# Area information into a separate dataframe
area_data = scor[['damage_grade', 'area_assesed_Both']]
area_data.groupby('damage_grade').sum()
plt.figure(figsize = (5, 4))

# KDE plot for area_assesed_Both
sns.kdeplot(scor.loc[scor['area_assesed_Both'] == 0, 'damage_grade'], label = 'area_assesed_both == 0')
# KDE plot for area_assesed_Both
sns.kdeplot(scor.loc[scor['area_assesed_Both'] == 1, 'damage_grade'], label = 'area_assesed_both == 1')

# Labeling of plot
plt.xlabel('Damage Grade'); plt.ylabel('Density'); plt.title('Area');
# Area information into a separate dataframe
area_data = scor[['damage_grade', 'has_repair_started_flag']]
area_data.groupby('damage_grade').sum()
plt.figure(figsize = (5, 4))

# KDE plot for has_repair_started_flag
sns.kdeplot(scor.loc[scor['has_repair_started_flag'] == 0, 'damage_grade'], label = 'area_assesed_both == 0')
# KDE plot for has_repair_started_flag
sns.kdeplot(scor.loc[scor['has_repair_started_flag'] == 1, 'damage_grade'], label = 'area_assesed_both == 1')

# Labeling of plot
plt.xlabel('Damage Grade'); plt.ylabel('Density'); plt.title('Area');
data_corrs = scor.corr()
data_corrs
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

train_labels = app_train['damage_grade']
# Drop the target from the training data
if 'damage_grade' in app_train:
    train = app_train.drop(columns = ['damage_grade'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
# imputer.fit(train)

# Transform both training and testing data
train = imputer.fit_transform(train)
test = imputer.transform(test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor

# Make the random forest classifier
clf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# print(clf.feature_importances_)

# Make the model with the specified regularization parameter
# clf = LogisticRegression(C = 0.0001)

#Use XGBooster
# clf = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
from sklearn import model_selection
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    train, train_labels, test_size=0.25)
# Train on the training data
# clf.fit(X_train, y_train, early_stopping_rounds=5, 
#              eval_set=[(X_test, y_test)], verbose=True)
clf.fit(X_train, y_train)

f1_score(y_test, clf.predict(X_test), average='weighted')
# clf.score(X_test, y_test)
# Extract feature importances
feature_importance_values = clf.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)
#Prediction with classifier
y=clf.predict(test)
prediction=pd.DataFrame({'building_id': test_building_id, 'damage_grade':y})
prediction.damage_grade = np.round(prediction.damage_grade)
target = {1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3', 4: 'Grade 4', 5: 'Grade 5'}
prediction.damage_grade.replace(target, inplace=True)
prediction.to_csv('submission.csv', index=False)
#prediction
