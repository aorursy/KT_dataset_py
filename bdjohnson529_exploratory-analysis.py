import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in datasets
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')

print('Training data features: \t\t', app_train.shape[1])
print('Testing data features: \t\t\t', app_test.shape[1])
print('\nTraining data observations: \t\t', app_train.shape[0])
print('Testing data observations: \t\t', app_test.shape[0])
# boolean function to verify whether all features are numeric
def TestNumericFeatures(df, feature):
    non_numeric = []

    unique_var = df[feature].unique()

    # verify that every unique variable is numeric
    for var in unique_var:
        vartype = type(var)
        if (vartype is not np.int64) and (vartype is not np.float64):
            non_numeric.append(feature)
            return False

    return True

# analyze features in a CSV
def AnalyzeFeatures(df):
    # list all features in dataset
    feature_list = df.columns.tolist()
    d = {'Feature': feature_list, 'Percentage_Full': 1}		# create dictionary
    feature_data = pd.DataFrame(data=d)

    # track non-numeric booleans
    non_numeric_bools = []
    
    # list features with null entries
    null_features = df.columns[df.isna().any()].tolist()

    # get percentage of filled entries
    for entry in null_features:
        feature = str(entry)
        x = df[feature].isnull().mean()
        feature_data.loc[(feature_data['Feature'] == feature), 'Percentage_Full'] = x

    # feature type: (0: boolean, 1: categorical, 2: continuous)
    feature_data['Type'] = -1
    feature_list = feature_data['Feature'].tolist()
    for entry in feature_list:
        feature = str(entry)
        unique_var = df[feature].unique()

        # label boolean variables
        if len(unique_var) == 2:
            if ( TestNumericFeatures(app_train, feature) ):
                feature_data.loc[(feature_data['Feature'] == feature), 'Type'] = 0
            else:
                non_numeric_bools.append(feature)
        else:
            num_var = len(unique_var)
            feature_data.loc[(feature_data['Feature'] == feature), 'Type'] = num_var

    return feature_data, non_numeric_bools
feature_data, non_numeric_bools = AnalyzeFeatures(app_train)

full_features = feature_data[['Feature', 'Percentage_Full']].groupby(['Percentage_Full']).get_group(1)

num_full = full_features.shape[0]
total = feature_data.shape[0]
num_partial = total - num_full

df = pd.DataFrame({'Number of Features': [num_full, num_partial] }, index=['Full Features', 'Partially Null'])
df['Number of Features'].plot(kind='pie', autopct='%.2f', figsize=(5, 5))

df
feature_list = full_features['Feature']

# check for features with date
date_features=[]
for feature in feature_list:
    if feature.find('DAY') >= 0:
        date_features.append(feature)
        
print("Features containing a date: ", date_features)
print('-'*20)
print("Sample data: ")
print( app_train['DAYS_BIRTH'].head() )
combined = [app_train, app_test]

for df in combined:
    df['AGE'] = abs(df['DAYS_BIRTH']) / 365
    df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365
    df['YEARS_REGISTRATION'] = abs(df['DAYS_REGISTRATION']) / 365
    df['YEARS_ID_PUBLISH'] = abs(app_train['DAYS_ID_PUBLISH']) / 365
    
    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'])
    
app_train = combined[0]
app_test = combined[0]

target0 = app_train[['TARGET', 'AGE', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH', 'YEARS_EMPLOYED']].groupby(['TARGET']).get_group(0)
target1 = app_train[['TARGET', 'AGE', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH', 'YEARS_EMPLOYED']].groupby(['TARGET']).get_group(1)

# plot histograms
plt.figure()
plot1 = target0['AGE'].plot(kind='kde', legend=True)
plot1 = target1['AGE'].plot(kind='kde', legend=True)
plot1.legend(('Succesful Repayments', 'Failed Repayments'))
plot1.set_xlabel("Age")

plt.figure()
plot2 = target0['YEARS_EMPLOYED'].plot(kind='kde', legend=True)
plot2 = target1['YEARS_EMPLOYED'].plot(kind='kde', legend=True)
plot2.legend(('Succesful Repayments', 'Failed Repayments'))
plot2.set_xlabel("Years Employed")

plt.figure()
plot3 = target0['YEARS_REGISTRATION'].plot(kind='kde', legend=True)
plot3 = target1['YEARS_REGISTRATION'].plot(kind='kde', legend=True)
plot3.legend(('Succesful Repayments', 'Failed Repayments'))
plot3.set_xlabel("Years Registered")

plt.figure()
plot4 = target0['YEARS_ID_PUBLISH'].plot(kind='kde', legend=True)
plot4 = target1['YEARS_ID_PUBLISH'].plot(kind='kde', legend=True)
plot4.legend(('Succesful Repayments', 'Failed Repayments'))
plot4.set_xlabel("Years with Published ID")

plt.show()
from scipy.stats import chi2_contingency

def ChiSquared(df, feature_name, target_name):
    contingency_table = []

    feature_values = df[feature_name].unique()
    target_values = df[target_name].unique()

    for x in feature_values:
        obs = df[[feature_name, target_name]].groupby([feature_name]).get_group(x)

        # only proceed if there is more than one value
        obs_list = obs['TARGET'].unique()
        if ( len(obs_list) < 2 ):
            chi2 = -1
            p = -1
            return chi2, p

        segregated_obs = []
        for y in target_values:
            obs_list = obs.groupby([target_name]).get_group(y)
            num_obs = len(obs_list)
            segregated_obs.append(num_obs)

        contingency_table.append(segregated_obs)

    contingency_table = np.array(contingency_table)
    contingency_table = np.transpose(contingency_table)
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return chi2, p
full_feature_list = full_features['Feature'].tolist()

bools = feature_data[['Feature', 'Type']].groupby(['Type']).get_group(0)
bool_list = bools['Feature'].tolist()

full_bool_list = [x for x in bool_list if x in full_feature_list]
full_bool_list.remove('TARGET')

chi2_table = []
for feature in full_bool_list:
    chi2, p = ChiSquared(app_train, feature, 'TARGET')
    chi2_table.append([feature, chi2, p])
    
chi2_arr = np.array(chi2_table)
# filter non-valid entries
features = [x[0] for x in chi2_arr if x[2].astype(float)>0]
pvalues = [x[2].astype(float) for x in chi2_arr if x[2].astype(float)>0]

selected_features = [x[0] for x in chi2_arr if (x[2].astype(float)>0) & (x[2].astype(float)<0.25)]
print("Number of full, boolean features: ", len(features) )
print("Selected features: ", len(selected_features) )


# plot histogram
fig,ax = plt.subplots(1)
ax.bar(features, pvalues)
ax.set_ylabel('P-value')
ax.set_xlabel('Features')
# Turn off labels
ax.set_xticklabels([])

plt.show()

print("Selected features: ", selected_features)
from sklearn.ensemble import RandomForestClassifier

# initialize test and training datasets
X_train = app_train[selected_features]
Y_train = app_train['TARGET']
X_test = app_test[selected_features]

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print( "random forest score:\t", acc_random_forest )

submission = pd.DataFrame({
    "Applicant": app_test['SK_ID_CURR'],
    "Success": Y_pred
    })
    
submission.to_csv('submission.csv', index=False)