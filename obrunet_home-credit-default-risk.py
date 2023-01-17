# usual data science stack in python

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
# imports of need modules in sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

import xgboost as xgb
# set options in this notebook

pd.set_option('display.max_columns', 300)



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
path_train = os.path.join('..', 'input', 'application_train.csv')

path_test = os.path.join('..', 'input', 'application_test.csv')
# load main datasets

app_train, app_test = pd.read_csv(path_train), pd.read_csv(path_test)
# 1st insight

app_train.tail()
app_train.shape, app_test.shape
app_train.TARGET.value_counts()
print(f'percentage of clients with payment difficulties: {app_train.TARGET.sum() / app_train.shape[0] * 100 :.2f}%')
plt.title('Distribution of the Target Column - 1 - client with payment difficulties / 0 - all other cases')

sns.countplot(x=app_train.TARGET, data=app_train)

plt.show()
app_train.dtypes.value_counts()
# Number of unique classes in each object column

app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# Function to calculate missing values by column# Funct // credits Will Koehrsen

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
# Missing values statistics

missing_values = missing_values_table(app_train)

missing_values.head(10)
# cols_to_drop = list((app_train.isnull().sum() > 75000).index)

cols_to_drop = [c for c in app_train.columns if app_train[c].isnull().sum() > 75000]
app_train, app_test = app_train.drop(cols_to_drop, axis=1), app_test.drop(cols_to_drop, axis=1)

app_test.isnull().sum().sort_values(ascending=False).head(10)
obj_cols = app_train.select_dtypes('object').columns

obj_cols
# filling string cols with 'Not specified' 

app_train[obj_cols] = app_train[obj_cols].fillna('Not specified')

app_test[obj_cols] = app_test[obj_cols].fillna('Not specified')
float_cols = app_train.select_dtypes('float').columns

float_cols
# filling float values with median of train (not test)

app_train[float_cols] = app_train[float_cols].fillna(app_train[float_cols].median())

app_test[float_cols] = app_test[float_cols].fillna(app_test[float_cols].median())
app_train.shape, app_test.shape
app_train.isnull().sum().sort_values(ascending=False).head()
app_test.isnull().sum().sort_values(ascending=False).head()
# Is there any duplicated rows ?
app_train.duplicated().sum()
app_test.duplicated().sum()
# Number of unique classes in each object column

app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
app_train['DAYS_EMPLOYED'].describe()
sns.distplot(app_train['DAYS_EMPLOYED'], kde=False);

plt.show()
print('The non-anomalies default on %0.2f%% of loans' % (100 * app_train[app_train['DAYS_EMPLOYED'] != 365243]['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * app_train[app_train['DAYS_EMPLOYED'] == 365243]['TARGET'].mean()))

print('There are %d anomalous days of employment' % len(app_train[app_train['DAYS_EMPLOYED'] == 365243]))
# Create an anomalous flag column

app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243



# Replace the anomalous values with nan

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



sns.distplot(app_train['DAYS_EMPLOYED'].dropna(), kde=False);
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243

app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)



print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
# refilling float values with median of train (not test)



app_train[float_cols] = app_train[float_cols].apply(pd.to_numeric, errors='coerce')

app_train = app_train.fillna(app_train.median())



app_test[float_cols] = app_test[float_cols].apply(pd.to_numeric, errors='coerce')

app_test = app_train.fillna(app_test.median())
correlations = app_train.corr()['TARGET'].sort_values()



print('Most Positive Correlations:\n', correlations.tail(10))

print('\n\nMost Negative Correlations:\n', correlations.head(10))
# Compute the correlation matrix

corr = app_train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(21, 19))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Find the correlation of the positive days since birth and target

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
plt.figure(figsize = (12, 6))



# KDE plot of loans that were repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')



# KDE plot of loans which were not repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# Age information into a separate dataframe

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



# Bin the age data

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head(10)
# Group by the bin and calculate averages

age_groups  = age_data.groupby('YEARS_BINNED').mean()

age_groups
plt.figure(figsize = (8, 6))



# Graph the age bins and the average of the target as a bar plot

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])



# Plot labeling

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')

plt.title('Failure to Repay by Age Group');
app_train = pd.get_dummies(data=app_train, columns=obj_cols)

app_test = pd.get_dummies(data=app_test, columns=obj_cols)
# back up of the target /  need to keep this information

y = app_train.TARGET

app_train = app_train.drop(columns=['TARGET'])
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train.shape, app_test.shape
feat_to_scale = list(float_cols).copy()

feat_to_scale.extend(['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START'])

feat_to_scale
scaler = StandardScaler()

app_train[feat_to_scale] = scaler.fit_transform(app_train[feat_to_scale])

app_test[feat_to_scale] = scaler.fit_transform(app_test[feat_to_scale])

app_train.head()
X_train, X_test, y_train, y_test = train_test_split(app_train, y, test_size=0.2)
# a simple RandomForrest Classifier without CV

rf = RandomForestClassifier(n_estimators=50)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

roc_auc_score(y_test, y_pred)
def submit(model, csv_name):

    

    # fit on the whole dataset of train

    model.fit(app_train, y)

    

    # Make predictions & make sure to select the second column only

    result = model.predict_proba(app_test)[:, 1]



    submit = app_test[['SK_ID_CURR']]

    submit['TARGET'] = result



    # Save the submission to a csv file

    submit.to_csv(csv_name, index = False)
# submit(rf, 'random_forrest_clf.csv')
importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(app_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the rf

plt.figure(figsize=(16, 8))

plt.title("Feature importances")

plt.bar(range(app_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(app_train.shape[1]), indices)

plt.xlim([-1, app_train.shape[1]])

plt.show()
(pd.Series(rf.feature_importances_, index=app_train.columns)

   .nlargest(15)

   .plot(kind='barh'))
rf_cv = RandomForestClassifier()

scores = cross_val_score(rf_cv, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

scores
rf_cv.fit(X_train, y_train)

roc_auc_score(y_test, rf_cv.predict(X_test))
#!pip install kaggle
#!kaggle competitions submit -c home-credit-default-risk -f randomforest_baseline.csv -m "My 1st submission - the baseline"
lgbm = lgb.LGBMClassifier(random_state = 50)

lgbm.fit(X_train, y_train, eval_metric = 'auc')

roc_auc_score(y_train, lgbm.predict(X_train))
roc_auc_score(y_test, lgbm.predict(X_test))
lgbm = lgb.LGBMClassifier(random_state = 50, n_jobs = -1, class_weight = 'balanced')

lgbm.fit(X_train, y_train, eval_metric = 'auc')

roc_auc_score(y_train, lgbm.predict(X_train))
roc_auc_score(y_test, lgbm.predict(X_test))
def submit_func(model, X_Test, file_name):

    model.fit(app_train, y)

    result = model.predict_proba(app_test)[:, 1]

    submit = app_test[['SK_ID_CURR']]

    submit['TARGET'] = result

    print(submit.head())

    print(submit.shape)

    submit.to_csv(file_name + '.csv', index=False)
submit_func(lgbm, app_test, 'lgbm_submission')
y.shape[0], y.sum()
ratio = (y.shape[0] - y.sum()) / y.sum()

ratio
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=50, eval_metric='auc', 

                              max_delta_step=2, scale_pos_weight=20)

xgb_model.fit(X_train, y_train)

roc_auc_score(y_train, xgb_model.predict(X_train))
roc_auc_score(y_test, xgb_model.predict(X_test))
submit_func(xgb_model, app_test, 'xgb_submission')