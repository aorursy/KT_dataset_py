# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Trun off warnings

import warnings

warnings.filterwarnings('ignore')
# Directory prefix

dir_prefix = "/kaggle/input/human-resources-data-set/"
# load v13 dataset

df_hrfull = pd.read_csv(dir_prefix + "HRDataset_v13.csv")
df_hrfull.head(10)
df_hrfull.dtypes
from scipy.stats import entropy, mode
df_hrfull.groupby('ManagerID')['PerfScoreID'].agg(mode)
df_hrfull.groupby('ManagerID')['PerfScoreID'].agg(lambda x: entropy(x.value_counts(normalize=True)) )
import matplotlib.pyplot as plt

import seaborn as sns
# Emplyment status

df_hrfull['EmploymentStatus'].value_counts()
# Select Active and on leave employes only

active_mask = (df_hrfull['EmploymentStatus'] == "Active") | (df_hrfull['EmploymentStatus']  == "Leave of Absence")

# print counts of these employess

df_hrfull.loc[active_mask, 'EmploymentStatus'].value_counts()
df_sub = df_hrfull.loc[active_mask].fillna({'Sex':'Unknown'})

sns.countplot(x="Sex", data=df_sub)
df_sub = df_hrfull.loc[active_mask].fillna({'RaceDesc':'Unknown'})



sns.countplot(y="RaceDesc", data=df_sub)
df_sub = df_hrfull.loc[active_mask].fillna({'MaritalDesc':'Unknown'})

sns.countplot(y="MaritalDesc", data=df_sub)
df_sub = df_hrfull.loc[active_mask].fillna({'Sex':'Unknown','RaceDesc':'Unknown'})



sns.countplot(y="RaceDesc", hue="Sex", data=df_sub)
df_sub = df_hrfull.loc[active_mask].fillna({'Sex':'Unknown','RaceDesc':'Unknown'})

# Create an interaction features of Sex and Race

df_sub['Sex_RaceDesc'] = df_sub["Sex"].str.strip() + "_" + df_sub["RaceDesc"].str.strip() 
df_sub['Sex_RaceDesc'].value_counts()
# Measure diversity as entropy of "Sex_RaceDesc" 

df_sub = pd.DataFrame(df_sub.groupby('RecruitmentSource')['Sex_RaceDesc'] \

                    .agg(lambda x: entropy(x.value_counts(normalize=True))))

df_sub = df_sub.rename(columns={"Sex_RaceDesc":"DiversityScore"}).sort_values(by="DiversityScore", ascending=False)

df_sub.reset_index(inplace=True)
plt.figure(figsize=(10,8))

ax = sns.barplot(x="DiversityScore", y="RecruitmentSource", data=df_sub)
# Import datetime to create column for days since hiring.

from datetime import datetime
df_hrfull['DaysSinceHire'] = (datetime.now()  - pd.to_datetime(df_hrfull['DateofHire'],infer_datetime_format=True)).dt.days
df_hrfull['DaysSinceHire'].describe()
# import iqr

from scipy.stats import iqr
# Let us only include employees that have a performance score of "Fully Meets"

perf_mask = (df_hrfull['PerformanceScore'] == "Fully Meets")
df_salary_pos = df_hrfull.loc[active_mask & perf_mask].groupby(['Position'])['PayRate'].agg(['median',iqr]).rename(lambda x: 'salary_' + x, axis=1)

df_salary_pos['numberOfEmployees'] = df_hrfull.loc[active_mask & perf_mask].groupby(['Position']).size()
df_salary_pos
df_salary_pos['relative_variability'] = df_salary_pos['salary_iqr'] / df_salary_pos['salary_median']

df_salary_pos.sort_values(by='relative_variability', ascending=False, inplace=True)

df_salary_pos
df_salary_pos.reset_index(inplace=True)
df_sub =  df_salary_pos.loc[df_salary_pos['relative_variability'] > 0.10]

df_sub
# replace df_sub with the new data frame

df_sub = pd.merge(df_sub, df_hrfull, on='Position', how='left')
# import linear regression from scipy

from scipy.stats import linregress
df_linreg_salary_time = df_sub.groupby('Position').apply(lambda x: pd.Series(linregress(x['DaysSinceHire'], x['PayRate'])) )

df_linreg_salary_time.columns = ["slope", "intercept", "r_value", "p_value", "std_err"]
df_linreg_salary_time
for idx, grup in df_sub.groupby('Position'):

    sns.regplot(x=grup['DaysSinceHire'], y=grup['PayRate'])

    plt.title(grup['Position'].unique()[0])

    plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)



for idx, grup in df_sub.groupby('Position'):

    # pipeline for regression

    lr_pipe = Pipeline([ ("std_scale", StandardScaler()) ,("lr",LinearRegression())])

    

    # include polynomial features

    poly = PolynomialFeatures(order=2)

    X_poly = poly.fit_transform(grup.loc[:,['DaysSinceHire', 'SpecialProjectsCount']])

    

    # fit pipeline

    lr_pipe.fit(X= X_poly, y=grup['PayRate'])

    

    # Print fit results

    print("Position: ", grup['Position'].unique()[0])

    print('Coefficients: \n', lr_pipe.named_steps['lr'].coef_)

    print("R^2 score: ", lr_pipe.score(X= X_poly, y=grup['PayRate']),'\n\n' )
df_hrfull['Termd'].value_counts()
active_or_termd_mask = ~(df_hrfull['EmploymentStatus'].isnull() | (df_hrfull['EmploymentStatus']=="Future Start"))
df_sub = df_hrfull.loc[active_or_termd_mask,:]
df_sub.columns
# days since termination

df_sub['DaysSinceTermd'] = (datetime.now()  - pd.to_datetime(df_sub['DateofTermination'],infer_datetime_format=True)).dt.days
df_sub.fillna({'DaysSinceTermd': 0}, inplace=True)
# are there any NaNs in "DaysSinceTermd"

df_sub.loc[ (df_sub['Termd']==1.0), 'DaysSinceTermd'].isnull().any()
df_sub['DaysWorked'] = df_sub['DaysSinceHire'] - df_sub['DaysSinceTermd']
# check if any nulls exist

df_sub['DaysWorked'].isnull().any()
df_sub['MaxAgeWhenEmployed'] = (datetime.now()  - pd.to_datetime(df_sub['DOB'],infer_datetime_format=True)).dt.days  -  df_sub['DaysSinceTermd']
df_sub['MaxAgeWhenEmployed'].isnull().any()
# Columns to drop for training the RandomForest model

drop_cols = ['Employee_Name', 'EmpID', 'MarriedID', 'GenderID','DOB','EmpStatusID',

               'DateofHire', 'DateofTermination', 'TermReason', 'EmploymentStatus','Position', 

               'Zip','Department', 'ManagerName', 'LastPerformanceReview_Date', 'DaysLateLast30', 

               'DaysSinceHire', 'DaysSinceTermd','Termd']



X = df_sub.drop(columns= drop_cols)

y = df_sub['Termd'].values
X.isnull().sum()
# Fillna in manager ID with a ID -99

X.fillna({'ManagerID':-99}, inplace=True)
from sklearn.preprocessing import LabelEncoder
# label encode categorical columns

for col in X.columns:

    if (X[col].dtype == 'O'):

        labenc = LabelEncoder()

        X[col] = labenc.fit_transform(X[col])
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, min_impurity_decrease=1e-8)
rf_params = {'max_depth': [2,4], 'min_samples_leaf': [3,10]}
gcv = GridSearchCV(estimator=rf, param_grid=rf_params, cv=StratifiedKFold(n_splits=5), scoring=['accuracy','roc_auc'], refit='roc_auc', verbose=3)
gcv.fit(X,y)
gcv.best_params_, gcv.best_score_
# Mean score

gcv.cv_results_['mean_test_accuracy'].mean().round(2), gcv.cv_results_['mean_test_roc_auc'].mean().round(2)
# Mean STD of folds

gcv.cv_results_['std_test_accuracy'].mean().round(2), gcv.cv_results_['std_test_roc_auc'].mean().round(2)
feature_inportance_idx = np.argsort(gcv.best_estimator_.feature_importances_)
# Features in increasing order of predictive power (to predict if an employee will terminate)

X.columns[feature_inportance_idx]