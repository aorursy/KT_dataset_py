# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

import seaborn as sns

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/determine-the-pattern-of-tuberculosis-spread/tubercolusis_from 2007_WHO.csv')

df.head()
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = df.corr(method='kendall')

plt.subplots(figsize=(8,6))

sns.heatmap(corrmat, vmax=0.9, square=True)
df.isnull().sum()
# filling missing values with NA

df[['Number of prevalent tuberculosis cases (End range)', 'Number of deaths due to tuberculosis, excluding HIV (Start range)', 'Number of deaths due to tuberculosis, excluding HIV (End range)','Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)', 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)']] = df[['Number of prevalent tuberculosis cases (End range)', 'Number of deaths due to tuberculosis, excluding HIV (Start range)', 'Number of deaths due to tuberculosis, excluding HIV (End range)', 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)', 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)']].fillna('NA')
df = df.fillna(value = {'Number of prevalent tuberculosis cases (End range)' : 'no_info', 

                            'Number of deaths due to tuberculosis, excluding HIV (Start range)' : 'no_info', 

                            'Number of deaths due to tuberculosis, excluding HIV (End range)':'no_info',

                           'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)' : 'no_info',

                           'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)' : 'no_info'})
from sklearn.preprocessing import LabelEncoder

categorical_col = ('Country', 'Number of deaths due to tuberculosis, excluding HIV', 'Number of deaths due to tuberculosis, excluding HIV (Start range)', 'Number of deaths due to tuberculosis, excluding HIV (End range)', 'Number of prevalent tuberculosis cases', 'Number of prevalent tuberculosis cases (Start range)', 'Number of prevalent tuberculosis cases (End range)', 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)', 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)', 'Prevalence of tuberculosis (per 100 000 population)', 'Prevalence of tuberculosis (per 100 000 population)(end range)')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
from scipy.stats import norm, skew

num_features = df.dtypes[df.dtypes != 'object'].index

skewed_features = df[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness.head(15)
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
# Save target value for later

y = df.Year.values



# In order to make imputing easier, we combine train and test data

df.drop(['Year'], axis=1, inplace=True)

df = pd.concat((df, test)).reset_index(drop=True)
#df = pd.get_dummies(df, drop_first=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.model_selection import KFold

# Indicate number of folds for cross validation

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



# Parameters for models

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

xgboost = make_pipeline(RobustScaler(),

                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 

                                     max_depth=3,min_child_weight=0 ,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,nthread=4,

                                     scale_pos_weight=1,seed=27, 

                                     reg_alpha=0.00006))



# Printing out XGBOOST Score and STD

xgboost_score = cross_val_score(xgboost, X, y, cv=kfolds, scoring='neg_mean_squared_error')

xgboost_rmse = np.sqrt(-xgboost_score.mean())

print("XGBOOST RMSE: ", xgboost_rmse)

print("XGBOOST STD: ", xgboost_score.std())
# Separate target variable



df_tunning = df

y = df_tunning.iloc[:,1]

X = pd.concat([df_tunning.iloc[:,0],df_tunning.iloc[:,2:30]], axis=1)
# Drop `baby` feature from data



#df_model = df_model.drop(['Number of deaths due to tuberculosis, excluding HIV (Start range)'], axis=1)
# Separate target variable for model building 



y_model = df.iloc[:,1]

X_model = pd.concat([df_tunning.iloc[:,0],df_tunning.iloc[:,2:30]], axis=1)

y_model.describe()
# Split to train and test with 70-30 ratio



X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.3, random_state=42, stratify = y)
from sklearn.preprocessing import StandardScaler

# Implement standart scaler method



standardScalerX = StandardScaler()

X_train = standardScalerX.fit_transform(X_train)

X_test = standardScalerX.fit_transform(X_test)
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold Cross Validation Method



kfold_cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)



for train_index, test_index in kfold_cv.split(X_model,y_model):

    X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]

    y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]
from xgboost import XGBClassifier

# Extreme Gradient Boosting Model Building



xgb_model = XGBClassifier(criterion = 'giny', learning_rate = 0.01, max_depth = 5, n_estimators = 100,

                          objective ='binary:logistic', subsample = 1.0)

# fit the model

xgb_model.fit(X_train, y_train)

#Predict Model

predict_xgb = xgb_model.predict(X_test)
from sklearn.metrics import classification_report

print("XGB", classification_report(y_test, predict_xgb))
from sklearn.metrics import confusion_matrix

XGB_matrix = confusion_matrix(y_test, predict_xgb)



fig, ax = plt.subplots(figsize=(15, 8))

sns.heatmap(XGB_matrix,annot=True, fmt="d", cbar=False, cmap="Pastel1")

plt.title("Gradient Boosting", weight='bold')

plt.xlabel('Predicted Labels')

plt.ylabel('Actual Labels')
#Using Label Encoder method for categorical features

from sklearn.preprocessing import LabelEncoder



#labelencoder = LabelEncoder()

#df['Country'] = labelencoder.fit_transform(df['Country'])

#df['Number of deaths due to tuberculosis, excluding HIV'] = labelencoder.fit_transform(df['Number of deaths due to tuberculosis, excluding HIV'])

#df['Number of deaths due to tuberculosis, excluding HIV (Start range)']= labelencoder.fit_transform(df['Number of deaths due to tuberculosis, excluding HIV (Start range)'])

#df['Number of deaths due to tuberculosis, excluding HIV (End range)']=labelencoder.fit_transform(df['Number of deaths due to tuberculosis, excluding HIV (End range)'])

#df['Number of prevalent tuberculosis cases'] = labelencoder.fit_transform(df['Number of prevalent tuberculosis cases'])

#df['Number of prevalent tuberculosis cases (Start range)'] = labelencoder.fit_transform(df['Number of prevalent tuberculosis cases (Start range)'])

#df['Number of prevalent tuberculosis cases (End range)'] = labelencoder.fit_transform(df['Number of prevalent tuberculosis cases (End range)'])

#df['Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)'] = labelencoder.fit_transform(df['Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)'])

#df['Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)'] = labelencoder.fit_transform(df['Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)'])

#df['Prevalence of tuberculosis (per 100 000 population)'] = labelencoder.fit_transform(df['Prevalence of tuberculosis (per 100 000 population)'])

#df['Prevalence of tuberculosis (per 100 000 population)(end range)'] = labelencoder.fit_transform(df['Prevalence of tuberculosis (per 100 000 population)(end range)'])
#df = pd.get_dummies(df, drop_first=True)