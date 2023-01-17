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
#df = df.fillna(value = {'Number of prevalent tuberculosis cases (End range)' : 'no_info', 

                           # 'Number of deaths due to tuberculosis, excluding HIV (Start range)' : 'no_info', 

                           # 'Number of deaths due to tuberculosis, excluding HIV (End range)':'no_info',

                          # 'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (Start range)' : 'no_info',

                           #'Deaths due to tuberculosis among HIV-negative people (per 100 000 population) (End range)' : 'no_info'})
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url ='data:image/png;base64,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',width=400,height=400)