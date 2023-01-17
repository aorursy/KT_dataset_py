# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





PREFIX = dirname + "/"

print("")

print(PREFIX)

#"/kaggle/input/covid19-global-forecasting-week-3/"




import pandas as pd

import numpy as np



from sklearn.ensemble import AdaBoostRegressor

from datetime import timedelta, datetime

from collections import defaultdict



from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline





from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import TransformedTargetRegressor

from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin):



    def fit(self, X, y=None, **fit_params):

        return self



    def transform(self, X, y=None, **fit_params):

        return X.todense()



pd.options.display.float_format = '{:.2f}'.format



#/kaggle/input/covid19-global-forecasting-week-2/

#PREFIX = "/home/mikeskim/Desktop/covid5/data/"

print("done")



#import sklearn as sk

#print(sk.__version__)





# In[2]:







train_df = pd.read_csv(PREFIX+"train.csv", parse_dates=["Date"])

print(train_df.describe())

train_df['TargetValue'].clip(0, inplace=True)

print(train_df.describe())

MIN_DATE_TRAIN = train_df["Date"].min()

test_df = pd.read_csv(PREFIX+"test.csv", parse_dates=["Date"])

sub_df = pd.read_csv(PREFIX+"submission.csv")



train_df.fillna("_", inplace=True)

test_df.fillna("_", inplace=True)



train_df['region_tuple'] = train_df[['County','Province_State','Country_Region']].sum(axis=1)

test_df['region_tuple'] = test_df[['County','Province_State','Country_Region']].sum(axis=1)

train_vc = train_df['region_tuple'].value_counts()

population_vc = train_df['Population'].map(np.log).round(0).value_counts()

train_weights = train_df['Weight'].tolist()

train_population = train_df['Population'].tolist()



def create_days(df):

    df = df.copy()

    df['target_dummy'] = 0

    df['region_tuple'] = df['region_tuple'].map(train_vc)

    df['pop_vc'] = df['Population'].map(np.log).round(0).map(population_vc)

    df.loc[df['Target']=='Fatalities','target_dummy']=1

    df['Days'] = (df["Date"]-MIN_DATE_TRAIN).dt.days.astype(int)

    df['Days2'] = df['Days']**0.5

    df['Days3'] = df['Days'].map(np.log1p)

    df['weekday'] = df["Date"].dt.dayofweek

    df['weekday_str'] = df['weekday'].astype(str)

    df['Weight'] =df['Weight']*df['Days']

    return df



train_df = create_days(train_df)

test_df = create_days(test_df)

train_days = train_df['Days'].tolist()



    

#   df['Days'] = (df["Date"]-MIN_DATE_TRAIN).dt.days.astype(int)

#    df['Days2'] = df['Days'].pow(2)

#   df['weekday'] = df["Date"].dt.dayofweek

#ef combine_regions(df):

#   df = df.copy()

#   df['C_P_S_C_R'] = df['County']+df['Province_State']+df['Country_Region'].astype(str)

#   return 



#rain_df = combine_regions(train_df)

#est_df = combine_regions(test_df)

print(MIN_DATE_TRAIN)





# In[3]:





sub_df.head(3)





# In[4]:





train_df_Id = train_df['Id'].tolist()

train_df.drop(["Id","Date"], axis=1, inplace=True)



train_df.head(3)





# In[5]:







#        model = make_pipeline(PolynomialFeatures(2), BayesianRidge()) #Ridge() worse

#        model.fit(prior_df[["ConfirmedCases","Fatalities","Days"]].shift(periods=2).dropna(),

#                  prior_df[["ConfirmedCases"]].shift(periods=-2).dropna().values.reshape(-1,))  

#        preds = model.predict(prior_df[["ConfirmedCases","Fatalities","Days"]])

#        features_list += [preds[-1]]





# In[6]:





test_df_ForecastId = test_df['ForecastId'].tolist()

test_df.drop(["ForecastId","Date"], axis=1, inplace=True)

test_df.head(3)





# In[7]:





train_df.dtypes





# In[8]:





from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler()),

                      ('interactions', PolynomialFeatures(2)),

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),



    ('onehot', OneHotEncoder(handle_unknown='ignore'))])





# In[9]:





numeric_features = train_df.select_dtypes(include=['int64', 'float64']).drop(['TargetValue'], axis=1).columns

categorical_features = train_df.select_dtypes(include=['object']).columns

print(numeric_features)

print(categorical_features)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])





# In[10]:





#https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf



transformed_ridge =  TransformedTargetRegressor(regressor=Ridge(alpha=0.1), # Lasso(alpha=0.01),

                                                func=np.log1p,

                                                inverse_func=np.expm1)



br = Pipeline(steps=[('preprocessor', preprocessor),

  #                    ('interactions', PolynomialFeatures(2)),

 #                        ('to_dense', DenseTransformer()), 

                      ('regressor', transformed_ridge)])



features_list = test_df.columns.tolist()
tw_list = [

           np.ones(len(train_weights)), 

           np.array(train_population),

           np.array(train_weights),

           np.array(train_days),

           np.array(train_weights)*np.array(train_population)*np.array(train_days),    

           np.array(train_weights)*np.array(train_population),

           np.array(train_weights)*(np.array(train_population)**0.5),

           np.array(train_population)*np.array(train_days),

           np.array(train_weights)*np.array(train_days),

           np.array(train_weights)*(np.array(train_days)**0.5),

           np.array(train_weights)*(np.array(train_days)**2),

          ]

preds_list = []

for tw in tw_list:

    br.fit(train_df[features_list], 

           train_df["TargetValue"],

           regressor__sample_weight=tw)

    preds = br.predict(test_df[features_list])

    preds_list.append(preds)





preds_df = pd.DataFrame(preds_list).T

preds_df.head(3)



output_df = pd.DataFrame()

output_df['min_pred'] = preds_df.min(axis=1)

output_df['median_pred'] = preds_df.median(axis=1)

output_df['max_pred'] = preds_df.max(axis=1)

output_df.head(3)



sub_list = output_df.values.flatten().tolist()

sub_list[0:5]



sub_df['TargetValue'] = sub_list

sub_df['TargetValue'] = sub_df['TargetValue'].clip(0)

sub_df.to_csv('submission.csv', index=False)

print(sub_df.describe())

sub_df.head(10)