import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV ,KFold, cross_val_score, train_test_split

from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MultiLabelBinarizer ,LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import xgboost

import math

import ast

from datetime import datetime

import calendar



import warnings

warnings.filterwarnings("ignore")



from scipy.special import boxcox1p

train_data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

test_data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
train_data.head()
test_data.head()
def nullColumns(train_data):

    list_of_nullcolumns =[]

    for column in train_data.columns:

        total= train_data[column].isna().sum()

        try:

            if total !=0:

                print('Total Na values is {0} for column {1}' .format(total, column))

                list_of_nullcolumns.append(column)

        except:

            print(column,"-----",total)

    print('\n')

    return list_of_nullcolumns





def percentMissingFeature(data):

    data_na = (data.isnull().sum() / len(data)) * 100

    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

    missing_data = pd.DataFrame({'Missing Ratio' :data_na})

    print(missing_data.head(20))

    return data_na





def plotMissingFeature(data_na):

    f, ax = plt.subplots(figsize=(15, 12))

    plt.xticks(rotation='90')

    if(data_na.empty ==False):

        sns.barplot(x=data_na.index, y=data_na)

        plt.xlabel('Features', fontsize=15)

        plt.ylabel('Percent of missing values', fontsize=15)

        plt.title('Percent missing data by feature', fontsize=15)



def extract_key_val(df,colname):

    

    for idx, row in df.iterrows():

        

        try:

            y =ast.literal_eval(row[colname])  

            z= []

            for i in y:

                z.append(i['name'])

            df[colname][idx] = z

        

        except Exception as e:

            print(idx ,e)

    

    return df

listOfNullColumns = nullColumns(train_data)
listOfNullColumns = nullColumns(test_data)
test_data['revenue'] = 0

combined_data = pd.concat([train_data,test_data],axis =0)

combined_data = combined_data.reset_index(drop = True)

combined_data.head()
target_column =combined_data.pop('revenue')[:3000]
combined_data =combined_data.drop(columns=['belongs_to_collection','homepage','poster_path','id'],axis =1)
def correct_year(df,colname):

    

    df[colname] = df[colname].apply(lambda x : (x-100) if x>2017 else x)

    

    return df
def generate_date_features(calendar,colname):

    

    df = pd.DataFrame()

    

    df['Year'] = pd.to_datetime(calendar[colname]).dt.year



    df['Month'] = pd.to_datetime(calendar[colname]).dt.month



    df['Day'] = pd.to_datetime(calendar[colname]).dt.day



    df['Dayofweek'] = pd.to_datetime(calendar[colname]).dt.dayofweek



    df['DayOfyear'] = pd.to_datetime(calendar[colname]).dt.dayofyear



    df['Week'] = pd.to_datetime(calendar[colname]).dt.week



    df['Quarter'] = pd.to_datetime(calendar[colname]).dt.quarter 



    df['Is_month_start'] = pd.to_datetime(calendar[colname]).dt.is_month_start



    df['Is_month_end'] = pd.to_datetime(calendar[colname]).dt.is_month_end



    df['Is_quarter_start'] = pd.to_datetime(calendar[colname]).dt.is_quarter_start



    df['Is_quarter_end'] = pd.to_datetime(calendar[colname]).dt.is_quarter_end



    df['Is_year_start'] = pd.to_datetime(calendar[colname]).dt.is_year_start



    #df['Is_year_end'] = pd.to_datetime(calendar[colname]).dt.is_year_end



    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)



    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)



    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)



    return df
date_features = generate_date_features(combined_data,'release_date')

date_features = correct_year(date_features,'Year')
date_features.head()
combined_data = pd.concat([date_features,combined_data],axis =1)
plt.figure(figsize =(20,20))

plt.xticks(rotation='90')

sns.pointplot(combined_data.loc[:3000,'Year'],target_column)
plt.figure(figsize = (20,20))

sns.barplot(combined_data.loc[:3000,'original_language'],target_column)
combined_data['genres'] = combined_data['genres'].fillna('[{"id": 9999, "name": "unknown1"}]')

#train_data['production_companies'] = train_data['production_companies'].fillna('[{"id": 9999, "name": "unknown2"}]')

#train_data['production_countries'] = train_data['production_countries'].fillna('[{"iso_3166_1": "unknown3", "name": "unknown4"}]')

#train_data['spoken_languages'] = train_data['spoken_languages'].fillna('[{"iso_639_1": "unknown5", "name": "unknown6"}]')

#train_data['Keywords'] = train_data['Keywords'].fillna('[{"id": "unknown7", "name": "unknown8"}]')

#train_data['cast'] = train_data['cast'].fillna('[{"cast_id": "unknown9", "name": "unknown10"}]')

#train_data['runtime'] =train_data['runtime'].fillna(train_data['runtime'].mean())
df = extract_key_val(combined_data,'genres')

#extract_vals(train_data,'production_companies')

#extract_vals(train_data,'production_countries')

#extract_vals(train_data,'spoken_languages')

#extract_vals(train_data,'Keywords')

#extract_vals(train_data,'cast')
df.head(5)
plt.figure(figsize = (20,20))

sns.distplot(target_column)
target_column = np.log1p(target_column[:3000])
plt.figure(figsize = (20,20))

sns.distplot(target_column)
#lb = LabelEncoder()

#df['original_language'] = lb.fit_transform(df['original_language'])

mlb = MultiLabelBinarizer()

#df['original_language'] = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["original_language"]),columns=mlb.classes_, index=df.index)],axis =1)

df = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["genres"]),columns=mlb.classes_, index=df.index)],axis =1)

#df = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["production_companies"]),columns=mlb.classes_, index=df.index)],axis =1)

#df = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["production_countries"]),columns=mlb.classes_, index=df.index)],axis =1)

#df = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["Keywords"].apply(lambda x :x[0:1])),columns=mlb.classes_, index=df.index)],axis =1)

#df = pd.concat([df,pd.DataFrame(mlb.fit_transform(df["cast"].apply(lambda x : x[0:2])),columns=mlb.classes_, index=df.index)],axis =1)
df['count_genres'] = df['genres'].apply(lambda x : len(x))

#df['noofPrCom'] = df["production_companies"].apply(lambda x : len(x))

#df['noofPrCou'] = df["production_countries"].apply(lambda x : len(x))

#df['noofkey'] = df["Keywords"].apply(lambda x : len(x))

#df['noofcast'] = df["cast"].apply(lambda x : len(x))

#df['noofspokenlang'] =df["spoken_languages"].apply(lambda x : len(x))

#df['sequel'] = mlb.fit_transform(df["Keywords"])[:,5869]
for col in df.columns:

    if df.dtypes[col] != 'O':

        #print(col)

        df[col] =boxcox1p(df[col],0.15)
df = df.drop(columns = ['release_date'

                                ,'imdb_id'

                                ,'title'

                                ,'overview'

                                ,'production_companies'

                                ,'production_countries'

                                ,'spoken_languages'

                                ,'status'

                                ,'tagline'

                                ,'title'

                                ,'cast'

                                ,'crew'

                                ,'Keywords'

                                ,'original_language'

                                ,'genres'

                                ,'original_title'

                                ,'TV Movie'

                                ,'unknown1'

                               ])
df.head()
X_train,X_val,y_train,y_val = train_test_split(df.iloc[:3000,:],target_column,test_size =0.2,random_state = 1001)
def RMSLE(y, y_pred):     

    assert len(y) == len(y_pred)

    terms_to_sum = []

    for p , a in zip(y_pred,y):

        terms_to_sum.append((math.log(p + 1) - math.log(a + 1)) ** 2.0)

    

    return 'RMSLE',(sum(terms_to_sum) * (1.0/len(y))) ** 0.5, False
def feature_importance(model, X_train=X_train):



    print(model.feature_importances_)

    names = X_train.columns.values

    ticks = [i for i in range(len(names))]

    plt.bar(ticks, model.feature_importances_)

    plt.xticks(ticks, names,rotation =90)

    plt.show()
def create_submission_file(model_list):

    preds = 0

    submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

    for model in model_list:

        preds = preds + np.expm1(model.predict(df.iloc[3000:,:]))

    submission.loc[:,'revenue'] = preds/len(model_list)

    !rm './submission.csv'

    submission.to_csv('submission.csv', index = False, header = True)

    print(submission.head())

        
model_xgb = xgboost.XGBRegressor(colsample_bytree=0.4, gamma=0.045, 

                             learning_rate=0.1, max_depth=6, 

                             min_child_weight=1.7817, n_estimators=1000,

                             reg_alpha=0.45, reg_lambda=0.8,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1,seed=42)



model_xgb.fit(X_train,y_train,eval_set=[(X_train, y_train), (X_val, y_val)],

        eval_metric='rmsle',

        early_stopping_rounds = 50,

        verbose=2)

plt.figure(figsize =(20,20))

feature_importance(model_xgb)
model_lgb = lgb.LGBMRegressor(bagging_fraction=0.8, bagging_frequency=4, boosting_type='gbdt',

              class_weight=None, colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.1, max_depth=3,

              min_child_samples=20, min_child_weight=30, min_data_in_leaf=70,

              min_split_gain=0.0001, n_estimators=200, n_jobs=-1,

              num_leaves=1200, objective='regression' ,random_state=101, reg_alpha=0.2,

              reg_lambda=0.6, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)



model_lgb.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_val, y_val)],

        eval_metric=RMSLE,

        early_stopping_rounds = 100,

        verbose=2)
plt.figure(figsize =(20,20))

feature_importance(model_xgb)
create_submission_file([model_lgb,model_xgb])