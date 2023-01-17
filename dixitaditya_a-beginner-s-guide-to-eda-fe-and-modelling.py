import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import ast

sns.set(rc={'figure.figsize':(15,5)})
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)

print(train.info())
print(train.head(1).T)
combined = pd.concat([train,test],axis=0,sort=False)

print(combined.shape)

combined.index = range(len(combined))


dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

combined = text_to_dict(combined)
t = pd.DatetimeIndex(combined['release_date'])

combined['release_date'] = t

combined['release_year'] = t.year

combined['release_month'] = t.month

combined['release_day_of_week'] = t.weekday
combined.groupby([combined.release_year // 10 * 10,'status'])['id'].count().unstack().fillna(0)

combined[(combined.release_year > 2010) & (combined.release_year < 2020)].groupby(['release_year','release_month'])['id'].count().unstack().plot(kind='bar')
combined.loc[combined.release_date > '2017-08-31','release_date'] = combined.loc[combined.release_date > '2017-08-31','release_date'].apply(lambda x: x - pd.DateOffset(years=100))
t = pd.DatetimeIndex(combined['release_date'])

combined['release_date'] = t

combined['release_year'] = t.year

combined['release_month'] = t.month

combined['release_day_of_week'] = t.weekday

combined['profit'] = combined.revenue - combined.budget
combined.groupby([combined.release_year // 10 * 10,'status'])['id'].count().unstack().fillna(0)


combined.groupby(combined.release_year // 10 * 10)[['revenue','budget','profit']].mean().plot(kind='bar',title='Revenue,Budget and Profit - Decade Wise')
# get list of unique genres and create columns for each, i.e. one hot encoding

def dict_to_cols(colname):

     for i in range(len(combined)):

        #running for each row

         for j in range(len(combined[colname][i])):

            #creating and filling values for each genre column based on current value

             combined.loc[i,str(colname) + "_" + str(combined.loc[i,colname][j]['name'])] = 1

    

    #fill 0 value where a genre doesn't exist for a movie

     t_col = combined.columns.str.startswith(colname)

     combined.iloc[:,t_col] = combined.iloc[:,t_col].fillna(0)



dict_to_cols('genres')

combined.drop('genres',axis=1,inplace=True)

combined.head().T
# get list of unique languages and create columns for each, i.e. one hot encoding

def dict_to_cols(colname,value):

     for i in range(len(combined)):

        #running for each row

         for j in range(len(combined[colname][i])):

            #creating and filling values for each genre column based on current value

             combined.loc[i,str(colname) + "_" + str(combined.loc[i,colname][j][value])] = 1

    

    #fill 0 value where a genre doesn't exist for a movie

     t_col = combined.columns.str.startswith(colname)

     combined.iloc[:,t_col] = combined.iloc[:,t_col].fillna(0)



dict_to_cols('spoken_languages','iso_639_1')

combined.drop('spoken_languages',axis=1,inplace=True)

combined.head(10).T
combined['prod_countries_count'] = combined.production_countries.apply(lambda x:len(x))

combined.drop('production_countries',axis=1,inplace=True)
combined['belongs_to_collection_flag'] = combined['belongs_to_collection'].apply(lambda x: 1 if len(x) > 0 else 0)

combined['homepage_exists'] = combined['homepage'].notnull().astype(int)

combined.drop('homepage',axis=1,inplace=True)
combined.groupby([combined.release_year // 10 * 10,'homepage_exists'])['revenue'].mean().unstack().fillna(0).plot(kind='bar',title='Decade wise counts')
combined['overview_length'] = combined.overview.apply(lambda x:len(str(x)))
# Let's drop the unwanted columns

combined.drop(['id','belongs_to_collection','release_date','overview','original_title','poster_path','tagline','title'],axis=1,inplace=True)

combined['production_companies'].head(10)
combined.Keywords.apply(lambda x: [x[i]['name'] for i in range(len(x))])
# %%time

# def get_names_from_dict(column):

#     df = pd.DataFrame(columns=['id','year','name','revenue','budget'])

#     for i in range(len(combined)):

#     #running for each row

#         for j in range(len(combined[column][i])):

#         #creating and filling values for the column based on current value

#             df = df.append({'name':combined.loc[i,column][j]['name'],

#                             'revenue':combined.loc[i,'revenue'],

#                             'budget':combined.loc[i,'budget'],

#                            'year':combined.loc[i,'release_year'],

#                            'id':combined.loc[i,'imdb_id']},

#                            ignore_index = True)

#     return(df)

# df = get_names_from_dict('production_companies')



# try:

#     df['production_roi'] = df.revenue / df.budget

# except:

#     df['production_roi'] = 0.00

# df[['budget','revenue']].head().T



# df.groupby(['name',df.year // 10 * 10])['revenue'].mean().fillna(0).sort_values(ascending=False)

# We now have the information that the movies from a given production house in a given decade earned what revenues on average

# For movies with multiple production houses, we can take the average of revenues in that decade and create a proxy for it

# This gives us an idea of whether the production house is big or small

combined.head().T
combined.drop(['imdb_id','original_language','production_companies','status','Keywords','cast','crew','profit'] \

              ,axis=1,inplace=True)

combined1 = combined.copy()

combined1.loc[combined1.runtime.isna(),'runtime'] = combined1.runtime.mean()
print(combined1.loc[combined.release_year.isna(),:])

print(combined1.loc[combined.release_year.isna(),:])

print(combined1.loc[combined.release_year.isna(),:])
combined1.loc[combined1.release_year.isna(),'release_year'] = 2000

combined1.loc[-combined1.release_year.isna(),'release_month'] = 5

combined1.loc[combined1.release_day_of_week.isna(),'release_day_of_week'] = 4
train_final = combined1.loc[-combined1.revenue.isna()].copy()

test_final = combined1.loc[combined1.revenue.isna()].copy()

print(train_final.shape)

print(test_final.shape)
test_final.drop('revenue',axis=1,inplace=True)

y = train_final['revenue']

X = train_final.drop('revenue',axis=1,inplace=True)



print(test_final.shape)

print(X.shape)
from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = train_test_split(train_final, y, test_size=0.33, random_state=42)
print(X_train.shape)

print(X_holdout.shape)

print(y_train.head())

print(y_holdout.head())

print(train_final.shape)
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=17)



forest_params = {

#     'max_depth': range(10, 21,5)

'max_features': range(20, 120,10)

                }

forest_grid = GridSearchCV(forest, forest_params,

                           cv=4, n_jobs=-1, verbose=True,scoring='neg_mean_squared_error')

forest_grid.fit(X_train, y_train)

forest_grid.best_params_, forest_grid.best_score_
holdout_pred = forest_grid.predict(X_holdout)

# print(holdout_pred)

from sklearn.metrics import mean_squared_error

100*mean_squared_error(y_holdout, holdout_pred)
test_pred = forest_grid.predict(test_final)
pred = pd.read_csv('../input/sample_submission.csv')

pred['revenue'] = test_pred

pred.to_csv("RFR.csv", index=False)

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

# from sklearn import cross_validation, metrics   

from sklearn.model_selection import GridSearchCV





xgb_params = {

    'learning_rate': [.01,.02,.03,.04,.05,.1,.15,.22,.3]}



xgb1 = XGBRegressor(n_estimators=100, gamma=0, subsample=0.75,

                           colsample_bytree=1,eval_metric='rmse',objective= 'reg:linear',

                      seed=27)



xgb_grid = GridSearchCV(xgb1, xgb_params,

                           cv=5, n_jobs=-1, verbose=True,scoring='neg_mean_squared_error')

xgb_grid.fit(X_train, y_train)
print(xgb_grid.best_params_, xgb_grid.best_score_)

holdout_pred = xgb_grid.predict(X_holdout)

mean_squared_error(y_holdout, holdout_pred)



from sklearn.metrics import mean_squared_error

mean_squared_error(y_holdout, holdout_pred)

test_pred = xgb_grid.predict(test_final)

pred_XG = pd.read_csv('../input/sample_submission.csv')

pred['revenue'] = test_pred

pred.to_csv("XGBR.csv", index=False)
