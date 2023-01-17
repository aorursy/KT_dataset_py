import numpy as np
import pandas as pd

#!pip install pandas-profiling

import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn_pandas import DataFrameMapper
from IPython.display import Image

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
tmdb = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
tmdb_test = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

tmdb.head()
tmdb.columns
tmdb.shape
tmdb_test.head()
tmdb_test.columns
tmdb_test.shape
tmdb.describe(include='all')
tmdb.dtypes
tmdb_test.dtypes
import pandas_profiling

pandas_profiling.ProfileReport(tmdb)
pandas_profiling.ProfileReport(tmdb_test)
for col in ['id', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview','poster_path','release_date','status','tagline','title']:
    tmdb[col] = tmdb[col].astype('category')
for col in ['id','homepage', 'imdb_id', 'original_language', 'original_title', 'overview','poster_path','release_date','status','tagline','title']:
    tmdb_test[col] = tmdb_test[col].astype('category')
tmdb.dtypes
tmdb_test.dtypes
tmdb.isnull().sum()

tmdb_test.isnull().sum()
missing=tmdb.isna().sum().sort_values(ascending=False)
sns.barplot(missing[:8],missing[:8].index)
plt.show()

### Checking Unique Values for all attribute
tmdb.nunique()
### Checking Unique Values for all attribute
tmdb_test.nunique()
## Checking Unique values = 1 for all columns in given data
tmdb.columns[tmdb.nunique() <= 1]


#Checking columns with all unique values 
tmdb.columns[tmdb.nunique() == 3000]
## Checking Unique values = 1 for all columns in test data
tmdb_test.columns[tmdb_test.nunique() <= 1]
#Checking columns with all unique values in test data
tmdb_test.columns[tmdb_test.nunique() == 4398]
tmdb = tmdb.drop(['id','imdb_id'],axis = 1)
tmdb_test = tmdb_test.drop(['id','imdb_id'],axis=1)
tmdb = tmdb.drop(['poster_path','overview','homepage','tagline','original_title','title','original_language','status'],axis = 1)
tmdb_test = tmdb_test.drop(['poster_path','overview','homepage','tagline','original_title','title','original_language', 'status'],axis = 1)
tmdb.select_dtypes(include=[np.number]).columns
tmdb.select_dtypes(include=[np.number]).head()

tmdb.describe(include=[np.number]).head()
tmdb['revenue'].value_counts()
tmdb['budget'].value_counts()
fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(tmdb['revenue']);
plt.title('Distribution of revenue');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(tmdb['revenue']));
plt.title('Distribution of log of revenue');

fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(tmdb['budget']);
plt.title('Distribution of budget');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(tmdb['budget']));
plt.title('Distribution of log of budget');

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(
                x=tmdb['budget'],
                y=tmdb['revenue'],
                mode='markers',
                marker=dict(
                     color='rgb(255, 178, 102)',
                     size=10,
                     line=dict(
                        color='DarkSlateGrey',
                        width=1
                      )
               )
))
fig.update_layout(
    title='Revenue by Budget',
    xaxis_title='budget ($)',
    yaxis_title='revenue ($)'
)
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(
                x=tmdb['runtime'],
                y=tmdb['revenue'],
                mode='markers',
                marker=dict(
                     color='rgb(48, 105, 152)',
                     size=10,
                     line=dict(
                        color='DarkSlateGrey',
                        width=1
                      )
               )
))
fig.update_layout(
    title='Revenue by Runtime',
    xaxis_title='runtime (minutes)',
    yaxis_title='revenue ($)'
)
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(
                x=tmdb['popularity'],
                y=tmdb['revenue'],
                mode='markers',
                marker=dict(
                     color='rgb(108, 198, 68)',
                     size=10,
                     line=dict(
                        color='DarkSlateGrey',
                        width=1
                      )
               )
))
fig.update_layout(
    title='Revenue by Popularity',
    xaxis_title='popularity',
    yaxis_title='revenue ($)'
)
fig.show()
def plot_corr(tmdb,filename):
    plt.subplots(figsize=(12, 9))
    sns.heatmap(tmdb.corr(),annot=True,linewidths=.5,annot_kws={"fontsize":15})
    plt.yticks(rotation=0,fontsize=15)
    plt.xticks(rotation=0,fontsize=15)
    plt.show()

plot_corr(tmdb[["revenue","budget","popularity","runtime"]],filename="corr.png")
tmdb['log_budget'] = np.log1p(tmdb['budget'])
tmdb_test['log_budget'] = np.log1p(tmdb_test['budget'])
tmdb = tmdb.drop(['budget'],axis = 1)
tmdb_test = tmdb_test.drop(['budget'],axis = 1)
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

import ast
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
tmdb = text_to_dict(tmdb)
tmdb_test = text_to_dict(tmdb_test)
for i, e in enumerate(tmdb['genres'][:5]):
    print(i, e)

print('Number of genres in films')
tmdb['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_genres = list(tmdb['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
## Finding top Genre
#!pip install WordCloud

from wordcloud import WordCloud
from collections import Counter
plt.figure(figsize = (12, 8))
text = ' '.join([i for j in list_of_genres for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()

## Using Counter
Counter([i for j in list_of_genres for i in j]).most_common()
tmdb['genres_names'] = tmdb['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
tmdb_test['genres_names'] = tmdb_test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
### So now I can drop 'genres' column

tmdb = tmdb.drop(['genres'], axis=1)
tmdb_test = tmdb_test.drop(['genres'], axis=1)
tmdb.head()
tmdb_test.head()
for i, e in enumerate(tmdb['belongs_to_collection'][:5]):
    print(i, e)

print('Number of collections in films')
tmdb['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_collection_names = list(tmdb['belongs_to_collection'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in list_collection_names for i in j]).most_common(15)
tmdb['collections_names'] = tmdb['belongs_to_collection'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
tmdb_test['collections_names'] = tmdb_test['belongs_to_collection'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
### So now I can drop 'belongs_to_collection' column

tmdb = tmdb.drop(['belongs_to_collection'], axis=1)
tmdb_test = tmdb_test.drop(['belongs_to_collection'], axis=1)
tmdb.head()
for i, e in enumerate(tmdb['production_companies'][:5]):
    print(i, e)
print('Number of production companies in films')
tmdb['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_companies = list(tmdb['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in list_of_companies for i in j]).most_common(20)
tmdb['production_names'] = tmdb['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
tmdb_test['production_names'] = tmdb_test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
### So now I can drop 'production_companies' column

tmdb = tmdb.drop(['production_companies'], axis=1)
tmdb_test = tmdb_test.drop(['production_companies'], axis=1)
tmdb.head()
tmdb_test.head()
for i, e in enumerate(tmdb['production_countries'][:5]):
    print(i, e)
print('Number of production countries in films')
tmdb['production_countries'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_countries = list(tmdb['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in list_of_countries for i in j]).most_common(25)
tmdb['production_countries_names'] = tmdb['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
tmdb_test['production_countries_names'] = tmdb_test['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
### So now I can drop 'production_countries' column

tmdb = tmdb.drop(['production_countries'], axis=1)
tmdb_test = tmdb_test.drop(['production_countries'], axis=1)
tmdb.head()
for i, e in enumerate(tmdb['spoken_languages'][:5]):
    print(i, e)
print('Number ofspoken languages in films')
tmdb['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_Spoken_Languages = list(tmdb['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
plt.figure(figsize = (12, 8))
text = ' '.join([i for j in list_of_Spoken_Languages for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Spoken Languages')
plt.axis("off")
plt.show()
Counter([i for j in list_of_Spoken_Languages for i in j]).most_common(25)
tmdb['language_names'] = tmdb['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
tmdb_test['language_names'] = tmdb_test['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
### So now I can drop 'spoken_languages' column

tmdb = tmdb.drop(['spoken_languages'], axis=1)
tmdb_test = tmdb_test.drop(['spoken_languages'], axis=1)
tmdb.head()
for i, e in enumerate(tmdb['cast'][:5]):
    print(i, e)
print('Number of casts in films')
tmdb['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_cast = list(tmdb['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in list_of_cast for i in j]).most_common(25)
tmdb['cast_names'] = tmdb['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
tmdb_test['cast_names'] = tmdb_test['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
### So now I can drop 'cast' column

tmdb = tmdb.drop(['cast'], axis=1)
tmdb_test = tmdb_test.drop(['cast'], axis=1)
tmdb.head()
tmdb_test.head()
## Dropping Key Words and Crew as they do not contribute to revenue 
tmdb = tmdb.drop(['Keywords','crew'], axis=1)
tmdb_test =  tmdb_test.drop(['Keywords','crew'],axis=1)
tmdb.columns
##Feature engineering
tmdb['release_datetime'] = pd.to_datetime(tmdb['release_date'])
tmdb['release_day'] = tmdb['release_datetime'].dt.day
tmdb['release_month']=tmdb['release_datetime'].dt.month
tmdb['release_year'] = tmdb['release_datetime'].dt.year
tmdb['release_weekday']=tmdb['release_datetime'].dt.weekday
tmdb[['release_datetime', 'release_day', 'release_month', 'release_year', 'release_weekday']]
day=tmdb['release_weekday'].value_counts().sort_index()
sns.barplot(day.index,day)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='45')
plt.ylabel('No of releases')
tmdb.head()
tmdb = tmdb.drop(['release_date','release_datetime','release_day','release_month','release_year'], axis=1)
## same thing for test set
tmdb_test['release_datetime'] = pd.to_datetime(tmdb_test['release_date'])
tmdb_test['release_day'] = tmdb_test['release_datetime'].dt.day
tmdb_test['release_month']=tmdb_test['release_datetime'].dt.month
tmdb_test['release_year'] = tmdb_test['release_datetime'].dt.year
tmdb_test['release_weekday']=tmdb_test['release_datetime'].dt.weekday
tmdb_test[['release_datetime', 'release_day', 'release_month', 'release_year', 'release_weekday']]
tmdb_test = tmdb_test.drop(['release_date','release_datetime','release_day','release_month','release_year'], axis=1)
sns.catplot(x='release_weekday',y='revenue',data=tmdb)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')
plt.show()
tmdb.head()
tmdb.dtypes
tmdb_test.dtypes
for col in ['genres_names','collections_names','production_names','release_weekday']:
    tmdb[col] = tmdb[col].astype('category')
for col in ['genres_names','collections_names','production_names','release_weekday']:
    tmdb_test[col] = tmdb_test[col].astype('category')
tmdb.dtypes
tmdb_test.dtypes
top_genre = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
for g in top_genre:
    tmdb['genre_' + g] = tmdb['genres_names'].apply(lambda x: 1 if g in x else 0)
for g in top_genre:
    tmdb_test['language_' + g] = tmdb_test['genres_names'].apply(lambda x: 1 if g in x else 0)
tmdb = tmdb.drop(['genres_names'], axis=1)
tmdb_test = tmdb_test.drop(['genres_names'], axis=1)
top_collections = [m[0] for m in Counter([i for j in list_collection_names for i in j]).most_common(15)]
for g in top_collections:
    tmdb['collection_' + g] = tmdb['collections_names'].apply(lambda x: 1 if g in x else 0)
for g in top_collections:
    tmdb_test['collection_' + g] = tmdb_test['collections_names'].apply(lambda x: 1 if g in x else 0)
tmdb = tmdb.drop(['collections_names'], axis=1)
tmdb_test = tmdb_test.drop(['collections_names'], axis=1)
top_production_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]
for g in top_production_countries:
    tmdb['production_country_name' + g] = tmdb['production_countries_names'].apply(lambda x: 1 if g in x else 0)

for g in top_production_countries:
    tmdb_test['production_country_name' + g] = tmdb_test['production_countries_names'].apply(lambda x: 1 if g in x else 0)
    
tmdb = tmdb.drop(['production_countries_names'], axis=1)
tmdb_test = tmdb_test.drop(['production_countries_names'], axis=1)
top_productions = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(15)]
for g in top_collections:
    tmdb['productions_' + g] = tmdb['production_names'].apply(lambda x: 1 if g in x else 0)
for g in top_productions:
    tmdb_test['productions_' + g] = tmdb_test['production_names'].apply(lambda x: 1 if g in x else 0)
tmdb = tmdb.drop(['production_names'], axis=1)
tmdb_test = tmdb_test.drop(['production_names'], axis=1)
top_languages = [m[0] for m in Counter([i for j in list_of_Spoken_Languages for i in j]).most_common(30)]
for g in top_languages:
    tmdb['language_' + g] = tmdb['language_names'].apply(lambda x: 1 if g in x else 0)
for g in top_languages:
    tmdb_test['language_' + g] = tmdb_test['language_names'].apply(lambda x: 1 if g in x else 0)
    
tmdb = tmdb.drop(['language_names'], axis=1)
tmdb_test = tmdb_test.drop(['language_names'], axis=1)
top_cast = [m[0] for m in Counter([i for j in list_of_cast for i in j]).most_common(30)]
for g in top_cast:
    tmdb['cast_' + g] = tmdb['cast_names'].apply(lambda x: 1 if g in x else 0)
for g in top_cast:
    tmdb_test['cast_' + g] = tmdb_test['cast_names'].apply(lambda x: 1 if g in x else 0)
    
tmdb = tmdb.drop(['cast_names'], axis=1)
tmdb_test = tmdb_test.drop(['cast_names'], axis=1)
pd.set_option('display.max_columns', None)

tmdb.head()
tmdb.shape
tmdb_test.head()
tmdb_test.head()
tmdb_test.dtypes
tmdb_test.shape
tmdb_test.isna().sum()
tmdb_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X = tmdb.drop(['revenue'], axis=1)
y = np.log1p(tmdb['revenue'])
X_test = tmdb_test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20,random_state=123)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
num_attr=X_train.select_dtypes(['int64','float64']).columns
num_attr
cat_attr = X_train.select_dtypes('category').columns
cat_attr
num_attr_test = X_test.select_dtypes(['int64','float64']).columns
num_attr_test
cat_attr_test = X_test.select_dtypes('category').columns
cat_attr_test
imputer = SimpleImputer(strategy='mean')

imputer = imputer.fit(X_train[num_attr])

X_train[num_attr] = imputer.transform(X_train[num_attr])
X_val[num_attr] = imputer.transform(X_val[num_attr])


X_test[num_attr_test] = imputer.transform(X_test[num_attr_test])

print(X_train.isnull().sum())
print(X_val.isnull().sum())
imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(X_train[cat_attr])

X_train[cat_attr] = imputer.transform(X_train[cat_attr])
X_val[cat_attr] = imputer.transform(X_val[cat_attr])

X_test[cat_attr_test] = imputer.transform(X_test[cat_attr_test])
# DataFrameMapper, a class for mapping pandas data frame columns to different sklearn transformations
mapper = DataFrameMapper(
  [([continuous_col], StandardScaler()) for continuous_col in num_attr] +
  [([categorical_col], OneHotEncoder(handle_unknown='error')) for categorical_col in cat_attr]
, df_out=True)
print(type(mapper))
mapper.fit(X_train)

X_train_final = mapper.transform(X_train)
X_val_final = mapper.transform(X_val)
# DataFrameMapper, a class for mapping pandas data frame columns to different sklearn transformations
mapper_test = DataFrameMapper(
  [([continuous_col], StandardScaler()) for continuous_col in num_attr_test] +
  [([categorical_col], OneHotEncoder(handle_unknown='error')) for categorical_col in cat_attr_test]
, df_out=True)
print(type(mapper_test))
mapper_test.fit(X_test)

X_test_final = mapper_test.transform(X_test)

X_test_final.head()
X_train_final.head()
X_train_final.columns
X_val_final.head()
X_val_final.columns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Model initialization
regression_model = LinearRegression()


# Fit the data(train the model)
regression_model.fit(X_train_final, y_train)
##Our model has now been trained. You can analyse each of the model’s coefficients using the following statement :
print(regression_model.coef_)
#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
pd.DataFrame(regression_model.coef_, X_train_final.columns, columns = ['Coeff'])
# Predict
predictions = regression_model.predict(X_train_final)
from sklearn import metrics
rmse = mean_squared_error(y_train,predictions)
r2 = r2_score(y_train,predictions)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)
X_test_final = X_test_final.replace([np.inf, -np.inf], 0).fillna(0)
predictions_test = np.expm1(regression_model.predict(X_test_final))
submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
submission['revenue'] = np.round(predictions_test)
submission.to_csv('submission_linear_regression.csv', index = False)
def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

def print_rf_score(model):
    print(f'Train R2:   {model.score(X_train_final, y_train)}')
    print(f'Valid R2:   {model.score(X_val_final, y_val)}')
    print(f'Train RMSE: {rmse(model.predict(X_train_final), y_train)}')
    print(f'Valid RMSE: {rmse(model.predict(X_val_final), y_val)}')
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 40, random_state = 25)
rf.fit(X_train_final,y_train)

print_rf_score(rf)
rf= RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)
rf.fit(X_train_final, y_train)

rf.fit(X_train_final, y_train)
print_rf_score(rf)
print(f'OOB Score:  {rf.oob_score_}')
feature_importances = pd.DataFrame(rf.feature_importances_, index = X_train_final.columns, columns=['importance'])
feature_importances
feature_importances = pd.DataFrame(rf.feature_importances_ , index = X_train_final.columns, columns=['importance'])
feature_importances = feature_importances.sort_values('importance', ascending=True)
feature_importances.plot(kind = 'barh', figsize = (15,60))
plt.show()
predictions_test = np.expm1(rf.predict(X_test_final))
submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
submission['revenue'] = np.round(predictions_test)
submission.to_csv('submission_simple_rf.csv', index = False)