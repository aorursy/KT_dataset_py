import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import warnings

warnings.filterwarnings("ignore")
data_cr = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

data_cr.columns = ['id', 'title', 'cast', 'crew']

data_mv = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

data = data_mv.merge(data_cr, on='id', suffixes=['', '_y'])

print(data.shape)

data.head(n=2)
data_explore = data.copy()
drop_cols = ['id', 'homepage', 'spoken_languages', 'overview', 'tagline', 'title', 'title_y']

data_explore = data_explore.drop(columns=drop_cols, axis=1)

data_explore.head(n=3)
data_explore.info()
data_explore['release_date'] = data_explore['release_date'].fillna('2000-01-01')

data_explore['release_year'] = pd.DatetimeIndex(data_explore['release_date']).year

data_explore['release_month'] = pd.DatetimeIndex(data_explore['release_date']).month

data_explore['release_day'] = pd.DatetimeIndex(data_explore['release_date']).day

data_explore['release_dow'] = pd.DatetimeIndex(data_explore['release_date']).dayofweek

data_explore = data_explore.drop('release_date', axis=1)
data_explore.head(n=3)
nan_cols = data_explore.isna().sum()

nan_cols[nan_cols>0]
data_explore['runtime'] = data_explore['runtime'].fillna(data_explore['runtime'].median()) 
data_explore.describe()
sum(data_explore['revenue']==0)
data_explore = data_explore[data_explore['revenue']!=0]

data_explore.shape
import ast

dict_cols = ['genres', 'production_companies', 'production_countries', 'keywords', 'cast', 'crew']



for col in dict_cols:

    data_explore[col] = data_explore[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
def get_names(x, col):

    """

        Get the name field from each JSON object.

        For crew field, considering the Director only.

        For cast field, considering the first 3 cast members. Generally they are the main roles from movie.

    """

    names = []

    for item in x:

        if col=='crew':

            if item['job']=='Director':

                names.append(item['name'])

        elif col=='cast':

            if item['order'] in (0, 1, 2):

                names.append(item['name'])

        else:

            names.append(item['name'])

    return names

    

for col in dict_cols:

    data_explore[col] = data_explore[col].apply(lambda x: get_names(x, col))
data_explore.head(n=3)
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

outliers = ((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()

outliers[outliers>0]
data_explore.hist(figsize=(15, 15))

plt.show()
most_popular_movies = data_explore.sort_values('popularity', ascending=False).head(n=20)

most_popular_movies['revenue(million)'] = most_popular_movies['revenue'].apply(lambda x : x//1000000)    # revenue in millions

most_popular_movies['budget(million)'] = most_popular_movies['budget'].apply(lambda x : x//1000000)    # revenue in millions

most_popular_movies[['genres', 'original_title', 'production_companies', 'popularity', 'cast', 'crew', 'vote_average', 'budget(million)', 'revenue(million)']]
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='popularity', data=most_popular_movies, order=most_popular_movies.sort_values('popularity', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(int(p.get_width())), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Top 20 Most Popular Movies', fontsize=12)

plt.ylabel('')

plt.show()
highest_revenue_movies = data_explore.sort_values('revenue', ascending=False).head(n=20)

highest_revenue_movies['revenue(million)'] = highest_revenue_movies['revenue'].apply(lambda x : x//1000000)    # revenue in millions

highest_revenue_movies['budget(million)'] = highest_revenue_movies['budget'].apply(lambda x : x//1000000)    # revenue in millions

highest_revenue_movies[['genres', 'original_title', 'production_companies', 'popularity', 'cast', 'crew', 'vote_average', 'budget(million)', 'revenue(million)']]
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='revenue(million)', data=highest_revenue_movies, order=highest_revenue_movies.sort_values('revenue(million)', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(int(p.get_width())), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Top 20 High Revenue(million) Movies', fontsize=12)

plt.ylabel('')

plt.show()
highest_budget_movies = data_explore.sort_values('budget', ascending=False).head(n=20)

highest_budget_movies['revenue(million)'] = highest_budget_movies['revenue'].apply(lambda x : x//1000000)    # revenue in millions

highest_budget_movies['budget(million)'] = highest_budget_movies['budget'].apply(lambda x : x//1000000)    # revenue in millions

highest_budget_movies[['genres', 'original_title', 'production_companies', 'popularity', 'cast', 'crew', 'vote_average', 'budget(million)', 'revenue(million)']]
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='budget(million)', data=highest_budget_movies, order=highest_budget_movies.sort_values('budget(million)', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(int(p.get_width())), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Top 20 High Budget(million) Movies', fontsize=12)

plt.ylabel('')

plt.show()
most_profit_movies = data_explore.copy()

most_profit_movies['revenue(million)'] = most_profit_movies['revenue'].apply(lambda x : x//1000000)    # revenue in millions

most_profit_movies['budget(million)'] = most_profit_movies['budget'].apply(lambda x : x//1000000)    # revenue in millions

most_profit_movies['profit(million)'] = most_profit_movies['revenue(million)']-most_profit_movies['budget(million)']

most_profit_movies = most_profit_movies.sort_values('profit(million)', ascending=False).head(n=20)

most_profit_movies[['genres', 'original_title', 'production_companies', 'popularity', 'cast', 'crew', 'vote_average', 'budget(million)', 'revenue(million)', 'profit(million)']]
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='profit(million)', data=most_profit_movies, order=most_profit_movies.sort_values('profit(million)', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(int(p.get_width())), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Top 20 Highest Grossing Movies', fontsize=12)

plt.ylabel('')

plt.show()
highest_rated_movies = data_explore.copy()

highest_rated_movies['revenue(million)'] = highest_rated_movies['revenue'].apply(lambda x : x//1000000)    # revenue in millions

highest_rated_movies['budget(million)'] = highest_rated_movies['budget'].apply(lambda x : x//1000000)    # revenue in millions

highest_rated_movies = highest_rated_movies.sort_values('vote_average', ascending=False).head(n=20)

highest_rated_movies[['genres', 'original_title', 'production_companies', 'popularity', 'cast', 'crew', 'vote_average', 'budget(million)', 'revenue(million)']]
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='vote_average', data=highest_rated_movies, order=highest_rated_movies.sort_values('vote_average', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(p.get_width()), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Top 20 Highest Rated Movies', fontsize=12)

plt.ylabel('')

plt.xlim(left=5)

plt.show()
plt.figure(figsize=(12, 10))

ax = sns.barplot(y='original_title', x='revenue(million)', data=highest_rated_movies, order=highest_rated_movies.sort_values('revenue(million)', ascending=False).original_title, orient='h')

for p in ax.patches:

        ax.annotate('{}'.format(p.get_width()), (p.get_width(), p.get_y()+0.5), fontsize=12)

plt.title('Revenue of Top 20 Highest Rated Movies', fontsize=12)

plt.ylabel('')

plt.show()
data_explore_enc = data_explore['genres'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')

data_explore_genres = pd.concat([data_explore, data_explore_enc], axis=1)

genres = data_explore_enc.columns

data_explore_genres.head(n=3)
genres_info = []

for col in genres:

    total_movies, total_budget, median_budget, total_revenue, median_revenue, median_popularity=0, 0, 0, 0, 0, 0

    total_movies = data_explore_genres[data_explore_genres[col]==1][col].count()

    total_budget = data_explore_genres[data_explore_genres[col]==1]['budget'].sum()

    median_budget = data_explore_genres[data_explore_genres[col]==1]['budget'].median()

    total_revenue = data_explore_genres[data_explore_genres[col]==1]['revenue'].sum()

    median_revenue = data_explore_genres[data_explore_genres[col]==1]['revenue'].median()

    median_popularity = data_explore_genres[data_explore_genres[col]==1]['popularity'].median()

    genres_info.append([col, total_movies, total_budget, median_budget, total_revenue, median_revenue, median_popularity])
genres_info = pd.DataFrame(genres_info, columns=['genres', 'movies_count', 'total_budget', 'median_budget', 'total_revenue', 'median_revenue', 'median_popularity'])

genres_info['total_budget(million)'] = genres_info['total_budget'].apply(lambda x : x//1000000)    # budget in millions

genres_info['median_budget(million)'] = genres_info['median_budget'].apply(lambda x : x//1000000)    # budget in millions

genres_info['total_revenue(million)'] = genres_info['total_revenue'].apply(lambda x : x//1000000)    # revenue in millions

genres_info['median_revenue(million)'] = genres_info['median_revenue'].apply(lambda x : x//1000000)    # revenue in millions

genres_info[['genres', 'movies_count', 'total_budget(million)', 'median_budget(million)', 'total_revenue(million)', 'median_revenue(million)', 'median_popularity']]
plt.figure(figsize=(15, 5))

ax = sns.barplot(x='genres', y='movies_count', data=genres_info, order=genres_info.sort_values('movies_count', ascending=False).genres)

for p in ax.patches:

        ax.annotate('{}'.format(int(p.get_height())), (p.get_x()+0.1, p.get_height()+10))

plt.xticks(rotation=45)

plt.ylabel('# of Movies', fontsize=12)

plt.xlabel('Genres', fontsize=12)

plt.show()
plt.figure(figsize=(15, 5))

ax = sns.barplot(x='genres', y='median_popularity', data=genres_info, order=genres_info.sort_values('median_popularity', ascending=False).genres)

for p in ax.patches:

        ax.annotate('{}'.format(np.round(p.get_height(), 2)), (p.get_x()+0.1, p.get_height()))

plt.xticks(rotation=45)

plt.ylabel('Median Popularity', fontsize=12)

plt.xlabel('Genres', fontsize=12)

plt.show()
plt.figure(figsize=(15, 6))

x_indexes = np.arange(len(genres))     

width = 0.35                            

genres_info = genres_info.sort_values('total_revenue(million)', ascending=False)

plt.bar(x_indexes,  genres_info['total_revenue(million)'], label="Total Movies Revenue", width=width)

plt.bar(x_indexes + width,  genres_info['total_budget(million)'], label="Total Movies Budget", width=width)

plt.legend(loc="upper right", fontsize=12)

plt.xticks(ticks=x_indexes+0.5, labels=genres_info['genres'].values, fontsize=12, rotation=-45)

plt.xlabel('Genres', fontsize=12)

plt.ylabel('Sum value(million)', fontsize=12)

plt.show()
plt.figure(figsize=(15, 6))

x_indexes = np.arange(len(genres))     

width = 0.35                            

genres_info = genres_info.sort_values('median_revenue(million)', ascending=False)

plt.bar(x_indexes,  genres_info['median_revenue(million)'], label="Median Movies Revenue", width=width)

plt.bar(x_indexes + width,  genres_info['median_budget(million)'], label="Median Movies Budget", width=width)

plt.legend(loc="upper right", fontsize=12)

plt.xticks(ticks=x_indexes+0.5, labels=genres_info['genres'].values, fontsize=12, rotation=-45)

plt.xlabel('Genres', fontsize=12)

plt.ylabel('Median value(million)', fontsize=12)

plt.show()
from wordcloud import WordCloud
plt.figure(figsize = (12, 12))

text = ' '.join(data_explore['original_title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Words in Movie Titles', fontsize=14)

plt.axis("off")

plt.show()
list_of_genres = list(data_explore['genres'].values)

plt.figure(figsize = (12, 12))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_genres for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False, width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Genres', fontsize=14)

plt.axis("off")

plt.show()
list_of_keywords = list(data_explore['keywords'].values)

plt.figure(figsize = (12, 12))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False, width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Keywords', fontsize=14)

plt.axis("off")

plt.show()
plt.figure(figsize=(7, 7))

sns.boxplot(x='revenue', data=data_explore, orient='v')

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_ylim(0, 300000000)

ax.set_title('Distribution of Revenue')
plt.figure(figsize=(10, 6))

sns.scatterplot(x='budget', y='revenue', data=data_explore)

ax = plt.gca()

ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Revenue Vs. Budget')

plt.show()
plt.figure(figsize=(10, 6))

sns.scatterplot(x='popularity', y='revenue', data=data_explore)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Revenue Vs. Popularity')

plt.xlim(0, 60)

plt.show()
plt.figure(figsize=(15, 5))

sns.scatterplot(x='runtime', y='revenue', data=data_explore)

plt.xticks(rotation=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Revenue Vs. Movie Runtime')

plt.xlim(50, 200)

plt.show()
plt.figure(figsize=(15, 5))

sns.barplot(x='release_year', y='revenue', data=data_explore, estimator=np.mean)

plt.xticks(rotation=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Avg. Revenue Each Year (Since 1960)')

ax.set_xlim(left=31.5)
plt.figure(figsize=(15, 5))

sns.barplot(x='release_month', y='revenue', data=data_explore, estimator=np.mean)

plt.xticks(rotation=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Avg. Revenue Each Month')
plt.figure(figsize=(15, 5))

sns.barplot(x='release_dow', y='revenue', data=data_explore, estimator=np.mean)

plt.xticks(rotation=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Avg. Revenue on Each Day of Week')
plt.figure(figsize=(15, 6))

sns.boxplot(x='original_language', y='revenue', data=data_explore)

# plt.xticks(rotation=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_ylim(top=400000000)

ax.set_xlim(right=15.5)
def boxplot_sorted(df, by, column):

    df_boxplot = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})

    meds = df_boxplot.mean().sort_values(ascending=False)

    return df_boxplot[meds.index].boxplot()
plt.figure(figsize=(15, 7))

axes = boxplot_sorted(data_explore, by = ['original_language'], column = 'revenue')

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_ylim((-10000000, 300000000))

ax.set_xlim(right=10.5)

plt.xlabel('Orignal Language')

plt.ylabel('Revenue')

plt.title('Distribution of Revenue for Top 10 Movies(by Average Revenue)')

plt.show()
plt.figure(figsize=(15, 7))

axes = boxplot_sorted(data_explore, by = ['release_year'], column = 'revenue')

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_ylim((-10000000, 400000000))

ax.set_xlim(right=20.5)

plt.xlabel('Release Year')

plt.ylabel('Revenue')

plt.title('Distribution of Revenue for Top 20 Years (by Average Revenue)')

plt.show()
plt.figure(figsize=(12, 7))

corr_matrix = data_explore.corr()

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)

plt.tight_layout()
corr_matrix['revenue'].sort_values(ascending=False)
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MultiLabelBinarizer
data = data[data['revenue']!=0]
X = data.drop(columns=['revenue'], axis=1).copy()

y = data['revenue'].copy()

X.shape, y.shape
from collections import Counter

top_30_values = dict()



list_of_genres_names = list(X['genres'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_30_genres = (Counter([i for j in list_of_genres_names for i in j]).most_common(30))

top_30_values['genres'] = [x for x, y in top_30_genres]



list_of_production_companies_names = list(X['production_companies'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_30_production_companies = (Counter([i for j in list_of_production_companies_names for i in j]).most_common(30))

top_30_values['production_companies'] = [x for x, y in top_30_production_companies]



list_of_production_countries_names = list(X['production_countries'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_30_production_countries = (Counter([i for j in list_of_production_countries_names for i in j]).most_common(30))

top_30_values['production_countries'] = [x for x, y in top_30_production_countries]



list_of_keywords = list(X['keywords'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_30_keywords = (Counter([i for j in list_of_keywords for i in j]).most_common(30))

top_30_values['keywords'] = [x for x, y in top_30_keywords]



list_of_cast_names = list(X['cast'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x if i['order'] in (0, 1, 2)] if x != {} else []).values)

top_30_cast = (Counter([i for j in list_of_cast_names for i in j]).most_common(30))

top_30_values['cast'] = [x for x, y in top_30_cast]



list_of_crew_names = list(X['crew'].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x)).apply(lambda x: [i['name'] for i in x if i['job'] in ('Director', 'Producer', 'Writer')] if x != {} else []).values)

top_30_crew = (Counter([i for j in list_of_crew_names for i in j]).most_common(30))

top_30_values['crew'] = [x for x, y in top_30_crew]
drop_cols = ['id', 'homepage', 'original_title', 'spoken_languages', 'overview', 'tagline', 'title', 'title_y']
class CustomAttr(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        try:

            

            X['release_date']= X['release_date'].fillna('2000-01-01')

            

            X['release_year'] = pd.DatetimeIndex(X['release_date']).year

#             print("release_year attribute added!")

            

            X['release_month'] = pd.DatetimeIndex(X['release_date']).month

#             print("release_month attribute added!")

            

            X['release_day'] = pd.DatetimeIndex(X['release_date']).day

#             print("release_day attribute added!")

            

            X['release_dow'] = pd.DatetimeIndex(X['release_date']).dayofweek

#             print("release_dow attribute added!")

            

            X = X.drop(['release_date'], axis=1)

#             print("belongs_to_collection, release_date attribute removed!")

            return X

        except Exception as e:

            print("CustomAttr: Exception caught: {}".format(e))
encoded_cols = [] # This will contain all the encoded column names of multivalued field



class JSONHandler(BaseEstimator, TransformerMixin):

    def __init__(self):

        """

        For each multivalued field, there will be a MultiLabelBinarizer.

        """

        self.mlbs = dict()

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        for col in list(X.columns):

            try:

                X[col] = X[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))

                X[col] = X[col].apply(lambda x: self.get_names(x, col))

                if not (col in self.mlbs.keys()):

                    self.mlbs[col] = MultiLabelBinarizer()

                    X_enc = pd.DataFrame(self.mlbs[col].fit_transform(X[col]),columns=self.mlbs[col].classes_, 

                                         index=X.index)

                    encoded_cols.extend(list(self.mlbs[col].classes_))

                else:

                    X_enc = pd.DataFrame(self.mlbs[col].transform(X[col]),columns=self.mlbs[col].classes_, 

                                         index=X.index)

                X = X.drop(col, axis=1)

                X = pd.concat([X, X_enc], axis=1)

            except Exception as e:

                print("JSONHandler: Exception caught for {}: {}".format(col,e))

        return X

       



    @staticmethod

    def get_names(x, col):

        """

            Get the name field value from JSON object.

        """

        names = []

        try:

            names = [item['name'] for item in x if item['name'] in top_30_values[col]]

            if len(names)==0:

                names.append('other_'+col)

            return names

        except Exception as e:

            print("JSONHandler: get_names() function -  exception caught {}: {}".format(x,e))
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),

                        ('scaler', PowerTransformer())])



cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),

                         ('cat_enc', OneHotEncoder(handle_unknown='ignore'))])



pre_process = ColumnTransformer([('drop_cols', 'drop', drop_cols),

                                 ('num_process', num_pipeline, ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']),

                                 ('add_custom_attrs', CustomAttr(), ['release_date']),

                                 ('cat_process', cat_pipeline, ['original_language', 'status']),

                                 ('jason_handler', JSONHandler(), ['genres', 'production_companies', 'production_countries', 'keywords', 'cast', 'crew'])], remainder='passthrough')



X_transformed = pre_process.fit_transform(X)

X_transformed.shape
from sklearn.model_selection import train_test_split



X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed, np.log1p(y), test_size=0.2, random_state=42)

X_train_transformed.shape, X_test_transformed.shape
feature_columns = ['budget', 'popularity', 'runtime', 'release_year', 'release_month', 'release_day', 'release_dow', 'vote_average', 'vote_count'] + list(pre_process.transformers_[3][1]['cat_enc'].get_feature_names(['original_language', 'status'])) + encoded_cols

len(feature_columns)
from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.model_selection import cross_val_score



results = []



def performance_measures(model, store_results=True):    

    train_rmses = cross_val_score(model, X_train_transformed, y_train, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)

    train_rmses *= -1

    train_mean_rmse = np.mean(train_rmses)

    

    test_rmses = cross_val_score(model, X_test_transformed, y_test, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)

    test_rmses *= -1

    test_mean_rmse = np.mean(test_rmses)

    

    print("Train Mean RMSE: {}\nTest Mean RMSE: {}".format(train_mean_rmse, test_mean_rmse))

    

    if store_results:

        results.append([model.__class__.__name__, train_mean_rmse, test_mean_rmse])
def plot_feature_importance(feature_columns, importance_values, top_n_features=15):

    feature_imp = [ col for col in zip(feature_columns, importance_values)]

    feature_imp.sort(key=lambda x:x[1], reverse=True)



    if top_n_features:

        imp = pd.DataFrame(feature_imp[:top_n_features], columns=['feature', 'importance'])

    else:

        imp = pd.DataFrame(feature_imp, columns=['feature', 'importance'])

        

    plt.figure(figsize=(10, 8))

    sns.barplot(y='feature', x='importance', data=imp, orient='h')

    plt.title('Most Important Features', fontsize=16)

    plt.ylabel("Feature", fontsize=16)

    plt.xlabel("")

    plt.show()
from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=100, random_state=42)

ridge_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, ridge_reg.coef_)
performance_measures(ridge_reg) 
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=300, max_depth=16, max_features=0.2, n_jobs=-1, random_state=42)

forest_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, forest_reg.feature_importances_)
performance_measures(forest_reg)
from xgboost import XGBRegressor



xgb_reg = XGBRegressor(objective='reg:squarederror', n_estimators = 600, max_depth = 8, learning_rate = 0.01, 

                       subsample = 0.5, colsample_bytree = 0.5, colsample_bylevel = 0.5, random_state=42, n_jobs=-1)

xgb_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, xgb_reg.feature_importances_)
performance_measures(xgb_reg)
from catboost import CatBoostRegressor



cat_boost_reg = CatBoostRegressor(loss_function='RMSE', bagging_temperature = 0.3, colsample_bylevel = 0.7, 

                                  depth = 9, eval_metric = 'RMSE', iterations = 1000, random_state=42, verbose=0)

cat_boost_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, cat_boost_reg.feature_importances_)
performance_measures(cat_boost_reg)
from lightgbm import LGBMRegressor



lgb_reg = LGBMRegressor(objective = 'regression', num_iterations = 500, max_depth = 10, learning_rate= 0.02, 

                        metric = 'rmse', colsample_bytree= 0.5, subsample_freq= 1, subsample= 0.5, n_jobs=-1, 

                        random_state=42)

lgb_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, lgb_reg.feature_importances_)
performance_measures(lgb_reg)
from sklearn.ensemble import VotingRegressor



named_estimators = [('cat_bost', cat_boost_reg), ('xgb_reg', xgb_reg), ('lgb_reg', lgb_reg), ('forest_reg', forest_reg), ('ridge_reg', ridge_reg)]



voting_reg = VotingRegressor(estimators=named_estimators, n_jobs=-1)

voting_reg.fit(X_train_transformed, y_train)
performance_measures(voting_reg)
pd.DataFrame(results, columns=['Model', 'Train RMSLE', 'Test RMSLE'])
predicted_revenue = voting_reg.predict(X_transformed)

overall_data = X.copy()

overall_data['revenue'] = y.copy()

overall_data['predicted_revenue'] = np.expm1(predicted_revenue)
overall_data.head(n=2)
plt.figure(figsize=(10, 5))

sns.scatterplot(x='budget', y='revenue', data=overall_data, label='Observed')

sns.scatterplot(x='budget', y='predicted_revenue', data=overall_data, color='red', label='Predicted')

plt.ylim(0, 750000000)

plt.xlim(0, 250000000)

plt.xlabel('Budget', fontsize=14)

plt.ylabel('Revenue', fontsize=14)

ax = plt.gca()

ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Revenue Vs. Budget')

plt.show()
plt.figure(figsize=(10, 5))

sns.scatterplot(x='popularity', y='revenue', data=overall_data, label='Observed')

sns.scatterplot(x='popularity', y='predicted_revenue', data=overall_data, color='red', label='Predicted')

plt.ylim(0, 500000000)

plt.xlim(0, 50)

plt.xlabel('Popularity', fontsize=14)

plt.ylabel('Revenue', fontsize=14)

ax = plt.gca()

ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.get_yaxis().get_major_formatter().set_scientific(False)

ax.set_title('Revenue Vs. Popularity')

plt.show()
year_info = overall_data[['release_date', 'revenue', 'predicted_revenue']].copy()

year_info['release_year'] = pd.DatetimeIndex(year_info['release_date']).year

year_info = year_info.groupby(['release_year']).median()

year_info = year_info.sort_values('release_year')



release_years = list(year_info.index)

x_indexes = np.arange(len(release_years))     

width = 0.25                            



plt.figure(figsize=(10, 5))

plt.bar(x_indexes,  year_info['revenue'], label="Median Observed Movies Revenue", width=width)

plt.bar(x_indexes + width,  year_info['predicted_revenue'], label="Median Predicted Movies Revenue", width=width)

plt.legend(loc="upper left", fontsize=12)

plt.xticks(ticks=x_indexes+0.5, labels=release_years, fontsize=12, rotation=-45)

plt.title('1990-2017')

plt.xlabel('Release Year', fontsize=12)

plt.ylabel('Revenue', fontsize=12)

plt.xlim(left=61.5, right=90)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.show()
x_indexes = np.arange(len(release_years))     

width = 0.35                            



plt.figure(figsize=(10, 5))

plt.bar(x_indexes,  year_info['revenue'], label="Median Observed Movies Revenue", width=width)

plt.bar(x_indexes + width,  year_info['predicted_revenue'], label="Median Predicted Movies Revenue", width=width)

plt.legend(loc="upper left", fontsize=12)

plt.xticks(ticks=x_indexes+0.5, labels=release_years, fontsize=12, rotation=-45)

plt.title('1960-1989')

plt.xlabel('Release Year', fontsize=12)

plt.ylabel('Revenue', fontsize=12)

plt.xlim(left=32, right=62)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.show()