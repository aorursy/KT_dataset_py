import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline, make_union

import seaborn as sns

import matplotlib.pyplot as plt



import random

random.seed(42)
credits = pd.read_csv('../input/tmdb_5000_credits.csv', index_col='movie_id')

movies = pd.read_csv('../input/tmdb_5000_movies.csv', index_col='id')



data = pd.merge(movies, credits)

print(data.shape)



data = data.loc[data['revenue'] != 0]

data['revenue'].dropna(inplace=True)

print(data.shape)





X_train, X_test, y_train, y_test = train_test_split(data.drop(['revenue'], axis=1), data['revenue']) 



Exploratory = X_train.copy() # I'm using the copy of the data (not the view!) just in case, not to mess with the original dataset.
nan_percent = Exploratory.isna().mean()*100

nan_count = Exploratory.isna().sum()

pd.concat([nan_count.rename('missing_count'), nan_percent.round().rename('missing_percent')], axis=1)
columns_to_drop = ['original_title', 'overview', 'tagline', 'title']

# original_title/title - not informative

# overview/tagline - similar features may be found in 'keywords'



Exploratory = Exploratory.drop(columns_to_drop, axis=1)
dtypes_description = pd.Series(['ratio', 'nominal', 'nominal', 'nominal', 'nominal', 'ratio', 'nominal', 'nominal', \

                     'interval', 'ratio', 'nominal', 'nominal', 'ratio', 'ratio', 'nominal', 'nominal'], \

                     index=Exploratory.dtypes.index)



pd.concat([Exploratory.dtypes.rename('dtype'), Exploratory.iloc[420].rename('example'), dtypes_description.rename('description')], axis=1)
Exploratory[['genres', 'spoken_languages', 'crew']].head()
Exploratory[['homepage', 'original_language', 'status']].head()
Exploratory['cast'].head().to_frame()
Exploratory['release_date'].head().to_frame()
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureSelector(BaseEstimator, TransformerMixin):



    def __init__(self, feature_names):

        self.feature_names = feature_names

        

    def fit(self, X, y=None):

        return self

        

    def transform(self, X):

        return X[self.feature_names]
prod_companies = FeatureSelector('production_companies').fit_transform(Exploratory)

prod_companies.to_frame().head()
from sklearn.feature_extraction.text import CountVectorizer

import re



def extract_items(list_, key, all_=True):

    sub = lambda x: re.sub(r'[^A-Za-z0-9]', '_', x)

    if all_:

        target = []

        for dict_ in eval(list_):

            target.append(sub(dict_[key].strip()))

        return ' '.join(target)

    elif not eval(list_):

        return 'no_data'

    else:

        return sub(eval(list_)[0][key].strip())



class DictionaryVectorizer(BaseEstimator, TransformerMixin):

    

    def __init__(self, key, all_=True):

        self.key = key

        self.all = all_

    

    def fit(self, X, y=None):

        genres = X.apply(lambda x: extract_items(x, self.key, self.all))

        self.vectorizer = CountVectorizer().fit(genres)        

        self.columns = self.vectorizer.get_feature_names()

        return self

        

    def transform(self, X):

        genres = X.apply(lambda x: extract_items(x, self.key))

        data = self.vectorizer.transform(genres)

        return pd.DataFrame(data.toarray(), columns=self.vectorizer.get_feature_names(), index=X.index)
prod_companies_vectorized = DictionaryVectorizer('name').fit_transform(prod_companies)

prod_companies_vectorized.head()
class TopFeatures(BaseEstimator, TransformerMixin):

    

    def __init__(self, percent):

        if percent > 100:

            self.percent = 100

        else:

            self.percent = percent

    

    def fit(self, X, y=None):

        counts = X.sum().sort_values(ascending=False)

        index_ = int(counts.shape[0]*self.percent/100)

        self.columns = counts[:index_].index

        return self

    

    def transform(self, X):

        return X[self.columns]
top_companies = TopFeatures(1).fit_transform(prod_companies_vectorized)

top_companies.head()
class SumTransformer(BaseEstimator, TransformerMixin):

    

    def __init__(self, series_name):

        self.series_name = series_name

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X):

        return X.sum(axis=1).to_frame(self.series_name)
companies_count = SumTransformer('companies_count').fit_transform(prod_companies_vectorized)

companies_count.head()
class Binarizer(BaseEstimator, TransformerMixin):

    

    def __init__(self, condition, name):

        self.condition = condition

        self.name = name

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X.apply(lambda x : int(self.condition(x))).to_frame(self.name)
missing_homepage = Binarizer(lambda x: isinstance(x, float), 'missing_homepage').fit_transform(Exploratory['homepage'])

missing_homepage.head(15)
from datetime import datetime



def get_year(date):

    return datetime.strptime(date, '%Y-%m-%d').year



def get_month(date):

    return datetime.strptime(date, '%Y-%m-%d').strftime('%b')



def get_weekday(date):

    return datetime.strptime(date, '%Y-%m-%d').strftime('%a')



class DateTransformer(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        year = X.apply(get_year).rename('year')

        month = pd.get_dummies(X.apply(get_month))

        day = pd.get_dummies(X.apply(get_weekday))

        return pd.concat([year, month, day], axis=1)        
date = DateTransformer().fit_transform(Exploratory['release_date'])

date.head()
def get_list_len(list_):

    return len(eval(list_))



class ItemCounter(BaseEstimator, TransformerMixin):

        

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X):

        return X.apply(lambda x: int(get_list_len(x)))
language_count = ItemCounter().fit_transform(Exploratory['spoken_languages'])

language_count.head().to_frame('language_count')
year = DateTransformer().fit_transform(Exploratory['release_date'])['year']

top_cast_count = make_pipeline(FeatureSelector('cast'), DictionaryVectorizer('name'), 

                               TopFeatures(0.25), SumTransformer('top_cast_count')).fit_transform(Exploratory)
notional_to_numeric = pd.concat([year, top_cast_count], axis=1)

notional_to_numeric.head(15)
numeric = pd.concat([Exploratory.select_dtypes(['int64', 'float64']), notional_to_numeric], axis=1)



numeric.hist(figsize=(15,15), bins=25)
numeric.corr().style.background_gradient(cmap='coolwarm')
numeric.plot(kind='scatter', x='popularity', y='vote_count')

possible_outliers = Exploratory[Exploratory['popularity'] > 400]



numeric[['popularity', 'vote_count']] = np.log(Exploratory[['popularity', 'vote_count']] + 1)

numeric.plot(kind='scatter', x='popularity', y='vote_count')
possible_outliers
numeric.corr().style.background_gradient(cmap='coolwarm')
class MeanTransformer(BaseEstimator, TransformerMixin):

    

    def __init__(self, name):

        self.name = name

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X.mean(axis=1).to_frame(self.name)
feature_mean = make_pipeline(FeatureSelector(['vote_count', 'popularity']), MeanTransformer('popularity_vote')).fit_transform(Exploratory)

feature_mean.head()
numeric['vote_popularity'] = feature_mean

numeric.drop(columns=['popularity', 'vote_count'], inplace=True)
sns.pairplot(numeric)
from scipy.stats import pearsonr



transformations = [lambda x: x, np.sqrt, lambda x: np.log(x+1)]

tran_description = [' no transformation', ' sqrt', ' log']

numeric_columns = numeric.columns



fig, axes = plt.subplots(len(numeric_columns), len(transformations), figsize=(20,15))

fig.tight_layout()



for col_idx, col in enumerate(numeric_columns):

    for tran_idx, tran in enumerate(transformations):

        axes[col_idx, tran_idx].scatter(x=numeric[col], y=tran(y_train))

        axes[col_idx, tran_idx].set_xticklabels([])

        axes[col_idx, tran_idx].set_xticks([]) 

        R2 = pearsonr(numeric[col], tran(y_train))[0]**2     

        axes[col_idx, tran_idx].title.set_text(f'{col}, {tran_description[tran_idx]} \n R2 coefficient: {R2:.2f}')

               

plt.show()
from sklearn.externals.joblib import Parallel, delayed

from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators

from scipy import sparse



import warnings

warnings.filterwarnings('ignore')



class PandasFeatureUnion(FeatureUnion):

    def fit_transform(self, X, y=None, **fit_params):

        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(

            delayed(_fit_transform_one)(

                transformer=trans,

                X=X,

                y=y,

                weight=weight,

                **fit_params)

            for name, trans, weight in self._iter())



        if not result:

            # All transformers are None

            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)

        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):

            Xs = sparse.hstack(Xs).tocsr()

        else:

            Xs = self.merge_dataframes_by_column(Xs)

        return Xs



    def merge_dataframes_by_column(self, Xs):

        return pd.concat(Xs, axis="columns", copy=False)



    def transform(self, X):

        Xs = Parallel(n_jobs=self.n_jobs)(

            delayed(_transform_one)(

                transformer=trans,

                X=X,

                y=None,

                weight=weight)

            for name, trans, weight in self._iter())

        if not Xs:

            # All transformers are None

            return np.zeros((X.shape[0], 0))

        if any(sparse.issparse(f) for f in Xs):

            Xs = sparse.hstack(Xs).tocsr()

        else:

            Xs = self.merge_dataframes_by_column(Xs)

        return Xs

    

def make_union(*transformers, **kwargs):

    n_jobs = kwargs.pop('n_jobs', None)

    verbose = kwargs.pop('verbose', False)

    if kwargs:

        # We do not currently support `transformer_weights` as we may want to

        # change its type spec in make_union

        raise TypeError('Unknown keyword arguments: "{}"'

                        .format(list(kwargs.keys())[0]))

    return PandasFeatureUnion(

        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
union = make_union(

    make_pipeline(

        FeatureSelector('genres'),

        DictionaryVectorizer('name')

    ),

    make_pipeline(

        FeatureSelector('homepage'),

        Binarizer(lambda x: isinstance(x, float), 'missing_homepage')

    ),

    make_pipeline(

        FeatureSelector('keywords'),

        DictionaryVectorizer('name'),

        TopFeatures(0.5)

    ),

    make_pipeline(

        FeatureSelector('original_language'),

        Binarizer(lambda x: x == 'en', 'en')

    ),

    make_pipeline(

        FeatureSelector('production_companies'),

        DictionaryVectorizer('name'),

        TopFeatures(1)

    ),

    make_pipeline(

        FeatureSelector('production_countries'),

        DictionaryVectorizer('name'),

        TopFeatures(25)

    ),

    make_pipeline(

        FeatureSelector('release_date'),

        DateTransformer()

    ),

    make_pipeline(

        FeatureSelector('spoken_languages'),

        ItemCounter(),

        Binarizer(lambda x: x > 1, 'multilingual')

    ),

    make_pipeline(

        FeatureSelector('original_language'),

        Binarizer(lambda x: x == 'Released', 'Released')

    ),    

    make_pipeline(

        FeatureSelector('cast'),

        DictionaryVectorizer('name'),

        TopFeatures(0.25),

        SumTransformer('top_cast_count')

    ),

    make_pipeline(

        FeatureSelector('crew'),

        DictionaryVectorizer('name', False),

        TopFeatures(1)

    ),

    make_pipeline(

        FeatureSelector(['budget', 'runtime', 'vote_average'])

    ),

    make_pipeline(

        FeatureSelector(['popularity', 'vote_count']),

        MeanTransformer('popularity_vote')

    )

)
union.fit(X_train)



X_train_T = union.transform(X_train)

X_test_T = union.transform(X_test)



print(X_train_T.shape)

print(X_test_T.shape)
X_train_T.head()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
lin_params = dict(alpha=np.logspace(1,7,7), normalize=(False, True))

for_params = dict(n_estimators=np.linspace(10,40,4).astype(int), min_samples_split=(2,3), min_samples_leaf=(1,2,3))

gbr_params = dict(n_estimators=np.linspace(100,300,3).astype(int), min_samples_split=(2,3))
ridge_grid = GridSearchCV(Ridge(random_state=42), lin_params, cv=10)

forest_grid = GridSearchCV(RandomForestRegressor(random_state=42), for_params, cv=10)

gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gbr_params, cv=10)
ridge_grid.fit(X_train_T, y_train)
forest_grid.fit(X_train_T, y_train)
gbr_grid.fit(X_train_T, y_train)
print(f'Ridge:\n\t *best params: {ridge_grid.best_params_}\n\t *best score: {ridge_grid.best_score_}')

print(f'Forest:\n\t *best params: {forest_grid.best_params_}\n\t *best score: {forest_grid.best_score_}')

print(f'Gradient Boost:\n\t *best params: {gbr_grid.best_params_}\n\t *best score: {gbr_grid.best_score_}')
best_ridge = Ridge(alpha=100, normalize=False)

best_forest = RandomForestRegressor(min_samples_leaf=3, min_samples_split=2, n_estimators=40)

best_gbr = GradientBoostingRegressor(min_samples_split=2, n_estimators=300)
from sklearn.metrics import r2_score
best_ridge.fit(X_train_T, y_train)

predicted = best_ridge.predict(X_test_T)



print(f'Ridge test score: {r2_score(y_test, predicted)}')



best_forest.fit(X_train_T, y_train)

predicted = best_forest.predict(X_test_T)



print(f'Random Forest test score: {r2_score(y_test, predicted)}')



best_gbr.fit(X_train_T, y_train)

predicted = best_gbr.predict(X_test_T)



print(f'Gradient Boosted Regressor test score: {r2_score(y_test, predicted)}')
ridge_coefs_df = pd.DataFrame(dict(score=best_ridge.coef_, column=X_test_T.columns))

ridge_coefs_df.sort_values(['score'], ascending=False).head(10)
print(f'Train target variable mean: ${round(y_train.mean()):,}.')
ridge_coefs_df.loc[136:]
pd.DataFrame(dict(score=best_forest.feature_importances_, column=X_test_T.columns)).sort_values(['score'], ascending=False).head(10)
pd.DataFrame(dict(score=best_gbr.feature_importances_, column=X_test_T.columns)).sort_values(['score'], ascending=False).head(10)