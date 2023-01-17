import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_json('../input/two-sigma-connect-rental-listing-inquiries/train.json.zip')

print(data.shape)

data.head(n=3)
data_explore = data.copy()
data_explore['num_photos'] = data_explore['photos'].apply(len)

data_explore['num_features'] = data_explore['features'].apply(len)

data_explore['num_description_words'] = data_explore['description'].apply(lambda x: len(x.split(' ')))
data_explore = data_explore.drop(['description', 'street_address', 'photos', 'listing_id'], axis=1)
data_explore.info()
null_cols = data_explore.isna().sum()

null_cols[null_cols>0]
for col in data_explore.columns:

    try:

        print(col, '\t\t' ,data_explore[col].nunique())

    except:

        pass
data_explore['created_year'] = pd.DatetimeIndex(data_explore['created']).year

data_explore['created_month'] = pd.DatetimeIndex(data_explore['created']).month

data_explore['created_day'] = pd.DatetimeIndex(data_explore['created']).day
data_explore.describe()
data_explore = data_explore.drop(['created_year'], axis=1)
def plot_histogram(data):

    ax = plt.gca()

    counts, _, patches = ax.hist(data)

    for count, patch in zip(counts, patches):

        if count>0:

            ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()+5))

    if data.name:

        plt.xlabel(data.name)
plt.figure(figsize=(15, 13))

i=1

for col in ['bathrooms', 'bedrooms', 'num_features', 'created_month', 'created_day', 'interest_level']:

    plt.subplot(3, 2, i)

    plot_histogram(data_explore[col])

    i+=1

plt.show()
# Box-Plot

plt.figure(figsize=(8, 4))

sns.boxplot(x='price', data=data_explore, orient='h')

plt.xlim(-1000, 10000)

ax = plt.gca()

ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set_title('Distribution of Price')
Q1 = data_explore['price'].quantile(0.25)

Q3 = data_explore['price'].quantile(0.75)

IQR = Q3 - Q1

((data_explore['price'] < (Q1 - 1.5 * IQR)) | (data_explore['price'] > (Q3 + 1.5 * IQR))).sum()
# Box-Plot

plt.figure(figsize=(7, 5))

sns.boxplot(x='interest_level', y='price',  data=data_explore, orient='v')

plt.ylim(-100, 10000)

ax = plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.show()
plt.figure(figsize=(10, 6))

sns.scatterplot(x='longitude', y='latitude', hue='interest_level', data=data_explore)

plt.xlim(-74.1, -73.7)

plt.ylim(40.55, 40.95)

plt.show()
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)

sns.barplot(x='interest_level', y='bedrooms',  data=data_explore, orient='v', estimator=np.median)

plt.ylabel('Median # of Bedrooms')

plt.subplot(2, 2, 2)

sns.barplot(x='interest_level', y='bathrooms',  data=data_explore, orient='v', estimator=np.median)

plt.ylabel('Median # of Bathrooms')

plt.subplot(2, 2, 3)

sns.barplot(x='interest_level', y='num_photos',  data=data_explore, orient='v', estimator=np.median)

plt.ylabel('Median # of Photos')

plt.subplot(2, 2, 4)

sns.barplot(x='interest_level', y='num_description_words',  data=data_explore, orient='v', estimator=np.median)

plt.ylabel('Median # of Words in Description')

plt.show()
from wordcloud import WordCloud



plt.figure(figsize = (12, 12))

text = ' '.join(data['description'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Words in Apartment Description', fontsize=14)

plt.axis("off")

plt.show()
list_of_features = list(data_explore['features'].values)

plt.figure(figsize = (10, 10))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_features for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False, width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Features', fontsize=14)

plt.axis("off")

plt.show()
plt.figure(figsize = (12, 12))

data['display_address'] = data['display_address'].apply(lambda x: x.replace(' ', '_'))

text = ' '.join(data['display_address'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Display Addresses', fontsize=14)

plt.axis("off")

plt.show()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
X = data.drop(columns=['interest_level'], axis=1).copy()

y = data['interest_level'].copy()

X.shape, y.shape
cat_attrs = ['building_id', 'display_address', 'manager_id',]
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(X, y):

    strat_train_set = data.iloc[train_index]

    strat_test_set = data.iloc[test_index]



X_train = strat_train_set.drop('interest_level', axis=1)

y_train = strat_train_set['interest_level'].copy()

X_test = strat_test_set.drop('interest_level', axis=1)

y_test = strat_test_set['interest_level'].copy()

X_train.shape, X_test.shape
import itertools

from collections import Counter

a = list(data['features'].values.flatten())

feature_list = list(itertools.chain.from_iterable(a))

top_25_features = [ x for x, y in Counter(feature_list).most_common(25)]

top_25_features
class CustomDateAttrs(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        X['created_month'] = pd.DatetimeIndex(X['created']).month

        X['created_day'] = pd.DatetimeIndex(X['created']).day

        X = X.drop(['created'], axis=1)

        return X
class CustomNumAttrs(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        X['photos'] = X['photos'].apply(len)

        X['description'] = X['description'].apply(lambda x: len(x.split(' ')))

        return X
encoded_features = []



class CustomMultiLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.mlb_enc = None

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        try:

            X['features'] = X['features'].apply(lambda x: ['no_feature', ] if len(x)==0 else self.get_features(x))

            if self.mlb_enc==None:

                self.mlb_enc = MultiLabelBinarizer()

                X_enc = pd.DataFrame(self.mlb_enc.fit_transform(X['features']), columns=self.mlb_enc.classes_, 

                                     index=X.index)

                encoded_features.append(self.mlb_enc.classes_)

            else:

                X_enc = pd.DataFrame(self.mlb_enc.transform(X['features']), columns=self.mlb_enc.classes_, 

                                     index=X.index)

            X = pd.concat([X, X_enc], axis=1)

            X = X.drop('features', axis=1)

        except Exception as e:

            print("CustomMultiLabelBinarizer: Exception caught for {}: {}".format(e))

        return X

    

    @staticmethod

    def get_features(x):

        if len(x)==0:

            return ['no_feature', ]

        

        features = [feature for feature in x if feature in top_25_features]

        if len(features)==0:

            features.append('other')

        return features

                
pre_process = ColumnTransformer([('drop_cols', 'drop', ['street_address', 'listing_id']),

                                 ('num_imputer', SimpleImputer(strategy='median'), ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude']),

                                 ('custom_date_attr', CustomDateAttrs(), ['created', ]),

                                 ('custom_num_attrs', CustomNumAttrs(), ['description', 'photos']),

                                 ('list_encoder', CustomMultiLabelBinarizer(), ['features', ]),

                                 ('cat_imputer', SimpleImputer(strategy='most_frequent'), cat_attrs)], remainder='passthrough')



X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)

X_train_transformed.shape, X_test_transformed.shape
feature_columns = ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude'] + ['created_month', 'created_day', ] + ['description', 'photos'] + list(encoded_features[0]) + cat_attrs

print(len(feature_columns), feature_columns)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_columns)

X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_columns)
X_train_transformed.head()
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
def performance_measures(model, store_results=True):

    train_log_loss = cross_val_score(model, X_train_transformed, y_train, scoring='neg_log_loss', cv=kf, n_jobs=-1)

    train_log_loss *= -1

    test_log_loss = cross_val_score(model, X_test_transformed, y_test, scoring='neg_log_loss', cv=kf, n_jobs=-1)

    test_log_loss *= -1

    print("Mean Train Log Loss: {}\nMean Test Log Loss: {}".format(train_log_loss.mean(), test_log_loss.mean()))

    
def plot_feature_importance(feature_columns, importance_values, top_n_features=0):

    feature_imp = [ col for col in zip(feature_columns, importance_values)]

    feature_imp.sort(key=lambda x:x[1], reverse=True)

    

    if top_n_features:

        imp = pd.DataFrame(feature_imp[0:top_n_features], columns=['feature', 'importance'])

    else:

        imp = pd.DataFrame(feature_imp, columns=['feature', 'importance'])

    plt.figure(figsize=(10, 8))

    sns.barplot(y='feature', x='importance', data=imp, orient='h')

    plt.title('Most Important Features', fontsize=16)

    plt.ylabel("Feature", fontsize=16)

    plt.xlabel("")

    plt.show()
from catboost import CatBoostClassifier





catboost_grid_params = [{'iterations':[1000, 1500, 2000], 'depth':[5, 6, 7, 8, 9]}] 



catboost_clf = CatBoostClassifier(task_type="GPU", loss_function='MultiClass', 

                                  cat_features=[36, 37, 38], random_state=42, verbose=0)



grid_search_results = catboost_clf.grid_search(catboost_grid_params, 

                                               X_train_transformed, y_train,

                                               cv=5, partition_random_seed=42, 

                                               calc_cv_statistics=True,

                                               search_by_train_test_split=True, refit=True, 

                                               shuffle=True,stratified=None, train_size=0.8, 

                                               verbose=0, plot=False)
grid_search_results['params']
catboost_clf.is_fitted()
plot_feature_importance(feature_columns, catboost_clf.feature_importances_)
performance_measures(catboost_clf)
X_trasformed = pre_process.transform(X)

predicted_interest = catboost_clf.predict(X_trasformed)

data['predicted_interest_level'] = predicted_interest
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plot_histogram(data['interest_level'])

plt.subplot(1, 2, 2)

plot_histogram(data['predicted_interest_level'])

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

sns.scatterplot(x='longitude', y='latitude', hue='interest_level', data=data)

plt.xlim(-74.1, -73.7)

plt.ylim(40.55, 40.95)

plt.subplot(1, 2, 2)

sns.scatterplot(x='longitude', y='latitude', hue='predicted_interest_level', data=data)

plt.xlim(-74.1, -73.7)

plt.ylim(40.55, 40.95)

plt.show()
final_model = Pipeline([('pre_process', pre_process),

                        ('catboost_clf', catboost_clf)])

final_model.fit(X_train, y_train)
import zipfile  



test_data = None  

with zipfile.ZipFile("../input/two-sigma-connect-rental-listing-inquiries/test.json.zip", "r") as z:

    for filename in z.namelist(): 

        with z.open(filename) as f:

            test_data = pd.read_json(f.read())

            

test_data.head()
test_data.info()
predictions = final_model.predict_proba(test_data)
output = pd.DataFrame(test_data['listing_id'])

output[["high", "medium", "low"]] = predictions.copy()
output.head()
output.to_csv("./submission.csv", index=False)