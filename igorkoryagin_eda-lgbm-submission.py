

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter





#preprocessing

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

from sklearn.feature_extraction.text import TfidfVectorizer

from math import floor



#visualization

import matplotlib.pyplot as plt

import seaborn as sns



import pickle

from sklearn.linear_model import LinearRegression

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor



from xgboost import plot_importance

import optuna

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

%matplotlib inline
train_df = pd.read_csv('../input/hse-aml-2020/books_train.csv')

test_df = pd.read_csv('../input/hse-aml-2020/books_test.csv')

sample_df = pd.read_csv('../input/hse-aml-2020/books_sample_submission.csv')
def export_res(X, filename='X'):

    try:

        with open('{}.pickle'.format(filename), 'wb') as fout:

            pickle.dump(X, fout)

        print(f'Preprocessed {filename} exported')

    except FileNotFoundError:

        print('File not found')





def load_saved_parameters(filename):

    try:

        with open('../input/hse-aml-params/{}.pickle'.format(filename), 'rb') as fin:

            X = pickle.load(fin)

        print('Parameters loaded')

    except FileNotFoundError:

        print('File with saved parameters not found')

    return X
train_df.info()
train_df.head()
test_df.info()
all_df = pd.concat([train_df, test_df])

all_df.info()
print('Numbers of isbn for bookID: ', all_df.groupby('bookID')['isbn'].count().unique())

print('Numbers of isbn13 for bookID: ',all_df.groupby('bookID')['isbn13'].count().unique())

print('Numbers of language_code for bookID: ',all_df.groupby('bookID')['language_code'].count().unique())

print('Numbers of publisher for bookID: ',all_df.groupby('bookID')['publisher'].count().unique())
def count_authors(authors_str):

    authors_str = str(authors_str)

    return authors_str.count('/') + 1



authors = np.array(train_df['authors'])

counted_auth = np.array(list(map(count_authors, authors)))

train_df['n_authors'] = counted_auth
most_rated = all_df.sort_values(by="ratings_count", ascending = False).head(10)



most_rated_titles = pd.DataFrame(most_rated.title).join(pd.DataFrame(most_rated.ratings_count))

most_rated_titles
def scatter_plot(x, y, title, x_label, y_label):

    plt.subplots(figsize=(8, 8))

    plt.scatter(x,

                y)

    plt.title(title)

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    

pages_count_and_average_rating_title = "Relation between pages count and average rating"

average_rating_label = "Average rating"

pages_count_label = "Pages count"

scatter_plot(all_df.average_rating,

             all_df['  num_pages'],

            pages_count_and_average_rating_title, average_rating_label, pages_count_label)
books_data = all_df.drop(all_df.index[all_df["  num_pages"] >= 2000], inplace=True)

pages_count_and_average_rating_title = "Relation between pages count and average rating"

average_rating_label = "Average rating"

pages_count_label = "Pages count"

scatter_plot(all_df.average_rating,

             all_df['  num_pages'],

            pages_count_and_average_rating_title, average_rating_label, pages_count_label)
all_df[all_df["  num_pages"] == 0]
all_df["  num_pages"] = all_df["  num_pages"].replace(0, np.nan)

all_df["  num_pages"].fillna(float(floor(all_df["  num_pages"].mean())), inplace=True)



print("Number 0s in num_pages:", len(all_df[all_df["  num_pages"] == 0]))

print("Is there any NaN in num_pages:" , all_df["  num_pages"].isna().any().any())
reviews_count_and_average_rating_title = "Relation between text_reviews_count and average rating"

average_rating_label = "Average rating"

reviews_count_label = "text_reviews_count"

scatter_plot(all_df.average_rating,

             all_df['text_reviews_count'],

            reviews_count_and_average_rating_title, average_rating_label, reviews_count_label)
# find text_reviews_count count outliers

sns.boxplot(x=all_df['text_reviews_count'])
all_df.drop(all_df.index[all_df["text_reviews_count"] >= 40000], inplace=True)
# find ratings count outliers

sns.boxplot(x=all_df['ratings_count'])
all_df.drop(all_df.index[all_df["ratings_count"] >= 1000000], inplace=True)
def preprocess_df(train_df):

    # removing num_pages outliers, replacing 0s with mean value

    #train_df.drop(train_df.index[train_df["  num_pages"] >= 2000], inplace=True)

    train_df["  num_pages"] = train_df["  num_pages"].replace(0, np.nan)

    train_df["  num_pages"].fillna(float(floor(train_df["  num_pages"].mean())), inplace=True)



    # removing text_reviews_count outliers

    #train_df.drop(train_df.index[train_df["text_reviews_count"] >= 40000], inplace=True)





    # removing ratings count outliers

    #train_df.drop(train_df.index[train_df["ratings_count"] >= 1000000], inplace=True)

    

    # Handling text data: titles, authors, publisher

    # encoding title column

    lb_encoder = LabelEncoder()

    train_df['title'] = lb_encoder.fit_transform(train_df['title'])

    

    # encoding authors column

    train_df['authors'] = lb_encoder.fit_transform(train_df['authors'])

    

    # encoding publisher column

    train_df['publisher'] = lb_encoder.fit_transform(train_df['publisher'])

    

    # encoding language column

    lang_enc = pd.get_dummies(train_df['language_code'])

    train_df = pd.concat([train_df, lang_enc], axis = 1)

    

    #Adding frequency rate for authors, publisher

    auth_freq = train_df['authors'].value_counts(normalize=True)

    train_df['authors_freq'] = train_df['authors'].map(lambda x: auth_freq[x])



    publisher_freq = train_df['publisher'].value_counts(normalize=True)

    train_df['publishers_freq'] = train_df['publisher'].map(lambda x: publisher_freq[x])

    

    

    

    return train_df
train_df['label'] = 'train'

test_df['label'] = 'test'



all_df_pre = pd.concat([train_df, test_df])



all_df_sub =  preprocess_df(all_df_pre)



train_df_sub = all_df_sub[all_df_sub['label']=='train']

test_df_sub = all_df_sub[all_df_sub['label']=='test']



# divide the data into attributes and labels

X_train_v = train_df_sub.drop(['average_rating', 'language_code', 'isbn', 'isbn13', 'publication_date','publisher','label'], axis = 1)

y_train_v = train_df_sub['average_rating']



test_df_sub = test_df_sub.drop(['language_code', 'isbn', 'isbn13', 'publication_date','publisher','label','average_rating'], axis = 1)

X_test_v = test_df_sub



results = load_saved_parameters('lgb_fin_1031')

lgb_params = results['LGB']['params']





model = LGBMRegressor(**lgb_params)

lgbm = model.fit(X_train_v, y_train_v,  verbose=False)

    

prediction = model.predict(X_test_v)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'bookID':test_df_sub['bookID'],'average_rating':prediction})



#Visualize the first 5 rows

submission.head()
submission.to_csv('submission_lgbm.csv', index=False)