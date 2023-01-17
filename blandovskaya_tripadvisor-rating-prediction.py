import os

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

import numpy as np

import pandas as pd

import re

pd.options.mode.chained_assignment = None





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip freeze > requirements.txt
random_seed = 42



current_date = pd.to_datetime('30/07/2020')
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train['sample'] = 1

df_test['sample'] = 0

df_test['Rating'] = 0



df = df_test.append(df_train, sort=False).reset_index(drop=True)
df.head()
df.info()
msno.bar(df)
df.columns = ['id', 'city', 'cuisine', 'ranking', 'price_range',

              'num_of_reviews', 'reviews', 'url_ta', 'id_ta', 'sample', 'rating']
# turning id to numeric format



df.id = df.id.apply(lambda x: x.split('_')[1]).astype(int)
# some of restaurants ids meets in dataset several times

# creating franchise feature



franchise = df[df['id'].isin(df['id'].value_counts()[

                             df['id'].value_counts() >= 2].index)]

franchise['franchise'] = 1

df['franchise'] = franchise['franchise']

df['franchise'] = df['franchise'].fillna(0)

df.franchise = df.franchise.astype(int)
restaurants_in_city = df.city.value_counts()

df['restaurants_count'] = df.city.apply(lambda x: restaurants_in_city[x])
is_capital = {'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0,

              'Berlin': 1, 'Milan': 0, 'Rome': 1, 'Prague': 1,

              'Lisbon': 1, 'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1,

              'Hamburg': 0, 'Munich': 0, 'Lyon': 0, 'Stockholm': 1,

              'Budapest': 1, 'Warsaw': 1, 'Dublin': 1, 'Copenhagen': 1,

              'Athens': 1, 'Edinburgh': 1, 'Zurich': 1, 'Oporto': 0,

              'Geneva': 1, 'Krakow': 1, 'Oslo': 1, 'Helsinki': 1,

              'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1}
df['capital'] = df['city'].apply(lambda x: 0 if is_capital[x] == 0 else 1)
population = {'Paris': 2.141, 'Stockholm': 0.973, 'London': 8.9, 'Berlin': 3.748,

              'Munich': 1.456, 'Oporto': 0.214, 'Milan': 1.352, 'Bratislava': 0.424,

              'Vienna': 1.889, 'Rome': 2.873, 'Barcelona': 5.515, 'Madrid': 6.55,

              'Dublin': 1.361, 'Brussels': 0.174, 'Zurich': 0.403, 'Warsaw': 1.708,

              'Budapest': 1.75, 'Copenhagen': 0.602, 'Amsterdam': 0.822, 'Lyon': 0.513,

              'Hamburg': 1.822, 'Lisbon': 0.505, 'Prague': 1.319, 'Oslo': 0.673,

              'Helsinki': 0.632, 'Edinburgh': 0.482, 'Geneva': 0.495, 'Ljubljana': 0.28,

              'Athens': 0.664, 'Luxembourg': 0.602, 'Krakow': 0.769}
df['population'] = df['city'].map(population)
for x in (df['city'].value_counts())[0:10].index:

    df['ranking'][df['city'] == x].hist(bins=100)

plt.show()
ranking_by_city = df.groupby(['city'])['ranking'].mean()

restaurants_in_city = df['city'].value_counts(ascending=False)

df['ranking_by_city'] = df['city'].apply(lambda x: ranking_by_city[x])

df['restaurants_in_city'] = df['city'].apply(lambda x: restaurants_in_city[x])

df['norm_ranking'] = (df['ranking'] - df['ranking_by_city']

                      ) / df['restaurants_in_city']
for x in (df['city'].value_counts())[0:10].index:

    df['norm_ranking'][df['city'] == x].hist(bins=100)

plt.show()
ranking_by_city_max = df.groupby(['city'])['ranking'].max()

df['ranking_by_city_max'] = df['city'].apply(lambda x: ranking_by_city_max[x])

df['norm_ranking_max'] = (

    df['ranking'] - df['ranking_by_city']) / df['ranking_by_city_max']
for x in (df['city'].value_counts())[0:10].index:

    df['norm_ranking_max'][df['city'] == x].hist(bins=100)

plt.show()
df['norm_ranking_pop'] = (

    df['ranking'] - df['ranking_by_city']) / df['population']



for x in (df['city'].value_counts())[0:10].index:

    df['norm_ranking_pop'][df['city'] == x].hist(bins=100)

plt.show()
df['cuisine'] = df.apply(lambda x: x['cuisine'].replace('[', '').replace(']', '').replace(

    "'", '').replace(' ', '') if type(x['cuisine']) != float else x['cuisine'], axis=1)
df['cuisine_isna'] = pd.isna(df['cuisine'])

df.cuisine = df.cuisine.fillna('Other')
df['cuisines_count'] = df['cuisine'].str.split(',').str.len().fillna(1)
df['special_menu'] = 0
special = ['VegetarianFriendly', 'VeganOptions', 'GlutenFreeOptions']



for option in special:

    for i in range(0, len(df)):

        if option in df.cuisine[i]:

            df.special_menu[i] = 1

        else:

            df.special_menu[i] = 0
cuisines = df['cuisine'].str.get_dummies(

    ',').sum().sort_values(ascending=False)

top_cuisines = [x for x in cuisines.index if cuisines[x] < 1000]





df = df.join(df['cuisine'].str.get_dummies(

    ',').drop(top_cuisines, axis=1), how='left')
price_range_dict = {"$": 1, "$$ - $$$": 2, "$$$$": 3, np.NaN: 2}
df['price_range_isna'] = df.price_range.isna()

df.price_range = df.price_range.map(price_range_dict)
price_in_city_dict = df.groupby('city')['price_range'].mean().to_dict()
df['price_in_city'] = df.city.map(price_in_city_dict)
df['num_of_reviews_isna'] = df.num_of_reviews.isna()

df.num_of_reviews = df.num_of_reviews.fillna(0)
# getting date of reviews and turning it to datetime format



df.reviews = df.reviews.apply(lambda x: x.replace(

    '[[], []]', 'NaN') if type(x) != float else 'NaN')

df['reviews'] = df['reviews'].apply(lambda x: str(x) if type(x) == list else x)

res = []

for i in df['reviews']:

    res.append(re.findall(r'(\d\d/\d\d/\d\d\d\d)', i))

reviews = pd.DataFrame(res)

reviews[0] = pd.to_datetime(reviews[0])

reviews[1] = pd.to_datetime(reviews[1])



# finding newness of last review



newest_review = []



for i in range(len(reviews)):

    if reviews.loc[i, 0] > reviews.loc[i, 1]:

        newest_review.append(current_date - reviews.loc[i, 0])

    elif reviews.loc[i, 0] < reviews.loc[i, 1]:

        newest_review.append(current_date - reviews.loc[i, 1])

    else:

        newest_review.append(0)
df['newest_review'] = pd.Series(newest_review)

df['newest_review'] = df['newest_review'].apply(

    lambda x: x.days if (type(x) != int) else 0)
df.reviews = df.reviews.apply(lambda x: np.nan if x == 'NaN' else x)



df['reviews_isna'] = df.reviews.isna()

df.reviews = df.reviews.fillna(0)
df.drop(['id_ta', 'url_ta', 'reviews', 'cuisine'], axis=1, inplace=True)
plt.rcParams['figure.figsize'] = (15, 10)

sns.heatmap(df.drop(['sample'], axis=1).corr())
df = pd.get_dummies(df, columns=['city'])
train_data = df.query('sample == 1').drop(['sample'], axis=1)



y = train_data.rating.values

X = train_data.drop(['rating'], axis=1)





X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=random_seed)
train_data.shape, X.shape, X_train.shape, X_test.shape
model = RandomForestRegressor(

    n_estimators=100, verbose=1, n_jobs=-1, random_state=random_seed)
model.fit(X_train, y_train)





y_pred = model.predict(X_test)
y_pred
y_pred_round = []

for item in y_pred:

    y_pred_round.append(round(item/0.5)*0.5)

y_pred_round = np.asarray(y_pred_round)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))
plt.rcParams['figure.figsize'] = (10, 15)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data = df.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['rating'], axis=1)



predict_submission = model.predict(test_data)





predict_submission_round = []

for item in predict_submission:

    predict_submission_round.append(round(item/0.5)*0.5)

predict_submission_round = np.asarray(predict_submission_round)



sample_submission['Rating'] = predict_submission_round

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()