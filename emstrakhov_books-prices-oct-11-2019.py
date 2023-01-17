# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_excel('../input/price-of-books/Data_Train.xlsx')

test_df = pd.read_excel('../input/price-of-books/Data_Test.xlsx')



train_df.head()
train_df.loc[0, 'Synopsis']
def clean_text(text):

    '''

    text: исходный текст

    '''

    text = text.lower()

    l = []

    for x in text:

        if x.isalpha() or x==' ':

            l.append(x)

        elif x=='\n':

            l.append(' ')

    return ''.join(l)
clean_text("THE HUNTERS return in their third brilliant novel from the Sunday Times Top Ten bestselling author Chris Kuzneski, whose writing James Patterson says has 'raw power'. The team are hunting Marco Polo's hidden treasure, but who is on their tail?\nTHE HUNTERS\nIf you seek, they will find...\n\nThe travels of Marco Polo are known throughout the world.\nBut what if his story isn't complete?\nWhat if his greatest adventure has yet to be discovered?\nGuided by a journal believed to have been dictated by Polo himself,\nthe Hunters set out in search of his final legacy:\nthe mythical treasure gathered during Polo's lifetime of exploration.\nBut as every ancient clue brings them closer to the truth,\neach new step puts them in increasing danger...\nExplosive action. Killer characters. Classic Kuzneski.")
train_df['CleanSynopsis'] = train_df['Synopsis'].map(clean_text)
train_df['Price'].hist();
train_df['LogPrice'] = np.log(train_df['Price'])

train_df['LogPrice'].hist();
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(stop_words='english', min_df=5)

text_features = bow.fit_transform(train_df['CleanSynopsis'])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', min_df=5)

text_features_tfidf = tfidf.fit_transform(train_df['CleanSynopsis'])
bow.get_feature_names()[:20]
tfidf.get_feature_names()[:20]
corpus = ['Студент написал заявление в деканате на имя декана.',

          'Студент получил студенческий билет.']

bow_text = bow.fit_transform(corpus)

bow.get_feature_names()
pd.DataFrame(bow_text.toarray(), columns=bow.get_feature_names())
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(text_features, 

                                                      train_df['LogPrice'], 

                                                      test_size=0.3, random_state=15)
from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(text_features_tfidf, 

                                                      train_df['LogPrice'], 

                                                      test_size=0.3, random_state=15)



ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_valid)

mean_squared_error(y_valid, y_pred)
from sklearn.metrics import mean_squared_error

y_pred = ridge.predict(X_valid)

mean_squared_error(y_valid, y_pred)
train_df['Price'].describe()
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(nrows=1, ncols=2)

# ax[0].hist(train_df['Price']);
import seaborn as sns

sns.boxplot(train_df['Price']);
train_df['Log Price'] = np.log(train_df['Price'])

sns.boxplot(train_df['Log Price']);
train_df['Log Price'].hist();
(np.log(13800) - np.log(14000))**2
train_df['Author'].nunique(), train_df.shape[0]
train_df['Edition'].values
train_df['Year'] = train_df['Edition'].map(lambda s: s[-4:])#.astype('int')

test_df['Year'] = test_df['Edition'].map(lambda s: s[-4:])#.astype('int')

train_df.head()
k = 0

for i in range(len(train_df['Year'].values)):

    x = train_df.loc[i, 'Year']

    if not x.isnumeric():

        train_df.loc[i, 'Year'] = 0

        

for i in range(len(test_df['Year'].values)):

    x = test_df.loc[i, 'Year']

    if not x.isnumeric():

        test_df.loc[i, 'Year'] = 0

        

train_df['Year'] = train_df['Year'].astype('int')

test_df['Year'] = test_df['Year'].astype('int')
sns.boxplot(train_df['Reviews'].map(lambda s: s.split(' ')[0]).astype('float'))
train_df['Rating'] = train_df['Reviews'].map(lambda s: s[:3]).astype('float')

test_df['Rating'] = test_df['Reviews'].map(lambda s: s[:3]).astype('float')

train_df.head()
train_df['NumReviews'] = train_df['Ratings'].map(lambda s: s.split(' customer')[0])

test_df['NumReviews'] = test_df['Ratings'].map(lambda s: s.split(' customer')[0])
def _(s):

    if ',' in s:

        s = int(s.split(',')[0]) * 1000 + int(s.split(',')[-1])

    return s
train_df['NumReviews'] = train_df['NumReviews'].map(_).astype('int')

test_df['NumReviews'] = test_df['NumReviews'].map(_).astype('int')
train_df.head()
numerical_features = ['Year', 'Rating', 'NumReviews']
train_df['BookCategory'].value_counts()
test_df['BookCategory'].value_counts()
set(test_df['BookCategory'].values) == set(train_df['BookCategory'].values)
train_df_1 = pd.get_dummies(train_df, columns=['BookCategory'])

test_df_1 = pd.get_dummies(test_df, columns=['BookCategory'])

train_df_1.head()
numerical_features = train_df_1.columns[-14:]

train_df_2 = train_df_1[numerical_features]

test_df_2 = test_df_1[numerical_features]

train_df_2.head()
X = train_df_2

y = train_df['Log Price']



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=15)
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=4, random_state=15)

tree.fit(X, y)



X_test = test_df_2

y_pred = tree.predict(X_test)



# from sklearn.metrics import mean_squared_log_error

# MSLE = mean_squared_log_error(y_valid, y_pred)

# RMSLE = np.sqrt(MSLE)

# 1 - RMSLE
y_pred_real = np.exp(y_pred)

pd.DataFrame(y_pred_real).to_excel('subm.xlsx', header='Price')
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=4, random_state=15)

tree.fit(X_train, y_train)



y_pred = tree.predict(X_valid)



from sklearn.metrics import mean_squared_log_error

MSLE = mean_squared_log_error(y_valid, y_pred)

RMSLE = np.sqrt(MSLE)

1 - RMSLE
train_df_1['Type of cover'] = train_df_1['Edition'].map(lambda s: s.split(',')[0])

train_df_1['Type of cover'].value_counts()
train_df_1.groupby('Type of cover')['Log Price'].mean().sort_values(ascending=False).plot(kind='bar');
train_df_1.groupby('Type of cover')['Log Price'].mean()
train_df_3 = pd.get_dummies(train_df_1, columns=['Type of cover'])

train_df_3.head()
def get_score(X, y, model):

    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                          test_size=0.3, 

                                                          random_state=15)

    

    model.fit(X_train, y_train)



    y_pred = model.predict(X_valid)



    from sklearn.metrics import mean_squared_log_error

    MSLE = mean_squared_log_error(y_valid, y_pred)

    RMSLE = np.sqrt(MSLE)

    return 1 - RMSLE
position = list(train_df_3.columns.values).index('Year')

position
numerical_features = train_df_3.columns.values[9:]

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=4, random_state=15)

get_score(train_df_3[numerical_features], y, tree)
train_df_3['Year'].value_counts()
train_df_3.groupby('Author')['Log Price'].mean().nlargest(20)
x = train_df_3.groupby('Author')['Log Price'].count()

top_authors = x[ x > 10 ].index
top_authors_df = train_df_3[ train_df_3['Author'].isin(top_authors) ]

top_authors_df.groupby('Author')['Price'].mean().nlargest(20).sort_values(ascending=False).plot(kind='bar');
train_df_3['Author'].nunique()
train_df_4 = pd.get_dummies(train_df_3, columns=['Author'])



numerical_features = train_df_4.columns.values[9:]

tree = DecisionTreeRegressor(max_depth=4, random_state=15)

get_score(train_df_4[numerical_features], y, tree)
def visualize_tree(tree):

    from sklearn.tree import export_graphviz

    tree_dot = export_graphviz(tree)

    print(tree_dot)
numerical_features = train_df_3.columns.values[9:]

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=4, random_state=15)

get_score(train_df_3[numerical_features], y, tree)
visualize_tree(tree)
train_df_3[numerical_features].columns.values[2]
train_df_3[numerical_features].columns.values[21]
train_df_3[numerical_features].columns.values[0]
train_df_3[numerical_features].columns.values[7]
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='neg_mean_squared_log_error') # кросс-валидация по 5 блокам



X = train_df_3[numerical_features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, 

                                                      random_state=15)

tree_grid.fit(X_train, y_train)
# Отрисовка графиков

import matplotlib.pyplot as plt



plt.plot(tree_params['max_depth'], tree_grid.cv_results_['mean_test_score']);
numerical_features = train_df_3.columns.values[9:]

from sklearn.tree import DecisionTreeRegressor

opt_tree = DecisionTreeRegressor(max_depth=7, random_state=15)

get_score(train_df_3[numerical_features], y, opt_tree)
numerical_features = train_df_3.columns.values[9:]

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, random_state=15)

get_score(train_df_3[numerical_features], y, rf)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=3)

get_score(train_df_3[numerical_features], y, knn)
from sklearn.model_selection import GridSearchCV



knn_params = {'n_neighbors': np.arange(3, 200, 2)}



knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='neg_mean_squared_log_error') # кросс-валидация по 5 блокам



X = train_df_3[numerical_features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, 

                                                      random_state=15)

knn_grid.fit(X_train, y_train)



plt.plot(knn_params['n_neighbors'], knn_grid.cv_results_['mean_test_score']);
knn_grid.best_params_
knn_opt = KNeighborsRegressor(n_neighbors=27)

get_score(train_df_3[numerical_features], y, knn_opt)
numerical_features = train_df_4.columns.values[9:]

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, random_state=15)

get_score(train_df_4[numerical_features], y, rf)
train_authors = set(train_df['Author'].values)

test_authors = set(test_df['Author'].values)

len(test_authors - train_authors)
len(test_authors & train_authors)