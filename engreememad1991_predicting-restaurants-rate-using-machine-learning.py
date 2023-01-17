import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix

#importing libraries

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

import time

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/zomato.csv')

df.head()
df.describe()
# Dimensions

print(f'This dataset has {df.shape[0]} rows and {df.shape[1]} columns.')

print(f'\nList of columns: \n{df.columns}')
# Null data

df.isnull().sum()
df.dtypes

df.describe()
# Approx cost

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',', '.'))

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)

print(f'{type(df["approx_cost(for two people)"][0])}')



# Transformando dados de ratings

df['rate_transformed'] = df['rate'].astype(str)

df['rate_transformed'] = df['rate_transformed'].apply(lambda x: x.split('/')[0])



# Cuidando de entradas inválidas

df['rate_transformed'] = df['rate_transformed'].apply(lambda x: x.replace('NEW', str(np.nan)))

df['rate_transformed'] = df['rate_transformed'].apply(lambda x: x.replace('-', str(np.nan)))



# Transformando em float

df['rate_transformed'] = df['rate_transformed'].astype(float)

df.drop(['rate'], axis=1, inplace=True)

print(f'{type(df["rate_transformed"][0])}')
df.dtypes
# Droping NA from rate_transformed

df_unrated = df[df['rate_transformed'].isnull()]

df.dropna(subset=['rate_transformed', 'approx_cost(for two people)'], inplace=True)



# Drop columns

df.drop(['url', 'phone'], axis=1, inplace=True)



# Verificando

df.isnull().sum()
def format_spines(ax, right_border=True):

    """

    this function sets up borders from an axis and personalize colors

    """    

    # Setting up colors

    ax.spines['bottom'].set_color('#CCCCCC')

    ax.spines['left'].set_color('#CCCCCC')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#CCCCCC')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')
import seaborn as sb

# Value ditribution

plt.figure(1, figsize=(18, 7))

sb.set(style="whitegrid")

sb.countplot( x= 'rate_transformed', data=df)

plt.title('distribution of all rates')

plt.show()
df['rate_transformed'].describe()
grouped_rest = df.groupby(by='name', as_index=False).mean()

top_rating = grouped_rest.sort_values(by='rate_transformed', ascending=False).iloc[:10, np.r_[0, -1]]

top_rating
# Adjusting a restaurant name

top_rating.iloc[1, 0] = 'Spa Cuisine'



# Plotting

fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.barplot(y='name', x='rate_transformed', data=top_rating, palette='Blues_d')

ax.set_xlim([4.7, 5])

format_spines(ax, right_border=False)



for p in ax.patches:

    width = p.get_width()

    ax.text(width+0.01, p.get_y() + p.get_height() / 2. + 0.2, '{:1.2f}'.format(width), 

            ha="center", color='grey')



ax.set_title('Top 10 Restaurants in Bengaluru', size=14)

plt.show()
df['approx_cost(for two people)'].describe()
high_cost = grouped_rest.sort_values(by='approx_cost(for two people)', 

                                     ascending=False).iloc[:10, np.r_[0, -1, -2]]

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(x='name', y='approx_cost(for two people)', data=high_cost, ax=ax, palette='PuBu')

ax2 = ax.twinx()

sns.lineplot(x='name', y='rate_transformed', data=high_cost, ax=ax2, color='crimson', sort=False)

ax.tick_params(axis='x', labelrotation=90)

ax.set_ylim(700, 980)

ax2.set_ylim([3, 5])

format_spines(ax, right_border=True)

format_spines(ax2, right_border=True)

ax.xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = high_cost['rate_transformed']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



ax.set_title('Higher Cost Restaurants and the Rates', size=14)

plt.tight_layout()

plt.show()
low_cost = grouped_rest.sort_values(by='approx_cost(for two people)', 

                                     ascending=True).iloc[:10, np.r_[0, -1, -2]]

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(x='name', y='approx_cost(for two people)', data=low_cost, ax=ax, palette='PuBu')

ax2 = ax.twinx()

sns.lineplot(x='name', y='rate_transformed', data=low_cost, ax=ax2, color='crimson', sort=False)

ax.tick_params(axis='x', labelrotation=90)

ax.set_ylim([0, 2])

ax2.set_ylim([0, 5])

format_spines(ax, right_border=True)

format_spines(ax2, right_border=True)

ax.xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = low_cost['rate_transformed']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



ax.set_title('Lower Cost Restaurants and the Rates', size=14)

plt.tight_layout()
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(x='rate_transformed', y='approx_cost(for two people)', data=df, ax=ax)

format_spines(ax, right_border=False)

ax.set_title('Correlation Between Rate and Approx Cost', size=14)

plt.show()
# Separating by Online Order and Book Table options

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

sns.scatterplot(x='rate_transformed', y='approx_cost(for two people)', hue='online_order', 

                data=df, ax=axs[0])

sns.scatterplot(x='rate_transformed', y='approx_cost(for two people)', hue='book_table', 

                data=df, ax=axs[1])

format_spines(axs[0], right_border=False)

format_spines(axs[1], right_border=False)

axs[0].set_title('Cost and Rate Distribution by Online Order Option', size=14)

axs[1].set_title('Cost and Rate Distribution by Book Table Option', size=14)

plt.show()
# Contagem de restaurantes por oferta de Delivery

fig, axs = plt.subplots(1, 2, figsize=(15, 5))



sns.countplot(x='online_order', data=df, ax=axs[0], palette='Blues_d')

format_spines(axs[0], right_border=False)

ncount = len(df)

for p in axs[0].patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        axs[0].annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

axs[0].set_title('Counting of Restaurants by Online Order Service', size=14)



# Contagem de restaurantes por agendamento de mesa

sns.countplot(x='book_table', data=df, ax=axs[1], palette='Blues_d')

format_spines(axs[1], right_border=False)

ncount = len(df)

for p in axs[1].patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        axs[1].annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

axs[1].set_title('Counting Restaurants by Book Table Service', size=14)



plt.tight_layout()

plt.show()
x=df['online_order'].value_counts()

colors = ['#FEBFB3', '#E1396C']



trace=go.Pie(labels=x.index,values=x,textinfo="value",

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

layout=go.Layout(title="Accepting vs not accepting online orders",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')
x=df['book_table'].value_counts()

colors = ['#96D38C', '#D0F9B1']



trace=go.Pie(labels=x.index,values=x,textinfo="value",

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

layout=go.Layout(title="Table booking",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')
# Online order restaurants comparison

df_delivery = df.groupby(by='online_order').mean()

df_delivery
# Book table restaurants comparison

df_delivery = df.groupby(by='book_table').mean()

df_delivery
df['listed_in(type)'].value_counts()
type_rest = df.groupby(by='listed_in(type)').mean().sort_values(by='rate_transformed', ascending=False)

type_rest
rest_params = df.groupby(by='listed_in(type)', as_index=False).mean().sort_values(by='rate_transformed', 

                                                                                  ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(x='listed_in(type)', y='approx_cost(for two people)', data=rest_params, ax=ax, palette='Blues_d')

ax2 = ax.twinx()

sns.lineplot(x='listed_in(type)', y='rate_transformed', data=rest_params, ax=ax2, color='crimson', sort=False)

ax.tick_params(axis='x', labelrotation=90)

format_spines(ax, right_border=True)

format_spines(ax2, right_border=True)

ax.xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = rest_params['rate_transformed']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



ax.set_title('Average Cost and Rating of Restaurants by Type', size=14)

plt.tight_layout()
df['listed_in(city)'].value_counts()
city_rest = df.groupby(by='listed_in(city)').mean().sort_values(by='rate_transformed', ascending=False)

city_rest
city_rest = df.groupby(by='listed_in(city)', as_index=False).mean().sort_values(by='rate_transformed', 

                                                                                  ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))

sns.barplot(x='listed_in(city)', y='approx_cost(for two people)', data=city_rest, ax=ax, palette='Blues_d')

ax2 = ax.twinx()

sns.lineplot(x='listed_in(city)', y='rate_transformed', data=city_rest, ax=ax2, color='crimson', sort=False)

ax.tick_params(axis='x', labelrotation=90)

format_spines(ax, right_border=True)

format_spines(ax2, right_border=True)

ax.xaxis.set_label_text("")



xs = np.arange(0,len(city_rest),1)

ys = city_rest['rate_transformed']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



ax.set_title('Average Cost and Rating of Restaurants by City', size=14)

plt.tight_layout()
# Numerical features distribution

sns.set(style='white', palette='muted', color_codes=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

sns.despine(left=True)

axs[0] = sns.distplot(df['votes'], bins=20, ax=axs[0])

axs[1] = sns.distplot(df['approx_cost(for two people)'], bins=20, ax=axs[1], color='g')

axs[2] = sns.distplot(df['rate_transformed'], bins=20, ax=axs[2], color='r')



fig.suptitle('Numerical Feature Distribution')

plt.setp(axs, yticks=[])

plt.tight_layout()

plt.show()
# Numerical features distribution

sns.set(style='white', palette='muted', color_codes=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

sns.despine(left=True)

axs[0] = sns.distplot(np.log1p(df['votes']), bins=20, ax=axs[0])

axs[1] = sns.distplot(np.log1p(df['approx_cost(for two people)']), bins=20, ax=axs[1], color='g')

axs[2] = sns.distplot(np.log1p(df['rate_transformed']), bins=20, ax=axs[2], color='r')



fig.suptitle('Numerical Feature Log Distribution')

plt.setp(axs, yticks=[])

plt.tight_layout()

plt.show()
plt.figure(figsize=(7,7))

Rest_locations=df['location'].value_counts()[:20]

sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")
from pandas.plotting import scatter_matrix

attributes = ['online_order', 'book_table', 'rate_transformed', 'votes',

       'approx_cost(for two people)']

scatter_matrix(df[attributes], figsize=(12, 8))

plt.show()
# Reading the data again to make a complete pipeline

new_df = pd.read_csv('../input/zomato.csv')

new_df.head(1)
# Filtering data

important_columns = ['online_order', 'book_table', 'rate', 'votes', 'approx_cost(for two people)', 'rest_type', 'dish_liked', 'cuisines',

                     'listed_in(type)', 'listed_in(city)']

data_filtered = new_df.loc[:, important_columns]

data_filtered.head()
# Class for filtering data

class attrSelect(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        cols = ['online_order', 'book_table', 'rate', 'votes', 'approx_cost(for two people)',

                'listed_in(type)', 'listed_in(city)']

        

        return X.loc[:, cols]
# Creating a class for making some transformation

class transformData(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):        

        return self

    

    def transform(self, X, y=None):

        

        # Cost column transforming

        X['approx_cost(for two people)'] = X['approx_cost(for two people)'].astype(str)

        X['approx_cost(for two people)'] = X['approx_cost(for two people)'].apply(lambda x: x.replace(',', '.'))

        X['approx_cost(for two people)'] = X['approx_cost(for two people)'].astype(float)

        

        # Rate column transforming

        X['rate'] = X['rate'].astype(str).apply(lambda x: x.split('/')[0])

        X['rate'] = X['rate'].apply(lambda x: x.replace('NEW', str(np.nan)))

        X['rate'] = X['rate'].apply(lambda x: x.replace('-', str(np.nan)))

        X['rate'] = X['rate'].astype(float)

        

        return X


# Creating a class for handle null data

class handleNullData(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):        

        return self

    

    def transform(self, X, y=None):        

        # For now we will just drop null data. In the future we can try another option

        return X.dropna()
# Class for log transformation

class logTransformation(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):        

        return self

    

    def transform(self, X, y=None):        

        return np.log1p(X)
# Splitting data

X = new_df.copy()

attr_selector = attrSelect()

X_filtered = attr_selector.fit_transform(X)



X_train, X_test = train_test_split(X, test_size=.20, random_state=42)



# Defining a pipeline

num_attribs = ['votes', 'approx_cost(for two people)']

cat_attribs = ['online_order', 'book_table', 'listed_in(type)', 'listed_in(city)']

all_attribs = num_attribs + cat_attribs

X_num = X_train.loc[:, num_attribs]

X_cat = X_train.loc[:, cat_attribs]



# Common pipeline

common_pipeline = Pipeline([

    ('attr_selector', attrSelect()),

    ('data_transformer', transformData()),

    ('null_handler', handleNullData()),

])



# Numerical pipeline

num_pipeline_first_approach = Pipeline([

    ('log_transformer', logTransformation()),

])



# Categorical pipeline

cat_pipeline_first_approach = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False)),

])



# Full pipeline

full_pipeline_first_approach = ColumnTransformer([

    ('num', num_pipeline_first_approach, num_attribs),

    ('cat', cat_pipeline_first_approach, cat_attribs),

])
# Preprocessing data

X_prep_temp = common_pipeline.fit_transform(new_df)

X = X_prep_temp.drop('rate', axis=1)

y = X_prep_temp['rate']

y_log = np.log1p(y)



#X = pd.get_dummies(X).values

#y = pd.get_dummies(y).values



# Spliting and preparing data

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=.20, random_state=42)

X_train_prepared = full_pipeline_first_approach.fit_transform(X_train)

X_test_prepared = full_pipeline_first_approach.fit_transform(X_test)

X_train_prepared[0]



#print(X_train_prepared.shape)

#X_train_prepared = X_train_prepared.reshape((18607, 6702))

#print(X_train_prepared.shape)

#X_test_prepared = X_test_prepared.reshape((18542872, 1))

#print(X_test_prepared.shape)
X_test_prepared


# Functions for report

def create_dataset():

    """

    This functions creates a dataframe to keep performance analysis

    """

    attributes = ['model', 'rmse_train', 'rmse_cv', 'rmse_test', 'total_time']

    model_performance = pd.DataFrame({})

    for col in attributes:

        model_performance[col] = []

    return model_performance



def model_results(models, X_train, y_train, X_test, y_test, df_performance, cv=5, 

                  scoring='neg_mean_squared_error'):

    for name, model in models.items():

        t0 = time.time()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)

        train_rmse = mean_squared_error(y_train, train_pred)

        score_train = r2_score(y_train, train_pred)  

        train_cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)

        train_cv_rmse = np.sqrt(-train_cv_scores).mean()

        

       # print("score: %.2f" % (score_train))

        print("Score:", score_train)

        print("Mean:", score_train.mean())

        print("Standard deviation:", score_train.std())



          

        

        test_pred = model.predict(X_test)

        test_rmse = mean_squared_error(y_test, test_pred)

        score_test = r2_score(y_test, test_pred)

        t1 = time.time()

        delta_time = t1-t0

        model_name = model.__class__.__name__

        # print("score: %.2f" % (score_train))

        #print("Score:", score_test)

        #print("Mean:", score_test.mean())

        #print("Standard deviation:", score_test.std())

        performances = {}

        performances['model'] = model_name

        performances['rmse_train'] = round(train_rmse, 4)

        performances['rmse_cv'] = round(train_cv_rmse, 4)

        performances['rmse_test'] = round(test_rmse, 4)

        performances['total_time'] = round(delta_time, 3)

        df_performance = df_performance.append(performances, ignore_index=True)

        

    return df_performance
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()



df_performance = create_dataset()

regressors = {

    'lin': LinearRegression(),

    'ridge': Ridge(),

    'lasso': Lasso(),

    'elastic': ElasticNet(),

    'forest': RandomForestRegressor(),

    'KNN': GridSearchCV(knn, params, cv=5)

}

df_performance = model_results(regressors, X_train_prepared, y_train, X_test_prepared, y_test, df_performance)

df_performance.set_index('model', inplace=True)

cm = sns.light_palette("cornflowerblue", as_cmap=True)

df_performance.style.background_gradient(cmap=cm)

def calc_rmse(model, X, y, cv=5, scoring='neg_mean_squared_error'):

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return np.sqrt(-scores).mean()
# Alpha analysis

alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

lasso_scores = []

for a in alphas:

    lasso_scores.append(calc_rmse(Lasso(alpha=a), X_train_prepared, y_train))

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(alphas, lasso_scores)

format_spines(ax, right_border=False)

ax.set_title('Alpha - Lasso Regression')

plt.show()
param_grid = [

    {'n_estimators': [30, 40, 50, 75, 90], 'max_features': [10, 12, 15, 20, 25]},

]



# Criando regressor

forest_reg = RandomForestRegressor()



# Treinando e procurando a melhor combinação

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train_prepared, y_train)

best_forest_rmse = np.sqrt(-grid_search.best_score_)

best_forest_rmse
# Alpha analysis

alphas = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

ridge_scores = []

for a in alphas:

    ridge_scores.append(calc_rmse(Ridge(alpha=a), X_train_prepared, y_train))

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(alphas, ridge_scores)

format_spines(ax, right_border=False)

ax.set_title('Alpha - Ridge Regression')

plt.show()
single_performance = create_dataset()

final_model = {

    'best_ridge': Ridge(alpha=30)

}

single_performance = model_results(final_model, X_train_prepared, y_train, X_test_prepared, y_test, 

                                   single_performance)

single_performance.set_index('model', inplace=True)

single_performance
# Training a definitive model

ridge_reg = Ridge(alpha=30)

ridge_reg.fit(X_train_prepared, y_train)
# Create function

def chooseOnlineOrder(rnd=True):

    if rnd:

        return 'Yes' if np.random.randint(1, 3) == 1 else 'No'  

    online_order_opt = 0

    while True:

        try:

            online_order_opt = int(input('Online Order? \n(1) Yes\n(2) No\n'))

            if online_order_opt in (1, 2):

                online_order = 'Yes' if online_order_opt == 1 else 'No'

                break

            else:

                print('Please, input a number between 1 and 2')

        except ValueError:

            print('Please, input a number between 1 and 2.')

    return online_order



def chooseBookTable(rnd=True):

    if rnd:

        return 'Yes' if np.random.randint(1, 3) == 1 else 'No'  

    book_table_opt = 0

    while True:

        try:

            book_table_opt = int(input('Book Table? \n(1) Yes\n(2) No\n'))

            if book_table_opt in (1, 2):

                book_table = 'Yes' if book_table_opt == 1 else 'No'

                break

            else:

                print('Please, input a number between 1 and 2')

        except ValueError:

            print('Please, input a number between 1 and 2.')

    return book_table



def chooseVotes(rnd=True):

    if rnd:

        return int(np.random.randint(1, 1001))

    while True:

        try:

            votes = int(input('Votes: '))

            if votes < 0:

                print('Please, insert a positive number')

            else:

                break

        except ValueError:

            print('Please, insert number.')

    return votes



def chooseApproxCost(rnd=True):

    if rnd:

        return float(np.random.randint(1, 1001))

    while True:

        try:

            approx_cost = int(input('Approx cost (for two people: '))

            if approx_cost < 0:

                print('Please, insert a positive number')

            else:

                break

        except ValueError:

            print('Please, insert a number.')

    return approx_cost



def chooseRestType(rnd=True):

    listed_in_select = list(X_train['listed_in(type)'].value_counts().index)

    idx_list = np.arange(len(list(X_train['listed_in(type)'].value_counts().index)))

    if rnd:

        return list(zip(idx_list, listed_in_select))[np.random.randint(1, 8)-1][1]

    print('\nChoose one option for Listed in (type): ')

    for idx, tipo in zip(idx_list, listed_in_select):

        print(f'({idx+1}) {tipo}')

    listed_in_opt = 0

    while True:

        try:

            listed_in_opt = int(input())

            if listed_in_opt in range(1, 8):

                listed_in_type = list(zip(idx_list, listed_in_select))[listed_in_opt-1][1]

                break

            else:

                print('Please, input a number between 1 and 7.')

        except ValueError:

            print('Please, input a number between 1 and 7.')

    return listed_in_type



def chooseRestCity(rnd=True):

    listed_in_select = list(X_train['listed_in(city)'].value_counts().index)

    idx_list = np.arange(len(list(X_train['listed_in(city)'].value_counts().index)))

    if rnd:

        return list(zip(idx_list, listed_in_select))[np.random.randint(1, 30)-1][1]

    print('\nChoose one option for Listed in (city): ')

    for idx, city in zip(idx_list, listed_in_select):

        print(f'({idx+1}) {city}')

    listed_in_opt = 0

    while True:

        try:

            listed_in_opt = int(input())

            if listed_in_opt in range(1, 31):

                listed_in_city = list(zip(idx_list, listed_in_select))[listed_in_opt-1][1]

                break

            else:

                print('Please, input a number between 1 and 30.')

        except ValueError:

            print('Please, input a number between 1 and 30.')

    return listed_in_city
# Generating new data

def generateNewData(qtd_sample, cols, random=True):

    new_data = pd.DataFrame({})

    for col in cols:

        new_data[col] = []

    new_data_dict = {}

    for i in range(qtd_sample):

        new_data_dict['online_order'] = chooseOnlineOrder(random)

        new_data_dict['book_table'] = chooseBookTable(random)

        new_data_dict['votes'] = chooseVotes(random)

        new_data_dict['approx_cost(for two people)'] = chooseApproxCost(random)

        new_data_dict['listed_in(type)'] = chooseRestType(random)

        new_data_dict['listed_in(city)'] = chooseRestCity(random)

        new_data = new_data.append(new_data_dict, ignore_index=True)

    return new_data  
cols = X_train.columns

new_data = generateNewData(50, cols, random=True)

new_data.head()
# Preparing new data and predicting

new_data_prepared = full_pipeline_first_approach.transform(new_data)

predictions = ridge_reg.predict(new_data_prepared)

rates = np.exp(predictions)

new_data['predicted_rate'] = rates

new_data.head()
rest_params = df.groupby(by='listed_in(type)', as_index=False).mean().sort_values(by='rate_transformed', 

                                                                                  ascending=False)

fig, axs = plt.subplots(2, 1, figsize=(11, 13))

sns.barplot(x='listed_in(type)', y='approx_cost(for two people)', data=rest_params, ax=axs[0], palette='Blues_d')

ax2 = axs[0].twinx()

sns.lineplot(x='listed_in(type)', y='rate_transformed', data=rest_params, ax=ax2, color='crimson', sort=False)

axs[0].tick_params(axis='x', labelrotation=90)

format_spines(axs[0], right_border=True)

format_spines(ax2, right_border=True)

axs[0].xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = rest_params['rate_transformed']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



# Data generated

rest_params = new_data.groupby(by='listed_in(type)', as_index=False).mean().sort_values(by='predicted_rate',                                                                                   ascending=False)

sns.barplot(x='listed_in(type)', y='approx_cost(for two people)', data=rest_params, ax=axs[1], palette='Blues_d')

ax3 = axs[1].twinx()

sns.lineplot(x='listed_in(type)', y='predicted_rate', data=rest_params, ax=ax3, color='crimson', sort=False)

axs[1].tick_params(axis='x', labelrotation=90)

format_spines(axs[1], right_border=True)

format_spines(ax3, right_border=True)

axs[1].xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = rest_params['predicted_rate']



for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')



axs[0].set_title('Average Cost and Rating of Restaurants by Type', size=14)

axs[1].set_title('Predicted Rate of Restaurants by Type', size=14)

plt.tight_layout()

plt.show()
data = X_prep_temp.copy()

bin_edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

bin_names = [1, 2, 3, 4, 5]

data['rate_category'] = pd.cut(data['rate'], bin_edges, labels=bin_names)

data.drop('rate', axis=1, inplace=True)

data.head()
# Splitting data

X = data.drop('rate_category', axis=1)

y = data['rate_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)



X_train_prepared = full_pipeline_first_approach.fit_transform(X_train)

X_test_prepared = full_pipeline_first_approach.transform(X_test)
y_train.value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from sklearn.model_selection import cross_val_predict



log_reg = LogisticRegression()

log_reg.fit(X_train_prepared, y_train)

predictions = cross_val_predict(log_reg, X_train_prepared, y_train, cv=5)
print(classification_report(y_train, predictions))
data = X_prep_temp.copy()

data['rounded_rate'] = data['rate'].apply(lambda x: round(x))

data.drop('rate', axis=1, inplace=True)

data.head()
# Balance of classification target

data['rounded_rate'].value_counts()
# Splitting data

X = data.drop('rounded_rate', axis=1)

y = data['rounded_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)



X_train_prepared = full_pipeline_first_approach.fit_transform(X_train)

X_test_prepared = full_pipeline_first_approach.transform(X_test)



log_reg = LogisticRegression()

log_reg.fit(X_train_prepared, y_train)

pred = log_reg.predict(X_train_prepared)

print(classification_report(y_train, pred))
predictions = cross_val_predict(log_reg, X_train_prepared, y_train, cv=5)

print(classification_report(y_train, predictions))
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier()

forest_clf.fit(X_train_prepared, y_train)

predictions = cross_val_predict(forest_clf, X_train_prepared, y_train, cv=5)

print(classification_report(y_train, predictions))
import numpy as np 

import pandas as pd

import os

import seaborn as sns

print(os.listdir("../input"))

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)

from wordcloud import WordCloud

from geopy.geocoders import Nominatim

from folium.plugins import HeatMap

import folium

from tqdm import tqdm

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

from collections import Counter

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.colors as mcolors

from sklearn.manifold import TSNE

from gensim.models import word2vec

import nltk

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/zomato.csv")
all_ratings = []



for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):

    ratings = eval(ratings)

    for score, doc in ratings:

        if score:

            score = score.strip("Rated").strip()

            doc = doc.strip('RATED').strip()

            score = float(score)

            all_ratings.append([name,score, doc])
rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])

rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))
rating_df.to_csv("Ratings.csv")
rating_df.head()
plt.figure(figsize=(7,6))

rating=rating_df['rating'].value_counts()

sns.barplot(x=rating.index,y=rating)

plt.xlabel("Ratings")

plt.ylabel('count')
rating_df['sent']=rating_df['rating'].apply(lambda x: 1 if int(x)>2.5 else 0)

stops=stopwords.words('english')

lem=WordNetLemmatizer()

corpus=' '.join(lem.lemmatize(x) for x in rating_df[rating_df['sent']==1]['review'][:3000] if x not in stops)

tokens=word_tokenize(corpus)
vect=TfidfVectorizer()

vect_fit=vect.fit(tokens)

    
id_map=dict((v,k) for k,v in vect.vocabulary_.items())

vectorized_data=vect_fit.transform(tokens)

gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)

ldamodel = gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)

counter=Counter(corpus)
stops=stopwords.words('english')

lem=WordNetLemmatizer()

corpus=' '.join(lem.lemmatize(x) for x in rating_df[rating_df['sent']==0]['review'][:3000] if x not in stops)

tokens=word_tokenize(corpus)
vect=TfidfVectorizer()

vect_fit=vect.fit(tokens)

id_map=dict((v,k) for k,v in vect.vocabulary_.items())

vectorized_data=vect_fit.transform(tokens)

gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)

ldamodel = gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)
counter=Counter(corpus)

out=[]

topics=ldamodel.show_topics(formatted=False)

for i,topic in topics:

    for word,weight in topic:

        out.append([word,i,weight,counter[word]])



dataframe = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        





# Plot Word Count and Weights of Topic Keywords

fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

for i, ax in enumerate(axes.flatten()):

    ax.bar(x='word', height="word_count", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')

    ax_twin = ax.twinx()

    ax_twin.bar(x='word', height="importance", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.2, label='Weights')

    ax.set_ylabel('Word Count', color=cols[i])

    #ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)

    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)

    ax.tick_params(axis='y', left=False)

    ax.set_xticklabels(dataframe.loc[dataframe.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')

    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')



fig.tight_layout(w_pad=2)    

fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=8, y=1.05)    

plt.show()
stops=set(stopwords.words('english'))

lem=WordNetLemmatizer()

corpus=[]

for review in tqdm(rating_df['review'][:10000]):

    words=[]

    for x in word_tokenize(review):

        x=lem.lemmatize(x.lower())

        if x not in stops:

            words.append(x)

            

    corpus.append(words)
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

             

    plt.figure(figsize=(10, 10)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
postive=rating_df[rating_df['rating']>3]['review'][:2000]

negative=rating_df[rating_df['rating']<2.5]['review'][:2000]



def return_corpus(df):

    corpus=[]

    for review in df:

        tagged=nltk.pos_tag(word_tokenize(review))

        adj=[]

        for x in tagged:

            if x[1]=='JJ':

                adj.append(x[0])

        corpus.append(adj)

    return corpus
corpus=return_corpus(postive)

model = word2vec.Word2Vec(corpus, size=100, min_count=10,window=20, workers=4)

tsne_plot(model)
rating_df['sent']=rating_df['rating'].apply(lambda x: 1 if int(x)>2.5 else 0)
max_features=3000

tokenizer=Tokenizer(num_words=max_features,split=' ')

tokenizer.fit_on_texts(rating_df['review'].values)

X = tokenizer.texts_to_sequences(rating_df['review'].values)

X = pad_sequences(X)
embed_dim = 32

lstm_out = 32



model = Sequential()

model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))

#model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
Y = pd.get_dummies(rating_df['sent'].astype(int)).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)


batch_size = 3200

history = model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size)

history.history
validation_size = 1500



X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
import json

with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss']].plot()

history_df[['acc']].plot()