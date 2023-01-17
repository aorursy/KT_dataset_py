import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

import warnings



%matplotlib inline

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,BayesianRidge

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from matplotlib import rcParams



warnings.filterwarnings('ignore')
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

    

    

# Class for log transformation

class logTransformation(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):        

        return self

    

    def transform(self, X, y=None):        

        return np.log1p(X)

    

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



        train_cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)

        train_cv_rmse = np.sqrt(-train_cv_scores).mean()



        test_pred = model.predict(X_test)

        test_rmse = mean_squared_error(y_test, test_pred)

        t1 = time.time()

        delta_time = t1-t0

        model_name = model.__class__.__name__



        performances = {}

        performances['model'] = model_name

        performances['rmse_train'] = round(train_rmse, 4)

        performances['rmse_cv'] = round(train_cv_rmse, 4)

        performances['rmse_test'] = round(test_rmse, 4)

        performances['total_time'] = round(delta_time, 3)

        df_performance = df_performance.append(performances, ignore_index=True)

    plotting_values("Root Mean Square Train Set",df_performance.model.values,df_performance.rmse_train.values)

    plt.show();

    plotting_values("Root Mean Square Cross Validation Set",df_performance.model.values,df_performance.rmse_cv.values)

    plt.show();

    plotting_values("Root Mean Square Test Set",df_performance.model.values,df_performance.rmse_test.values)

    plt.show();

    plotting_values("Total Time",df_performance.model.values,df_performance.total_time.values)

    plt.show();

        

    return df_performance



def plotting_values(title,x,values):

    

    plt.title(title)

    plt.bar(x,values)

    

def calc_rmse(model, X, y, cv=5, scoring='neg_mean_squared_error'):

    scores = cross_val_score(

        model, X, y, cv=cv, scoring=scoring)

    return np.sqrt(-scores).mean()

data = pd.read_csv("../input/zomato.csv")
del data['url']

del data['address']

del data['phone']

del data['location']
data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality','listed_in(type)': 'restaurant_type'}, inplace=True)

data.head()
print("Percentage null or na values in df")

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
len(data)
# values_to_drop = data.columns.values

# values_to_drop = values_to_drop[values_to_drop != 'dish_liked']
data.rate = data.rate.replace("NEW", np.nan)

data.dropna(how ='any', inplace = True)
print("Percentage null or na values in df")

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
data.rate = data.rate.astype(str)

data.rate = data.rate.apply(lambda x: x.replace('/5',''))

data.rate = data.rate.apply(lambda x: float(x))

data.head()
data.average_cost = data.average_cost.str.replace(',','').astype('float')
data = data.drop_duplicates(subset='name',keep='last') 
fig = plt.figure(figsize=(16,12))

g = sns.countplot(x="locality",data=data, palette = "Set1" ,order = data['locality'].value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('locality',size = 20);
dummy = data.cuisines.str.get_dummies(sep=', ')
dummy.columns
dummy["Tea"] = dummy["Tea"] + dummy["Bubble Tea"]

del dummy['Bubble Tea']
cuisine_of_res = {}



for typ in dummy.columns.values[13:]:

    cuisine_of_res[typ] = dummy[typ].value_counts()[1]



fig = plt.figure(figsize=(16,12))

plt.xticks(rotation='vertical')

plt.bar(cuisine_of_res.keys(), cuisine_of_res.values(), color='g');
data  = pd.concat([data,dummy],axis=1)
dummy2 = data.dish_liked.str.get_dummies(sep=', ')
type_of_dish = {}



for typ in dummy2.columns.values[13:]:

    type_of_dish[typ] = dummy2[typ].value_counts()[1]





import operator

sorted_x = sorted(type_of_dish.items(), key=operator.itemgetter(1),reverse=True)



print(sorted_x[:20])
dummy3 = data.restaurant_type.str.get_dummies(sep=', ')
type_of_res = {}



for typ in dummy3.columns.values:

    type_of_res[typ] = dummy3[typ].value_counts()[1]

fig = plt.figure(figsize=(16,12))

plt.xticks(rotation='vertical')

plt.bar(type_of_res.keys(), type_of_res.values(), color='rgb');
df  = pd.concat([data,dummy3],axis=1)
del df['name']

del df['dish_liked']

del df['cuisines']

del df['reviews_list']

del df['menu_item']

del df['restaurant_type']

del df['rest_type']
dummy4 = pd.get_dummies(df['locality'])

del df['locality']

dummy5 = pd.get_dummies(df['online_order'],prefix='online_order')

dummy6 = pd.get_dummies(df['book_table'],prefix='book_table')

del df['online_order']

del df['book_table']

df  = pd.concat([df,dummy5,dummy6],axis=1)
df.head()
df.columns.values 
y = df.loc[:,'rate'].values

X = df.drop('rate', axis=1).values
sns.kdeplot(y);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
df_performance = create_dataset()

regressors = {

    'lin': LinearRegression(),

    'forest': RandomForestRegressor(),

    'SVR':SVR(),

}

df_performance = model_results(regressors, X_train, y_train, X_test, y_test, df_performance)

df_performance.set_index('model', inplace=True)

cm = sns.light_palette("cornflowerblue", as_cmap=True)

df_performance.style.background_gradient(cmap=cm)
rf  = RandomForestRegressor()

pred = rf.fit(X_train, y_train)



y_pred = pred.predict(X_test)



df_pred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df_pred.head()
from sklearn import metrics





print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.kdeplot(y_pred);
fig = plt.figure(figsize=(16,12))

sns.kdeplot(y_test,label='y_test')

sns.kdeplot(y_pred,label='y_pred')

plt.show()