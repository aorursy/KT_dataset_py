#import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Load the data

movies = pd.read_csv('../input/train.csv')
#print shape of the dataframe

print(movies.shape)
movies.info()
#show top five observations of the data

pd.options.display.max_columns = 23

pd.set_option('max_colwidth', 100)



movies.head()
##dataframe containing percentage of null values for each of the variables

import warnings

warnings.filterwarnings('ignore')

no_of_null = movies.isnull().sum().sort_values(ascending=False)

percent = (no_of_null*100/len(movies)).sort_values(ascending=False)

percent_null_df = pd.concat([no_of_null, percent], axis=1, keys=['total','percent'])

percent_null_df
#getting rid of columns with more than 10% as null

movies = movies[percent_null_df[percent_null_df['percent'] < 10].index]
movies.head()
movies.isnull().sum()
#null value imputation with mean

ind = movies[movies['runtime'].isnull()].index

movies.iloc[ind]['runtime'] = movies['runtime'].mean()
movies.describe()
#dropping null observations

movies = movies.dropna()

movies = movies.reset_index()
movies.shape
from tqdm import tqdm_notebook as tqdm
#using the 'eval()' function to convert from string type to list type

#eval() evaluates the passed string as a Python expression and returns the result.

dictionary = eval(movies['production_companies'][0])

dictionary
#convert from string to list type

columns = ['genres','production_companies','production_countries','spoken_languages','Keywords','cast']

for c in tqdm(columns):

    movies[c] = movies[c].apply(lambda x: eval(x))
def name(dict_list):

    lst = []

    for d in dict_list:

        lst.append(d['name'])

    return lst





for c in tqdm(columns):

    movies[c] = movies[c].apply(name)

    
#dictionary storing unique categories

unique_dict = {}

for c in tqdm(columns):

    temp_list = []

    for d in tqdm(movies[c]):

        for i in d:

            if i not in temp_list:

                temp_list.append(i)

    unique_dict[c] = temp_list
#number of unique values for each categorical column

for c in columns:

    print(c, ' : ', len(unique_dict[c]))
#create binary variables for the categories

for c in tqdm(columns):

    for d in tqdm(unique_dict[c]):

        movies[d] = movies[c].apply(lambda x: 1 if d in x else 0)
movies['original_language'].value_counts()
#Create binary variables for the 'original language' column

movies = pd.concat([movies,pd.get_dummies(movies['original_language'])], axis=1)
#current shape of the dataframe after feature extraction

movies.shape
#dataframe of numeric columns

movies_numeric = movies.select_dtypes(include=['float','int','int64','float64', 'bool'])

movies_numeric.shape
#Count number of features whose sum is less than 5

low = []



for col, val in movies_numeric.sum().iteritems():

    if val < 5:

        low.append(col)

                

print("Binary features that appear less than 5 times in the data-set:", len(low))

#drop features whose sum is less than 5

movies = movies.drop(low, axis=1)

movies.shape
movies['crew'][0]
crew = movies['crew'].apply(lambda x: eval(x))
crew[0][5]['gender']
#extracting unique department names

u_departments = []

for l in crew:

    for i in l:

        if i['department'] not in u_departments:

            u_departments.append(i['department'])
u_departments
def dep_counts(x, dep):

    count = 0

    for d in x:

        if dep in d['department']:

            count += 1

    return count
#Applying function to count number of each department in each of the rows and column

movies['crew_team'] = crew



for ud in tqdm(u_departments):

    ##creating features showing department strength for each row

    movies[ud] = movies['crew_team'].apply(lambda x: dep_counts(x, ud))
movies.shape
#function to calculate gender strength in a movies crew team

def gen_counts(x, gen):

    count = 0

    for g in x:

        if gen == g['gender']:

            count += 1

    return count



for ug in [0,1,2]:

    ##creating features showing gender strength

    movies[ug] = movies['crew_team'].apply(lambda x: gen_counts(x, ug))
movies['crew_size'] = movies[0]+movies[1]+movies[2]
#calculate percentage of gender count

for gender in [0,1,2]:

    movies[gender] = movies[gender]/movies['crew_size']
plt.figure(figsize=(12,8))

g = sns.distplot(movies['revenue'], kde_kws={"color": (61/255,75/255,222/255), "lw": 1.5, "label": "KDE"},

            hist_kws={"color": (100/255,221/255,225/255), "lw": 0.5})

g.set_ylabel('Density', fontsize=15)

g.set_xlabel('Revenue',  fontsize=15, labelpad=20)

g.yaxis.label.set_color((120/255,120/255,120/255))

g.xaxis.label.set_color((120/255,120/255,120/255))

g.spines['bottom'].set_color((120/255,120/255,120/255))

g.spines['left'].set_color((120/255,120/255,120/255))

g.tick_params(axis='x', colors=(120/255,120/255,120/255))

g.tick_params(axis='y', colors=(120/255,120/255,120/255))

g.set_title("Distribution of revenue")

sns.despine(offset=10, trim=True)
plt.figure(figsize=(10,6))

sns.distplot(np.log1p(movies['revenue']))
movies['log_revenue'] = np.log1p(movies['revenue'])
fig = plt.figure(figsize=(15,5))

g1 = fig.add_subplot(1,2,1)

g1 = sns.scatterplot(x=movies['budget'], y=movies['revenue'], color = (23/255,201/255,106/255))

g1.set_ylabel('Revenue', fontsize=15)

g1.set_xlabel('Budget',  fontsize=15, labelpad=20)

g1.yaxis.label.set_color((120/255,120/255,120/255))

g1.xaxis.label.set_color((120/255,120/255,120/255))

g1.spines['bottom'].set_color((120/255,120/255,120/255))

g1.spines['left'].set_color((120/255,120/255,120/255))

g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

g1.set_title("Revenue vs Budget")

sns.despine(offset=10, trim=True)



g2 = fig.add_subplot(1,2,2)

g2 = sns.scatterplot(x=movies['runtime'], y=movies['revenue'], color = (74/255,144/255,185/255))

g2.set_ylabel('Revenue', fontsize=15)

g2.set_xlabel('Runtime',  fontsize=15, labelpad=20)

g2.yaxis.label.set_color((120/255,120/255,120/255))

g2.xaxis.label.set_color((120/255,120/255,120/255))

g2.spines['bottom'].set_color((120/255,120/255,120/255))

g2.spines['left'].set_color((120/255,120/255,120/255))

g2.tick_params(axis='x', colors=(120/255,120/255,120/255))

g2.tick_params(axis='y', colors=(120/255,120/255,120/255))

g2.set_title("Revenue vs Runtime")

sns.despine(offset=10, trim=True)
movies[u_departments].describe()
fig = plt.figure(figsize=(25,20))

plt.suptitle("Revenue vs department size")



for i,d in tqdm(enumerate(u_departments)):

    g1=fig.add_subplot(4,3,i+1)

    g1=sns.lineplot(movies[d], movies['revenue'], color=(255/255,145/255,0/255))

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=10, trim=True)

    
fig = plt.figure(figsize=(15,3))

plt.suptitle("Revenue vs gender strength")



for i,d in tqdm(enumerate([0,1,2])):

    g1=fig.add_subplot(1,3,i+1)

    g1=sns.lineplot(movies[d], movies['revenue'], color=(255/255,100/255,100/255))

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=10, trim=True)
fig = plt.figure(figsize=(25,35))

plt.suptitle('Box plot of revenue vs genre')

for i, c in tqdm(enumerate([col for col in movies.columns if col in unique_dict['genres']])):

    g1=fig.add_subplot(7,3,i+1)

    g1 = sns.boxplot(movies[c], movies['revenue'], palette = 'coolwarm', width=0.5, linewidth = 0.5)

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=5, trim=True)
unique_dict.keys()


fig = plt.figure(figsize=(25,40))

plt.suptitle('Box plot of revenue for top countries Country')

for i, c in tqdm(enumerate([col for col in movies.columns if col in unique_dict['production_countries']])):

    g1=fig.add_subplot(10,4,i+1)

    sns.boxplot(movies[c], movies['revenue'], width=0.5, linewidth = 0.5, )

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=5, trim=True)

    #plt.legend()
#Top 30 production companies

u_prodc = [col for col in movies.columns if col in unique_dict['production_companies']]

top_30_prodc = movies[u_prodc].sum().sort_values(ascending=False).iloc[:30].index


    

fig = plt.figure(figsize=(25,40))

plt.suptitle('Box plot of revenue for the top 30 production companies')

for i, c in tqdm(enumerate(top_30_prodc)):

    g1=fig.add_subplot(10,3,i+1)

    sns.boxplot(movies[c], movies['revenue'], width=0.5, linewidth = 0.5, )

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=5, trim=True)
movies['num_companies'] = 0

for p in top_30_prodc:

    movies['num_companies'] = movies['num_companies'] + movies[p]

    

plt.figure(figsize=(10,6))

sns.boxplot(movies['num_companies'], movies['revenue'], palette='terrain')

plt.xlabel('Number of production companies')

sns.despine()
fig = plt.figure(figsize=(20,15))

plt.suptitle("Revenue vs gender strength")



for i,d in tqdm(enumerate([0,1,2])):

    g1=fig.add_subplot(3,1,i+1)

    g1=sns.lineplot(movies[d], movies['revenue'], color=(255/255,145/255,0/255))

    g1.spines['bottom'].set_color((120/255,120/255,120/255))

    g1.spines['left'].set_color((120/255,120/255,120/255))

    g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

    g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

    sns.despine(offset=10, trim=True)

    
fig = plt.figure(figsize=(20,10))

plt.suptitle("Revenue vs Crew strength")



g1=sns.lineplot(movies['crew_size'], movies['revenue'], color=(255/255,145/255,0/255))

g1.spines['bottom'].set_color((120/255,120/255,120/255))

g1.spines['left'].set_color((120/255,120/255,120/255))

g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

sns.despine(offset=10, trim=True)
import datetime as dt

movies['release_date'] = pd.to_datetime(movies['release_date'])

movies['release_year'] = movies['release_date'].dt.year
plt.figure(figsize=(6,4))

g1=sns.lineplot(movies['release_year'], movies['revenue'], color=(255/255,145/255,0/255))

g1.spines['bottom'].set_color((120/255,120/255,120/255))

g1.spines['left'].set_color((120/255,120/255,120/255))

g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

sns.despine(offset=10, trim=True)
movies[movies['release_year']>2018][['release_year', 'release_date', 'status', 'title']]
correct_year = movies.iloc[movies[movies['release_year']>2019].index,:]

movies.drop(movies[movies['release_year']>2019].index, axis=0, inplace=True)

correct_year['release_year'] = correct_year['release_year'] - 100

movies = pd.concat([movies, correct_year], axis = 0)
movies['release_month'] = movies['release_date'].dt.month
plt.figure(figsize=(6,4))

g1=sns.lineplot(movies['release_year'], movies['revenue'], color=(255/255,145/255,0/255))

g1.spines['bottom'].set_color((120/255,120/255,120/255))

g1.spines['left'].set_color((120/255,120/255,120/255))

g1.tick_params(axis='x', colors=(120/255,120/255,120/255))

g1.tick_params(axis='y', colors=(120/255,120/255,120/255))

sns.despine(offset=10, trim=True)



plt.figure(figsize=(6,4))

g2=sns.lineplot(movies['release_month'], movies['revenue'], color=(255/255,145/255,0/255))

g2.spines['bottom'].set_color((120/255,120/255,120/255))

g2.spines['left'].set_color((120/255,120/255,120/255))

g2.tick_params(axis='x', colors=(120/255,120/255,120/255))

g2.tick_params(axis='y', colors=(120/255,120/255,120/255))

sns.despine(offset=10, trim=True)
movies.head()
movies.drop(['index','genres','imdb_id','original_title','overview','poster_path','production_companies','production_countries',

'spoken_languages','status','title','Keywords','cast','crew','release_date','release_year','id','index','log_revenue','original_language'], axis=1, inplace=True)
rev_corr = movies.corr()['revenue'].sort_values(ascending=False)
cor25 = movies[list(rev_corr[rev_corr>0.25].index) + list(rev_corr[rev_corr < -0.25].index)].corr()
plt.figure(figsize=(25,25))

sns.heatmap(cor25, annot=True)
#iterlist = list(cor25.columns)

#iterlist.remove('revenue')

#iterlist



print('Mean revenue', np.mean(movies['revenue']))

print('Minimum revenue', np.min(movies['revenue']))

print('Revenue range:', np.max(movies['revenue']) - np.min(movies['revenue']))



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

lr = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(movies.drop('revenue', axis=1), movies['revenue'], test_size=0.25)



feat1 = ['crew_size', 'adventure', 'Stan Lee', '3d', 'Walt Disney Pictures']

feat2 = ['budget',  'adventure', 'Stan Lee', '3d', 'Walt Disney Pictures']

#cor_series = cor25.loc[feat]

feat3 = ['budget', 'Stan Lee']



for feat in [feat1, feat2, feat3]:

    lr.fit(x_train[feat], y_train)

    predictions = lr.predict(x_test[feat])

    rmse = mean_squared_error(y_test, predictions)**0.5

    r2 = r2_score(y_test, predictions)**0.5

    print('\n')

    print(feat)

    print('RMSE', rmse)

    print('R2', r2)

log_movies = movies.select_dtypes(include=['int','int64','float','float64','bool'])

log_movies = log_movies.transform(np.log1p)



print('Mean revenue', np.mean(log_movies['revenue']))

print('Minimum revenue', np.min(log_movies['revenue']))

print('Revenue range:', np.max(log_movies['revenue']) - np.min(log_movies['revenue']))



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

lr = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(log_movies.drop('revenue', axis=1), log_movies['revenue'], test_size=0.25)



feat1 = ['crew_size', 'adventure', 'Stan Lee', '3d', 'Walt Disney Pictures']

feat2 = ['budget',  'adventure', 'Stan Lee', '3d', 'Walt Disney Pictures']

feat3 = ['budget', 'Stan Lee']



for feat in [feat1, feat2, feat3]:

    lr.fit(x_train[feat], y_train)

    predictions = lr.predict(x_test[feat])

    rmse = mean_squared_error(np.exp(y_test), np.exp(predictions))**0.5

    r2 = r2_score(np.exp(y_test), np.exp(predictions))

    print('\n')

    print(feat)

    print('RMSE', rmse)

    print('R2', r2)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(min_samples_split=10)

from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(movies.drop(['revenue','crew_team'], axis=1), movies['revenue'], test_size=0.25)



rfr.fit(x_train, y_train)

predictions = rfr.predict(x_test)



RMSE = mean_squared_error(y_test, predictions)**0.5

RMSE
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(min_samples_split=10)

from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(log_movies.drop(['revenue'], axis=1), log_movies['revenue'], test_size=0.25)



rfr.fit(x_train, y_train)

predictions = rfr.predict(x_test)



RMSE = mean_squared_error(np.exp(y_test), np.exp(predictions))**0.5

RMSE
X = movies['budget']

y = movies['revenue']

X_grid = np.arange(min(X), max(X), 1000) 

  

# reshape for reshaping the data into  

# a len(X_grid)*1 array, i.e. to make 

# a column out of the X_grid values 

X_grid = X_grid.reshape((len(X_grid), 1))  

  

plt.figure(figsize=(20,10))

# scatter plot for original data 

plt.scatter(X, y, color = 'red') 

  

regressor = RandomForestRegressor()

regressor.fit(movies[['budget']], movies['revenue'])

# plot predicted data 

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')  

  

# specify title 

plt.title('Revenue to budget (Random Forest Regression)')  

  

# specify X axis label 

plt.xlabel('Budget') 

  

# specify Y axis label 

plt.ylabel('Revenue') 

  

# show the plot 

plt.show() 