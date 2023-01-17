#import the necessary python libraries



import pandas as pd

import numpy as np

from scipy import stats as st

import math

import datetime

import os

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

from IPython.display import display

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor 

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from itertools import combinations

from collections import Counter

import random

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
RANDOM_SEED = 42

!pip freeze > requirements.txt

CURRENT_DATE = pd.to_datetime('21/09/2020')
#charting function



def diagram_bar(data, column):

    fig = plt.figure()

    main_axes = fig.add_axes([0,0,1,1])

    data[column].hist(bins = 20)

    insert_axes = fig.add_axes([1.1,0,0.5,1])

    data.boxplot(column = column)

    
# function that determines the correctness of a hypothesis



def hypothyroidism(data, column, alpha):

    

    if (st.ttest_ind(df[(df['Sample'] == 1) & df[column] == 1].Rating, df[(df['Sample'] == 1) & df[column] == 0].Rating).pvalue < alpha):

        print("Отвергаем нулевую гипотезу")

        

    else:

        print("Не получилось отвергнуть нулевую гипотезу")
# function that determines the correctness of the hypothesis in the context of various combinations



def hypothyroidism_1(data, column, alpha):

    

    for row in combinations(data[column].unique(), 2):

        print(row)

        

        if (st.ttest_ind(data[(data['Sample'] == 1) & data[column] == row[0]].Rating, data[(df['Sample'] == 1) & data[column] == row[1]].Rating).pvalue < alpha):

            print("Отвергаем нулевую гипотезу равенства рейтингов ", row)

            

        else:

            print("Не получилось отвергнуть нулевую гипотезу равенства ", row)

    
#function for predicting the objective function using the random forest method and calculating the MAE metric





def model_random_forest(X, y):

    

    RANDOM_SEED = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    MAE = metrics.mean_absolute_error(y_test, y_pred)

    print('MAE:', MAE)

    

    plt.rcParams['figure.figsize'] = (12,10)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    feat_importances.nlargest(15).plot(kind='barh')
# loading data



path_to_file = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(path_to_file+'main_task.csv')

df_test = pd.read_csv(path_to_file+'kaggle_task.csv')

sample_submission = pd.read_csv(path_to_file+'/sample_submission.csv')



# view the data



pd.set_option('display.max_columns', 200)

display(df_train.head(2))

display(df_test.head(2))
display(df_train.info())

display(df_test.info())
# checking data for duplicates



df_train.drop('Reviews', axis = 1).duplicated().sum()
df_test.drop('Reviews', axis = 1).duplicated().sum()
# build a diagram of the distribution of data in the Rating column



diagram_bar(df_train, 'Rating')
# connect two DFs for processing



df_train['Sample'] = 1 # метка для обучающей DF

df_test['Sample'] = 0 # метка для тестовой DF

df_test['Rating'] = 0 # т.к. в тестовой выборке отсутствует столбец Rating, то создаем и заполняем его столбец 0



df = df_test.append(df_train, sort=False).reset_index(drop=True) 
df.info()
# determine the number of unique Restaurant_id



len(df.Restaurant_id.value_counts())
# restaurants account for 13094 unique values.

# Made decisions to just code them



le = LabelEncoder()

le.fit(df['Restaurant_id'])

df['Restaurant_id_code'] = le.transform(df['Restaurant_id'])
restaurant = df.Restaurant_id.value_counts().reset_index()
#we will introduce a new parameter chain or non-chain restaurant



restaurant = restaurant[restaurant.Restaurant_id > 1]

df['restaurant_chain'] = df['Restaurant_id'].where(df['Restaurant_id'].isin(restaurant['index']), 0)

df.loc[df.restaurant_chain != 0, 'restaurant_chain'] = 1
# test the hypothesis about the equality of ratings of chain and chain restaurants



hypothyroidism(df, 'restaurant_chain', 0.05)
# define the unique meaning of cities



len(df['City'].value_counts())
# cities have 31 unique values. Previous analysis showed

# that it makes no sense to reduce the number of cities.

# So I decided to just encode them.



le = LabelEncoder()

le.fit(df['City'])

df['City_code'] = le.transform(df['City'])
#создадим dummy-переменные из столбца City

#проверим на модели и выделим города, влияющие на итоговую оценку



City = pd.get_dummies(df['City'])

model_random_forest(City, df.Rating)
df = pd.concat([df, City['Rome']], axis = 1)
# highlight a new sign whether the city belongs to the capital or not





dict_Сity_capital = {'London' : 1, 'Paris' : 1, 'Madrid' : 1, 'Barcelona' : 0, 

                        'Berlin' : 1, 'Milan' : 0, 'Rome' : 1, 'Prague' : 1, 

                        'Lisbon' : 1, 'Vienna' : 1, 'Amsterdam' : 1, 'Brussels' : 1, 

                        'Hamburg' : 0, 'Munich' : 0, 'Lyon' : 0, 'Stockholm' : 0, 

                        'Budapest' : 1, 'Warsaw' : 1, 'Dublin' : 1, 

                        'Copenhagen' : 1, 'Athens' : 1, 'Edinburgh' : 1, 

                        'Zurich' : 1, 'Oporto' : 0, 'Geneva' : 0, 'Krakow' : 0, 

                        'Oslo' : 1, 'Helsinki' : 1, 'Bratislava' : 1, 

                        'Luxembourg' : 1, 'Ljubljana' : 1}

df['capital'] = df.apply(lambda row: dict_Сity_capital[row['City']], axis = 1)



hypothyroidism(df, 'capital', 0.05)
# add a new feature - the population of cities



dict_Сity_population= {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

df['Сity_population'] = df.apply(lambda row: dict_Сity_population[row['City']], axis = 1)
# add a new feature - average per capita income before taxes



dict_Сity_income = {'London' : 2511, 'Paris' : 3617, 'Madrid' : 2651, 'Barcelona' : 2663, 

                        'Berlin' : 4521, 'Milan' : 3183, 'Rome' : 2843, 'Prague' : 1400, 

                        'Lisbon' : 1526, 'Vienna' : 2646, 'Amsterdam' : 4612, 'Brussels' : 3401, 

                        'Hamburg' : 5604, 'Munich' : 5181, 'Lyon' : 2400, 'Stockholm' : 2391, 

                        'Budapest' : 670, 'Warsaw' : 1259, 'Dublin' : 3000, 

                        'Copenhagen' : 5000, 'Athens' : 1100, 'Edinburgh' : 2050, 

                        'Zurich' : 6758, 'Oporto' : 1288, 'Geneva' : 2100, 'Krakow' : 1027, 

                        'Oslo' : 4048, 'Helsinki' : 3691, 'Bratislava' : 1176, 

                        'Luxembourg' : 5000, 'Ljubljana' : 1807}

df['Сity_income'] = df.apply(lambda row: dict_Сity_income[row['City']], axis = 1)
# select a new feature country of location



dict_Сountries = {'London' : 'England', 'Paris' : 'France', 'Madrid' : 'Spain', 

                  'Barcelona' : 'Spain', 'Berlin' : 'Germany', 'Milan' : 'Italy', 

                  'Rome' : 'Italy', 'Prague' : 'Czech_c', 'Lisbon' : 'Portugal', 

                  'Vienna' : 'Austria', 'Amsterdam' : 'Holland', 

                  'Brussels' : 'Belgium', 'Hamburg' : 'Germany', 'Munich' : 'Germany', 

                  'Lyon' : 'France', 'Stockholm' : 'Sweden', 'Budapest' : 'Romania', 

                  'Warsaw' : 'Poland', 'Dublin' : 'Ireland', 'Copenhagen' : 'Denmark', 

                  'Athens' : 'Greece', 'Edinburgh' : 'Scotland', 'Zurich' : 'Switzerland', 

                  'Oporto' : 'Portugal', 'Geneva' : 'Switzerland', 'Krakow' : 'Poland', 

                  'Oslo' : 'Norway', 'Helsinki' : 'Finland', 'Bratislava' : 'Slovakia', 

                  'Luxembourg' : 'Luxembourg_c', 'Ljubljana' : 'Slovenia'}

df['Сountry'] = df.apply(lambda row: dict_Сountries[row['City']], axis = 1)



# encode data by country



le = LabelEncoder()

le.fit(df['Сountry'])

df['code_Сountry'] = le.transform(df['Сountry'])
# there are more than 22% missing values in the order column

# this feature can be a good feature for the model



df['NAN_Cuisine'] = pd.isna(df['Cuisine Style']).astype('int')
hypothyroidism(df, 'NAN_Cuisine', 0.05)
# let's preprocess the values



df['Cuisine Style'] = df['Cuisine Style'].fillna("['Other']")

df['Cuisine Style'] = df['Cuisine Style'].str.findall(r"'(\b.*?\b)'") 



# create a new feature number of cuisine styles



df['quantity_Cuisine_Style'] = df.apply(lambda row: len(row['Cuisine Style']), axis = 1)
diagram_bar(df[df['NAN_Cuisine'] != 1], 'quantity_Cuisine_Style')
df.loc[(df['NAN_Cuisine'] == 1), 'quantity_Cuisine_Style'] = df.loc[(df['NAN_Cuisine'] != 1), 'quantity_Cuisine_Style'].median()
diagram_bar(df, 'quantity_Cuisine_Style')
df.Ranking.describe()
diagram_bar(df, 'Ranking')
# a large amount of emissions. Let's look at the distribution up to 11000 and more

fig = plt.figure()

main_axes = fig.add_axes([0,0,1,1])

df[df['Ranking'] < 11000]['City'].value_counts(ascending=True).plot(kind='barh')

insert_axes = fig.add_axes([1.1,0,1,1])

df[df['Ranking'] >= 11000]['City'].value_counts(ascending=True).plot(kind='barh')
# emissions are found in 2 cities London and Paris. Emissions can be related to the number of establishments in cities

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
# Normalizing the Ranking feature



Ranking_City_mean = df.groupby(['City'])['Ranking'].mean()

Restorant_City_count = df['City'].value_counts(ascending=False)

df['Ranking_City_mean'] = df['City'].apply(lambda x: Ranking_City_mean[x])

df['Restorant_City_count'] = df['City'].apply(lambda x: Restorant_City_count[x])

df['Ranking_Rest_City_norm'] = (df['Ranking'] - df['Ranking_City_mean']) / df['Restorant_City_count']
for x in (df['City'].value_counts())[0:10].index:

    df['Ranking_Rest_City_norm'][df['City'] == x].hist(bins=100)

plt.show()
# encode data from the Price Range column



dict_Price = {'$':1,'$$ - $$$':2,'$$$$':3}

df['Price Range']=df['Price Range'].map(lambda x: dict_Price.get(x,x))
# for data gaps, I suggest filling in the gaps mod



df['Price Range'] = df['Price Range'].fillna(2)
df['Number of Reviews'].describe()
diagram_bar(df, 'Number of Reviews')
for element in [0,1]:

    diagram_bar(df[df['Sample'] == element], 'Number of Reviews')

    print(df[df['Sample'] == element]['Number of Reviews'].describe())
# re-rhyme the column and see how the data distribution changes



df['LOG_Number_Reviews'] = df['Number of Reviews'].apply(lambda x: math.log1p(x))

df['LOG_Number_Reviews'].hist(bins=50)
for x in df['City'].value_counts().index[0:10]:

    df['Number of Reviews'][df['City'] == x].hist(bins=20)

plt.show()
for x in df['City'].value_counts().index[0:10]:

    df['LOG_Number_Reviews'][df['City'] == x].hist(bins=100)

plt.show()
# создадим новый признак 



df['NAN_Number_Reviews'] = pd.isna(df['Number of Reviews']).astype('float64')
df['Number of Reviews'] = df['Number of Reviews'].fillna(df['Number of Reviews'].median())



df['LOG_Number_Reviews'] = df['LOG_Number_Reviews'].fillna(df['LOG_Number_Reviews'].median())
# there are no gaps in the review, but more than 6000 lines with the value [[], []]. In fact, these are empty lines, let's save them

df['Reviews'] = df['Reviews'].fillna('[[], []]')



# analysis of the test base revealed two gaps, despite the fact that pandas.profiling did not reveal any gaps on the training base, fill them in '[[], []]' and drop them into empty_Reviewsdf['Reviews'] = df['Reviews'].fillna('[[], []]')

df['empty_Reviews'] = (df['Reviews']=='[[], []]').astype('float64')
# pull the date out of the review and create new criteria



df['date_of_Review'] = df['Reviews'].str.findall('\d+/\d+/\d+')
df['len_date'] = df['date_of_Review'].apply(lambda x: len(x))

df[df['len_date']>2]
df.loc[df['len_date']>2]['Reviews']
# that people indicated dates in reviews and these dates were processed. throw data from DF



for row in df.loc[df['len_date']>2].index:

    date_list = df.loc[row]['date_of_Review']

    del date_list[0]

    df.loc[row]['date_of_Review'] = date_list
df.loc[df['len_date'] == 1]
# turned out to be reviews with one (1) review and a lot of them 5680

# so I suggest working with one (first date)

# I suggest using the month and day of the week of the most recent review as signs



df['week_day'] = df['date_of_Review'].apply(lambda x: pd.to_datetime(x).max().weekday())

df['month'] = df['date_of_Review'].apply(lambda x: pd.to_datetime(x).max().month)

diagram_bar(df, 'week_day')
df['week_day'] = df['week_day'].fillna(df['week_day'].value_counts().index[0])
diagram_bar(df, 'week_day')
# create dummy variables



day_week = pd.get_dummies(df['week_day'])

day_week.columns = ['mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

df = pd.concat([df,day_week], axis=1)
for column in day_week.columns:

    print(column)

    hypothyroidism(df, column, 0.05)
diagram_bar(df, 'month')
df['month'] = df['month'].fillna(df['month'].value_counts().index[0])
diagram_bar(df, 'month')
# create dummy variables



month = pd.get_dummies(df['month'])

month.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

df = pd.concat([df,month], axis=1)
for column in month.columns:

    print(column)

    hypothyroidism(df, column, 0.05)
# introduce new signs the difference between dates, and the statute of limitations



def delta_data(data):

    if data['date_of_Review'] == []:

        return None

    return pd.to_datetime(data['date_of_Review']).max() - pd.to_datetime(data['date_of_Review']).min()



def delta_data_now(data):

    if data['date_of_Review'] == []:

        return None

    return datetime.datetime.now() - pd.to_datetime(data['date_of_Review']).max()



def delta_data_start(data):

    if data['date_of_Review'] == []:

        return None

    return abs(pd.to_datetime(data['date_of_Review']).min() - pd.to_datetime('01/01/2000'))
df['delta_data'] = df.apply(delta_data, axis = 1).dt.days

df['delta_data_now'] = df.apply(delta_data_now, axis = 1).dt.days

df['delta_data_start'] = df.apply(delta_data_start, axis = 1).dt.days

df['delta_data_start'] = df['delta_data_start'].fillna(0)

df['delta_data_now'] = df['delta_data_now'].fillna(0)
# create polynomial features based on columns



pf = PolynomialFeatures(2)

delta_data = pf.fit_transform(df[['delta_data_now', 'delta_data_start']])
# create a new feature, column length Reviews



df['LEN_Reviews'] = df['Reviews'].apply(lambda x: len(x))
diagram_bar(df, 'LEN_Reviews')
for x in df['Restaurant_id'].value_counts().index[0:10]:

    df['LEN_Reviews'][df['Restaurant_id'] == x].hist(bins=100)

plt.show()
df.drop(['Rome', 'Greece', 'Restaurant_id', 'City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'date_of_Review', 'Сountry', 'Сity_population', 'mean_Ranking_on_City', 'count_Restorant_in_City', 'max_Ranking_on_City'], axis=1, inplace=True, errors='ignore')

df_model = df
display(df_model.describe())
#standardize two columns 



def StandardScaler_column(d_col):

    scaler = StandardScaler()

    scaler.fit(df_model[[d_col]])

    return scaler.transform(df_model[[d_col]])



for column in df_model.columns:

    if column not in ['Rating','Sample']:

        df_model[column] = StandardScaler_column(column)

        if len(df_model[df_model[column].isna()]) < len(df_model):

            df_model[column] = df_model[column].fillna(0)

    

train_data = df_model.query('Sample == 1').drop(['Sample'], axis=1)

test_data = df_model.query('Sample == 0').drop(['Sample'], axis=1)
# Checking the model on the training set



y = train_data.Rating.values            

X = train_data.drop(['Rating'], axis=1)



# Let's use the special function train_test_split to split test data

# allocate 20% of the data for validation (parameter test_size)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# check



test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# building the model



model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Train the model on a test dataset



model.fit(X_train, y_train)



# We use a trained model to predict restaurant ratings in a test sample.

# The predicted values are written to the y_pred variable



y_pred = model.predict(X_test)
# standard mathematical rounding function

def classic_round(d_num):

    return int(d_num + (0.5 if d_num > 0 else -0.5))



# rounding function is a multiple of 0.5

def my_round(d_pred):

    result = classic_round(d_pred*2)/2

    if result <=5:

        return result

    else:

        return 5

    

# round off forecast values

my_vec_round = np.vectorize(my_round)
y_pred = my_vec_round(y_pred)
# Compare the predicted values (y_pred) with the real ones (y_test), and see how much they differ on average

# The metric is called the Mean Absolute Error (MAE) and shows the average deviation of the predicted values from the actual ones.



MAE = metrics.mean_absolute_error(y_test, y_pred)

print('MAE:', MAE)
# in RandomForestRegressor it is possible to display the most important features for the model



plt.rcParams['figure.figsize'] = (12,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
# Checking the correlation of important variables



df_temp = df_model.loc[df_model['Sample'] == 1, list(feat_importances.nlargest(15).index[0:15])]

plt.rcParams['figure.figsize'] = (12,6)

ax = sns.heatmap(df_temp.corr(), annot=True, fmt='.2g')

i, k = ax.get_ylim()

ax.set_ylim(i+0.5, k-0.5)
df_model.drop(['Ranking_City_mean'], axis=1, inplace=True, errors='ignore')
train_data = df.query('Sample == 1').drop(['Sample'], axis=1)

test_data = df.query('Sample == 0').drop(['Sample','Rating'], axis=1)
y = train_data.Rating.values    

X = train_data.drop(['Rating'], axis=1)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
model.fit(X, y)
predict_submission = model.predict(test_data)
predict_submission = my_vec_round(predict_submission)#
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)