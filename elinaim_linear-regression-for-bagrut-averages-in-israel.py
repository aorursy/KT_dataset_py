import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Binarizer,FunctionTransformer, OneHotEncoder,LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression , SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error as MSE

from pandas.tools.plotting import scatter_matrix

from sklearn.metrics import f1_score

from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.mixture import GaussianMixture

sns.set(style="whitegrid", color_codes=True)

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]



print(os.listdir("../input"))
nRowsRead = None

df = pd.read_csv('../input/bagrut-israel/israel_bagrut_averages.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df = df.dropna()
nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
# GET CITIES DATA SET

places = pd.read_csv('../input/places/cities.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = places.shape

print(f'There are {nRow} rows and {nCol} columns')
places.head()
print(df['grade'].describe())

print(df.grade.plot.hist(figsize=(16,12)))
# 1. REMOVE EMPTY VALUES

df = df.where(df['grade'] >=0).dropna()

nRow, nCol = df.shape

print(f'There are {nRow} rows\n')



year,yNames = len(df['year'].unique()),df['year'].unique()

print(f'There are {year} unique years {yNames}\n')



studyunits,uNames = len(df['studyunits'].unique()),df['studyunits'].unique()

print(f'There are {studyunits} unique study units {uNames}\n')



citeis_count = df['city'].unique().size

print(f'There are {citeis_count} unique cities')

subject = df['subject'].unique().size

print(f'There are {subject} unique subjects')

# GOOGLE MAPS FUNCTION

# import csv

# def write_to_file(city):

# #     r = requests.get("https://maps.googleapis.com/maps/api/geocode/json?address=" + city + "&key=" + apiKey)

#     d = json.loads(r.content)

#     results_len = len(d['results'])

#     if(results_len > 0 ):

#         lat = d['results'][0]['geometry']['location']['lat']

#         lng = d['results'][0]['geometry']['location']['lng']

#         # print(lat, lng)

#         row = [city, lat, lng]

#         with open('cities.csv', 'a') as csvFile:

#             writer = csv.writer(csvFile)

#             writer.writerow(row)



#         csvFile.close()



# for city in cities:

#     city = city.strip()

#     write_to_file(city)
#MERGE DATAFRAMES

df['city'] = df['city'].apply(lambda x: x.strip())

df = pd.merge(df, places, on='city')

df.head()
def show_loca(tmp_df):

    data = pd.DataFrame({'x': tmp_df['lng'],'y':  tmp_df['lat']},columns=['x', 'y'])

#     print(tmp_df.groupby(['lat','lng','city']).agg({'takers':'sum', 'grade':'mean'}).reset_index().sort_values(by=['lng'], ascending=True).head())

    data.plot('x', 'y', kind='scatter', s=100 , figsize=(20,12))
# PLOT LOCATIONS

show_loca(df)
# REMOVE OUTLYERS

df = df[(df['lng'] > 25) & (df['lat'] < 34)]
show_loca(df)
new_df = df.groupby(['lat', 'lng'], as_index=False).mean()

new_df = new_df.drop(['takers', 'studyunits','year','semel'], axis=1) 

print(new_df.head())

color_theme = np.array(['blue','red','green'])

plt.figure(figsize=(20, 12), dpi=80)

plt.scatter(x=df['lng'], y = df['lat'],c=df['grade'],s=100)
# PLOT BY REGION

from ipyleaflet import Map, Heatmap

# from random import uniform

accdf = df

accdf=accdf.groupby(['lat','lng']).agg({'grade':'mean'}).reset_index()

accdf['grade'].fillna((accdf['grade'].mean()), inplace=True)

lats=accdf[['lat','lng','grade']].values.tolist()

m = Map(center=(accdf['lat'].mean(), accdf['lng'].mean()), zoom=8)

heat = Heatmap(locations=lats, radius=20, blur=20)

m.add_layer(heat)



# # Change some attributes of the heatmap

# heat.radius = 10

# heat.blur = 10

# heat.max = 0.5

# heat.gradient = {0.7: 'red', 0.8: 'cyan', 1.0: 'blue'}

# m.layout.width = '100%'

m.layout.height = '800px'

# m
# PLOT TOP SCORES BY REGION



accdf_small = accdf[accdf['grade']>83]

accdf_small=accdf_small.groupby(['lat','lng']).agg({'grade':'mean'}).reset_index()

accdf_small['grade'].fillna((accdf_small['grade'].mean()), inplace=True)

lats=accdf_small[['lat','lng','grade']].values.tolist()

m = Map(center=(accdf_small['lat'].mean(), accdf_small['lng'].mean()), zoom=8)

heat = Heatmap(locations=lats, radius=20, blur=20)

m.add_layer(heat)

m.layout.height = '500px'

# m
k = 6

model = GaussianMixture(n_components=6)

model.fit_predict(df[['lng', 'lat']])



df = df.dropna()



colors = {0: 'red', 1: 'lightgreen', 2: 'blue', 

          3:'yellow', 4: 'orange', 5: 'purple', 

          -1: 'black'}



cluster = np.array(df[['lng', 'lat']])

df['cluster'] = model.predict(cluster)

c = df['cluster'].apply(lambda x: colors[x])

df.plot('lng', 'lat', kind='scatter', c=c, s=20, figsize=(20,12))

#Shuffle DATA

df.sample(frac=1).head(10)
#HOW MANY AVR NOT PASSED

df_small = df[df['grade']<=55]

sSmall = len(df_small)

print(f'There are {sSmall} rows\n')

print(#WHERE MOST OF SCHOOLLS DIDNT PASS

df_small.groupby('city').size().sort_values(ascending=False).head())

print(df_small.grade.plot.hist(figsize=(16,12)))
sns.set(rc={'figure.figsize':(16,12)})

sns.boxplot(x="year", y="grade", data=df, linewidth=1)
sns.boxplot(x="year", y="grade", hue='studyunits', data=df, linewidth=1)
# See DIFF between random cities to check if years is relevant

import random

store_data = pd.DataFrame()

def reverse_city(value):

    return value[::-1]

df['reverse_city'] =  df['city'].apply(reverse_city)

unique_cities = df['city'].unique()



def gen_chart():

    random_city = random.choices(unique_cities,k=1)[0]

    city_data = df[df['city'] == random_city].groupby(['year']).mean().reset_index()

    print(city_data.plot.line(x='year', y='grade',ylim=(0,100),title=random_city[::-1],figsize=(6,4)))

    return df[df['city'] == random_city]





for _ in range(10):

    city_data = gen_chart()

    store_data = store_data.append(city_data)

# store_data.head()
# MOST TAKErS by 'subject','studyunits'

df_exploration  = df.groupby(['subject','studyunits']).agg({'takers':'sum', 'grade':'mean'}).reset_index()

df_exploration.sort_values(by=['takers'], ascending=False).head(10)
# LEAST TAKErS by 'subject','studyunits'

df_exploration.sort_values(by=['takers'], ascending=True).head(10)
#BOTTOM BY GRADES by 'subject','studyunits'

df_exploration.sort_values(by=['grade'], ascending=True).head(10)
#TOP BY GRADES by 'subject','studyunits'

df_exploration.sort_values(by=['grade'], ascending=False).head(10)
def trim(row):

    return row['subject'].strip()
#DATA ABOUT TOP GRRADES by 'subject','studyunits'

df['subject'] = df.apply(trim,axis=1)

df[df['subject']=='אומנות'].groupby(['school','city','studyunits']).agg({'grade':'mean','takers':'sum'}).reset_index()
df_mm = df[df['subject']=='תכנון ותכנות מערכות'].groupby(['school','city','studyunits']).agg({'grade':'mean','takers':'sum'}).reset_index()

print(len(df_mm))

df_mm.head(10)
#MOST TAKERS BY CITY & SUBJECT

df_exploration_city  = df.groupby(['city','subject']).agg({'takers':'sum', 'grade':'mean'}).reset_index()

df_exploration_city.sort_values(by=['takers'], ascending=False).head(10)
#TOP GRADES BY BY CITY & SUBJECT

df_exploration_city.sort_values(by=['grade'], ascending=False).head(10)
def f(row):

    return row['takers'] * 100 / df_exploration['takers'].sum()



df_exploration['precent'] = df_exploration.apply(f,axis=1)

df_exploration.sort_values(by=['precent'], ascending=False).head(20)
# GET FIRST WORD 

from collections import Counter

unique_school = df['school'].unique()

unique_school_list = []



for sc in unique_school:

    new_sc = sc.split()[0]

    unique_school_list.append(new_sc)



x = Counter(unique_school_list)  

tups = x.most_common()

tups[:40]
def get_first_word(school):

    return school.split()[0]



df['school_word'] = df['school'].apply(get_first_word)

df.head()
df.groupby(['school','city']).agg({'takers':'sum', 'grade':'mean'}).reset_index().sort_values(by=['grade'], ascending=False).head(10)
#MOST TAKERS BY SCHOOL WORD

df_by_grade = df.groupby(['school_word']).agg({'city':'size', 'grade':'mean'}).reset_index().sort_values(by=['grade'], ascending=False).rename(columns={'grade':'avg grade'})

df_by_count = df.groupby(['school_word']).agg({'takers':'sum', 'grade':'mean'}).reset_index()

most_takers_df = df_by_count.sort_values(by=['takers'], ascending=False).head(30)

most_takers_df.head(30)
#TOP GRADES BY SCHOOL WORD

most_takers_df.sort_values(by=['grade'], ascending=False).head(10)
# df[df['school_word'] == 'נזירות'].groupby(['school','subject']).mean()

df[df['school_word'] == 'נזירות'].groupby('school').mean()
#BOTTOM GRADES BY SCHOOL WORD

most_takers_df.sort_values(by=['grade'], ascending=True).head(10)
df[df['school_word'] == 'ברנקו'].groupby('school').mean()
df.hist(bins=100, layout=(2, 8), figsize=(20,12))
X = df.drop('grade', axis=1)

y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
X_train.head()
def get_cols_4_ss(df):

    return df[['grade']]



def get_cols_4_mas(df):

    return df[['semel','year','studyunits']]



def get_cols_4_bins(df):

    return df[['studyunits']]



def get_takers(df):

    return df[['takers']]



def count_vectorize_school(df):

    corpus = df['school']

    vectorizer = CountVectorizer(encoding="utf-8",max_features=880)

    X = vectorizer.fit_transform(corpus)

#     print(vectorizer.get_feature_names())

    return X.toarray()



def count_vectorize_subject(df):

    corpus = df['subject']

    vectorizer = CountVectorizer(encoding="utf-8",max_features=159)

    X = vectorizer.fit_transform(corpus)

#     print(vectorizer.get_feature_names())

    return X.toarray()



def count_vectorize_school_word(df):

    corpus = df['school_word']

    vectorizer = CountVectorizer(encoding="utf-8",max_features=189)

    X = vectorizer.fit_transform(corpus)

#     print(vectorizer.get_feature_names())

    return X.toarray()



def count_vectorize_studyunits(df):

    corpus = df['studyunits']

    lb = LabelEncoder()

    y = lb.fit_transform(corpus)

    return y





# def count_vectorize_cluster(df):

#     corpus = df['cluster']

#     lb = LabelEncoder()

#     y = lb.fit_transform(corpus)

#     return y



def count_vectorize_cluster(df):

    return pd.get_dummies(df['cluster'],prefix=('cluster'))



ss_selector = FunctionTransformer(func=get_cols_4_ss, validate=False)

mas_selector = FunctionTransformer(func=get_cols_4_mas, validate=False)

studyunits_selector = FunctionTransformer(func=get_cols_4_bins, validate=False)

takers_selector = FunctionTransformer(func=get_takers, validate=False)



# OneHotEncoderTransformer = OneHotEncoder(sparse = False, categories='auto')

count_vectorize_school = FunctionTransformer(func=count_vectorize_school, validate=False)

count_vectorize_subject = FunctionTransformer(func=count_vectorize_subject, validate=False)

count_vectorize_school_word = FunctionTransformer(func=count_vectorize_school_word, validate=False)

count_vectorize_studyunits = FunctionTransformer(func=count_vectorize_studyunits, validate=False)

count_vectorize_cluster = FunctionTransformer(func=count_vectorize_cluster, validate=False)



# get_subject_dummies = FunctionTransformer(func=get_subject_dummies, validate=False)

# get_studyunits_dummies = FunctionTransformer(func=get_studyunits_dummies, validate=False)



# def get_subject_dummies(df):

#     df_subject = df[['subject']].copy()

#     return pd.get_dummies(df_subject, columns=['subject'], drop_first=True)



# def get_studyunits_dummies(df):

#     df_subject = df[['studyunits']].copy()

#     return pd.get_dummies(df_subject, columns=['studyunits'], drop_first=True)
xx = df.groupby(['subject']).agg({'takers':'count', 'grade':'mean'}).reset_index()

xx.sort_values(by=['takers'], ascending=False).head()
ss_pipeline = Pipeline([('ss_selector', ss_selector), 

                        ('ss', StandardScaler())])



mas_pipeline = Pipeline([('mas_selector', mas_selector), 

                         ('mas', MaxAbsScaler())])



takers_pipeline = Pipeline([('takers_selector', takers_selector), 

                           ('takers_bin', Binarizer())])



count_vectorize_school = Pipeline([('count_vectorize_school'

                                    , count_vectorize_school)])



count_vectorize_subject = Pipeline([('count_vectorize_subject'

                                    , count_vectorize_subject)])



count_vectorize_school_word = Pipeline([('count_vectorize_school_word'

                                    , count_vectorize_school_word)])



count_vectorize_studyunits = Pipeline([('count_vectorize_studyunits'

                                    , count_vectorize_studyunits)])



count_vectorize_cluster = Pipeline([('count_vectorize_cluster'

                                    , count_vectorize_cluster)])

print('school')

print(count_vectorize_school.fit_transform(X_train).shape)

print(count_vectorize_school.fit_transform(X_test).shape)

print('subject')

print(count_vectorize_subject.fit_transform(X_train).shape)

print(count_vectorize_subject.fit_transform(X_test).shape)

print('count_vectorize_school_word')

print(count_vectorize_school_word.fit_transform(X_train).shape)

print(count_vectorize_school_word.fit_transform(X_test).shape)

print('count_vectorize_cluster')

print(count_vectorize_cluster.fit_transform(X_train).shape)

print(count_vectorize_cluster.fit_transform(X_test).shape)

trans_pipeline = FeatureUnion([

                                ('mas_pipeline', mas_pipeline), 

                                ('count_vectorize_school', count_vectorize_school), 

                                ('count_vectorize_subject', count_vectorize_subject),

                                ('count_vectorize_cluster', count_vectorize_cluster),    

                               ('takers_pipeline', takers_pipeline)

                            ])
# prepared_train = trans_pipeline.fit_transform(X_train)

# prepared_train
lr = LinearRegression()

dt = DecisionTreeRegressor(max_depth=5)

rf = RandomForestRegressor(max_depth=20, n_estimators=10, n_jobs=-1)
full_pipeline = Pipeline([('trans_pipeline', trans_pipeline), ('lr', lr)])

full_pipeline.fit(X_train, y_train)

y_train_pred = full_pipeline.predict(X_train)

y_test_pred = full_pipeline.predict(X_test)

rmse = np.sqrt(MSE(y_test, y_test_pred))

print(rmse)
plt.plot(y_train, y_train_pred, '.', label='Data')

plt.plot([0, 100], [0, 100], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
full_pipeline = Pipeline([('trans_pipeline', trans_pipeline), ('dt', dt)])

full_pipeline.fit(X_train, y_train)

y_train_pred = full_pipeline.predict(X_train)

y_test_pred = full_pipeline.predict(X_test)

rmse = np.sqrt(MSE(y_test, y_test_pred))

print(rmse)
plt.plot(y_test, y_test_pred, '.', label='Data')

plt.plot([0, 100], [0, 100], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
full_pipeline = Pipeline([('trans_pipeline', trans_pipeline), ('rf', rf)])

full_pipeline.fit(X_train, y_train)

y_train_pred = full_pipeline.predict(X_train)

y_test_pred = full_pipeline.predict(X_test)

rmse = np.sqrt(MSE(y_test, y_test_pred))

print(rmse)
plt.plot(y_test, y_test_pred, '.', label='Data')

plt.plot([0, 100], [0, 100], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()