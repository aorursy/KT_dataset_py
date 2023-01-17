#thêm thư viện 

import numpy as np # Đại số tuyến tính

import pandas as pd # load file .CSV(I/O)

import matplotlib.pyplot as plt #vẽ

import seaborn as sns #vẽ

sns.set()

from sklearn.feature_extraction.text import TfidfVectorizer

#1 load dữ liệu

train_orig = pd.read_csv('../input/train.csv')

test_orig = pd.read_csv('../input/test.csv')

subm = pd.DataFrame()

subm['id'] = test_orig.id.values

train_orig.head()
#2 

train_orig['bool_belongs_to_collection'] = (train_orig['belongs_to_collection'].notnull()).astype(int)

test_orig['bool_belongs_to_collection'] = (test_orig['belongs_to_collection'].notnull()).astype(int)

#3

train_orig['split'] = 'train'

test_orig['split'] = 'test'

#4

train_test = pd.concat([train_orig[['popularity','budget','split','bool_belongs_to_collection']], test_orig[['popularity','budget','split','bool_belongs_to_collection']]])

#5

train_test.shape
#6 biểu đồ giữa train và test (sự nổi tiếng và ngân sách)

fig, ax = plt.subplots()

sns.scatterplot(x="popularity", y="budget", hue="split", data=train_test,ax=ax)
#7 fig 

fig, ax = plt.subplots()

fig.set_size_inches(18.5, 10.5)

sns.scatterplot(x="popularity", y="budget", hue="split",style='bool_belongs_to_collection', data=train_test,ax=ax, alpha=0.4)

ax.set_xlim([0,100])
#8 biểu đồ về sự nổi tiếng(budget)

g = sns.catplot(x='split',y='budget',data=train_test, kind='box' )

g.set_axis_labels("Split", "budget")

#9 Biêu đồ về sự nổi tiếng và doanh thu (popularity và revenue)

from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.models import ColumnDataSource

output_notebook()

#gán

x = train_orig.popularity

y = train_orig.revenue



source = ColumnDataSource(data=dict(

    popularity=train_orig.popularity,

    revenue=train_orig.revenue,

    original_language=train_orig.original_language,

))



#xuất file html

output_file("popularity_revenue.html", title="Popularity, Revenue", mode="cdn")

TOOLTIPS = [

    ("Popularity", "@popularity"),

    ("Revenue", "@revenue"),

    ("Original Language", "@original_language"),  

]



p = figure(tooltips=TOOLTIPS,y_axis_type="log")

p.circle('popularity', 'revenue',fill_alpha=0.6, line_color=None, source = source)

p.xaxis.axis_label = "popularity"

p.yaxis.axis_label = "revenue"

show(p)
#10 biểu đồ về doanh thu và ngân sách(revenue và budget)

from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.models import ColumnDataSource

output_notebook()



x = train_orig.budget

y = train_orig.revenue



source = ColumnDataSource(data=dict(

    budget=train_orig.budget,

    revenue=train_orig.revenue,

    original_language=train_orig.original_language,

))



output_file("budget_revenue.html", title="Budget, Revenue", mode="cdn")

TOOLTIPS = [

    ("Budget", "@budget"),

    ("Revenue", "@revenue"),

    ("Original Language", "@original_language"),

    

]



p = figure(tooltips=TOOLTIPS,y_axis_type="log")



p.circle('budget', 'revenue',fill_alpha=0.6, line_color=None, source = source)

p.xaxis.axis_label = "budget"

p.yaxis.axis_label = "revenue"

show(p)
#11 out ngôn ngữ và số lượng phim > 5 của tập train

print(len(train_orig.columns))#độ dài cột train

print(len(test_orig.columns))#độ dài cột test

olang = train_orig.original_language.value_counts()[train_orig.original_language.value_counts()>5].index.tolist()

print(olang)

print(len(olang))

train_orig_sample = train_orig[train_orig.original_language.isin(olang)].copy()

print(train_orig_sample.original_language.value_counts())

#12

train_orig_sample.loc[:,'revenue'] = np.log(train_orig_sample['revenue'].fillna(0)+1)

#13 biểu đồ Doanh thu theo ngôn ngữ

g = sns.catplot(x='original_language',y='revenue',data=train_orig_sample, kind='box', aspect=2 )

g.set_axis_labels("Original language", "Log of revenue")
#14 out thông tin (các cột và số dòng non-null)

train_orig.info()
#15 

train_olang = pd.get_dummies(train_orig.original_language)[olang]

train_orig = pd.concat([train_orig,train_olang], axis=1)
#16

def extract_id(cell):

    return yaml.load(cell)[0]['id']
#17 các cột của train

train_orig.columns
#18 tính toán bên test

test_olang = pd.get_dummies(test_orig.original_language)[olang]

test_orig = pd.concat([test_orig,test_olang], axis=1)

test_orig.head(2)
#19

train_orig['cast_crew'] = train_orig.cast + ' ' + train_orig.crew 

test_orig['cast_crew'] = test_orig.cast + ' ' + test_orig.crew
#20 diễn viên và phi hành đoàn

vec = TfidfVectorizer(analyzer='word',max_features=450,token_pattern=r"'name': '(.*?)'")

vec.fit(train_orig.cast_crew.fillna(''))

vocab = vec.get_feature_names()

vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)

train_crew_w = vec.fit_transform(train_orig.cast_crew.fillna(''))

test_crew_w = vec.transform(test_orig.cast_crew.fillna(''))

train_crew_w_cols = vec.get_feature_names()

train_crew_w_cols = ['crew_'+a for a in train_crew_w_cols]

print(train_crew_w.shape)

print(test_crew_w.shape)

print(train_crew_w_cols)

train_crew_w = pd.DataFrame(train_crew_w.toarray(),columns=train_crew_w_cols)

test_crew_w = pd.DataFrame(test_crew_w.toarray(),columns=train_crew_w_cols)
#21 tên công ty

vec = TfidfVectorizer(analyzer='word',max_features=100,token_pattern=r"'name': '(.*?)'")

vec.fit(train_orig.production_companies.fillna(''))

vocab = vec.get_feature_names()

vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)

train_production_companies_w = vec.fit_transform(train_orig.production_companies.fillna(''))

test_production_companies_w = vec.transform(test_orig.production_companies.fillna(''))

train_production_companies_w_cols = vec.get_feature_names()

train_production_companies_w_cols = ['prod_comp_'+a for a in train_production_companies_w_cols]

print(train_production_companies_w.shape)

print(test_production_companies_w.shape)

print(train_production_companies_w_cols)

train_production_companies_w = pd.DataFrame(train_production_companies_w.toarray(),columns=train_production_companies_w_cols)

test_production_companies_w = pd.DataFrame(test_production_companies_w.toarray(),columns=train_production_companies_w_cols)  

#22 quốc gia 

vec = TfidfVectorizer(analyzer='word',max_features=20,token_pattern=r"'name': '(.*?)'")

vec.fit(train_orig.production_countries.fillna(''))

vocab = vec.get_feature_names()

vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)

train_production_countries_w = vec.fit_transform(train_orig.production_countries.fillna(''))

test_production_countries_w = vec.transform(test_orig.production_countries.fillna(''))

train_production_countries_w_cols = vec.get_feature_names()

train_production_countries_w_cols = ['prod_country_'+a for a in train_production_countries_w_cols]

print(train_production_countries_w.shape)

print(test_production_countries_w.shape)

print(train_production_countries_w_cols)

train_production_countries_w = pd.DataFrame(train_production_countries_w.toarray(),columns=train_production_countries_w_cols)

test_production_countries_w = pd.DataFrame(test_production_countries_w.toarray(),columns=train_production_countries_w_cols)
#23 bộ sưu tập

vec = TfidfVectorizer(analyzer='word',max_features=50,token_pattern=r"'name': '(.*?)'")



train_belongs_to_collection_w = vec.fit_transform(train_orig.belongs_to_collection.fillna(''))

test_belongs_to_collection_w = vec.transform(test_orig.belongs_to_collection.fillna(''))

train_belongs_to_collection_w_cols = vec.get_feature_names()

train_belongs_to_collection_w_cols = ['collection_'+a for a in train_belongs_to_collection_w_cols]

print(train_belongs_to_collection_w.shape)

print(test_belongs_to_collection_w.shape)

print(train_belongs_to_collection_w_cols)

train_belongs_to_collection_w = pd.DataFrame(train_belongs_to_collection_w.toarray(),columns=train_belongs_to_collection_w_cols)

test_belongs_to_collection_w = pd.DataFrame(test_belongs_to_collection_w.toarray(),columns=train_belongs_to_collection_w_cols)
#24 thể loại(genres)

vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')

train_genres_w = vec.fit_transform(train_orig.genres.fillna(''))

test_genres_w = vec.transform(test_orig.genres.fillna(''))

train_genres_w_cols = vec.get_feature_names()

train_genres_w_cols = ['genre_'+a for a in train_genres_w_cols]

print(train_genres_w.shape)

print(test_genres_w.shape)

print(train_genres_w_cols)

train_genres_w = pd.DataFrame(train_genres_w.toarray(),columns=train_genres_w_cols)

test_genres_w = pd.DataFrame(test_genres_w.toarray(),columns=train_genres_w_cols)

print(train_genres_w.shape)

print(test_genres_w.shape)
#25

#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo

train_orig['Keywords_tagline_overview'] = train_orig.title + ' ' + train_orig.Keywords +' ' + train_orig.tagline + ' ' + train_orig.overview

test_orig['Keywords_tagline_overview'] = test_orig.title + ' ' + test_orig.Keywords + ' ' + test_orig.tagline + ' ' + test_orig.overview

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=60,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')

train_tagline_keyword_w = vec.fit_transform(train_orig.Keywords_tagline_overview.fillna(''))

train_tagline_keyword_w_cols = vec.get_feature_names()

print(train_tagline_keyword_w.shape)

test_tagline_w = vec.transform(test_orig.Keywords_tagline_overview.fillna(''))

print(test_tagline_w.shape)

train_tagline_keyword_w_cols = ['kw_tg_ow_' + a for a in train_tagline_keyword_w_cols]

train_tagline_keyword_w = pd.DataFrame(train_tagline_keyword_w.toarray(),columns=train_tagline_keyword_w_cols)

test_tagline_keyword_w = pd.DataFrame(test_tagline_w.toarray(),columns=train_tagline_keyword_w_cols)

train = pd.concat([train_orig,train_tagline_keyword_w,train_genres_w,train_belongs_to_collection_w,

                   train_production_companies_w,train_crew_w,train_production_countries_w], axis=1)

test = pd.concat([test_orig,test_tagline_keyword_w,test_genres_w,test_belongs_to_collection_w,

                  test_production_companies_w,test_crew_w,test_production_countries_w], axis=1)

print(train.shape)

print(test.shape)
#26

train['bool_belongs_to_collection'] = (train['belongs_to_collection'].notnull()).astype(int)

test['bool_belongs_to_collection'] = (test['belongs_to_collection'].notnull()).astype(int)
#27

len(train_tagline_keyword_w_cols)
#28

train['release_date'] = pd.to_datetime(train['release_date'] )

test['release_date'] = pd.to_datetime(test['release_date'] )
#29

train['release_month'] = train['release_date'].dt.month

#print(train['release_month'].value_counts())

test['release_month'] = test['release_date'].dt.month

#print(test['release_month'].value_counts())
#30

train['release_year'] = train['release_date'].dt.year

#print(train['release_year'].value_counts())

test['release_year'] = test['release_date'].dt.year

#print(test['release_year'].value_counts())
#31 --> 33

train['release_dayofyear'] = train['release_date'].dt.dayofyear

test['release_dayofyear'] = test['release_date'].dt.dayofyear

train['release_day_of_week'] = train['release_date'].dt.dayofweek

#print(train['release_day_of_week'].value_counts())

test['release_day_of_week'] = test['release_date'].dt.dayofweek

#print(test['release_day_of_week'].value_counts())

train['release_week'] = train['release_date'].dt.week

test['release_week'] = test['release_date'].dt.week

test['release_month'].mode()
#34-->36

test['release_year'].mode()

test['release_week'].mode()

test['release_year'].min()
#37-->46

test['release_month'] = test['release_month'].fillna(9.0)

test['release_year'] = test['release_year'].fillna(2014.0)

test['release_week'] = test['release_week'].fillna(36.0)

train['bool_homepage'] = (train['homepage'].notnull()).astype(int)

test['bool_homepage'] = (test['homepage'].notnull()).astype(int)

train['production_companies_len'] = train['production_companies'].str.len()

test['production_companies_len'] = test['production_companies'].str.len()

train['production_countries_len'] = train['production_countries'].str.len()

test['production_countries_len'] = test['production_countries'].str.len()

train['Keywords_len'] = train['Keywords'].str.len()

test['Keywords_len'] = test['Keywords'].str.len()

train['title_len'] = train['title'].str.len()

test['title_len'] = test['title'].str.len()

train['genres_len'] = train['genres'].str.len() 

test['genres_len'] = test['genres'].str.len() 

train['cast_crew_len'] = train['cast'].str.len() + train['crew'].str.len()

test['cast_crew_len'] = test['cast'].str.len() + test['crew'].str.len()

train['cast_crew_len'].fillna(train['cast_crew_len'].median(),inplace=True)

train['runtime'].fillna(train['runtime'].median(),inplace=True)

train['genres_len'].fillna(train['genres_len'].median(),inplace=True)

train['production_companies_len'].fillna(train['production_companies_len'].median(),inplace=True)

train['production_countries_len'].fillna(train['production_countries_len'].median(),inplace=True)

train['Keywords_len'].fillna(train['Keywords_len'].median(),inplace=True)

(train['release_year']>2019).sum()
#47-->49

train.loc[(train['release_year']>2019),'release_year']=train['release_year'].median()

train['month_into_year'] = train['release_month']*train['release_year']

(test['release_year']>2019).sum()
#50 -->54

test.loc[(test['release_year']>2019),'release_year']=test['release_year'].median()

vcast_crew_len = test['cast_crew_len'].median()

test['cast_crew_len'].fillna(vcast_crew_len, inplace=True)

test['runtime'].fillna(test['runtime'].median(),inplace=True)

test['release_month'].fillna(test['release_month'].median(),inplace=True)

test['title_len'].fillna(test['title_len'].median(),inplace=True)

test['release_year'].fillna(test['release_year'].median(),inplace=True)

test['release_day_of_week'].fillna(test['release_day_of_week'].median(),inplace=True)

test['release_dayofyear'].fillna(test['release_dayofyear'].median(),inplace=True)

test['genres_len'].fillna(test['genres_len'].median(),inplace=True)

test['production_companies_len'].fillna(test['production_companies_len'].median(),inplace=True)

test['production_countries_len'].fillna(test['production_countries_len'].median(),inplace=True)

test['Keywords_len'].fillna(test['Keywords_len'].median(),inplace=True)

test['month_into_year'] = test['release_month']*test['release_year']

features = ['bool_homepage', 'release_dayofyear','production_companies_len', 'production_countries_len', 'Keywords_len' , 'cast_crew_len','budget','popularity','runtime','release_month','release_day_of_week','release_week','genres_len','bool_belongs_to_collection', 'title_len','release_year']

train['log_revenue'] = np.log(train['revenue'].fillna(0)+1)
#55 Biểu đồ revenue với cast_crew_len

from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.models import ColumnDataSource

output_notebook()





source = ColumnDataSource(data=dict(

    cast_crew_len=train.cast_crew_len,

    revenue=train.revenue,

    original_language=train.original_language,

))





output_file("cast_crew_len_revenue.html", title="cast_crew_len, Revenue", mode="cdn")

TOOLTIPS = [

    ("cast_crew_len", "@cast_crew_len"),

    ("Revenue", "@revenue"),

    ("Original Language", "@original_language"),

    

]



p = figure(tooltips=TOOLTIPS,y_axis_type="log")



p.circle('cast_crew_len', 'revenue',fill_alpha=0.6, line_color=None, source = source)

p.xaxis.axis_label = "cast_crew_len"

p.yaxis.axis_label = "revenue"

show(p)
#56  Biểu đồ về tổng  doanh thu có homepage hoặc không 

g = sns.catplot(x='bool_homepage',y='log_revenue',data=train, kind='box', aspect=1 )

g.set_axis_labels("Is there a homepage", "Log of Revenue")
#57 Biểu đồ doanh thu theo tháng

g = sns.catplot(x='release_month',y='log_revenue',data=train, kind='box', aspect=2 )

g.set_axis_labels("Release month", "Log of Revenue")
#58 Biểu đồ doanh thu theo ngày trong tuần 

g = sns.catplot(x='release_day_of_week',y='log_revenue',data=train, kind='box', aspect=2 )

g.set_axis_labels("Release day of week", "Log of Revenue")
#59 Biểu đồ doanh thu Tuần trong năm

#https://www.kaggle.com/jlove5/avocados-usa-prices



g = sns.catplot(x='release_week',y='log_revenue',data=train, kind='box', aspect=3 )

g.set_axis_labels("Release week", "Log of Revenue")
#60 Doanh thu theo thể loại có drama hay k

train['is_genre_drama'] = (train['genre_drama']>0).astype(int)

g = sns.catplot(x='is_genre_drama',y='log_revenue', data=train,kind='box' )

g.set_axis_labels("is_genre_drama", "Log of Revenue")
#61

train['is_kw_tg_ow_death'] = (train['kw_tg_ow_death']>0).astype(int)

g = sns.catplot(x='is_kw_tg_ow_death',y='log_revenue', data=train,kind='box' )

g.set_axis_labels("is_kw_tg_ow_death", "Log of Revenue")
#62

train['is_genre_thriller'] = (train['genre_thriller']>0).astype(int)

g = sns.catplot(x='is_genre_thriller',y='log_revenue', data=train,kind='box' )

g.set_axis_labels("is_genre_thriller", "Log of Revenue")
#63 Doanh thu theo năm

#https://www.kaggle.com/jlove5/avocados-usa-prices



g = sns.catplot(x='release_year',y='log_revenue',data=train, kind='box', aspect=3 )

g.set_axis_labels("Release year", "Log of Revenue")

g.set_xticklabels(rotation=30)
#64

len(features)
#65

features = features+olang+train_tagline_keyword_w_cols+train_genres_w_cols+train_belongs_to_collection_w_cols \

+train_production_companies_w_cols + train_crew_w_cols + train_production_countries_w_cols
#66

len(features)
#67

test.release_year.min()
#68

train[features].info()
#69

test[features].info()
#70

target_column = 'revenue'

columns_for_prediction=features

X = train[columns_for_prediction].copy()

import sklearn.preprocessing as preprocessing

y_scale = preprocessing.PowerTransformer()

#y = np.log(train[target_column])

#https://stackoverflow.com/questions/26584971/how-to-not-standarize-target-data-in-scikit-learn-regression

y = y_scale.fit_transform(train[target_column].values.reshape(-1, 1) )



X_unseen = test[columns_for_prediction].copy()

scale = preprocessing.PowerTransformer()

X = pd.DataFrame(scale.fit_transform(X),columns=columns_for_prediction)

X_unseen = pd.DataFrame(scale.transform(test[columns_for_prediction]),columns=columns_for_prediction)



budget_min = X['budget'].quantile(0.28)

X['budget'] = X['budget'].replace(0,budget_min)



X_unseen['budget'] = X_unseen['budget'].replace(0,budget_min)

#71

columns_for_prediction
#73

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2019)

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

params = {'n_estimators': 700, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}



reg = GradientBoostingRegressor(**params).fit(X_train, y_train)

score = reg.score(X_test, y_test)

print('Test score %d'%score)

preds = reg.predict(X_test)

err = mean_squared_error(y_test, preds)

print('Test mse %d'%err)

reg = GradientBoostingRegressor(n_estimators=700).fit(X, y)

score = reg.score(X, y)

print('Train score %d'%score)

preds_first = reg.predict(X_unseen)
#75 dùng hồi quy để tính toán 

#https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



#Chuẩn hóa dữ liệu

def norm(x):

  return (x - train_stats.loc['mean']) / train_stats.loc['std']

train_dataset = X

#train_labels = y.values

train_labels = y

test_dataset = X_unseen

train_stats = train_dataset.describe()

normed_train_data = train_dataset

normed_test_data = test_dataset



#Xây dựng model

def build_model():

  model = keras.Sequential([

    layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer='normal', input_shape=[len(train_dataset.keys())]),

    layers.Dropout(.8),  

    layers.Dense(100, activation=tf.nn.leaky_relu, kernel_initializer='normal'),

    layers.Dropout(.6), 

    layers.Dense(50, activation=tf.nn.leaky_relu, kernel_initializer='normal'),

    layers.Dropout(.4),   

    layers.Dense(20, activation=tf.nn.leaky_relu, kernel_initializer='normal'),

    layers.Dropout(.2),   

    layers.Dense(1, activation='linear', kernel_initializer='normal')

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  #optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='mean_squared_error',

                optimizer=optimizer,

                metrics=['mean_absolute_error', 'mean_squared_error'])

  return model



model = build_model()

# summary method để in mô tả về đơn giản về mô hình

model.summary()



# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')



EPOCHS = 1000



history = model.fit(

  normed_train_data, train_labels,batch_size = 100,

  epochs=EPOCHS, validation_split = 0.1, verbose=1,

 callbacks=[PrintDot()])



hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()





def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch

    

  

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error ')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

  

  plt.legend()

  

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error ')

  plt.plot(hist['epoch'], hist['mean_squared_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_error'],

           label = 'Val Error')

  

  plt.legend()

  plt.show()





plot_history(history)

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,

                    validation_split = 0.2, verbose=0, callbacks=[early_stop])

plot_history(history)



preds_estop = model.predict(normed_test_data).flatten()
#80

preds[:10]
#82 Xuất file kết quả

median_revenue = train[target_column].median()

preds = preds_estop

preds = y_scale.inverse_transform(preds.reshape(-1, 1))

preds[preds < 0] = median_revenue

subm['revenue'] = preds

subm.to_csv('Submission.csv', index=False)

print(subm.head())
#83

len(train)
#84

import seaborn as sns

#sns.distplot(train['revenue'] )

train['revenue'].hist(log=True)
#85

len(subm)
#86

#sns.distplot(subm['revenue'] )

subm['revenue'].hist(log=True)
#87

ax = sns.scatterplot(x="popularity", y="revenue",

                     hue="release_year", 

                     data=train)
#88

ax = sns.scatterplot(x=test.popularity, y=subm.revenue,

                     hue=test.release_year)
#89

ax = sns.scatterplot(x="budget", y="revenue",

                     hue="release_year", 

                     data=train)
#90

ax = sns.scatterplot(x=test.budget, y=subm.revenue,

                     hue=test.release_year)
#91

train[ ['release_date', 'revenue']].set_index('release_date').resample('A').mean()[:'2019'].plot(style='--')

#92

test['revenue'] = subm['revenue']
#93

test[ ['release_date', 'revenue']].set_index('release_date').resample('A').mean()[:'2019'].plot(style='--')