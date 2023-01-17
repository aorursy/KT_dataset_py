import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.decomposition import TruncatedSVD

color = sns.color_palette()
%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)
#train_period = pd.read_csv("../input/periods_train.csv", parse_dates=["activation_date", "date_from", "date_to"])
#test_period = pd.read_csv("../input/periods_test.csv", parse_dates=["activation_date", "date_from", "date_to"])
#print("Period Train file rows and columns are : ", train_period.shape)
#print("Period Test file rows and columns are : ", test_period.shape)
#train_period.head()
#test_period['item_id'].nunique()
#Data Manipulation converting Russian to English

from io import StringIO

temp_data = StringIO("""
region,region_en
Свердловская область, Sverdlovsk oblast
Самарская область, Samara oblast
Ростовская область, Rostov oblast
Татарстан, Tatarstan
Волгоградская область, Volgograd oblast
Нижегородская область, Nizhny Novgorod oblast
Пермский край, Perm Krai
Оренбургская область, Orenburg oblast
Ханты-Мансийский АО, Khanty-Mansi Autonomous Okrug
Тюменская область, Tyumen oblast
Башкортостан, Bashkortostan
Краснодарский край, Krasnodar Krai
Новосибирская область, Novosibirsk oblast
Омская область, Omsk oblast
Белгородская область, Belgorod oblast
Челябинская область, Chelyabinsk oblast
Воронежская область, Voronezh oblast
Кемеровская область, Kemerovo oblast
Саратовская область, Saratov oblast
Владимирская область, Vladimir oblast
Калининградская область, Kaliningrad oblast
Красноярский край, Krasnoyarsk Krai
Ярославская область, Yaroslavl oblast
Удмуртия, Udmurtia
Алтайский край, Altai Krai
Иркутская область, Irkutsk oblast
Ставропольский край, Stavropol Krai
Тульская область, Tula oblast
""")

region_df = pd.read_csv(temp_data)
train_df = pd.merge(train_df, region_df, how="left", on="region")
test_df = pd.merge(test_df, region_df, how="left", on="region")
temp_data = StringIO("""
parent_category_name,parent_category_name_en
Личные вещи,Personal belongings
Для дома и дачи,For the home and garden
Бытовая электроника,Consumer electronics
Недвижимость,Real estate
Хобби и отдых,Hobbies & leisure
Транспорт,Transport
Услуги,Services
Животные,Animals
Для бизнеса,For business
""")

temp_df = pd.read_csv(temp_data)
train_df = pd.merge(train_df, temp_df, on="parent_category_name", how="left")
test_df = pd.merge(test_df, temp_df, how="left", on="parent_category_name")

temp_data = StringIO("""
category_name,category_name_en
"Одежда, обувь, аксессуары","Clothing, shoes, accessories"
Детская одежда и обувь,Children's clothing and shoes
Товары для детей и игрушки,Children's products and toys
Квартиры,Apartments
Телефоны,Phones
Мебель и интерьер,Furniture and interior
Предложение услуг,Offer services
Автомобили,Cars
Ремонт и строительство,Repair and construction
Бытовая техника,Appliances
Товары для компьютера,Products for computer
"Дома, дачи, коттеджи","Houses, villas, cottages"
Красота и здоровье,Health and beauty
Аудио и видео,Audio and video
Спорт и отдых,Sports and recreation
Коллекционирование,Collecting
Оборудование для бизнеса,Equipment for business
Земельные участки,Land
Часы и украшения,Watches and jewelry
Книги и журналы,Books and magazines
Собаки,Dogs
"Игры, приставки и программы","Games, consoles and software"
Другие животные,Other animals
Велосипеды,Bikes
Ноутбуки,Laptops
Кошки,Cats
Грузовики и спецтехника,Trucks and buses
Посуда и товары для кухни,Tableware and goods for kitchen
Растения,Plants
Планшеты и электронные книги,Tablets and e-books
Товары для животных,Pet products
Комнаты,Room
Фототехника,Photo
Коммерческая недвижимость,Commercial property
Гаражи и машиноместа,Garages and Parking spaces
Музыкальные инструменты,Musical instruments
Оргтехника и расходники,Office equipment and consumables
Птицы,Birds
Продукты питания,Food
Мотоциклы и мототехника,Motorcycles and bikes
Настольные компьютеры,Desktop computers
Аквариум,Aquarium
Охота и рыбалка,Hunting and fishing
Билеты и путешествия,Tickets and travel
Водный транспорт,Water transport
Готовый бизнес,Ready business
Недвижимость за рубежом,Property abroad
""")

temp_df = pd.read_csv(temp_data)
train_df = pd.merge(train_df, temp_df, on="category_name", how="left")
test_df = pd.merge(test_df, temp_df, on="category_name", how="left")

train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))
train_df["description"].fillna("NA", inplace=True)
train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))

test_df["description"].fillna("NA", inplace=True)
test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))
### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
#ngram_range defines how you want to have words in your dictionary. (min,max) = (1,2) will mean you will have unigrams and bigrms in your vocabulary. 
#Example String: "The old fox"
#Vocabulary: "The", "old", "fox", "The old", "old fox"

full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
#train_df['title'].values.tolist() this converts all the values in the title column into a list. '+' operator appends two lists with each other

train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())
### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['description'].values.tolist() + test_df['description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
train_df['param123'] = train_df['param_1'].fillna('') + " " + train_df['param_2'].fillna('') + " " + train_df['param_3'].fillna('') 
test_df['param123'] = test_df['param_1'].fillna('') + " " + test_df['param_2'].fillna('') + " " + test_df['param_3'].fillna('') 
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['param123'].values.tolist() + test_df['param123'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['param123'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['param123'].values.tolist())

### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_params_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_params_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
# New variable on weekday #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday
train_df.head()
test_df.head()
train_df["price_new"] = train_df["price"].values
train_df["price_new"].fillna(np.nanmedian(train_df["price"].values), inplace=True)

test_df["price_new"] = test_df["price"].values
test_df["price_new"].fillna(np.nanmedian(train_df["price"].values), inplace=True)

#Feature Scaling
train_df["price_new"] = (train_df["price_new"]/sum(train_df["price_new"]))
test_df["price_new"] = test_df["price_new"]/sum(train_df["price_new"])

train_df["price"] = train_df["price_new"]
test_df["price"] = test_df["price_new"]
trn = train_df
tst = test_df
train_df.head()
test_df.head()
#train_df = trn
#test_df = tst
train_df['image_top_1'] = train_df['image_top_1'].fillna(0)
train_df['image_top_1'] = train_df['image_top_1'].astype('float32')
print(train_df.isnull().sum())
# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

#train_df = train_df.dropna()

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image", "param_1", "param_2", 
                "param_3", "param123", "region_en", "parent_category_name_en", "category_name_en", "price_new"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values
test_X['image_top_1'] = test_X['image_top_1'].fillna(0)
test_X['image_top_1'] = test_X['image_top_1'].astype('float32')
print(test_X.isnull().sum())
test_X.head()
#split the train into development and validation sample. Take the last 100K rows as validation sample.
# Splitting the data for model training#
dev_X = train_X.iloc[:-100000,:]
val_X = train_X.iloc[-100000:,:]
dev_y = train_y[:-100000]
val_y = train_y[-100000:]
print(dev_X.shape, val_X.shape, test_X.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
np.random.seed(123)

#def create_model():
# create model
model = Sequential()
model.add(Dense(192, input_dim=26, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, kernel_initializer='normal', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer='normal', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

#return model
history = model.fit(dev_X, dev_y, validation_split=0.1, epochs=10, batch_size=100, verbose=1)
from sklearn.metrics import mean_squared_error
res = model.predict(val_X)
print(res)
score = mean_squared_error(val_y, res)
print(score)
# Making a submission file #
pred_test = model.predict(test_X)
print(pred_test)
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("baseline_mlp.csv", index=False)
#print(os.listdir("../working"))