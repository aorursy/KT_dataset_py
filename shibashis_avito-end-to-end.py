# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)



import numpy as np
import pandas as pd
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
train_df.info()
train_df.head()
parent_category_name_map = {"Личные вещи" : "Personal belongings",
                            "Для дома и дачи" : "For the home and garden",
                            "Бытовая электроника" : "Consumer electronics",
                            "Недвижимость" : "Real estate",
                            "Хобби и отдых" : "Hobbies & leisure",
                            "Транспорт" : "Transport",
                            "Услуги" : "Services",
                            "Животные" : "Animals",
                            "Для бизнеса" : "For business"}

region_map = {"Свердловская область" : "Sverdlovsk oblast",
            "Самарская область" : "Samara oblast",
            "Ростовская область" : "Rostov oblast",
            "Татарстан" : "Tatarstan",
            "Волгоградская область" : "Volgograd oblast",
            "Нижегородская область" : "Nizhny Novgorod oblast",
            "Пермский край" : "Perm Krai",
            "Оренбургская область" : "Orenburg oblast",
            "Ханты-Мансийский АО" : "Khanty-Mansi Autonomous Okrug",
            "Тюменская область" : "Tyumen oblast",
            "Башкортостан" : "Bashkortostan",
            "Краснодарский край" : "Krasnodar Krai",
            "Новосибирская область" : "Novosibirsk oblast",
            "Омская область" : "Omsk oblast",
            "Белгородская область" : "Belgorod oblast",
            "Челябинская область" : "Chelyabinsk oblast",
            "Воронежская область" : "Voronezh oblast",
            "Кемеровская область" : "Kemerovo oblast",
            "Саратовская область" : "Saratov oblast",
            "Владимирская область" : "Vladimir oblast",
            "Калининградская область" : "Kaliningrad oblast",
            "Красноярский край" : "Krasnoyarsk Krai",
            "Ярославская область" : "Yaroslavl oblast",
            "Удмуртия" : "Udmurtia",
            "Алтайский край" : "Altai Krai",
            "Иркутская область" : "Irkutsk oblast",
            "Ставропольский край" : "Stavropol Krai",
            "Тульская область" : "Tula oblast"}

category_map = {"Одежда, обувь, аксессуары":"Clothing, shoes, accessories",
"Детская одежда и обувь":"Children's clothing and shoes",
"Товары для детей и игрушки":"Children's products and toys",
"Квартиры":"Apartments",
"Телефоны":"Phones",
"Мебель и интерьер":"Furniture and interior",
"Предложение услуг":"Offer services",
"Автомобили":"Cars",
"Ремонт и строительство":"Repair and construction",
"Бытовая техника":"Appliances",
"Товары для компьютера":"Products for computer",
"Дома, дачи, коттеджи":"Houses, villas, cottages",
"Красота и здоровье":"Health and beauty",
"Аудио и видео":"Audio and video",
"Спорт и отдых":"Sports and recreation",
"Коллекционирование":"Collecting",
"Оборудование для бизнеса":"Equipment for business",
"Земельные участки":"Land",
"Часы и украшения":"Watches and jewelry",
"Книги и журналы":"Books and magazines",
"Собаки":"Dogs",
"Игры, приставки и программы":"Games, consoles and software",
"Другие животные":"Other animals",
"Велосипеды":"Bikes",
"Ноутбуки":"Laptops",
"Кошки":"Cats",
"Грузовики и спецтехника":"Trucks and buses",
"Посуда и товары для кухни":"Tableware and goods for kitchen",
"Растения":"Plants",
"Планшеты и электронные книги":"Tablets and e-books",
"Товары для животных":"Pet products",
"Комнаты":"Room",
"Фототехника":"Photo",
"Коммерческая недвижимость":"Commercial property",
"Гаражи и машиноместа":"Garages and Parking spaces",
"Музыкальные инструменты":"Musical instruments",
"Оргтехника и расходники":"Office equipment and consumables",
"Птицы":"Birds",
"Продукты питания":"Food",
"Мотоциклы и мототехника":"Motorcycles and bikes",
"Настольные компьютеры":"Desktop computers",
"Аквариум":"Aquarium",
"Охота и рыбалка":"Hunting and fishing",
"Билеты и путешествия":"Tickets and travel",
"Водный транспорт":"Water transport",
"Готовый бизнес":"Ready business",
"Недвижимость за рубежом":"Property abroad"}

train_df['region_en'] = train_df['region'].apply(lambda x : region_map[x])
train_df['parent_category_name_en'] = train_df['parent_category_name'].apply(lambda x : parent_category_name_map[x])
train_df['category_name_en'] = train_df['category_name'].apply(lambda x : category_map[x])


test_df['region_en'] = test_df['region'].apply(lambda x : region_map[x])
test_df['parent_category_name_en'] = test_df['parent_category_name'].apply(lambda x : parent_category_name_map[x])
test_df['category_name_en'] = test_df['category_name'].apply(lambda x : category_map[x])
#result = pd.concat([df1, df4], axis=1, sort=False)
#train_df1=pd.concat([train_df, train_df['region_en']], axis=1, sort=False)
train_df.head()
test_df.head()
# df.plot(x='col_name_1', y='col_name_2', style='o')
#train_df.plot(x='category_name_en', y='deal_probability', style='o')
#a.groupby('user')['num1', 'num2'].average()
#m1 = (df['SibSp'] > 0) | (df['Parch'] > 0)
#m= (train_df1['deal_probability'])

#grouped = df.groupby('mygroups').sum().reset_index()
#grouped.sort_values('mygroups', ascending=False)

grouped=train_df.groupby('category_name_en')['deal_probability'].sum()
grouped.sort_values(ascending=False)
#train_df.head()
#m.head()
## Description charecter count
# df['NAME_Count'] = df['NAME'].str.len()
train_df['des_str_count']=train_df['description'].str.len()
test_df['des_str_count']=test_df['description'].str.len()
train_df.head()
## Title charecter count
# df['NAME_Count'] = df['NAME'].str.len()
train_df['title_str_count']=train_df['title'].str.len()
test_df['title_str_count']=test_df['title'].str.len()
train_df.head()
#encoding text columns
#column = column.astype('category')
#column_encoded = column.cat.codes

train_df['region_en1'] = train_df['region_en'].astype('category')
train_df['region_en_encoded']=train_df['region_en1'].cat.codes


train_df['parent_category_name_en1'] = train_df['parent_category_name_en'].astype('category')
train_df['parent_category_name_en_encoded']=train_df['parent_category_name_en1'].cat.codes

train_df['category_name_en1'] = train_df['category_name_en'].astype('category')
train_df['category_name_en_encoded']=train_df['category_name_en1'].cat.codes


test_df['region_en1'] = test_df['region_en'].astype('category')
test_df['region_en_encoded']=test_df['region_en1'].cat.codes


test_df['parent_category_name_en1'] = test_df['parent_category_name_en'].astype('category')
test_df['parent_category_name_en_encoded']=test_df['parent_category_name_en1'].cat.codes

test_df['category_name_en1'] = test_df['category_name_en'].astype('category')
test_df['category_name_en_encoded']=test_df['category_name_en1'].cat.codes
test_df.head()
#new table with selected columns
train_df1=train_df[['title_str_count','des_str_count','region_en_encoded','parent_category_name_en_encoded','category_name_en_encoded','deal_probability']]

test_df1=test_df[['title_str_count','des_str_count','region_en_encoded','parent_category_name_en_encoded','category_name_en_encoded']]
train_df1.head()
test_df1.head()
#df.isnull().any().any() - This returns a boolean value
train_df1.isnull().any()
#df = df[np.isfinite(df['EPS'])]
#dat.dropna()
#train_df1=train_df1[np.isfinite(train_df1['des_str_count'])]
#train_df1=train_df1.dropna()
#train_df1.dropna(subset=train_df1['deal_probability'], how='all')

train_df1 = train_df1.dropna(how='any',axis=0) 
test_df1 = test_df1.dropna(how='any',axis=0) 
#train_df1.loc[:, train_df1.isna().any()]
train_df1.info()
train_df1.head()
#from sklearn import preprocessing
#from sklearn import utils

#lab_enc = preprocessing.LabelEncoder()
#encoded = lab_enc.fit_transform(train_df1['deal_probability'])

#train_df1['deal_probability'] = train_df1['deal_probability'].astype(int)
#train_df1['deal_probability'] = train_df1['deal_probability'].astype(float)
#print(utils.multiclass.type_of_target(train_df1['deal_probability']))
train_df1['deal_probability']=train_df1['deal_probability']*100
train_df1['deal_probability']
train_df1['deal_probability'] = train_df1['deal_probability'].astype(int)
#df1['c2'] = df1.c2.astype(np.int64)
train_df1['des_str_count'] = train_df1['des_str_count'].astype(np.int64)
train_df1['deal_probability']
#Splitting the Training Data
from sklearn.model_selection import train_test_split

predictors = train_df1[['title_str_count','des_str_count','region_en_encoded','parent_category_name_en_encoded','category_name_en_encoded']]
#predictors = train_df1[['title_str_count','des_str_count','parent_category_name_en_encoded','category_name_en_encoded']]



target = train_df1['deal_probability']
#y=y.astype('int')
#target=target.astype('float64')
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#y_train=y_train.astype('float64')
x_test=test_df1
#x_train
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# decesion tree
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(max_depth = 10, random_state = 0)
#tree.fit(x_train, y_train)


# Logistic reg
#from sklearn.linear_model import LogisticRegression
# Create logistic regression object
#model = LogisticRegression()
# Train the model using the training sets and check score
#model.fit(x_train, y_train)
#model.score(x_train, y_train)
#Equation coefficient and Intercept
#print('Coefficient: \n', model.coef_)
#print('Intercept: \n', model.intercept_)
#Predict Output
#predicted= model.predict(x_val)
#x_val
#predicted
y_pred
# Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score

#gaussian = GaussianNB()
#gaussian.fit(x_train, y_train)
#y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_gaussian)
# decesion tree
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(max_depth = 10, random_state = 0)
#tree.fit(x_train, y_train)
y_pred
##np.set_printoptions(threshold=np.inf)
y_pred=y_pred/100
test_df1.head()
y_test_pred = gaussian.predict(test_df1)
y_test_pred=y_test_pred/100
test_df1.head()
test_df.head()
# pd.concat([df1['c'], df2['c']], axis=1, keys=['df1', 'df2'])
submit_data= pd.concat([test_df['item_id'], pd.DataFrame(y_test_pred)], axis=1)
#submit_data
# data.rename(columns={'gdp':'log(gdp)'}, inplace=True)
#submit_data.rename(columns={0:'deal_probablity'}, inplace=True)  D:\personal\kaggle\Avito  funded.to_csv(path+'greenl.csv')
# import os
# funded.to_csv(os.path.join(path,r'green1.csv'))
submit_data.rename(columns={0:'deal_probability'}, inplace=True)
submit_data.head()
#train_df1.head()
#df1['c2'] = df1.c2.astype(np.int64)
#train_df1['des_str_count'] = train_df1['des_str_count'].astype(np.int64)
train_df1.head()

#x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)


#x_train
#Prediction
y_pred=clf.predict(x_val)
#y_val
#Accuracy

#from sklearn.metrics import accuracy_score
#accuracy=accuracy_score(y_pred,y_val)
#test prediction
y_test_pred_lgbm = clf.predict(test_df1)
y_test_pred_lgbm = y_test_pred_lgbm/100
# pd.concat([df1['c'], df2['c']], axis=1, keys=['df1', 'df2'])
submit_data_lgbm= pd.concat([test_df['item_id'], pd.DataFrame(y_test_pred_lgbm)], axis=1)
submit_data_lgbm.rename(columns={0:'deal_probability'}, inplace=True)
#path='D:\\personal\\kaggle\\Avito\\'
#import os
#submit_data.to_csv(os.path.join(path,r'submission.csv'))
#submit_data.to_csv(path+"submission.csv")
#submit_data.to_csv("D:\\personal\\kaggle\\Avito\\submission.csv", index=False)

#submit_data.to_csv("submission.csv", index=False)
submit_data_lgbm.to_csv("submission.csv", index=False)
#submit_data_lgbm.head()
#kaggle competitions submit -c avito-demand-prediction -f submission.csv -m "Message"

#print('Saved file: ' + "submission.csv")
#ls