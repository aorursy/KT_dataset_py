# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/avito-demand-prediction/train.csv")
test = pd.read_csv("../input/avito-demand-prediction/test.csv")
# Any results you write to the current directory are saved as output.
#아니면 도시별 평균값을 나눠서, 도시마다 평균값 자체를 attribute으로 해도 될듯. Or 평균값의 범위를 나누어도됨
#그다음 test data에는 특정도시에 대해 그 값을 할당해 새 feature로 사용

df_sorted = df.sort_values(by = ["image"])
test_sorted = test.sort_values(by = ["image"])
#Load image blurrness for train


tr_1 = pd.read_csv("../input/train-data-image-blurrness/1.csv")
tr_2 = pd.read_csv("../input/train-data-image-blurrness/2.csv")
tr_3 = pd.read_csv("../input/train-data-image-blurrness/3.csv")
tr_4 = pd.read_csv("../input/train-data-image-blurrness/4.csv")
tr_5 = pd.read_csv("../input/train-data-image-blurrness/5.csv")
tr_6 = pd.read_csv("../input/train-data-image-blurrness/6.csv")
tr_7 = pd.read_csv("../input/train-data-image-blurrness/7.csv")
tr_8 = pd.read_csv("../input/train-data-image-blurrness/8.csv")
tr_9 = pd.read_csv("../input/train-data-image-blurrness/9.csv")
tr_10 = pd.read_csv("../input/train-data-image-blurrness/10.csv")
tr_11 = pd.read_csv("../input/train-data-image-blurrness/11(12_13.5).csv")
tr_12 = pd.read_csv("../input/train-data-image-blurrness/last.csv")

frames = [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10, tr_11, tr_12]
new = pd.concat(frames)
new["File"] = new["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new = new.sort_values(by = ["File"])

scores = list(new["Score"].values) + [-1] * (len(df)-len(new))

df_sorted["image_blurrness_score"] = scores

df = df_sorted.sort_index()
#For test
te_1 = pd.read_csv("../input/image-blurrness-test/test_1.csv")
te_2 = pd.read_csv("../input/image-blurrness-test/test_2.csv")
te_3 = pd.read_csv("../input/image-blurrness-test/test_3.csv")
te_4 = pd.read_csv("../input/image-blurrness-test/test_4.csv")
te_5 = pd.read_csv("../input/image-blurrness-test/test_5.csv")

frames_te = [te_1, te_2, te_3, te_4, te_5]
new_te = pd.concat(frames_te)
new_te["File"] = new_te["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new_te = new_te.sort_values(by = ["File"])
scores_te = list(new_te["Score"].values) + [-1] * (len(test)-len(new_te))

test_sorted["image_blurrness_score"] = scores_te
test = test_sorted.sort_index()
df_train= df
df_test = test
#Copied this from others' kernel

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


params_top35_map = {'Женская одежда':"Women's clothing",
                    'Для девочек':'For girls',
                    'Для мальчиков':'For boys',
                    'Продам':'Selling',
                    'С пробегом':'With mileage',
                    'Аксессуары':'Accessories',
                    'Мужская одежда':"Men's Clothing",
                    'Другое':'Other','Игрушки':'Toys',
                    'Детские коляски':'Baby carriages', 
                    'Сдам':'Rent',
                    'Ремонт, строительство':'Repair, construction',
                    'Стройматериалы':'Building materials',
                    'iPhone':'iPhone',
                    'Кровати, диваны и кресла':'Beds, sofas and armchairs',
                    'Инструменты':'Instruments',
                    'Для кухни':'For kitchen',
                    'Комплектующие':'Accessories',
                    'Детская мебель':"Children's furniture",
                    'Шкафы и комоды':'Cabinets and chests of drawers',
                    'Приборы и аксессуары':'Devices and accessories',
                    'Для дома':'For home',
                    'Транспорт, перевозки':'Transport, transportation',
                    'Товары для кормления':'Feeding products',
                    'Samsung':'Samsung',
                    'Сниму':'Hire',
                    'Книги':'Books',
                    'Телевизоры и проекторы':'Televisions and projectors',
                    'Велосипеды и самокаты':'Bicycles and scooters',
                    'Предметы интерьера, искусство':'Interior items, art',
                    'Другая':'Other','Косметика':'Cosmetics',
                    'Постельные принадлежности':'Bed dress',
                    'С/х животные' :'Farm animals','Столы и стулья':'Tables and chairs'}

df_train['region_en'] = df_train['region'].apply(lambda x : region_map[x])
df_train['parent_category_name_en'] = df_train['parent_category_name'].apply(lambda x : parent_category_name_map[x])
df_train['category_name_en'] = df_train['category_name'].apply(lambda x : category_map[x])

del df_train['region']
del df_train['parent_category_name']
del df_train['category_name']

df_train

df_test['region_en'] = df_test['region'].apply(lambda x : region_map[x])
df_test['parent_category_name_en'] = df_test['parent_category_name'].apply(lambda x : parent_category_name_map[x])
df_test['category_name_en'] = df_test['category_name'].apply(lambda x : category_map[x])

del df_test['region']
del df_test['parent_category_name']
del df_test['category_name']
df_test

df_train
regional = pd.read_csv("../input/regionaldata/regional.csv", index_col = [0])

rDense = regional["Density_of_region(km2)"]
rRural = regional["Rural_%"]
rTime_zone = regional["Time_zone"]
rPopulation = regional["Total_population"]
rUrban = regional["Urban%"]

reg_index = np.array([regional.index[i].lower() for i in range(len(regional))])
rDense.index = reg_index
rRural.index = reg_index
rTime_zone.index = reg_index
rPopulation.index = reg_index
rUrban.index = reg_index

df_region = df_train["region_en"]

reg_dense = np.array([rDense[df_region[i].lower()] for i in range(len(df_train))])
reg_rural = np.array([rRural[df_region[i].lower()] for i in range(len(df_train))])
reg_Time_zone = np.array([rTime_zone[df_region[i].lower()] for i in range(len(df_train))])
reg_Population = np.array([rPopulation[df_region[i].lower()] for i in range(len(df_train))])
reg_Urban = np.array([rUrban[df_region[i].lower()] for i in range(len(df_train))])

df_train["reg_dense"] = reg_dense
df_train["rural"] = reg_rural
df_train["reg_Time_zone"] = reg_Time_zone
df_train["reg_Population"] = reg_Population
df_train["reg_Urban"] = reg_Urban

df_train

reg_dense = np.array([rDense[df_region[i].lower()] for i in range(len(df_test))])
reg_rural = np.array([rRural[df_region[i].lower()] for i in range(len(df_test))])
reg_Time_zone = np.array([rTime_zone[df_region[i].lower()] for i in range(len(df_test))])
reg_Population = np.array([rPopulation[df_region[i].lower()] for i in range(len(df_test))])
reg_Urban = np.array([rUrban[df_region[i].lower()] for i in range(len(df_test))])

df_test["reg_dense"] = reg_dense
df_test["rural"] = reg_rural
df_test["reg_Time_zone"] = reg_Time_zone
df_test["reg_Population"] = reg_Population
df_test["reg_Urban"] = reg_Urban

df_train
#Cut out unimportant features
new_df_train = df_train.copy()
new_df_test = df_test.copy()
del new_df_test["image"]
del new_df_train["image"]

del new_df_test["activation_date"]
del new_df_train["activation_date"]
#image 제외함
#vectorize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
import gc

stopWords_en = stopwords.words('english')
stopWords_ru = stopwords.words('russian')

tr =new_df_train.copy()
te =new_df_test.copy()

tri=tr.shape[0]
y = tr.deal_probability.copy()
lb=LabelEncoder()
List_Var=['item_id', 'description']

def Concat_Text(df,Columns,Name):
    df=df.copy()
    df.loc[:,Columns].fillna(" ",inplace=True)
    df[Name]=df[Columns[0]].astype('str')
    for col in Columns[1:]:
        df[Name]=df[Name]+' '+df[col].astype('str')
    return df

def Ratio_Words(df):
    df=df.copy()
    df['description']=df['description'].astype('str')
    df['num_words_description']=df['description'].apply(lambda x:len(x.split()))
    Unique_Words=df['description'].apply(lambda x: len(set(x.split())))
    df['Ratio_Words_description']=Unique_Words/df['num_words_description']
    return df

def Lenght_Columns(df,Columns):
    df=df.copy()
    Columns_Len=['len_'+s for s in Columns]
    for col in Columns:
        df[col]=df[col].astype('str')
    for x,y in zip(Columns,Columns_Len):
        df[y]=df[x].apply(len)
    return df


####
tr_te=tr[tr.columns.difference(["deal_probability"])].append(te)\
     .pipe(Concat_Text,['city','param_1'],'txt1')\
     .pipe(Concat_Text,['title','description'],'txt2').pipe(Ratio_Words).pipe(Lenght_Columns,['title','description','param_1']).assign( category_name_en=lambda x: pd.Categorical(x['category_name_en']).codes,
              parent_category_name_en=lambda x:pd.Categorical(x['parent_category_name_en']).codes,
              region_en=lambda x:pd.Categorical(x['region_en']).codes, reg_Time_zone=lambda x:pd.Categorical(x['reg_Time_zone']).codes,
              user_type=lambda x:pd.Categorical(x['user_type']).codes, image_top_1=lambda x:pd.Categorical(x['image_top_1']).codes,
              param_1=lambda x:lb.fit_transform(x['param_1'].fillna('-1').astype('str')),
            param_2=lambda x:lb.fit_transform(x['param_2'].fillna('-1').astype('str')), param_3=lambda x:lb.fit_transform(x['param_3'].fillna('-1').astype('str')),
              user_id=lambda x:lb.fit_transform(x['user_id'].astype('str')), reg_dense=lambda x: np.log1p(x['reg_dense'].fillna(0)),
              city=lambda x:lb.fit_transform(x['city'].astype('str')), rural=lambda x: np.log1p(x['rural'].fillna(0)),
            reg_Urban=lambda x: np.log1p(x['reg_Urban'].fillna(0)),
             price=lambda x: np.log1p(x['price'].fillna(0)), reg_Population=lambda x: np.log1p(x['reg_Population'].fillna(0)),
             image_blurrness_score=lambda x: np.log1p(x['image_blurrness_score'].fillna(0)),
             
            title=lambda x: x['title'].astype('str')).drop(labels=List_Var,axis=1)

tr_te.price.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
tr_te.reg_Population.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
tr_te.reg_Urban.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
tr_te.reg_dense.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
tr_te.rural.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
tr_te.image_blurrness_score.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)

tr_te



##
del tr,te
gc.collect()

tr_te.loc[:,'txt2']=tr_te.txt2.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))

print("Processing Text")
print("Text 1")

vec1=CountVectorizer(ngram_range=(1,2),dtype=np.uint8,min_df=5, binary=True,max_features=3000) 
m_tfidf1=vec1.fit_transform(tr_te.txt1)
tr_te.drop(labels=['txt1'],inplace=True,axis=1)

print("Text 2")

vec2=TfidfVectorizer(ngram_range=(1,2),stop_words=stopWords_ru,min_df=3,max_df=0.4,sublinear_tf=True,norm='l2',max_features=5500,dtype=np.uint8)
m_tfidf2=vec2.fit_transform(tr_te.txt2)
tr_te.drop(labels=['txt2'],inplace=True,axis=1)

print("Title")
vec3=CountVectorizer(ngram_range=(3,6),analyzer='char_wb',dtype=np.uint8,min_df=5, binary=True,max_features=2000) 
m_tfidf3=vec3.fit_transform(tr_te.title)
tr_te.drop(labels=['title'],inplace=True,axis=1)

data  = hstack((tr_te.values,m_tfidf1,m_tfidf2,m_tfidf3)).tocsr()

print(data.shape)
del tr_te,m_tfidf1,m_tfidf2,m_tfidf3
gc.collect()

dtest=data[tri:]
X=data[:tri]

del data
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=23)


#LGB
dtrain =lgb.Dataset(data = X_train, label = y_train)
dval =lgb.Dataset(data = X_valid, label = y_valid)

Dparam = {'objective' : 'regression',
          'boosting_type': 'gbdt',
          'metric' : 'rmse',
          'nthread' : 4,
          'shrinkage_rate':0.02,
          'max_depth':18,
          'min_child_weight': 8,
          'bagging_fraction':0.75,
          'feature_fraction':0.75,
          'lambda_l1':0,
          'lambda_l2':0,
          'num_leaves':31, 
         'verbosity' : -1} 

print("Training Model")
m_lgb=lgb.train(params=Dparam,train_set=dtrain,num_boost_round=40000, early_stopping_rounds=500, valid_sets=[dtrain,dval], verbose_eval=50,valid_names=['train','valid'])
#하고서 0.03, 15000라운드 해보기

#Plot
fig, ax = plt.subplots(figsize=(10, 14))
lgb.plot_importance(m_lgb, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")


Pred=m_lgb.predict(dtest)
Pred[Pred<0]=0
#이것도 하지 말아보기 (0보내는거)
Pred[Pred>1]=1


print("Output Model")
LGB_text=pd.read_csv("../input/avito-demand-prediction/sample_submission.csv")
LGB_text['deal_probability']=Pred
LGB_text.to_csv("no_Actdate.csv", index=False)