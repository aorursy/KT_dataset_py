import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'])

print("Shape train: ", df_train.shape)
df_train.info()
is_null = df_train.isnull().sum() / len(df_train) * 100
print("NaN values in train Dataset")
print(is_null[is_null > 0].sort_values(ascending=False))
plt.figure(figsize=(16, 5))

cols = df_train.columns

uniques = [len(df_train[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, palette='hls', log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
# I have copied some dictionary's from another fellows Kernels.
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
df_train.head()
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
ax = sns.distplot(df_train["deal_probability"].values, bins=100, kde=False)
ax.set_xlabel('Deal Probility', fontsize=15)
ax.set_ylabel('Deal Probility', fontsize=15)
ax.set_title("Deal Probability Histogram", fontsize=20)

plt.subplot(1,2,2)
plt.scatter(range(df_train.shape[0]), np.sort(df_train['deal_probability'].values))
plt.xlabel('Index', fontsize=15)
plt.ylabel('Deal Probability', fontsize=15)
plt.title("Deal Probability Distribution", fontsize=20)
plt.xticks(rotation=45)
plt.show()

interval = (-0.99, .02, .05, .1, .15, .2, .35, .50, .70,.85,2)
cats = ['0 -.02%', '.02%-.05%', '.05-.10', '.10-.15', '.15-.20', '.20-.35', '.35-.50', '.50-.70', '.70-.85','.85+']

df_train["deal_prob_cat"] = pd.cut(df_train.deal_probability, interval, labels=cats)
prob_cat_percent = df_train["deal_prob_cat"].value_counts() / len(df_train['deal_probability'])* 100

plt.figure(figsize=(12,5))
g = sns.barplot(prob_cat_percent.index, prob_cat_percent.values)
g.set_xlabel('The Deal Probability Categorical Dist',fontsize=16)
g.set_ylabel('% of frequency',fontsize=16)
g.set_title('Deal Probability Categorical % Frequency',fontsize= 20)
plt.show()
df_train['price_log'] = np.log(df_train['price'] + 1)

plt.figure(figsize=(12,5))

g = sns.boxplot(x='deal_prob_cat', y='price_log', data=df_train)
g.set_xlabel('The Deal Probability Categorical Dist',fontsize=16)
g.set_ylabel('Price Log Dist',fontsize=16)
g.set_title('Looking the Price Log of each deal_prob_cat',fontsize= 20)

plt.show()
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
g = sns.distplot(np.log(df_train['price'].dropna() + 1))
g.set_xlabel('Price Log', fontsize=15)
g.set_ylabel('Probility', fontsize=15)
g.set_title("Price Histogram", fontsize=20)

plt.subplot(1,2,2)
plt.scatter(range(df_train.shape[0]), np.sort(np.log(df_train['price']+1).values))
plt.xlabel('Index ', fontsize=15)
plt.ylabel('Price Log', fontsize=15)
plt.title("Price Log Distribution", fontsize=20)
plt.xticks(rotation=45)
plt.show()

plt.show()
print("User Type % Proportion")
print(round(df_train['user_type'].value_counts() / len(df_train) * 100, 2))

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
g = sns.countplot(x='user_type', data=df_train, )
g.set_xlabel('User Type',fontsize=16)
g.set_ylabel('Count',fontsize=16)
g.set_title('User Type Count',fontsize= 20)

plt.subplot(1,3,2)
g1 = sns.boxplot(x='user_type', y='deal_probability', data=df_train)
g1.set_xlabel('User Type',fontsize=16)
g1.set_ylabel('Deal probability',fontsize=16)
g1.set_title('User Type x Prob Dist',fontsize= 20)

plt.subplot(1,3,3)
g1 = sns.boxplot(x='user_type', y='price_log', data=df_train)
g1.set_xlabel('User Type',fontsize=16)
g1.set_ylabel('Price Log',fontsize=16)
g1.set_title('User Type x Price Log',fontsize= 20)

plt.show()
cols = ['user_type','deal_prob_cat']
colmap = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cols[0]], df_train[cols[1]]).style.background_gradient(cmap = colmap)
plt.figure(figsize=(12,14))

plt.subplot(3,1,1)
g = sns.countplot(x='parent_category_name_en', data=df_train)
g.set_xlabel('User Type',fontsize=16)
g.set_ylabel('Count',fontsize=16)
g.set_title('Category of Ad',fontsize= 20)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(x='parent_category_name_en',y='deal_probability', data=df_train)
g1.set_xlabel("Category's Name",fontsize=16)
g1.set_ylabel('Deal Probability',fontsize=16)
g1.set_title('Category of Ad',fontsize= 20)
g1.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(x='parent_category_name_en', y='price_log', data=df_train)
g2.set_xlabel("Category's Name",fontsize=16)
g2.set_ylabel('Price Log',fontsize=16)
g2.set_title('Category of Ad',fontsize= 20)
g2.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()
cols = ['parent_category_name_en','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cols[0]], df_train[cols[1]]).style.background_gradient(cmap = cm)
plt.figure(figsize=(16,20))
plt.subplot(3,1,1)
g = sns.countplot(x='region_en', data=df_train)
g.set_xlabel('Ad Regions',fontsize=16)
g.set_ylabel('Count',fontsize=16)
g.set_title('Ad Regions Count',fontsize= 20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(3,1,2)
g1 = sns.boxplot(x='region_en', y='deal_probability',data=df_train, orient='')
g1.set_xlabel('Ad Regions',fontsize=16)
g1.set_ylabel('Deal Probability',fontsize=16)
g1.set_title('Ad Regions Deal Prob Distribuition',fontsize= 20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)

plt.subplot(3,1,3)
g2 = sns.boxplot(x='region_en', y='price_log',data=df_train, orient='')
g2.set_xlabel('Ad Regions',fontsize=16)
g2.set_ylabel('Price Log Distribuition',fontsize=16)
g2.set_title('Ad Regions Price Distribuition',fontsize= 20)
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()
cols = ['region_en','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cols[0]], df_train[cols[1]]).style.background_gradient(cmap = cm)
lower_probs = ['0 -.02%', '.02%-.05%', '.05-.10']
higher_probs = ['.70-.85', '.85+']

plt.figure(figsize=(15,12))

plt.subplot(2,1,1)
g = sns.countplot(x='region_en', data=df_train[df_train.deal_prob_cat.isin(lower_probs)])
g.set_title("Region with lower deal probs", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Regions", fontsize=15)
g.set_ylabel("Count", fontsize=15)


plt.subplot(2,1,2)
g1 = sns.countplot(x='region_en', data=df_train[df_train.deal_prob_cat.isin(higher_probs)])
g1.set_title("Regions with higher deal probability", fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel("Regions",fontsize=15)
g1.set_ylabel("Count", fontsize=15)

plt.subplots_adjust(hspace = 0.8,top = 0.9)

plt.show()
city_count = df_train['city'].value_counts()[:35].index.values

plt.figure(figsize=(15,6))

g = sns.countplot(x='city', data=df_train[df_train.city.isin(city_count)])
g.set_xlabel('Ad Citys',fontsize=16)
g.set_ylabel('Count',fontsize=16)
g.set_title("Ad City's Count",fontsize= 20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.show()
cities_top_35 = ['Краснодар', 'Екатеринбург', 'Новосибирск', 'Ростов-на-Дону',
       'Нижний Новгород', 'Челябинск', 'Пермь', 'Казань', 'Самара', 'Омск',
       'Уфа', 'Красноярск', 'Воронеж', 'Волгоград', 'Саратов', 'Тюмень',
       'Калининград', 'Барнаул', 'Ярославль', 'Иркутск', 'Оренбург', 'Сочи',
       'Ижевск', 'Тольятти', 'Кемерово', 'Белгород', 'Тула', 'Ставрополь',
       'Набережные Челны', 'Новокузнецк', 'Владимир', 'Сургут', 'Магнитогорск',
       'Нижний Тагил', 'Новороссийск']
plt.figure(figsize=(16,10))

plt.subplot(2,1,1)
g = sns.boxplot(x='city', y='deal_probability', data=df_train[df_train.city.isin(cities_top_35)])
g.set_xlabel("", fontsize=15)
g.set_ylabel("Deal Probability", fontsize=15)
g.set_title("Deal Probability of TOP 35 City's", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='city', y='price_log', data=df_train[df_train.city.isin(cities_top_35)])
g1.set_xlabel("TOP 35 City's", fontsize=15)
g1.set_ylabel("Price Log", fontsize=15)
g1.set_title("Price Log of TOP 35 City's", fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()
plt.figure(figsize=(16,12))

plt.subplot(2,1,1)
g = sns.countplot(x='category_name_en', data=df_train)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Category Names', fontsize=15)
g.set_ylabel('Count', fontsize=15)
g.set_title('Category Name Count', fontsize=20)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='category_name_en', y='price_log', data=df_train)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel('Category Names', fontsize=15)
g1.set_ylabel('Price log Dist', fontsize=15)
g1.set_title('Category Name Count', fontsize=20)

plt.subplots_adjust(hspace = 0.9,top = 0.9)

plt.show()
cols = ['category_name_en','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cols[0]], df_train[cols[1]]).style.background_gradient(cmap = cm)
params = df_train.param_1.value_counts().head(35)

params.index = ["Women's Clothing", 'For Girls', 'For Boys', 'Selling',
                'With mileage', 'Accessories', "Men's clothing", 'Other', 'Toys',
                'Baby carriages', 'Rent', 'Repair, construction', 'Building materials',
                'iPhone', 'Beds, sofas and armchairs', 'Tools', 'For the kitchen',
                'Accessories', "Children's Furniture", 'Cabinets and Chests', 
                'Devices and accessories', 'For the house', 'Transport, transportation',
                'Nursing Items', 'Samsung', 'Hire', 'Books',
                'TVs and projectors', 'Bicycles and scooters',
                'Interior items, art', 'Other', 'Cosmetics',
                'Bedding', 'Farm animals', 'Tables and chairs']
plt.figure(figsize=(16,5))

g = sns.barplot(x=params.index, y=params.values)
g.set_xlabel("Translated params", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_title("Most Frequent params in Ads", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.show()
print("The top five Ad params in %")
print(round((params / len(df_train) * 100).head(n=5),2))
# I used the google translate to translate this all words in params
russian_param_names = ["Женская одежда","Для девочек","Для мальчиков","Продам", "С пробегом","Аксессуары",
"Мужская одежда","Другое","Игрушки","Детские коляски","Сдам","Ремонт, строительство","Стройматериалы",
"iPhone","Кровати, диваны и кресла","Инструменты","Для кухни","Комплектующие","Детская мебель","Шкафы и комоды",
"Приборы и аксессуары","Для дома","Транспорт, перевозки","Товары для кормления","Samsung","Сниму",
"Книги","Телевизоры и проекторы","Велосипеды и самокаты","Предметы интерьера, искусство","Другая",
"Косметика","Постельные принадлежности","С/х животные","Столы и стулья"]

subset_param = df_train[df_train.param_1.isin(russian_param_names)]

subset_param['param_en'] = subset_param['param_1'].apply(lambda x : params_top35_map[x])
plt.figure(figsize=(16,12))
plt.subplot(2,1,1)
g = sns.boxplot(x='param_en', y='price_log', data=subset_param)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Price Dist(log)", fontsize=15)
g.set_title("Price of TOP 35 params_1", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='param_en', y='deal_probability', data=subset_param)
g1.set_xlabel("TOP 35 Params", fontsize=15)
g1.set_ylabel("Deal Probability", fontsize=15)
g1.set_title("Deal Probability of TOP 35 params_1", fontsize=20)
g1.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.9,top = 0.9)

plt.show()
cols = ['param_en','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(subset_param[cols[0]], subset_param[cols[1]]).style.background_gradient(cmap = cm)
df_train['day'] = df_train['activation_date'].dt.day

time_count = df_train['day'].value_counts()

plt.figure(figsize=(16,5))

g = sns.barplot(time_count.index, time_count.values)
g.set_xlabel("Date Distribuition of Dataset", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_title("Ads Date of Avitos dataset", fontsize=20)

plt.show()
cols = ['day','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cols[0]], df_train[cols[1]]).style.background_gradient(cmap = cm)
#nlp
import string
import re    #for any necessary regex
import nltk
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
from nltk.corpus import stopwords

#Word count in each comment:
df_train['count_word'] = df_train["title"].apply(lambda x: len(str(x).split()))
df_train['count_word_desc'] = df_train["description"].apply(lambda x: len(str(x).split()))

#Unique word count
df_train['count_unique_word'] = df_train["title"].apply(lambda x: len(set(str(x).split())))
df_train['count_unique_word_desc']= df_train["description"].apply(lambda x: len(set(str(x).split())))

#Letter count
df_train['count_letters'] = df_train["title"].apply(lambda x: len(str(x)))
df_train['count_letters_desc']= df_train["description"].apply(lambda x: len(str(x)))

#punctuation count
df_train["count_punctuations"] = df_train["title"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_train["count_punctuations_desc"] = df_train["description"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count
df_train["count_words_upper"] = df_train["title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df_train["count_words_upper_desc"] = df_train["description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count
df_train["count_words_title"] = df_train["title"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df_train["count_words_title_desc"] = df_train["description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Average length of the words
df_train["mean_word_len"] = df_train["title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_train["mean_word_len_desc"] = df_train["description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
plt.figure(figsize = (12,18))

plt.subplot(421)
g1 = sns.distplot(np.log(df_train['count_word']), 
                  hist=False, label='Title')
g1 = sns.distplot(np.log(df_train['count_word_desc']), 
                  hist=False, label='Description')
g1.set_title("COUNT WORDS DISTRIBUITION", fontsize=16)

plt.subplot(422)
g2 = sns.distplot(np.log(df_train['count_unique_word']),
                  hist=False, label='Title')
g2 = sns.distplot(np.log(df_train['count_unique_word_desc']), 
                  hist=False, label='Description')
g2.set_title("COUNT UNIQUE DISTRIBUITION", fontsize=16)

plt.subplot(423)
g3 = sns.distplot(np.log(df_train['count_letters']), 
                  hist=False, label='Title')
g3 = sns.distplot(np.log(df_train['count_letters_desc']), 
                  hist=False, label='Description')
g3.set_title("COUNT LETTERS DISTRIBUITION", fontsize=16)

plt.subplot(424)
g4 = sns.distplot(np.log(df_train["count_punctuations"]), 
                  hist=False, label='Title')
g4 = sns.distplot(np.log(df_train["count_punctuations_desc"]), 
                  hist=False, label='Description')
g4.set_xlim([-2,50])
g4.set_title('COUNT PONCTUATIONS DISTRIBUITION', fontsize=16)

plt.subplot(425)
g5 = sns.distplot(np.log(df_train["count_words_upper"] + 1) , 
                  hist=False, label='Title')
g5 = sns.distplot(np.log(df_train["count_words_upper_desc"] + 1) , 
                  hist=False, label='Description')
g5.set_title('COUNT WORDS UPPER DISTRIBUITION', fontsize=16)

plt.subplot(426)
g6 = sns.distplot(np.log(df_train["count_words_title"] + 1), 
                  hist=False, label='Title')
g6 = sns.distplot(np.log(df_train["count_words_title_desc"]  + 1), 
                  hist=False, label='Tags')
g6.set_title('WORDS DISTRIBUITION', fontsize=16)

plt.subplot(427)
g7 = sns.distplot(np.log(df_train["mean_word_len"]  + 1), 
                  hist=False, label='Title')
g7 = sns.distplot(np.log(df_train["mean_word_len_desc"] + 1), 
                  hist=False, label='Description')
g7.set_xlim([-2,100])
g7.set_title('MEAN WORD LEN DISTRIBUITION', fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.legend()
plt.show()
df_train['count_word_log'] = np.log(df_train['count_word'])

(sns
  .FacetGrid(df_train, 
             hue='deal_prob_cat', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'count_word_log', shade=True)
 .add_legend()
)
plt.show()
df_train['count_word_desc_log'] = np.log(df_train['count_word_desc'])

(sns
  .FacetGrid(df_train, 
             hue='deal_prob_cat', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'count_word_desc_log', shade=True)
 .add_legend()
)
plt.show()
df_train['count_unique_word_log'] = np.log(df_train['count_unique_word'])

(sns
  .FacetGrid(df_train, 
             hue='deal_prob_cat', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'count_unique_word_log', shade=True)
 .add_legend()
)
plt.show()
df_train['count_unique_word_desc_log'] = np.log(df_train['count_unique_word_desc'])

(sns
  .FacetGrid(df_train, 
             hue='deal_prob_cat', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'count_unique_word_desc_log', shade=True)
 .add_legend()
)
plt.show()
stopWords = set(stopwords.words('russian'))
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopWords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_train['title'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLE",fontsize=25)
plt.axis('off')
plt.show()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopWords,
                          max_words=1000,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_train['description'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)
plt.axis('off')
plt.show()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopWords,
                          max_words=1500,
                          max_font_size=250, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_train['param_1'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - PARAM_1",fontsize=25)
plt.axis('off')
plt.show()
title_freq = df_train.title.value_counts()[:35]

title_freq.index = ["Dress","Shoes","Jacket","Coat","Jeans","Overalls","Sneakers","Costume",
                    "Boots","Sandals","Skirt","Blouse","Windbreaker","I'll rent one apartment",
                    "Boots","Wedding Dress","Shirt","A bag","Stroller","Blouse","Sandals","Sofa",
                    "Pants","Cloak","Ankle Booties","A bike","The plot is 10 hundred. (IZhS)","Sneakers",
                    "Jacket demi-season","Hire a house","Selling dress","A jacket",
                    "I'll rent a 2-room apartment","T-shirt","Footwear"]
plt.figure(figsize=(16,6))

g = sns.barplot(title_freq.index, title_freq.values)
g.set_xlabel("TOP 35 Titles", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_title("TOP 35 titles frequency", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.show()
title_freq = df_train.title.value_counts()[:35]

plt.figure(figsize=(16,12))

plt.subplot(2,1,1)
g = sns.boxplot(x='title', y='price_log', 
                data=df_train[df_train.title.isin(title_freq.index.values)])
g.set_xlabel("", fontsize=15)
g.set_ylabel("Price Log", fontsize=15)
g.set_title("TOP 35 titles by Price_log", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='title', y='deal_probability', 
                data=df_train[df_train.title.isin(title_freq.index.values)])
g1.set_xlabel("TOP 35 Titles", fontsize=15)
g1.set_ylabel("Deal Probability", fontsize=15)
g1.set_title("TOP 35 titles by Deal Probability", fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()
cols = ['title','deal_prob_cat']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[df_train.title.isin(title_freq.index.values)][cols[0]], 
            df_train[df_train.title.isin(title_freq.index.values)][cols[1]]).style.background_gradient(cmap = cm)
import squarify 

plt.figure(figsize = (16,5)) 

plt.subplot(1,2,1)
grouped_prob_cat = np.log1p(df_train.groupby(['parent_category_name_en']).mean())
grouped_prob_cat['cat'] = grouped_prob_cat.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_prob_cat['deal_probability'], \
              label = grouped_prob_cat.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 12)
plt.axis('off')
plt.title("SQUARIFY OF CATEGORY AND DEAL PROB MEAN ")

plt.subplot(1,2,2)
grouped_prob_cat = np.log1p(df_train.groupby(['parent_category_name_en']).mean())
grouped_prob_cat['cat'] = grouped_prob_cat.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_prob_cat['price'], \
              label = grouped_prob_cat.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 12)
plt.axis('off')
plt.title("SQUARIFY OF CATEGORY AND PRICE MEAN ")
plt.show()
plt.figure(figsize = (16,5)) 

plt.subplot(1,2,1)
grouped_prob_cat = np.log1p(df_train.groupby(['user_type']).mean())
grouped_prob_cat['cat'] = grouped_prob_cat.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_prob_cat['deal_probability'].values, 
              label = grouped_prob_cat.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 12)
plt.axis('off')
plt.title("SQUARIFY OF USER TYPE filled by Deal probability")

plt.subplot(1,2,2)
grouped_prob_cat = np.log1p(df_train.groupby(['user_type']).sum())
grouped_prob_cat['cat'] = grouped_prob_cat.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_prob_cat['price'].values, 
              label = grouped_prob_cat.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 12)
plt.axis('off')
plt.title("SQUARIFY weighted by Mean Price")

plt.show()

