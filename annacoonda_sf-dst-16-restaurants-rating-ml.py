import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Зафиксируем RANDOM_SEED для воспроизводимости экспериментов
RANDOM_SEED = 42
#Настройки представления
pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.nunique(dropna=False)
data['Restaurant_id'].value_counts()
data[data['Restaurant_id']=='id_436']
data['City'].value_counts()
cities=set(list(data['City']))
len(cities)
#Создаём словарь, где ключами являются наименования городов
cit_cap={'London': 'Capital', 'Paris': 'Capital', 'Madrid': 'Capital','Barcelona': 'Town', 'Berlin': 'Capital', 
 'Milan': 'Town', 'Rome': 'Capital','Prague': 'Capital','Lisbon': 'Capital', 'Vienna': 'Capital', 
 'Amsterdam': 'Capital', 'Brussels': 'Capital', 'Hamburg': 'Town', 'Munich': 'Town', 'Lyon': 'Town', 
 'Stockholm': 'Capital', 'Budapest': 'Capital', 'Warsaw': 'Capital', 'Dublin': 'Town', 'Copenhagen': 'Capital', 
 'Athens': 'Capital', 'Edinburgh': 'Town', 'Zurich': 'Town', 'Oporto': 'Town', 'Geneva': 'Capital',  'Krakow': 'Town',
 'Oslo': 'Capital', 'Helsinki': 'Capital', 'Bratislava': 'Capital',  'Luxembourg': 'Capital', 'Ljubljana': 'Town'}
len(cit_cap)
#Создаём новый столбец, значения которого являются значениями созданного словаря по ключам из колонки City
data['Capital']=data['City'].map(cit_cap)
data['Capital'].value_counts()
cit_cap_0 = {'Capital':1, 'Town':0}
data['Capital']=data['Capital'].map(cit_cap_0)
data.Capital.value_counts()
DATA_DIR1 = '/kaggle/input/city-destinations/'
df_c = pd.read_csv(DATA_DIR1+'/Tourism.csv')
df_c.head()
data = pd.merge(data, df_c, on='City')
data.info()
data['Cuisine Style'][0]
type(data['Cuisine Style'][0])
data['Cuisine Style'].fillna("[]",inplace=True)
#Функция, автоматически оценивающая строку и преобразующая её в предполагаемый формат
data['Cuisine Style']=data['Cuisine Style'].apply(lambda x: ast.literal_eval(x))
data['Cuisines'] = data['Cuisine Style'].apply(lambda x: len(x))
data['Cuisines'].value_counts()
cus_set=set() #Создаём пустой сет
for c in data['Cuisine Style']:
    for i in c: #Проходим по элементам списка в каждой записи
        cus_set.add(i) #Добавляем элемент в сет, повторяющиеся значения "схлопнутся"
len(cus_set)
print(cus_set)
asian = ['Yunnan',  'Vietnamese',  'Japanese',  'Tibetan',  'Bangladeshi',  'Asian',  'Thai',  'Malaysian','Xinjiang',
          'Burmese',  'Pakistani',  'Taiwanese',  'Nepali',  'Korean',  'Mongolian',  'Southwestern', 'Chinese', 
          'Indian',  'Sri Lankan',   'Central Asian',   'Filipino','Sushi',  'Singaporean','Indonesian',  
          'Minority Chinese',  'Fujian',  'Cambodian',  'Asian']
latin = ['Argentinean',  'Latin',  'South American',  'Venezuelan',  'Mexican',  'Central American',  'Ecuadorean', 
          'Cuban', 'Brazilian',  'Colombian',  'Salvadoran',  'Chilean',  'Peruvian']
inter = ['African',  'Fusion',  'Caribbean',  'Caucasian',   'International',  'New Zealand',  'Hungarian',  
          'Russian',  'Georgian',   'Polynesian',  'Cajun & Creole',   'American',  'Azerbaijani',  'Israeli', 
          'Afghani',  'Caribbean',   'Armenian',   'Jamaican',   'Hawaiian',   'Uzbek',   'Ethiopian',  'Albanian',  
          'Native American',  'Australian',   'Canadian', 'International']
eur = ['French',  'Latvian',  'Swiss', 'Polish',  'Scottish',  'Central European', 'Scandinavian',  'German', 
        'Danish', 'Belgian', 'Austrian', 'Ukrainian',  'Norwegian',  'Czech',  'Dutch', 'British',  'Croatian',  
        'Swedish',  'Slovenian',  'Irish',  'Eastern European',  'Romanian',   'Welsh', 'European']
arab = ['Arabic',  'Middle Eastern',  'Egyptian',  'Tunisian',  'Lebanese',   'Moroccan']
medter = ['Persian',  'Greek',  'Spanish',  'Portuguese',   'Italian',   'Turkish',  'Mediterranean']

# Добавляем новые столбцы
data['Arab'] = 0
data['Asian'] = 0 
data['Europe']=0 
data['Inter']= 0 
data['Latin']=0 
data['Mediter']=0
data['Vegetarian']=0
data['Bar']=0
data.info()
v='Vegetarian Friendly' #Строковые переменные
vg='Vegan Options'
b='Bar'
wb='Wine Bar'
for i in range(50000):
    s=set(data['Cuisine Style'][i]) #Переменная, создающая сет из списка стилей кухни
    data['Arab'][i]=len(list(s&set(arab)))
    data['Asian'][i]=len(list(s&set(asian)))
    data['Europe'][i]=len(list(s&set(eur)))
    data['Inter'][i]=len(list(s&set(inter)))
    data['Mediter'][i]=len(list(s&set(medter)))
    data['Latin'][i]=len(list(s&set(latin)))
    if v in data['Cuisine Style'][i] or vg in data['Cuisine Style'][i]:
        data['Vegetarian'][i]=1
    elif b in data['Cuisine Style'][i] or wb in data['Cuisine Style'][i]:
        data['Bar'][i]=1

#Посмотрим, что получилось на примере столбца с азиатской кухней
data['Asian'].value_counts()
data.Ranking.describe()
data.Rating.value_counts()
data['Price Range'].value_counts()
dct_pr = {} #Пустой словарь для ценовых категорий
dct_pr['$']=1
dct_pr['$$ - $$$']=2
dct_pr['$$$$']=3
dct_pr
# Заменим значения по словарю и посмотрим, что получилось.
data['Price Range']=data['Price Range'].map(dct_pr)
data['Price Range'].value_counts()
data['Price Range'].fillna(0, inplace=True) #Записываем в пропуски нули для условия
for i in range(50000):
    if data['Price Range'][i]==0:
        x=round(data[data['City']==data['City'][i]]['Price Range'].mean()) #Среднее округлённое значение
        data['Price Range'][i]=x
data['Price Range'].value_counts()
data['Number of Reviews'].value_counts()
data['Reviews'].value_counts()
type(data['Reviews'][0])
data[data['Reviews']=="[['Good service and clean', 'Chinese fusion cuisine - ok to try if you...'], ['11/21/2017', '05/23/2017']]"]
data['Reviews'].fillna('[[], []]', inplace=True)
data[data['Reviews']== '[[], []]']['Number of Reviews'].fillna(0, inplace=True)
data['Number of Reviews'].fillna(1, inplace=True)
data['Number of Reviews city']=0
for i in range(len(data['Reviews'])):
    data['Number of Reviews city'][i]=data[data['City']==data['City'][i]]['Number of Reviews'].sum()
data['Number of Reviews'][144]
for i in range(50000):
    if data['Reviews'][i].find('nan]')>0 or data['Reviews'][i].find('nan,')>0:
        data['Reviews'][i]=data['Reviews'][i].replace('nan',"'no review'")
        print(data['Reviews'][i])
for i in range(50000):
    data['Reviews'][i]=ast.literal_eval(data['Reviews'][i])

data['Review_text']=data['Reviews'].apply(lambda x: x[0])
type(data['Review_text'][0])
type(data['Review_text'][1][0])
# Вставим столбец Review_words со значением по умолчанию 0.
data['Review_words']=0
i = -1 #Счётчик
words = [] #Пока пустое множество слов
for reviews in data['Review_text']:
    i += 1
    a = 0 # Обнуляем количество слов в первом и втором элементе списка
    b = 0
    if len(reviews)==2:
        # Удаляем все небуквенные символы
        word1 = re.sub(r'\W', ' ', reviews[0])
        word2 = re.sub(r'\W', ' ', reviews[1])
        # Удаляем все одиночные буквы (могли остаться от удаления апострофа)
        word1 = re.sub(r'\s+[a-zA-Z]\s+', ' ', word1)
        word2 = re.sub(r'\s+[a-zA-Z]\s+', ' ', word2)
        # Удаляем одиночные символы с начала
        word1 = re.sub(r'\^[a-zA-Z]\s+', ' ', word1)
        word2 = re.sub(r'\^[a-zA-Z]\s+', ' ', word2)
        # Заменяем множественные пробелы одним
        word1 = re.sub(r'\s+', ' ', word1, flags=re.I)
        word2 = re.sub(r'\s+', ' ', word2, flags=re.I)
        # Удаляем префикс 'b'
        word1 = re.sub(r'^b\s+', '', word1)
        word2 = re.sub(r'^b\s+', '', word2)
        # Переводим все слова в строчной регистр
        word1 = word1.lower()
        word2 = word2.lower()
        # Считаем количество слов по принципу количество пробелов +1
        a = word1.count(' ') + 1
        b = word2.count(' ') + 1
        data['Review_words'][i] = a + b 
        word = word1 + ' ' + word2
        word_list = list(map(str, word.split()))
        words.extend(word_list)
    elif len(reviews)==1:
        word1 = re.sub(r'\W', ' ', reviews[0])
        word1 = re.sub(r'\s+[a-zA-Z]\s+', ' ', word1)
        word1 = re.sub(r'\^[a-zA-Z]\s+', ' ', word1)
        word1 = re.sub(r'\s+', ' ', word1, flags=re.I)
        word1 = re.sub(r'^b\s+', '', word1)
        word1 = word1.lower()
        a = word1.count(' ') + 1
        data['Review_words'][i] = a
        word_list = list(map(str, word1.split()))
        words.extend(word_list)

len(words)
data['Review_words'].value_counts()
words = set(words)
print(len(words), words)
#Список негативных слов
minus = ['absurdly', 'abundance', 'abundant', 'abusive', 'aggresive', 'aggressive', 'agressive', 'angry', 'annoyed', 'annoying', 'annoyingly', 'anxiety', 'argumentive', 'arrogance', 'arrogant', 'ashame', 'ashamed','awefull', 'awful','awfull','awfully', 'bad','badly','betrug', 'boring', 'cancel','canceled','cancellation', 'cancelled','catastrophe', 'catastrophic', 'catastrophy','chaos', 'chaot', 'chaotic','chaotically','closed', 'closes','collapse','conflicting', 'conflictual','confused','confusing', 'confusio','confusion','danger', 'dangerous','dangerously','delay', 'delayed','depressing','desappointing', 'dirtiest','dirty','disapointing', 'disapointment','disapoointing','disappear', 'disappearing','disappo','disappoi', 'disappoin','disappoing','disappoinment', 'disappoint','disappointe','disappointed', 'disappointin','disappointing', 'disappointingly','disappointme','disappointmen','disappointment','disappointments','disappoints','disapponiting','disapponted','disaster','disasterous','disastrous','disguise','disgusted','disgusti','disgusting','disgustingly','dispointed','disppointing','dissapoint','dissapointed','dissapointing','dissapointment','dissapoints','expencieve','expencive','expenive','expensiv','expensive','exspencive','flavorless','horribly','hysterically','impolite','inadequate','inappropriate','inattentive','inauthentic','incompetent','incomplete','incomprehensible','inconsiderate','inconsistent','inconspicuous','incontournable','inconvenient','inedible','irritated','miserable','mistake','mistakes','overcharge','overcharged','overcharging','overcrowded','overestimated','poisening','poison','poisoned','poisoning','poisson','rubbish','rubish','rud','rude','rudely','rudeness','sexist','sick','spoil','spoiled','spoils','spoilt','stole','stolen','tasteless','tastless','theft','thief','thiefs','thieves','trash','trashy','ugliest','ugly','unacceptable','unacceptably','unaccomodating','unappealing','unappetising','unattractive','unclean','uncomfortable','unconvincing','unfirendly','unforgiving','unfortunate','unfortunately','unfortunetly','unfriendliest','unfriendly','unhappy','unhealthy','unhelpful','unhygenic','unhygienic','unimpressible','unimpressive','unintentionally','unintereste','uninterested','uninteresting','unlucky','unorganized','unpleasant','unpleasent','unpolite','unprofessi','unprofessional','unprofessionnal','unreasonable','unreasonably','unreliable','unsanitary','unsatisfactory','unsatisfied','unsatisfying','unstable','untasetiness','untasty','untrained','untrendy','unwanted','unwelcome','unwelcoming','urin','urinals','vandalism','waste','wasted','wasting']
# Список позитивных слов
plus = ['delicieux','délicieux','delicious','deliciouse','deliciously','deliciuos','delicius','delicous','delighful','delight','delighted','delightful','delightfully','delighthful','delights','excelent','excelente','excell','excellant','excelle','excellect','excellen','excellence','excellenct','excellency','excellent','excellente','excellento','excellet','excellient','excit','excite','excited','excitement','exciting','excitingly','excllent','execellent','execerlent','execllent','exelent','exellence','exellent','extroardinary','exzellent','faboulous','fabukous','fabul','fabulius','fabulo','fabulos','fabulous','fabulously','fabuloussss','favorites','favour','favourite','favourites','favours','favurite','flavorful','flavorfull','friendl','friendlier','friendliest','friendliness','friendlly','friendly','friéndly','friends','friendy','good','goodfood','goood','gooood','goooood','gooooood','horrific','hospitality','hospitallity','hospitalty','impressing','impressions','impressive','impressively','accceptable',' advantage',' alright',' alright',' amazed',' amazi',' amazin',' amazinf',' amazing','amazinggggg','amazingly','amazingness','amazlingly','amazng','amazzzzzing','ammmmazzzzingggg','attractive','autenthic','autentic','autenticity','auwsome','awesom','awesome','awesomer','awesone','awessome','awseome','awsom','awsome','beatiful','beaultiful','beaut','beautful','beauti','beautifu','beautiful','beautifull','beautifully','beauty','beaux','bellisima','bellisimo','bellisomo','bellissimo','bello','best','bestt','besttt','blessed','blessing','bonito','bonne','bonnes','breathtaking','briliant','brill','brillante','brilliance','brilliant','brilliantly','buon','buona','buonísimo','buonissimo','buono','charm','charmed','charmig','charmin','charming','charmy','classic','classical','classically','classy','comfort','comforta','comfortable','comfortably','compliment','compliments','congrat','congratulations','convenient','conveniently','delcious','delecious','delicetessen','délicieuse','incredible','incredibly','incredicle','inviting','irresistible','loveley','lovelier','loveliest','lovely','ĺovely','luxe','luxurious','luxuruious','mmmhhhhh','mmmm','mmmmm','mmmmmm','mmmmmmhh','mmmmmmmm','mmmmmmmmmmmmmmm','outsanding','outstandin','outstanding','outstandingly','outstanging','perfect','perfecto','perfecton','phantastic','perfect','perfecto','perfecton','phantastic','phenomenal','pleasant','pleasantl','pleasantly','pleasat','pleased','pleasent','pleasently','pleaser','pleaseure','pleasing','pleasur','pleasurable','pleasurably','pleasure','pleasures','plesant','pleseant','reccomendation','recomend','recomended','recommendable','sensation','sensational','sensationnel','spectacular','spectacularly','splendid','splendiferous','splendour','super','sympatic','tasteful','tastefull','tastefully','tasty','tastyy','tastyyy','tastyyyyy','thank','thankf','thankful','thankfully','thanks','tidy','unbeatable','underestimated','underrated','unforgetable','unforgettable','unglaublich','unmemorable','wondeful','wondefully','wonder','wonderdful','wonderfu','wonderful','wonderfull','wonderland','wondersful','wonderul','wonderwull','wondrful','wow','wowsers','woww','wowwwwww','wunderbar','wunderfull','wunderschön','yam','yami','yammi','yammy','yeah','yeap','yes','yess','yum','yumm','yummi','yummie','yummiest','yummii','yumminess','yummm','yummmm','yummmmm','yummmmmm','yummmmmmmmmmmmmmmmmmmmmmmmmmmmm','yummmmmy','yummy','yummyy','yummyyyyy','yums','yumy','yuuuum']

data['Reviews_eval'] = 1
for i in range(len(data['City'])):
    if len(data['Review_text'][i])==1:
        for m in minus:
            if m in data['Review_text'][i][0]:
                data['Reviews_eval'][i] = 0
        for p in plus:
            if p in data['Review_text'][i][0]:
                data['Reviews_eval'][i] = 2
    elif len(data['Review_text'][i])==2:
        a = 1
        b = 1
        for m in minus:
            if m in data['Review_text'][i][0]:
                a = 0
            if m in data['Review_text'][i][1]:
                b = 0
        for p in plus:
            if p in data['Review_text'][i][0]:
                a = 2
            if p in data['Review_text'][i][1]:
                b = 2
        data['Reviews_eval'][i] = round((a+b)/2)
data['Reviews_eval'].value_counts()
data['Review_Date']=data['Reviews'].apply(lambda x: x[1])
data['Latest Review']=5500
date = datetime.now() #Сегодняшняя дата
i=-1
for ddd in data['Review_Date']: #Список дат в строковом виде 
    i+=1
    for dd in ddd: #Элемент списка
        if len(ddd)==1:
            d=datetime.strptime(dd, '%m/%d/%Y')
            z = date-d      #Разница с единственной датой
            data['Latest Review'][i]=int(z.days) 
        elif len(ddd)==2:
            x=datetime.strptime(ddd[0], '%m/%d/%Y')
            y=datetime.strptime(ddd[1], '%m/%d/%Y')
            z=date-max(x,y) #Разница с позднейшей из двух дат
            data['Latest Review'][i]=int(z.days)
data['Latest Review'].value_counts()
data['URL_TA'].head(10)
data['ID_TA'].head(10)
data.info()
data.drop(data.columns[[2,6,7,8,23,26]], axis=1, inplace=True)
data.info()
#Распределение значений по Ranking
plt.rcParams['figure.figsize'] = (10,7)
data['Ranking'].plot(kind = 'hist', grid = True, title = 'Ranking')
#Распределение значений по Rating
plt.rcParams['figure.figsize'] = (10,7)
data['Rating'].plot(kind='hist', grid=True, title='Rating')
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(data.drop(['sample'], axis=1).corr(), xticklabels=data.drop(['sample'], axis=1).corr().columns, yticklabels=data.drop(['sample'], axis=1).corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Тепловая матрица корреляций для проекта Рестораны', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='Rating', 
                data=data.loc[data.loc[:, column].isin(data.loc[:, column].value_counts().index[:10])],
               ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()
data = pd.get_dummies(data, columns=['City'], dummy_na=True)
data=data.drop(['Restaurant_id','City_nan'], axis = 1)
# Выделим тестовую часть
train_data = data.query('sample == 1').drop(['sample'], axis=1)
test_data = data.query('sample == 0').drop(['sample'], axis=1)
y = train_data.Rating.values            # наш таргет
X = train_data.drop(['Rating'], axis=1)
# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
