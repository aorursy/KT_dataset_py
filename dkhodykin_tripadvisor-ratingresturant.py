import pandas as pd
df = pd.read_csv('../input/main_task.csv')
# Cоздадим спсок всех городов

cities = []



for i in df['City']:

    if i not in cities:

        cities.append(i) #Всего будет добавлен список из 31 город = len(cities)



        

# Отметим, какие из них являются столицами

capitals = ['Paris','Stockholm','London','Berlin','Vienna','Rome','Madrid','Dublin','Brussels','Bratislava',

           'Warsaw','Budapest','Copenhagen','Amsterdam','Lisbon','Prague','Oslo','Helsinki','Ljubljana',

           'Athens','Luxembourg','Edinburgh']
# Функция для создания фиктивных переменных, отражающих "столичность" ресторана

def capital_resturant(city):

    if city in capitals:

        return 1

    else:

        return 0
# Добавим в датасет столбец с признаком, отражающим "столичность" ресторана

df['Capital'] = df.City.apply(capital_resturant)
#Добавим столбцы с принадлежностью ресторана к городу

df['Paris'] = df['City'].apply(lambda x: 1 if 'Paris' in str(x) else 0)

df['London'] = df['City'].apply(lambda x: 1 if 'London' in str(x) else 0)

df['Stockholm'] = df['City'].apply(lambda x: 1 if 'Stockholm' in str(x) else 0)

df['Berlin'] = df['City'].apply(lambda x: 1 if 'Berlin' in str(x) else 0)

df['Munich'] = df['City'].apply(lambda x: 1 if 'Munich' in str(x) else 0)

df['Oporto'] = df['City'].apply(lambda x: 1 if 'Oporto' in str(x) else 0)

df['Milan'] = df['City'].apply(lambda x: 1 if 'Milan' in str(x) else 0)

df['Bratislava'] = df['City'].apply(lambda x: 1 if 'Bratislava' in str(x) else 0)

df['Vienna'] = df['City'].apply(lambda x: 1 if 'Vienna' in str(x) else 0)

df['Rome'] = df['City'].apply(lambda x: 1 if 'Rome' in str(x) else 0)

df['Barcelona'] = df['City'].apply(lambda x: 1 if 'Barcelona' in str(x) else 0)
df['Madrid'] = df['City'].apply(lambda x: 1 if 'Madrid' in str(x) else 0)

df['Dublin'] = df['City'].apply(lambda x: 1 if 'Dublin' in str(x) else 0)

df['Brussels'] = df['City'].apply(lambda x: 1 if 'Brussels' in str(x) else 0)

df['Zurich'] = df['City'].apply(lambda x: 1 if 'Zurich' in str(x) else 0)

df['Warsaw'] = df['City'].apply(lambda x: 1 if 'Warsaw' in str(x) else 0)

df['Budapest'] = df['City'].apply(lambda x: 1 if 'Budapest' in str(x) else 0)

df['Copenhagen'] = df['City'].apply(lambda x: 1 if 'Copenhagen' in str(x) else 0)

df['Amsterdam'] = df['City'].apply(lambda x: 1 if 'Amsterdam' in str(x) else 0)

df['Lyon'] = df['City'].apply(lambda x: 1 if 'Lyon' in str(x) else 0)

df['Hamburg'] = df['City'].apply(lambda x: 1 if 'Hamburg' in str(x) else 0)

df['Lisbon'] = df['City'].apply(lambda x: 1 if 'Lisbon' in str(x) else 0)
df['Prague'] = df['City'].apply(lambda x: 1 if 'Prague' in str(x) else 0)

df['Oslo'] = df['City'].apply(lambda x: 1 if 'Oslo' in str(x) else 0)

df['Helsinki'] = df['City'].apply(lambda x: 1 if 'Helsinki' in str(x) else 0)

df['Edinburgh'] = df['City'].apply(lambda x: 1 if 'Edinburgh' in str(x) else 0)

df['Geneva'] = df['City'].apply(lambda x: 1 if 'Geneva' in str(x) else 0)

df['Ljubljana'] = df['City'].apply(lambda x: 1 if 'Ljubljana' in str(x) else 0)

df['Athens'] = df['City'].apply(lambda x: 1 if 'Athens' in str(x) else 0)

df['Luxembourg'] = df['City'].apply(lambda x: 1 if 'Luxembourg' in str(x) else 0)

df['Krakow'] = df['City'].apply(lambda x: 1 if 'Krakow' in str(x) else 0)
worst = ['poor', 'terrible', 'disappoint', 'horrible', 'overpriced', 'strange', 'refused', 'unappealing', 'worst', 'boring', 'racism',

         'bad', 'expensive','careful', 'appalling', ':(', 'amateur', 'dirty', 'rude', 'wretched', 'mean', 'amiss']
best = ['unique', 'great', 'good', 'heavenly', 'brilliant', 'amazing', 'tasty', 'delicious', 'healthy', 'perfect', 'nice', 

       'wonderful', 'delight', 'enjoy', 'awesome', 'super', 'charm', 'excellent', 'lovely', ':)', 'friendly', 'yum', 'beautiful',

       'fresh', 'fantastic', 'talent', 'grand', 'relax', 'best', 'modern', 'cosy', 'right']
# Функция для создания фиктивных переменных, отражающих тональность отзывов о ресторане

def reviews_quality(reviews):

    for i in worst:

        if i in str(reviews).lower():

            return -1

    for j in best:

        if j in str(reviews).lower():

            return 1

        else:

            return 0
# Создадим столбец, отражающий тональность отзывов

df['Reviews_quality'] = df.Reviews.apply(reviews_quality)
import re



# Очистим список кухонь от лишшних символов и преобразуем в список

df['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: re.sub("'", "", str(x)[1:-1]).split(', '))
# Создадим столбец, указывающий на максимальное количество типовов кухонь в ресторане

df['Cuisine Max'] = df['Cuisine Style'].apply(lambda x: len(x))
# Заполним пропуски в количествах отзывов - по среднему значению количества отзывов

df['Number of Reviews'] = df['Number of Reviews'].fillna(df['Number of Reviews'].mean())
# Если количество отзывов более Х, добавим признак "1"

df['Many_Rev'] = df['Number of Reviews'].apply(lambda x: 1 if x>300 else 0)
#Создадим столбцы, отражающие наличие в ресторане популярнейших кухонь и подберем те из них, которые в большей степени уменьшают MAE

df['Cuisine_Vegeterian'] = df['Cuisine Style'].apply(lambda x: 1 if 'Vegetarian' in str(x) else 0)

df['Cuisine_European'] = df['Cuisine Style'].apply(lambda x: 1 if 'European' in str(x) else 0)

# df['Cuisine_Mediterranean'] = df['Cuisine Style'].apply(lambda x: 1 if 'Mediterranean' in str(x) else 0)

# df['Cuisine_Italian'] = df['Cuisine Style'].apply(lambda x: 1 if 'Italian' in str(x) else 0)

df['Cuisine_Vegan'] = df['Cuisine Style'].apply(lambda x: 1 if 'Vegan' in str(x) else 0)

df['Cuisine_Gluten'] = df['Cuisine Style'].apply(lambda x: 1 if 'Gluten' in str(x) else 0)

# df['Cuisine_French'] = df['Cuisine Style'].apply(lambda x: 1 if 'French' in str(x) else 0)

# df['Cuisine_Asian'] = df['Cuisine Style'].apply(lambda x: 1 if 'Asian' in str(x) else 0)
# Функция для очистки "мусора" в столбце с датами отзывов

from datetime import datetime



def time_clean(list_oftime):

    try:

        for i in list_oftime:

            return datetime.strptime(list_oftime[0], '%m/%d/%Y')

    except ValueError:

        ''
# Очистим отзывы от лишних символов для извлечения информации о датах отзывов

list_date_last = df['Reviews'].apply(lambda x: x[-27:-17].split(', ') if len(x)>9 else "")

list_date_prev = df['Reviews'].apply(lambda x: x[-13:-3].split(', ') if len(x)>9 else "")
# Создадим итерабельные столбцы с датами

df['Rewiews_date_prev'] = list_date_prev.apply(time_clean)

df['Rewiews_date_last'] = list_date_last.apply(time_clean)
# Данных в столбце с датами предыдущих отзывов - больше, т.к. видимо по некоторым ресторанам был всего один отзыв, заполним пропуски

df['Rewiews_date_last'] = df['Rewiews_date_last'].fillna(value=df['Rewiews_date_prev'])
# Определим дату самого "свежего отзыва"

df['Rewiews_date_last'].max()
from datetime import datetime



# Добавим столбец, отражающий "свежесть" отзывов

df['Fresh_Rew'] = (datetime.today() - df['Rewiews_date_last']).dt.days

df['Fresh_Rew'] = df['Fresh_Rew'].fillna(df['Fresh_Rew'].mean())

df['Fresh_Rew'] = df['Fresh_Rew'].apply(lambda x: 1 if x < 800 else 0)
kaggle = pd.read_csv('../input/kaggle_task.csv')

kaggle = kaggle.drop(['Name'], axis=1)

kaggle.tail(3)
# Добавим в датасет столбец с признаком, отражающим "столичность" ресторана

kaggle['Capital'] = df.City.apply(capital_resturant)
#Добавим столбцы с принадлежностью ресторана к городу

kaggle['Paris'] = kaggle['City'].apply(lambda x: 1 if 'Paris' in str(x) else 0)

kaggle['London'] = kaggle['City'].apply(lambda x: 1 if 'London' in str(x) else 0)

kaggle['Stockholm'] = kaggle['City'].apply(lambda x: 1 if 'Stockholm' in str(x) else 0)

kaggle['Berlin'] = kaggle['City'].apply(lambda x: 1 if 'Berlin' in str(x) else 0)

kaggle['Munich'] = kaggle['City'].apply(lambda x: 1 if 'Munich' in str(x) else 0)

kaggle['Oporto'] = kaggle['City'].apply(lambda x: 1 if 'Oporto' in str(x) else 0)

kaggle['Milan'] = kaggle['City'].apply(lambda x: 1 if 'Milan' in str(x) else 0)

kaggle['Bratislava'] = kaggle['City'].apply(lambda x: 1 if 'Bratislava' in str(x) else 0)

kaggle['Vienna'] = kaggle['City'].apply(lambda x: 1 if 'Vienna' in str(x) else 0)

kaggle['Rome'] = kaggle['City'].apply(lambda x: 1 if 'Rome' in str(x) else 0)

kaggle['Barcelona'] = kaggle['City'].apply(lambda x: 1 if 'Barcelona' in str(x) else 0)
kaggle['Madrid'] = kaggle['City'].apply(lambda x: 1 if 'Madrid' in str(x) else 0)

kaggle['Dublin'] = kaggle['City'].apply(lambda x: 1 if 'Dublin' in str(x) else 0)

kaggle['Brussels'] = kaggle['City'].apply(lambda x: 1 if 'Brussels' in str(x) else 0)

kaggle['Zurich'] = kaggle['City'].apply(lambda x: 1 if 'Zurich' in str(x) else 0)

kaggle['Warsaw'] = kaggle['City'].apply(lambda x: 1 if 'Warsaw' in str(x) else 0)

kaggle['Budapest'] = kaggle['City'].apply(lambda x: 1 if 'Budapest' in str(x) else 0)

kaggle['Copenhagen'] = kaggle['City'].apply(lambda x: 1 if 'Copenhagen' in str(x) else 0)

kaggle['Amsterdam'] = kaggle['City'].apply(lambda x: 1 if 'Amsterdam' in str(x) else 0)

kaggle['Lyon'] = kaggle['City'].apply(lambda x: 1 if 'Lyon' in str(x) else 0)

kaggle['Hamburg'] = kaggle['City'].apply(lambda x: 1 if 'Hamburg' in str(x) else 0)

kaggle['Lisbon'] = kaggle['City'].apply(lambda x: 1 if 'Lisbon' in str(x) else 0)
kaggle['Prague'] = kaggle['City'].apply(lambda x: 1 if 'Prague' in str(x) else 0)

kaggle['Oslo'] = kaggle['City'].apply(lambda x: 1 if 'Oslo' in str(x) else 0)

kaggle['Helsinki'] = kaggle['City'].apply(lambda x: 1 if 'Helsinki' in str(x) else 0)

kaggle['Edinburgh'] = kaggle['City'].apply(lambda x: 1 if 'Edinburgh' in str(x) else 0)

kaggle['Geneva'] = kaggle['City'].apply(lambda x: 1 if 'Geneva' in str(x) else 0)

kaggle['Ljubljana'] = kaggle['City'].apply(lambda x: 1 if 'Ljubljana' in str(x) else 0)

kaggle['Athens'] = kaggle['City'].apply(lambda x: 1 if 'Athens' in str(x) else 0)

kaggle['Luxembourg'] = kaggle['City'].apply(lambda x: 1 if 'Luxembourg' in str(x) else 0)

kaggle['Krakow'] = kaggle['City'].apply(lambda x: 1 if 'Krakow' in str(x) else 0)
# Создадим столбец, отражающий тональность отзывов

kaggle['Reviews_quality'] = kaggle.Reviews.apply(reviews_quality)
# Очистим список кухонь от лишшних символов и преобразуем в список

kaggle['Cuisine Style'] = kaggle['Cuisine Style'].apply(lambda x: re.sub("'", "", str(x)[1:-1]).split(', '))
# Создадим столбец, указывающий на максимальное количество типовов кухонь в ресторане

kaggle['Cuisine Max'] = kaggle['Cuisine Style'].apply(lambda x: len(x))
# Заполним пропуски в количествах отзывов

kaggle['Number of Reviews'] = kaggle['Number of Reviews'].fillna(kaggle['Number of Reviews'].mean())
# Если количество отзывов более Х, добавим признак "1"

kaggle['Many_Rev'] = kaggle['Number of Reviews'].apply(lambda x: 1 if x>300 else 0)
#Создадим столбцы, отражающие наличие в ресторане популярнейших кухонь и подберем те из них, которые в большей степени уменьшают MAE

kaggle['Cuisine_Vegeterian'] = kaggle['Cuisine Style'].apply(lambda x: 1 if 'Vegetarian' in str(x) else 0)

kaggle['Cuisine_European'] = kaggle['Cuisine Style'].apply(lambda x: 1 if 'European' in str(x) else 0)

kaggle['Cuisine_Vegan'] = kaggle['Cuisine Style'].apply(lambda x: 1 if 'Vegan' in str(x) else 0)

kaggle['Cuisine_Gluten'] = kaggle['Cuisine Style'].apply(lambda x: 1 if 'Gluten' in str(x) else 0)
# Очистим отзывы от лишних символов для извлечения информации о датах отзывов

list_date_last_k = kaggle['Reviews'].apply(lambda x: x[-27:-17].split(', ') if len(str(x))>9 else "")

list_date_prev_k = kaggle['Reviews'].apply(lambda x: x[-13:-3].split(', ') if len(str(x))>9 else "")
# Создадим итерабельные столбцы с датами

kaggle['Rewiews_date_prev'] = list_date_prev_k.apply(time_clean)

kaggle['Rewiews_date_last'] = list_date_last_k.apply(time_clean)
# Данных в столбце с датами предыдущих отзывов - больше, т.к. видимо по некоторым ресторанам был всего один отзыв, заполним пропуски

kaggle['Rewiews_date_last'] = kaggle['Rewiews_date_last'].fillna(value=kaggle['Rewiews_date_prev'])
# Определим дату самого "свежего отзыва"

kaggle['Rewiews_date_last'].max()
# Добавим столбец, отражающий "свежесть" отзывов

kaggle['Fresh_Rew'] = (datetime.today() - kaggle['Rewiews_date_last']).dt.days

kaggle['Fresh_Rew'] = kaggle['Fresh_Rew'].fillna(kaggle['Fresh_Rew'].mean())

kaggle['Fresh_Rew'] = kaggle['Fresh_Rew'].apply(lambda x: 1 if x < 800 else 0)
# Сверим совпадение количества столбцов датасетов

if len(df.T)==len(kaggle.T):

    print('Количество столбцов датасетов совпадает')
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

X_train = df.drop(['Restaurant_id', 

             'City', 

             'Cuisine Style', 

             'Rating',

             'Rewiews_date_prev',

             'Price Range', 

             'Reviews',

             'URL_TA',

             'Rewiews_date_last',

             'ID_TA'], axis = 1)

y_train = df['Rating']
X_test = kaggle.drop(['Restaurant_id', 

             'City', 

             'Cuisine Style', 

             'Rewiews_date_prev',

             'Price Range', 

             'Reviews',

             'URL_TA',

             'Rewiews_date_last',

             'ID_TA'], axis = 1)
# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
# Внесем данные с предсказанием рейтинга в целевой датасет

kaggle['Rating'] = y_pred
# Сохраним результаты

result = kaggle[['Restaurant_id', 'Rating']]

result.to_csv('data.csv', index = False)