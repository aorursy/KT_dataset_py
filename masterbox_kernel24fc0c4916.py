import numpy as np

import pandas as pd

import re

from datetime import datetime, timedelta



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
RANDOM_SEED = 42
df_train = pd.read_csv('../input/sf-begin01/main_task.csv')

df_test = pd.read_csv('../input/sf-begin01/kaggle_task.csv')

sample_submission = pd.read_csv('../input/sf-begin01/sample_submission.csv')
df_train['sample'] = 1

df_test['sample'] = 0

df_test['Rating'] = 0

data = df_test.append(df_train, sort=False).reset_index(drop=True)
# Добавление признака столицы из добавленного датасета



df_capitals = pd.read_csv('../input/begin01/capitals.csv')

capitals = set(df_capitals['capital'])

data['Capital'] = data['City'].apply(lambda x: 1 if x in capitals else 0)
# Обработка дат в отзывах



def rev_dates(temp):

    max_date = datetime.strptime('01/01/0001', "%m/%d/%Y")

    i_dates = temp.split('], [')[1]

    result = re.findall('\d+/\d+/\d+', i_dates)

    for i in result:

        cur_date = datetime.strptime(i, '%m/%d/%Y')

        if cur_date > max_date:

            max_date = cur_date

    return max_date



def rev_delta(temp):

    delta = timedelta(0)

    count = 0

    i_dates = temp.split('], [')[1]

    result = re.findall('\d+/\d+/\d+', i_dates)

    for i in result:

        if count == 0:

            count = 1

            cur_date = datetime.strptime(i, '%m/%d/%Y')

        else:

            next_date = datetime.strptime(i, '%m/%d/%Y')

            cur_delta = abs(next_date - cur_date)

            if cur_delta > delta:

                delta = cur_delta

            cur_date = next_date

    return delta.days
# Форматирование пустых отзывов и вычисление дельты между отзывами



data['Reviews'].fillna(value='[[], []]', inplace=True)

data['Last Review'] = data['Reviews'].apply(rev_dates)

data['Delta Review'] = data['Reviews'].apply(rev_delta)
# Поисх самых встречающихся слов в отзывах



import collections

reviews = set()

words_col = collections.Counter()



for k in data['Reviews']:

    if k is not np.nan:

        i_rev = k.split('], [')[0]

        result = re.findall(r'[\'\"](.*)[\'\"]', i_rev)

        if result:

            for i in result:

                i_revs = i.split('\', \'')

                for j in i_revs:

                    reviews.add(j)

                    x_worlds = j.split(' ')

                    for x in x_worlds:

                        x = x.lower()

                        x = re.sub('[!.,]', '', x)

                        if len(x) > 3:

                            words_col[x] += 1



words = list(dict(words_col.most_common(1000)).keys())
# Добавление признака присутствия слова в отзывах



for i in words:

    data[i] = 0
# Заполнение признака присутствия слова в отзывах



for i in range(len(data)):

    dd = data.iloc[i]['Reviews']

    i_dd = dd.split('], [')[0]

    body = i_dd[2:]

    if len(body) > 0:

        for k in body.split(', '):

            review = k[1:-1]

            for j in review.split(' '):

                word = j.lower()

                if word in words:

                    data.at[i,word] = 1        
data['Number of Reviews'].fillna(0, inplace=True)
data = data.drop(['Reviews', 'Last Review'], axis = 1)
data.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)
# Фукнция оразмеривания признака Price



def prices(data):

    res = 0

    if data == '$':

        res = 1

    elif data == '$$$$':

        res = 4

    elif data == '$$ - $$$':

        res = 2.5

    return res
# Добавление числового значения признака Price



data['Prices'] = data['Price Range'].apply(prices)
data = data.drop(['Price Range'], axis = 1)



# Заполнение пустых значений Prices средним значением в городе



data2 = data[data['Prices'] > 0][['City','Prices']].groupby(['City']).mean()



for i in data[data['Prices'] == 0].index:

    city = data['City'].iloc[i]

    data.at[i,['Prices']] = round(float(data2.loc[city]),1)
# Добавление признака зависимости ранка от конкуренции в городе



cities = data['City'].value_counts()

data['competitors'] = data['City'].apply(lambda x: cities[x])

data['comp_rank'] = data['competitors'] - data['Ranking']

delta_comp_rank = abs(round(data['comp_rank'].min())) + 1

data['comp_rank'] = data['comp_rank'] + delta_comp_rank
# Признак количества кухонь



data['Cuisines'] = data['Cuisine Style'].apply(lambda x: len(str(x).split(',')))
# нормализация данных

data['Number of Reviews'] = (data['Number of Reviews'] - data['Number of Reviews'].mean()) / np.linalg.norm(data['Number of Reviews'] - data['Number of Reviews'].mean())

data['Delta Review'] = (data['Delta Review'] - data['Delta Review'].mean()) / np.linalg.norm(data['Delta Review'] - data['Delta Review'].mean())

data['Ranking'] = (data['Ranking'] - data['Ranking'].mean()) / np.linalg.norm(data['Ranking'] - data['Ranking'].mean())

data['Prices'] = (data['Prices'] - data['Prices'].mean()) / np.linalg.norm(data['Prices'] - data['Prices'].mean())

data['competitors'] = (data['competitors'] - data['competitors'].mean()) / np.linalg.norm(data['competitors'] - data['competitors'].mean())

data['comp_rank'] = (data['comp_rank'] - data['comp_rank'].mean()) / np.linalg.norm(data['comp_rank'] - data['comp_rank'].mean())
# Применяем метод главных компонент для показателей comp_rank и Ranking

# corr (comp_rank, Ranking) = -0.77619



c = np.array([[1,-0.77619],[-0.77619,1]])

np.linalg.eig(c)
# айгенпара 1.77619, (0.70710678, -0.70710678)

data['gl_rank'] = 0.70710678 * data['comp_rank'] + 0.70710678 * data['Ranking']
# Надо бы удалить пару признаков после нахождения нового признака методом главных компанент,

# но без них результаты хуже :-)



#data.drop(['comp_rank','Ranking','competitors'], axis = 1, inplace=True)
# Разложение признака города



data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
# Определение количества кухонь



cuisines = set()

for k in data['Cuisine Style']:

    if k is not np.nan:

        l = str(k).split(',')

        for i in l:

            cuisine = i.split('\'')[1]

            cuisines.add(cuisine)
# Добавление признака присутствия кухни



for i in cuisines:

    data[i] = 0
# Заполнение признака кухни



for i in range(len(data)):

    dd = data.iloc[i]['Cuisine Style']

    if dd is not np.nan:

        for k in list(dd.split(',')):

            j = k.split('\'')[1]

            data.at[i,j] = 1
data.drop(['URL_TA','Cuisine Style'], axis = 1, inplace=True)
from sklearn.model_selection import train_test_split

train_data = data.query('sample == 1').drop(['sample'], axis=1)

X = train_data.drop(['Rating'], axis=1)

y = train_data.Rating.values   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Функция округления целевых значений



def round_half(num):

    if num < 1.25:

        num = 1.0

    elif num < 1.75:

        num = 1.5

    elif num < 2.25:

        num = 2

    elif num < 2.75:

        num = 2.5

    elif num < 3.25:

        num = 3.0

    elif num < 3.75:

        num = 3.5

    elif num < 4.25:

        num = 4.0

    elif num < 4.75:

        num = 4.5

    else:

        num = 5.0

    return num



vfunc = np.vectorize(round_half)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = vfunc(y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = data.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
predict_submission = vfunc(predict_submission)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)
plt.rcParams['figure.figsize'] = (10,5)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')