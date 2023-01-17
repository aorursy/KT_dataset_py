import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from tqdm.notebook import tqdm

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder
print('Python       :', sys.version.split('\n')[0])

print('Numpy        :', np.__version__)
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
VERSION    = 15

DIR_TRAIN  = '../input/parsing-all-moscow-auto-ru-09-09-2020/' # подключил к ноутбуку внешний датасет

DIR_TEST   = '../input/sf-dst-car-price-prediction/'

VAL_SIZE   = 0.20   # 20%



# CATBOOST

ITERATIONS = 5000

LR         = 0.1
!ls '../input'
train = pd.read_csv(DIR_TRAIN+'all_auto_ru_09_09_2020.csv') # датасет для обучения модели

test = pd.read_csv(DIR_TEST+'test.csv')

sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
train.head(5)
train.info()
test.head(5)
test.info()
# ... 
train.dropna(subset=['productionDate','mileage'], inplace=True)

train.dropna(subset=['price'], inplace=True)
# для baseline просто возьму пару схожих признаков без полной обработки

df_train = train[['bodyType', 'brand', 'productionDate', 'engineDisplacement', 'mileage']]

y = train['price']



df_test = test[['bodyType', 'brand', 'productionDate', 'engineDisplacement', 'mileage']]
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
for colum in ['bodyType', 'brand', 'engineDisplacement']:

    data[colum] = data[colum].astype('category').cat.codes
data
X = data.query('sample == 1').drop(['sample'], axis=1)

X_sub = data.query('sample == 0').drop(['sample'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
model = CatBoostRegressor(iterations = ITERATIONS,

                          learning_rate = LR,

                          random_seed = RANDOM_SEED,

                          eval_metric='MAPE',

                          custom_metric=['R2', 'MAE']

                         )

model.fit(X_train, y_train,

         #cat_features=cat_features_ids,

         eval_set=(X_test, y_test),

         verbose_eval=100,

         use_best_model=True,

         plot=True

         )
model.save_model('catboost_single_model_baseline.model')
predict_submission = model.predict(X_sub)

predict_submission
sample_submission['price'] = predict_submission

sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)

sample_submission.head(10)