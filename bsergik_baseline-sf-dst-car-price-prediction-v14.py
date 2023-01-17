import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from tqdm.notebook import tqdm

from catboost import CatBoostRegressor
print('Python       :', sys.version.split('\n')[0])

print('Numpy        :', np.__version__)
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
VERSION    = 11

DIR_TRAIN  = '../input/autoru-parsed-0603-1304/' # подключил к ноутбуку свой внешний датасет

DIR_TEST   = '../input/sf-dst-car-price/'

VAL_SIZE   = 0.33   # 33%

N_FOLDS    = 5



# CATBOOST

ITERATIONS = 2000

LR         = 0.1
!ls ../input/autoru-parsed-0603-1304/
train = pd.read_csv(DIR_TRAIN+'new_data_99_06_03_13_04.csv')

test = pd.read_csv(DIR_TEST+'test.csv')

sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
def vehicleConfiguration(row):

    for w in ['MECHANICAL', 'AUTOMATIC', 'ROBOT', 'VARIATOR', 0]:

        for s in str(row).split():

            if w == s:

                return w

    return 0



def color(row):

    if row == 'CACECB': return 'серебристый'

    elif row == 'FAFBFB': return 'белый'

    elif row == 'EE1D19': return 'красный'

    elif row == '97948F': return 'серый'

    elif row == '660099': return 'пурпурный'

    elif row == '040001': return 'чёрный'

    elif row == '4A2197': return 'фиолетовый'

    elif row == '200204': return 'коричневый'

    elif row == '0000CC': return 'синий'

    elif row == '007F00': return 'зелёный'

    elif row == 'C49648': return 'бежевый'

    elif row == '22A0F8': return 'голубой'

    elif row == 'DEA522': return 'золотистый'

    elif row == 'FFD600': return 'жёлтый'

    elif row == 'FF8649': return 'оранжевый'

    elif row == 'FFC0CB': return 'розовый'

    return row



def pts(row):

    if row == 'ORIGINAL': return 'Оригинал'

    elif row == 'DUPLICATE': return 'Дубликат'

    return row



def rul(row):

    if row == 'LEFT': return 'Левый'

    elif row == 'RIGHT': return 'Правый'

    return row



def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### Предобработка ############################################################## 

    # убираем ненужные для модели признаки

    df_output['Владение'] = df_output['Владение'].fillna('nodata')

    df_output['Владельцы'] = df_output['Владельцы'].apply(lambda x: float(str(x).split()[0]))

    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: float(str(x).split()[0]))

    df_output['ПТС'].fillna(0, inplace=True)

    df_output['Руль'].fillna('Левый', inplace=True)



    df_output['Руль'] = df_output['Руль'].apply(rul).fillna('Левый')

    df_output['ПТС'] = df_output['ПТС'].apply(pts).fillna('Оригинал')

    df_output['color'] = df_output['color'].apply(color)

    df_output['vehicleConfiguration'] = df_output['vehicleConfiguration'].apply(vehicleConfiguration)



    df_output['description'] = df_output['description'].fillna('[]')

    df_output['description_len'] = df_output['description'].apply(lambda x: len(x.split()))

    df_output['description_word'] = df_output['description'].apply(lambda x: [str(i).lower() for i in x.split()])



    df_output['leather']= df_output['description_word'].apply(lambda x: 1 if ('темный' and 'салон') in x else 0)

    df_output['carter']= df_output['description_word'].apply(lambda x: 1 if ('защита' and 'картера') in x else 0)

    df_output['ABS']= df_output['description_word'].apply(lambda x: 1 if ('антиблокировочная' and 'система') in x else 0)

    df_output['airbags']= df_output['description_word'].apply(lambda x: 1 if ('подушки' and 'безопасности') in x else 0)

    df_output['immob']= df_output['description_word'].apply(lambda x: 1 if ('иммобилайзер') in x else 0)

    df_output['central_locking']= df_output['description_word'].apply(lambda x: 1 if ('центральный' and 'замок') in x else 0)

    df_output['on_board_computer']= df_output['description_word'].apply(lambda x: 1 if ('бортовой' and 'компьютер') in x else 0)

    df_output['cruise_control']= df_output['description_word'].apply(lambda x: 1 if ('круиз-контроль') in x else 0)

    df_output['climat_control']= df_output['description_word'].apply(lambda x: 1 if ('климат-контроль') in x else 0)

    df_output['multi_rudder']= df_output['description_word'].apply(lambda x: 1 if ('мультифункциональный' and 'руль') in x else 0)

    df_output['power_steering']= df_output['description_word'].apply(lambda x: 1 if ('гидроусилитель' or 'гидро' or 'усилитель' and 'руля') in x else 0)

    df_output['light_and_rain_sensors']= df_output['description_word'].apply(lambda x: 1 if ('датчики' and 'света' and 'дождя') in x else 0)

    df_output['сarbon_body_kits']= df_output['description_word'].apply(lambda x: 1 if ('карбоновые' and 'обвесы') in x else 0)

    df_output['rear_diffuser_rkp']= df_output['description_word'].apply(lambda x: 1 if ('задний' and 'диффузор') in x else 0)

    df_output['door_closers']= df_output['description_word'].apply(lambda x: 1 if ('доводчики' and 'дверей') in x else 0)

    df_output['rear_view_camera']= df_output['description_word'].apply(lambda x: 1 if ('камера' or 'видеокамера' and 'заднего' and 'вида') in x else 0)

    df_output['amg']= df_output['description_word'].apply(lambda x: 1 if ('amg') in x else 0)

    df_output['bi_xenon_headlights']= df_output['description_word'].apply(lambda x: 1 if ('биксеноновые' and 'фары') in x else 0)

    df_output['from_salon']= df_output['description_word'].apply(lambda x: 1 if ('рольф' or 'панавто' or 'дилер' or 'кредит' or 'ликвидация') in x else 0)

    df_output['alloy_wheels']= df_output['description_word'].apply(lambda x: 1 if ('легкосплавные' or 'колесные' or 'диски') in x else 0)

    df_output['parking_sensors']= df_output['description_word'].apply(lambda x: 1 if ('парктроник' or 'парктронник') in x else 0)

    df_output['dents']= df_output['description_word'].apply(lambda x: 1 if ('вмятины' or 'вмятина' or 'царапина' or 'царапины' or 'трещина') in x else 0)

    df_output['roof_with_panoramic_view']= df_output['description_word'].apply(lambda x: 1 if ('панорамная' and 'крыша') in x else 0)

    

    # ################### fix ############################################################## 

    # Переводим признаки из float в int (иначе catboost выдает ошибку)

    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate', 'Владельцы', 'enginePower',

            'leather', 'carter', 'ABS', 'airbags', 'immob', 'central_locking', 'on_board_computer', 'cruise_control', 

            'climat_control', 'multi_rudder', 'power_steering', 'light_and_rain_sensors', 'сarbon_body_kits', 

            'rear_diffuser_rkp', 'door_closers', 'rear_view_camera', 'amg', 'bi_xenon_headlights', 'from_salon', 

            'alloy_wheels', 'parking_sensors', 'dents', 'roof_with_panoramic_view']:

        where_are_NaNs = np.isnan(df_output[feature])

        df_output[where_are_NaNs] = 0

        df_output[feature]=df_output[feature].astype('int32')

    

    # ################### Feature Engineering ####################################################



    # df_output['horses'] = df_output['name'].str.extract(r'\((\d+) л\.с\.\)', expand=False)

    # df_output = pd.get_dummies(df_output, columns=['vehicleConfiguration', 'fuelType', 'brand', 'Привод', 'Руль', 'ПТС'])

    # df_output['Владельцы'].fillna(0)

    # df_output['enginePower']

    

    # ################### Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    df_output.drop(['Комплектация', 'description', 'description_len', 'description_word', 

        'Владение', 'name', 'id', 'Unnamed: 0', 'vehicleTransmission',

        'Таможня', 'Состояние', 'id'], axis=1, inplace=True, errors='ignore')

        # 'Unnamed: 0', 'vehicleConfiguration_0', 'fuelType_0', 'brand_0', 'Привод_0', 'Руль_0', 'ПТС_0',

        # 'name', 'vehicleConfiguration', 'fuelType', 'brand', 'Привод', 'Руль', 'ПТС'], axis=1, inplace=True, errors='ignore')

    

    return df_output
# preproc_data(train).isna().any()

# preproc_data(train).info()

# preproc_data(train)['ПТС'].unique()
train_preproc = preproc_data(train)

X_sub = preproc_data(test)
X = train_preproc.drop(['Price'], axis=1,)

y = train_preproc.Price.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
# Keep list of all categorical features in dataset to specify this for CatBoost

cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
model = CatBoostRegressor(iterations = ITERATIONS,

                          learning_rate = LR,

                          random_seed = RANDOM_SEED,

                          eval_metric='MAPE',

                          custom_metric=['R2', 'MAE']

                         )

model.fit(X_train, y_train,

         cat_features=cat_features_ids,

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