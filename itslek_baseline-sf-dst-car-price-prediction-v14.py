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

DIR_TRAIN  = '../input/sf-autoru-solve-v4/' # подключил к ноутбуку свой внешний датасет

DIR_TEST   = '../input/sf-dst-car-price/'

VAL_SIZE   = 0.33   # 33%

N_FOLDS    = 5



# CATBOOST

ITERATIONS = 2000

LR         = 0.1
!ls ../input/
train = pd.read_csv(DIR_TRAIN+'train.csv') # мой подготовленный датасет для обучения модели

test = pd.read_csv(DIR_TEST+'test.csv')

sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Таможня', 'Состояние', 'id'], axis=1, inplace=True,)

    

    

    # ################### fix ############################################################## 

    # Переводим признаки из float в int (иначе catboost выдает ошибку)

    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:

        df_output[feature]=df_output[feature].astype('int32')

    

    

    # ################### Feature Engineering ####################################################

    # тут ваш код на генерацию новых фитчей

    # ....

    

    

    # ################### Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    df_output.drop(['Комплектация', 'description', 'Владение'], axis=1, inplace=True,)

    

    return df_output
train_preproc = preproc_data(train)

X_sub = preproc_data(test)
train_preproc.drop(['car_url'], axis=1, inplace=True,) # убрал лишний столбец, которого нет в testе
X = train_preproc.drop(['price'], axis=1,)

y = train_preproc.price.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
# чтобы не писать весь список этих признаков, просто вывел их через nunique(). и так сойдет)

X_train.nunique()
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
def cat_model(y_train, X_train, X_test, y_test):

    model = CatBoostRegressor(iterations = ITERATIONS,

                              learning_rate = LR,

                              eval_metric='MAPE',

                              random_seed = RANDOM_SEED,)

    model.fit(X_train, y_train,

              cat_features=cat_features_ids,

              eval_set=(X_test, y_test),

              verbose=False,

              use_best_model=True,

              plot=False)

    

    return(model)





def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели

score_ls = []

splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))



for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):

    # use the indexes to extract the folds in the train and validation data

    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]

    # model for this fold

    model = cat_model(y_train, X_train, X_test, y_test,)

    # score model on test

    test_predict = model.predict(X_test)

    test_score = mape(y_test, test_predict)

    score_ls.append(test_score)

    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")

    # submissions

    submissions[f'sub_{idx+1}'] = model.predict(X_sub)

    model.save_model(f'catboost_fold_{idx+1}.model')

    

print(f'Mean Score: {np.mean(score_ls):0.3f}')

print(f'Std Score: {np.std(score_ls):0.4f}')

print(f'Max Score: {np.max(score_ls):0.3f}')

print(f'Min Score: {np.min(score_ls):0.3f}')
submissions.head(10)
submissions['blend'] = (submissions.sum(axis=1))/len(submissions.columns)

sample_submission['price'] = submissions['blend'].values

sample_submission.to_csv(f'submission_blend_v{VERSION}.csv', index=False)

sample_submission.head(10)