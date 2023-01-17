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
VERSION    = 1
DIR_TRAIN  = '../input/avito-auto-train/' # подключил к ноутбуку свой внешний датасет
DIR_TEST   = '../input/sf-dst-car-price/'
VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1
!ls ../input/
train = pd.read_csv(DIR_TRAIN+'avito_auto_train.csv') # мой подготовленный датасет для обучения модели
test = pd.read_csv(DIR_TEST+'test.csv')
sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
train = train.dropna()
train.info()
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    #df_output.drop(['Таможня', 'Состояние', 'id'], axis=1, inplace=True,)
    df_output.drop(['Таможня', 'ПТС', 'id', 'description', 'Комплектация'], axis=1, inplace=True,) #Комплектация - слишком разные знач-я в моем датасете и тесте
    df_output.drop(['name', 'vehicleConfiguration'], axis=1, inplace=True,)
    
    
    # ################### fix ############################################################## 
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:
        df_output[feature]=df_output[feature].astype('int32')
    
    
    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # ....
    
    
    # ################### Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    #df_output.drop(['Комплектация', 'description', 'Владение'], axis=1, inplace=True,)
    
    return df_output
# test.info()
train_preproc = preproc_data(train)
X_sub = preproc_data(test)
train_preproc.drop(['url'], axis=1, inplace=True,) # убрал лишний столбец, которого нет в testе
X = train_preproc.drop(['price'], axis=1,)
y = train_preproc.price.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
X_sub.info()
X_train.info()
# чтобы не писать весь список этих признаков, просто вывел их через nunique(). и так сойдет)
X_train.nunique()
# Keep list of all categorical features in dataset to specify this for CatBoost

# Все фичи
# ITERATIONS = 2000
# LR         = 0.1
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 30000)[0].tolist()
# cat_features_ids
# old value: bestTest = 0.2594671266
# old value: bestIteration = 1990

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.3002506984
# bestIteration = 1995
# Убраны фичи, у которых уникальных значений <3000
# X_train = X_train.drop(['name', 'description', 'mileage'], axis=1)
# X_test = X_test.drop(['name', 'description', 'mileage'], axis=1)
# ITERATIONS = 2000
# LR         = 0.1
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
# cat_features_ids
# old value: bestTest = 0.2666901984
# old value: bestIteration = 1999
ITERATIONS = 4000
LR         = 0.1

# Оставляю все признаки, категориальные из них перечислены вручную
cat_features_ids = [0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2699388592
# bestIteration = 1022

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 7000, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2983751802
# bestIteration = 1999

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 7000, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.291609672
# bestIteration = 2613

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2794647076
# bestIteration = 1636
# Убрал name (3000+)
# X_train = X_train.drop(['name'], axis=1)
# X_test = X_test.drop(['name'], axis=1)

ITERATIONS = 4000
LR         = 0.1
cat_features_ids = [0, 1, 2, 3, 7, 8, 9, 10, 12, 13, 14, 15, 16]

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2611273872
# bestIteration = 1474

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 7000, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2983175387
# bestIteration = 1446

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 3000, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2983175387
# bestIteration = 1446

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2723449069
# bestIteration = 1996

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2682154873
# bestIteration = 2176

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE', 'bagging_temperature': 5}
# bestTest = 0.2682154873
# bestIteration = 2176

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE', 'bagging_temperature': 10}
# bestTest = 0.2682154873
# bestIteration = 2176

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2652452448
# bestIteration = 1647
# Убрал vehicleConfiguration (475)
# X_train = X_train.drop(['vehicleConfiguration'], axis=1)
# X_test = X_test.drop(['vehicleConfiguration'], axis=1)

ITERATIONS = 4000
LR         = 0.1
cat_features_ids = [0, 1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15]

# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2603763223
# bestIteration = 1673

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2771858909
# bestIteration = 1509

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2597116225
# bestIteration = 1780

LR         = 0.01
# model params: {'iterations': 10000, 'learning_rate': 0.01, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.3110392118
# bestIteration = 7499

LR         = 0.05
# model params: {'iterations': 8000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2617761191
# bestIteration = 3400

LR         = 0.07
# model params: {'iterations': 4000, 'learning_rate': 0.07, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2518687933
# bestIteration = 3649

LR         = 0.07
# model params: {'iterations': 4000, 'learning_rate': 0.07, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE', 'bagging_temperature': 10}
# bestTest = 0.2518687933
# bestIteration = 3649

# model params: {'iterations': 4000, 'learning_rate': 0.07, 'l2_leaf_reg': 4.0, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE', 'bagging_temperature': 10}
# bestTest = 0.2509877959
# bestIteration = 3999

ITERATIONS = 6000

# !!! Беру эти параметры !!!
# model params: {'iterations': 6000, 'learning_rate': 0.07, 'l2_leaf_reg': 4.0, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.248731864
# bestIteration = 4239
# Убрал enginePower (338)
# X_train = X_train.drop(['enginePower'], axis=1)
# X_test = X_test.drop(['enginePower'], axis=1)

# ITERATIONS = 4000
# LR         = 0.1
# cat_features_ids = [0, 1, 2, 3, 7, 8, 10, 11, 12, 13, 14]

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.3001611289
# bestIteration = 769

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 120, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2734358937
# bestIteration = 2097

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2635431819
# bestIteration = 2622

# model params: {'iterations': 4000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'one_hot_max_size': 100, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2635431819
# bestIteration = 2622
X_train.nunique()
# Убраны фичи, у которых уникальных значений <3000
# X_train = X_train.drop(['name', 'description', 'mileage'], axis=1)
# X_test = X_test.drop(['name', 'description', 'mileage'], axis=1)
# Изменены параметры
# ITERATIONS = 3000
# LR         = 0.05
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
# cat_features_ids
# bestTest = 0.2683242514
# bestIteration = 2988
# Убраны фичи, у которых уникальных значений <7000
# ITERATIONS = 2000
# LR         = 0.1
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 7000)[0].tolist()
# cat_features_ids
# bestTest = 0.2572848724
# bestIteration = 1999
# Убраны фичи, у которых уникальных значений <7000
# ITERATIONS = 4000
# LR         = 0.1
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 7000)[0].tolist()
# cat_features_ids
# bestTest = 0.2517722065
# bestIteration = 3836
# Убраны фичи, у которых уникальных значений <7000
# ITERATIONS = 8000
# LR         = 0.05
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 7000)[0].tolist()
# cat_features_ids
# model params: {'iterations': 8000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2490244432
# bestIteration = 7944
# Убраны фичи, у которых уникальных значений <7000
# ITERATIONS = 2000
# LR         = 0.1
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 7000)[0].tolist()
# cat_features_ids
# model params: {'iterations': 2000, 'learning_rate': 0.1, 'loss_function': 'RMSE', 'od_wait': 101, 'od_type': 'Iter', 'random_seed': 42, 'custom_metric': ['R2', 'MAE'], 'eval_metric': 'MAPE'}
# bestTest = 0.2638938549
# bestIteration = 1615
model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE'],
#                           metric_period=500,
#                           bagging_temperature=10,
                          one_hot_max_size = 100,
                          l2_leaf_reg = 4.0,
                          od_type='Iter',
                          od_wait=101
                         )
print('model params:', model.get_params())
model.fit(X_train, y_train,
         cat_features=cat_features_ids,
         eval_set=(X_test, y_test),
         verbose_eval=100,
         use_best_model=True,
         plot=True
         )
model.save_model('catboost_single_model_baseline.model')
X_sub.info()
X_sub['Владение'] = X_sub['Владение'].apply(lambda x: 'Нет данных' if pd.isna(x) else x)
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


