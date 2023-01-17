import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
import re

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42
VERSION    = 5
DIR_TRAIN  = '../input/sfcarprice/' # подключил к ноутбуку свой внешний датасет
DIR_TEST   = '../input/sf-dst-car-price/'
VAL_SIZE   = 0.1   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 6000
LR         = 0.05
!ls ../input/
train = pd.read_csv(DIR_TRAIN+'BMW_train.csv') # мой подготовленный датасет для обучения модели
test = pd.read_csv(DIR_TEST+'test.csv')
sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
train = train.dropna(subset = ['price', 'name'])
train['Владельцы'].fillna('3 или более', inplace = True)
train['ПТС'].fillna('Оригинал', inplace = True)

def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    df_output.drop(['id', 'Таможня', 'Состояние', 'vehicleConfiguration'], axis=1, inplace=True,)
    
    
    # в явном виде переводим в INT объем двигателя и мощность
    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: int(x[:-4]))
    df_output['engineDisplacement'] = df_output['engineDisplacement'].apply(lambda x: 
                                                                            0 if x == 'undefined LTR' else 10 * float(x[:-4]))

    # ################### fix ############################################################## 
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate', 'enginePower', 'engineDisplacement']:
        df_output[feature]=df_output[feature].astype('int32')
    

    
    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # Добавим признак = количеству опций
    df_output['lenConfiguration'] = df_input['Комплектация'].apply(lambda x: len(configuration_parsing(x)))
    
    # ################### Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    df_output.drop(['description', 'Владение', 'Комплектация'], axis=1, inplace=True,)
    
    # V5 - уберем name, вся информация содержится в других столбцах
    #df_output.drop(['name'], axis=1, inplace=True)
    
    return df_output

###################################################
def configuration_parsing(txt):
    configuration = []
    pattern = r'\"values\":\[(.+?)\]}'
    for txt_elem in re.findall(pattern, txt):
        pattern_txt_elem = r'\"(.+?)\"'
        for config_item in re.findall(pattern_txt_elem, txt_elem):
            configuration.append(config_item)
    return configuration
# код для one-hot-encoding данных комплектации
set_configuration = set()
for i in range(train.shape[0]):
    txt = train['Комплектация'].iloc[i]
    lst = configuration_parsing(txt)
    set_configuration.update(set(lst))

for i in range(test.shape[0]):
    txt = test['Комплектация'].iloc[i]
    lst = configuration_parsing(txt)
    set_configuration.update(set(lst))
    
train = train.reindex(columns = train.columns.tolist() + list(set_configuration))
test = test.reindex(columns = test.columns.tolist() + list(set_configuration))

for col in set_configuration:
    train[col] = train['Комплектация'].apply(lambda x: 1 if col in configuration_parsing(x) else 0)
    test[col] = test['Комплектация'].apply(lambda x: 1 if col in configuration_parsing(x) else 0)
train_preproc = preproc_data(train)
X_sub = preproc_data(test)
train_preproc.drop(['URL'], axis=1, inplace=True,) # убрал лишний столбец, которого нет в testе
X = train_preproc.drop(['price'], axis=1,)
y = 0.95 * train_preproc.price.values
cat_features_ids = ['bodyType', 'brand', 'color', 'fuelType', 'name',
         'vehicleTransmission', 'Привод', 'Руль', 'Владельцы', 'ПТС']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
# Keep list of all categorical features in dataset to specify this for CatBoost
# cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
cat_features_ids = ['bodyType', 'brand', 'color', 'fuelType', 'name',
         'vehicleTransmission', 'Привод', 'Руль', 'Владельцы', 'ПТС']
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
features_importances = pd.DataFrame(data = model.feature_importances_, index = X.columns, columns = ['FeatImportant'])
features_importances.sort_values(by = 'FeatImportant', ascending = False).head(20)

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
# sample_submission['price'] = np.around(submissions['blend'].values, decimals = -4) # округлили до 10 тыс
sample_submission['price'] = np.ceil(submissions['blend'].values / 10000) * 10000 # округлили до 10 тыс
# sample_submission['price'] = submissions['blend'].values

sample_submission.to_csv(f'submission_blend_v{VERSION}_BMW.csv', index=False)
sample_submission.head(10)
def compute_meta_feature(model, X_train, X_test, y_train, cv):
    """
    Computes meta-features usinf the classifier cls
    
    :arg model: scikit-learn classifier
    :arg X_train, y_train: training set
    :arg X_test: testing set
    :arg cv: cross-validation folding
    """
    
    X_meta_train = np.zeros_like(y_train, dtype = np.float32)
    for train_fold_index, predict_fold_index in cv.split(X_train):
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        
        folded_model = clone(model)
        folded_model.fit(X_fold_train, y_fold_train)
        X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)
        
    meta_model = clone(model)
    meta_model.fit(X_train, y_train)
    
    X_meta_test = meta_model.predict_proba(X_test)[:,1]
    
    return X_meta_train, X_meta_test
n_foldes = 5
cv = KFold(n_splits=n_foldes, shuffle=True)


X_meta_train_features = []
X_meta_test_features = []

# 1 - catboost
# тут нужно переделать на разбивку от всего Х,у - без тестовой выборки, кот вначале сделали. Мы так 20% данных отбросили
# и 

model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE']
                         )

X_meta_train = np.zeros_like(y, dtype = np.float32)
X_meta_test = np.zeros(len(X_sub), dtype = np.float32)
for train_fold_index, predict_fold_index in cv.split(X):
    X_fold_train, X_fold_predict = X.iloc[train_fold_index], X.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train,
                     cat_features=cat_features_ids,
                     eval_set=(X_test, y_test),
                     verbose_eval=1000,
                     use_best_model=True,
                     plot=False
)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)
    X_meta_test += folded_model.predict(X_sub)
    
'''
# Для предсказаний на тесте я переобучала модель, но попробую взять среднее по фолад
meta_model = clone(model)
meta_model.fit(X_train, y_train,
             cat_features=cat_features_ids,
             eval_set=(X_test, y_test),
             verbose_eval=1000,
             use_best_model=True,
             plot=True
)
'''

X_meta_test = X_meta_test / n_foldes

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)

# randomForestRegressor

# LinearRegressor l1

# LinearRegressor l2
X.drop(cat_features_ids, axis = 1).info()
# 2 - randomForestRegressor

model = RandomForestRegressor(n_estimators=300, random_state=42)

X_meta_train = np.zeros_like(y, dtype = np.float32)
X_train_num = X.drop(cat_features_ids, axis = 1)
#X_test_num = X_test.drop(cat_features_ids, axis = 1)
X_sub_num = X_sub.drop(cat_features_ids, axis = 1)

for train_fold_index, predict_fold_index in cv.split(X_train_num):
    X_fold_train, X_fold_predict = X_train_num.iloc[train_fold_index], X_train_num.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)

meta_model = clone(model)
meta_model.fit(X_train_num, y)

X_meta_test = meta_model.predict(X_sub_num)

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)


# 3 LinearRegressor
# что-то ерунда получилась, нужно проверить, что с ней - в ответах теста было отриц значение в степени 20!!!
# может взять просто минимум числовых - год, пробег, мощности


model = LinearRegression(normalize = True)

X_meta_train = np.zeros_like(y, dtype = np.float32)

for train_fold_index, predict_fold_index in cv.split(X_train_num):
    X_fold_train, X_fold_predict = X_train_num.iloc[train_fold_index], X_train_num.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)

meta_model = clone(model)
meta_model.fit(X_train_num, y)

X_meta_test = meta_model.predict(X_sub_num)

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)


stacked_features_train = np.vstack(X_meta_train_features[:2]).T
stacked_features_test = np.vstack(X_meta_test_features[:2]).T
stacked_features_test[:10,:]
X_meta_test_features[0].max()
final_model = LinearRegression()
final_model.fit(stacked_features_train, y)
sample_submission['price'] = np.floor(final_model.predict(stacked_features_test) / 10000) * 10000 
sample_submission.to_csv(f'submission_stack_v{VERSION}_BMW.csv', index=False)
sample_submission.head(10)




