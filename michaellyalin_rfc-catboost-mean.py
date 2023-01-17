import catboost

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from catboost import Pool, cv
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 100)
df = pd.read_csv('train.csv')
# заменим NaNы в категориальных признаках

for col in df.select_dtypes('object').columns:

    df[col] = df[col].fillna('UNKNOWN')
# заменим NaNы в числовых признаках

for col in df.select_dtypes(['int64','float64']).columns:

    df[col] = df[col].fillna(df[col].median())
# Определим фичи с низкой корреляцией с 'churn'

# Далее отбросим величины, корреляция которых лежит в интервале [-0.006768, 0.001665]

df.corrwith(df['churn'], method='spearman').sort_values(ascending=False)
# Список всех числовых фич

num_features = df.select_dtypes(['int64','float64']).columns.tolist()
# наблюдения показали, что исключение фичи 'Customer_ID' из списка используемых увеличивает скор

# предположительно из-за того, что фича не несет полезной инфы

num_features = ['rev_Mean',

 'mou_Mean',

 'totmrc_Mean',

 'da_Mean',

 'ovrmou_Mean',

 'ovrrev_Mean',

 'vceovr_Mean',

 'datovr_Mean',

 'roam_Mean',

 'change_mou',

 'change_rev',

 'drop_vce_Mean',

 'drop_dat_Mean',

 'blck_vce_Mean',

 'blck_dat_Mean',

 'unan_vce_Mean',

 'unan_dat_Mean',

 'plcd_vce_Mean',

 'plcd_dat_Mean',

 'recv_vce_Mean',

 'recv_sms_Mean',

 'comp_vce_Mean',

 'comp_dat_Mean',

 'custcare_Mean',

 'ccrndmou_Mean',

 'cc_mou_Mean',

 'inonemin_Mean',

 'threeway_Mean',

 'mou_cvce_Mean',

 'mou_cdat_Mean',

 'mou_rvce_Mean',

 'owylis_vce_Mean',

 'mouowylisv_Mean',

 'iwylis_vce_Mean',

 'mouiwylisv_Mean',

 'peak_vce_Mean',

 'peak_dat_Mean',

 'mou_peav_Mean',

 'mou_pead_Mean',

 'opk_vce_Mean',

 'opk_dat_Mean',

 'mou_opkv_Mean',

 'mou_opkd_Mean',

 'drop_blk_Mean',

 'attempt_Mean',

 'complete_Mean',

 'callwait_Mean',

 'months',

 'uniqsubs',

 'actvsubs',

 'totcalls',

 'totmou',

 'totrev',

 'adjrev',

 'adjmou',

 'adjqty',

 'avgmou',

 'avgqty',

 'avg3mou',

 'avg3qty',

 'avg3rev',

 'avg6mou',

 'avg6qty',

 'avg6rev',

 'hnd_price',

 'phones',

 'models',

 'lor',

 'income',

 'eqpdays']
# будем использовать только численные фичи из списка выше

X_train, X_valid, y_train, y_valid = train_test_split(

    df[num_features], df['churn'], test_size=0.33, random_state=42)
# Основываясь на предсказании roc_auc, подобрали "оптимальные" параметры для RFC

clf = RandomForestClassifier(

    max_depth = 13, 

    min_samples_leaf = 6,

    n_estimators = 30,

    random_state = 13,

    max_features = 20

)

#clf.fit(X_train, y_train)

# Постороили модель по всей тренировочной выборке

clf.fit(df[num_features], df['churn'])

# Оценка обучения модели по предсказанию на части из тренировочной выборки

# Параллельно можно проверять по разбиению X_train/X_valid, результат roc_auc близок к результату с kaggle

predict_test = clf.predict_proba(X_valid[num_features])[:,1]

roc_auc_score(y_valid, predict_test)
test_df = pd.read_csv('test.csv')



for col in test_df.select_dtypes(['int64','float64']).columns:

    test_df[col] = test_df[col].fillna(test_df[col].median())

    

test_df['pred_proba'] = clf.predict_proba(test_df[num_features])[:,1]

submission_df = test_df[['Customer_ID']]

submission_df['churn'] = test_df['pred_proba']

submission_df.to_csv(

    'RFC_cut_customer_ID_full_train_cut_num.csv',

    index = False

)
# заменим NaNы в категориальных признаках

for col in df.select_dtypes('object').columns:

    df[col] = df[col].fillna('UNKNOWN')
# заменим NaNы в числовых признаках

# НО! catboost дает лучший результат, если НЕ заменять NaN в числовых фичах

for col in df.select_dtypes(['int64','float64']).columns:

    df[col] = df[col].fillna(df[col].median())
# замутим список всех фич, целевую выкинули оттуда

features = df.columns.tolist()
# Аналогично предыдущему, выбираем фичи

features= ['Customer_ID',

 'rev_Mean',

 'mou_Mean',

 'totmrc_Mean',

 'da_Mean',

 'ovrmou_Mean',

 'ovrrev_Mean',

 'vceovr_Mean',

 'datovr_Mean',

 'roam_Mean',

 'change_mou',

 'change_rev',

 'drop_vce_Mean',

 'drop_dat_Mean',

 'blck_vce_Mean',

 'blck_dat_Mean',

 'unan_vce_Mean',

 'unan_dat_Mean',

 'plcd_vce_Mean',

 'plcd_dat_Mean',

 'recv_vce_Mean',

 'recv_sms_Mean',

 'comp_vce_Mean',

 'comp_dat_Mean',

 'custcare_Mean',

 'ccrndmou_Mean',

 'cc_mou_Mean',

 'inonemin_Mean',

 'threeway_Mean',

 'mou_cvce_Mean',

 'mou_cdat_Mean',

 'mou_rvce_Mean',

 'owylis_vce_Mean',

 'mouowylisv_Mean',

 'iwylis_vce_Mean',

 'mouiwylisv_Mean',

 'peak_vce_Mean',

 'peak_dat_Mean',

 'mou_peav_Mean',

 'mou_pead_Mean',

 'opk_vce_Mean',

 'opk_dat_Mean',

 'mou_opkv_Mean',

 'mou_opkd_Mean',

 'drop_blk_Mean',

 'attempt_Mean',

 'complete_Mean',

 'callwait_Mean',

 'months',

 'uniqsubs',

 'actvsubs',

 'new_cell',

 'crclscod',

 'asl_flag',

 'totcalls',

 'totmou',

 'totrev',

 'adjrev',

 'adjmou',

 'adjqty',

 'avgmou',

 'avgqty',

 'avg3mou',

 'avg3qty',

 'avg3rev',

 'avg6mou',

 'avg6qty',

 'avg6rev',

 'prizm_social_one',

 'area',

 'dualband',

 'refurb_new',

 'hnd_price',

 'phones',

 'models',

 'hnd_webcap',

 'ownrent',

 'lor',

 'dwlltype',

 'marital',

 'adults',

 'infobase',

 'HHstatin',

 'dwllsize',

 'forgntvl',

 'ethnic',

 'kid0_2',

 'kid3_5',

 'kid6_10',

 'kid11_15',

 'kid16_17',

 'creditcd',

 'eqpdays']
# получим список категориальных фич

cat_features = df[features].select_dtypes('object').columns.tolist()
X_train, X_valid, y_train, y_valid = train_test_split(

    df[features], df['churn'], test_size=0.33, random_state=42)
# Интуитивно подобраны параметры для проведения кросс-валидации (из анализа графика)

cv_dataset = Pool(data=df[features],

                  label=df['churn'],

                  cat_features=cat_features)



params = {"iterations": 200,

          "depth": 3,

          "loss_function": "Logloss",

          "verbose": False,

           'learning_rate':0.3,

         'use_best_model':True}



scores = cv(cv_dataset,

            params,

            fold_count=3, 

            plot="True")
# Некоторые ученые говорят, что итераций многовато, да и скорость обучения высоковата, но что имеем, то и продаем

model = catboost.CatBoostClassifier(iterations = 600, depth=3, verbose= False, learning_rate=0.3)

model.fit(df[features], df['churn'], cat_features=cat_features, plot=True, verbose=False)

#model.fit(X_train, y_train, cat_features=cat_features, plot=True, verbose=False)

# Оценка точности по части той выборки, на которой обучали

# Ясно, что это лишь способ проверить согласованность модели

predict_test = model.predict_proba(X_valid[features])[:,1]

roc_auc_score(y_valid, predict_test)
test_df = pd.read_csv('test.csv')

#test_df = test_df.drop(['numbcars','dwllsize' ,'HHstatin','ownrent'], axis=1)



for col in test_df.select_dtypes('object').columns:

    test_df[col] = test_df[col].fillna('UNKNOWN')   

    

test_df['pred_proba'] = model.predict_proba(test_df[features])[:,1]

submission_df = test_df[['Customer_ID']]

submission_df['churn'] = test_df['pred_proba']

submission_df.to_csv(

    'Catboost_0.69555.csv',

    index = False

)
RFC_submission_df = submission_df

RFC_submission_df
CB_submission_df = pd.read_csv('Catboost_0.69555.csv')

CB_submission_df
Mean = CB_submission_df[['Customer_ID']]

Mean['churn'] = (CB_submission_df['churn']+RFC_submission_df['churn'])/2
Mean.to_csv(

    'CB_0.69555_RFC_mean.csv',

    index = False

)
Mean