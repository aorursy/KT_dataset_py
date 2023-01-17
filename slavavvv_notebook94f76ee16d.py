# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



from sklearn.preprocessing import OneHotEncoder    

from sklearn import tree



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier



# импортируем функцию roc_auc_score()

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import PolynomialFeatures, PowerTransformer

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV





#########



df = pd.DataFrame()

df = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')

df_test = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')





#########

# берем катег признаки



cat_features = df.dtypes[df.dtypes == 'object'].index





#########



# смотрим редкие категории

# for i in cat_features:

#     abs_freq = df[i].value_counts(dropna = False)

    

# избавляемся от редких категорий



r=50



for i in cat_features:

    abs_freq = df[i].value_counts(dropna = False)

    df[i] = np.where(df[i].isin(abs_freq[abs_freq >= r].index.tolist()), df[i],'Other')

    print(df[i].value_counts())





# >>>>>>>>>>>>>>>>>>



########



# сделаем get_dummies в трейн и тест и удалим лишние столбцы из трейн/тест

df = pd.get_dummies(df, columns = cat_features, drop_first = True)



# избавляемся от редких категорий

# сразу вычислим пересечение столбцов трейна и теста, для этого загрузим test выборку

df_test = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')

cat_features_df_test = df_test.dtypes[df_test.dtypes == 'object'].index



# избавляемся от редких категорий

for i in cat_features_df_test:

    abs_freq = df_test[i].value_counts(dropna = False)

    df_test[i] = np.where(df_test[i].isin(abs_freq[abs_freq >= r].index.tolist()), df_test[i], 'Other')

    print(df_test[i].value_counts())





#########    

# делаем get dummies

df_test = pd.get_dummies(df_test, columns = cat_features_df_test, drop_first = True)



print('df.shape, df_test.shape : ', df.shape, df_test.shape)



# определяем общие признаки для обоих датасетов df / df_test и оставляем только их

df_list = list(df.columns)

df_test_list = list(df_test.columns)



common_cols = set.intersection(set(df_test_list) & set(df_list))

print('# число общих колонок после get_dummies: / ', len(common_cols))





# >>>>>>>>>>>>>>>>>>





#########



# Split the train data into train and test data

x_train, x_test, y_train, y_test = train_test_split(

    df.drop(labels = ['target'], axis = 1),

    df['target'],

    test_size = 0.3,

    random_state = 0

)



# >>>>>>>>>>>>>>>>>>

x_train = x_train[common_cols]

x_test = x_test[common_cols]

df_test = df_test[common_cols]

# >>>>>>>>>>>>>>>>>>







print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



#########



# заменим все NAN в train выборке на средние .mean()  - ничего не дает



# numerical_cols = x_train.dtypes[x_train.dtypes != 'object'].index

# for i in numerical_cols:

#     x_train[i].fillna(x_train[i].median(), inplace = True)

#     x_test[i].fillna(x_train[i].median(), inplace = True)





#########



# # используем PowerTransformer для нормализации данных - ничего не дает



# pt = PowerTransformer()

# pt.fit(x_train)                       ## Fit the PT on training data

# x_train_pt = pt.transform(x_train)    ## Then apply on all data

# x_test_pt = pt.transform(x_test)

# df_test_pt = pt.transform(df_test)



# x_train_pt = pd.DataFrame(x_train_pt)

# x_test_pt = pd.DataFrame(x_test_pt)

# df_test_pt = pd.DataFrame(df_test_pt)



# # удалим нан-ы



# for i in x_train_pt.columns:

#     x_train_pt[i].fillna(x_train_pt[i].median(), inplace = True)

#     x_test_pt[i].fillna(x_train_pt[i].median(), inplace = True)

#     df_test_pt[i].fillna(x_train_pt[i].median(), inplace = True)



# for i in x_train_pt.columns:

#     x_train[i].fillna(x_train[i].median(), inplace = True)

#     x_test[i].fillna(x_train[i].median(), inplace = True)

#     df_test[i].fillna(x_train[i].median(), inplace = True)







#########

# удалим нан-ы



train_features = x_train.dtypes[x_train.dtypes != 'object'].index



for i in train_features:

    x_train[i].fillna(x_train[i].median(), inplace = True)

    x_test[i].fillna(x_train[i].median(), inplace = True)

    df_test[i].fillna(x_train[i].median(), inplace = True)



# >>>>>>>>>>>>>>>>>>

x_train.columns = train_features

x_test.columns = train_features

df_test.columns = train_features

# >>>>>>>>>>>>>>>>>>







print('x_train.columns: \n', x_train.columns, 'df_test.columns: \n', df_test.columns)



#########

# # удалим столбцы констант

# x_train_pt = x_train_pt.loc[:,x_train_pt.apply(pd.Series.nunique) != 1]

# x_train_pt.apply(pd.Series.nunique) != 1



# Mutual information



# мера, показывает как 2 переменные взаимо зависимы друг от друга. х от у и наоборот. 

# условно - это сколько информации дает нам знание одной переменной Х об другой переменной У. и наоборот.



from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.feature_selection import SelectKBest, SelectPercentile

from scipy import stats

from sklearn.preprocessing import MinMaxScaler



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import os

import glob

import datetime

import dateutil.tz





# x_train[train_features].isna().sum().sum()



##################

# нормализация

# minmaxscaler (x - xmin / xmax - xmin)



# scaler = MinMaxScaler()

# print(scaler.fit(x_train))

# print(scaler.data_max_)

# x_train = scaler.transform(x_train)

# x_train = pd.DataFrame(x_train)





# x_test =  scaler.transform(x_test)

# x_test = pd.DataFrame(x_test)









##################

# # удаляем дубли из датасета



# df = df.drop_duplicates(subset = None,

#                    keep = 'first',

#                    inplace = True)





##################

# # сделаем box cox



# for i in numerical_cols:

#     positive_data = x_train[i][x_train[i] > 0]

#     positive_data, lam = stats.boxcox(positive_data)

#     print(lam, positive_data)

# #     x_train[i], fitted_lambda = stats.boxcox(x_train[i])

# #     x_test[i] = stats.boxcox(x_test[i], fitted_lambda)





# from sklearn.model_selection import GridSearchCV



# ##################

# # бустинг, подбираем лучшие темп через использование GS



# Lgb = LGBMClassifier(n_estimators = 400, 

#                      random_state = 94, 

#                      max_depth = 5,

#                      verbose = 1)



# param_grid = {'learning_rate': [0.01, 0.05, 0.07, 0.09, 0.1]}



# gs = GridSearchCV(Lgb,

#                   param_grid,

#                   scoring = 'roc_auc',

#                   cv = 5,

#                   n_jobs = -1,

#                   return_train_score = False)



# gs.fit(x_train, y_train)



# print('лучшие значения параметров learn rate: \n', gs.best_params_) #0.05

# print('значение скора roc auc: \n', gs.best_score_)  #0.7528472850502619





# # print(accuracy_score(y_test, [i[1] for i in Lgb_pred])) 

# # print(roc_auc_score(y_test, [i[1] for i in Lgb_pred])) #0.7515020019901595

# # print(roc_auc_score(y_test, Lgb_pred)) #0.7515020019901595



# gs.cv_results_
# # делаем GridSearchCV (минусы - долго, плюсы - перебирает все, высокая точность)



# Lgb12 = LGBMClassifier(n_estimators = 400, 

#                      random_state = 94, 

#                      max_depth = 5,

#                      learning_rate = 0.07,

#                      verbose = 1)



# param_grid2 = {'lambda_l1': [0, 5, 10],

#               'bagging_fraction': [0.3, 0.5, 1],

#               'feature_fraction': [0.3, 0.5, 1]}



# gs2 = GridSearchCV(Lgb12,

#                   cv = 5,

#                   scoring = 'roc_auc',

#                   param_grid = param_grid2,

#                   n_jobs = -1,

#                   return_train_score = False)



# gs2.fit(x_train, y_train)



# # 0.753937566934885 - 0.752869411327





# print('лучшие значения параметров: \n {}'.format(gs2.best_params_)) # {'bagging_fraction': 0.3, 'feature_fraction': 0.5, 'lambda_l1': 0}

# print('значение скора roc auc: \n {}'.format(gs2.best_score_)) #0.7539945492050786



# лучшие значения параметров: 

#  {'bagging_fraction': 0.3, 'feature_fraction': 1, 'lambda_l1': 5}

# значение скора roc auc: 

#  0.7537644557477048
# # делаем RandomizedSearchCV (минусы - рандомный ответ, плюсы - быстро)



# from sklearn.model_selection import RandomizedSearchCV





# Lgb13 = LGBMClassifier(n_estimators = 400, 

#                      random_state = 94, 

#                      max_depth = 5,

#                      learning_rate = 0.07,

#                      verbose = 1)



# param_dist = {'lambda_l1': [0, 1, 2, 5, 10],

#               'bagging_fraction': [0.3, 0.5, 0.7, 1],

#               'feature_fraction': [0.3, 0.5, 0.7, 1]}



# rs1 = RandomizedSearchCV(Lgb13,

#                          cv = 5,

#                          scoring = 'roc_auc',

#                          n_iter = 10,

#                          param_distributions = param_dist,

#                          n_jobs = -1,

#                          return_train_score = False,

#                          verbose = 1)



# rs1.fit(x_train, y_train)



# print('rs1 лучшие значения параметров: \n {}'.format(rs1.best_params_)) # {'bagging_fraction': 0.3, 'feature_fraction': 0.5, 'lambda_l1': 0}

# print('rs1 значение скора roc auc: \n {}'.format(rs1.best_score_)) #0.7539945492050786 - 0.7549460699800374



# # rs1 лучшие значения параметров: 

# #  {'lambda_l1': 5, 'feature_fraction': 1, 'bagging_fraction': 0.3}

# # rs1 значение скора roc auc: 

# #  0.7549460699800374



############### 

# доделываем тюнинг параметров 

# далее нам надо будет выполнить перекрестную проверку на всем x_train/y_train через

# функцию cross_validate





# мы сразу делаем модель с параметрами, которые нашли исходя из randomizedsearchCV 

# rs1 лучшие значения параметров: 

#  {learning_rate = 0.07, 'lambda_l1': 5, 'feature_fraction': 1, 'bagging_fraction': 0.3} // n_estimators = 400  // rs1 значение скора roc auc: 0.7549460699800374





from sklearn.model_selection import cross_validate



lgb14 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       learning_rate = 0.07,

                       lambda_l1 = 5,

                       feature_fraction = 1,

                       bagging_fraction = 0.3,

                       importance_type='gain')





cv = cross_validate(lgb14, 

                    x_train, 

                    y_train, 

                    cv = 5, 

                    scoring = 'roc_auc', 

                    return_estimator = True, 

                    verbose = 1)





# дальше можно посмотреть cv['estimator'][i].feature_importances_
# print(cv['estimator'][0], cv['test_score'].mean())

cv.keys()



cv['test_score'].mean()

# ?cross_validate
fi = []



# добавим все feature_importances_ в датафрейм



for i in range(len(cv['estimator'])):

    fi.append(cv['estimator'][i].feature_importances_)    

fi_pd = pd.DataFrame(index = x_train.columns.values, columns = range(len(fi)))



for i in fi_pd.columns:

    fi_pd[i] = fi[i]

    

# for i in range(fi_pd.shape[0]):

#     fi_pd['mean'][i] = fi_pd.iloc[i][0:5].mean()

fi_pd['mean'] = fi_pd.mean(axis = 1)



lgbm14_best_feat = fi_pd['mean'].sort_values(ascending = False).head(100)



plt.figure(figsize = (15,14))

plt.barh(lgbm14_best_feat.index, lgbm14_best_feat, color='g', height=0.4, align='center', alpha=0.4)

# plt.yticks(fi_pd['mean'])

# plt.xlabel(lgbm14_best_feat.index)



# You can control the size of the figure using plt.figure (e.g., plt.figure(figsize = (6,12)))



# оставим только ТОП300 признаков и из них уже будем удалять плохие

# отсортируем по возрастанию важности признаков, вверху будут самые слабые, их начнем удалять по одному



lgbm14_test_feat = fi_pd['mean'].sort_values(ascending = False).iloc[:-377] #мы знаем что последние 377 признаков имеют важность 0

lgbm14_test_feat_0 = fi_pd['mean'].sort_values(ascending = False).iloc[296:] #мы знаем что последние 377 признаков имеют важность 0

lgbm14_test_feat = lgbm14_test_feat.sort_values(ascending = True)

lgbm14_test_feat = list(lgbm14_test_feat.index.values)



#len(lgbm14_test_feat) = 295

print(lgbm14_test_feat_0.describe(), lgbm14_test_feat )
# # feature_to_remove_auc_score.append(auc_score_mean)



# print(feature_to_remove_auc_score, max(feature_to_remove_auc_score), auc_score_all, auc_score_mean)

# теперь будем удалять по одному признаку и считать модель скор на каждой итерации



# для этого создаем три списка:

# 1 - для удаляемых признаков 2 - для скоров 3 - для подсчета разницы скоров



# 1 - для удаляемых признаков

feature_to_remove = []

# 2 - для скоров

feature_to_remove_auc_score = []

# 3 - для подсчета разницы скоров

feature_to_remove_auc_score_diff = []



count = 1

#  {learning_rate = 0.07, 'lambda_l1': 5, 'feature_fraction': 1, 'bagging_fraction': 0.3} // rs1 значение скора roc auc: 0.7549460699800374

# len(lgbm14_bad_feat)



auc_score_all = cv['test_score'].mean()

feature_to_remove_auc_score.append(auc_score_all)





# запишем все в датафрейм, для него создадим отдельные списки

features_for_df = []

scores_for_df1_after = []

scores_for_df2_before = []

diff_for_df = []



for i in range(1):

    

    feature_to_remove.append(lgbm14_test_feat[i])

    

    print('\n', 'счет номер = ', count, '\n', 'удаляем признак ', lgbm14_test_feat[i], '\n', 'список feature_to_remove пополнили:', feature_to_remove)

    model = LGBMClassifier(n_estimators = 400,

                           learning_rate = 0.07, 

                           lambda_l1 = 3, 

                           feature_fraction = 1, 

                           bagging_fraction = 0.3,

                           random_state = 94)



    auc_score = cross_val_score(model,

                                x_train[lgbm14_test_feat].drop(feature_to_remove, axis = 1),

                                y_train,

                                scoring = 'roc_auc',

                                verbose = 1)



# auc_score_mean - аук модели в цикле на каждой итерации считается свой

    auc_score_mean = auc_score.mean()



    diff = auc_score_mean - max(feature_to_remove_auc_score)



    

    # запишем все в датафрейм, для него создадим отдельные списки



    features_for_df.append(lgbm14_test_feat[i])

    scores_for_df1_after.append(auc_score_mean)

    scores_for_df2_before.append(max(feature_to_remove_auc_score))

    diff_for_df. append(diff)

    

#     all тоже надо пересчитывать!

    count = count + 1

      

    print('auc модели после удаления', '\n', 

          auc_score_mean, '\n',

          'auc модели со всеми признаками :', '\n', 

          max(feature_to_remove_auc_score), '\n')

    

    feature_to_remove_auc_score.append(auc_score_mean)

    

    if diff > 0:

        print('видим, что разница скоров > 0.0001, значит стало лучше, от признака можно избавиться')

        

    else:

        print('видим, что разница скоров < 0.0001 или отрицательна, значит стало хуже, признак следует оставить')

        feature_to_remove.remove(lgbm14_test_feat[i])

        feature_to_remove_auc_score.remove(auc_score_mean)



# создадим датафрейм, в который будем записывать все значения

df_feat_to_remove = pd.DataFrame({'features': features_for_df,

                                  'scores before remove': scores_for_df2_before,

                                  'scores after remove': scores_for_df1_after,

                                  'diff': diff_for_df})





# # 75 признаков удаляли

# auc модели после удаления 

#  0.7548826898632182 

#  auc модели со всеми признаками : 

#  0.755370837432026 



# видим, что разница скоров < 0.0001 или отрицательна, значит стало хуже, признак следует оставить



#  счет номер =  75 

#  удаляем признак  v48 

#  список feature_to_remove пополнили: ['v112_R', 'v113_AF', 'v47_F', 'v113_F', 'v79_M', 'v79_O', 'v56_DH', 'v91_F', 'v48']



# видим, что разница скоров < 0.0001 или отрицательна, значит стало хуже, признак следует оставить



#  счет номер =  63 

#  удаляем признак  v79_K 

#  список feature_to_remove пополнили: ['v22_QNA', 'v113_U', 'v125_CF', 'v125_R', 'v125_BI', 'v125_CD', 'v56_DJ', 'v30_F', 'v22_WRI', 'v112_B', 'v113_L', 'v125_Z', 'v79_K']





#  счет номер =  250 

#  список feature_to_remove пополнили: ['v22_QNA', 'v113_U', 'v125_CF', 'v125_R', 'v125_BI', 'v125_CD', 'v56_DJ', 'v30_F', 'v22_WRI', 'v112_B', 'v113_L', 'v125_Z', 'v112_P']

print('список feature_to_remove: ', '''['v22_QNA', 'v113_U', 'v125_CF', 'v125_R', 'v125_BI', 'v125_CD', 'v56_DJ', 'v30_F', 'v22_WRI', 'v112_B', 'v113_L', 'v125_Z', 'v112_P']''')

feature_to_remove = ['v22_QNA', 'v113_U', 'v125_CF', 'v125_R', 'v125_BI', 'v125_CD', 'v56_DJ', 'v30_F', 'v22_WRI', 'v112_B', 'v113_L', 'v125_Z', 'v112_P']



print('удаляем feature_to_remove из train_features ', '\n', 'длина train_features до удаления', len(train_features))



train_features = [x for x in train_features if x not in feature_to_remove and x not in lgbm14_test_feat_0]

train_features



print('длина train_features после удаления', len(train_features))

# import numpy

# from sklearn.model_selection import RandomizedSearchCV

# from sklearn.model_selection import cross_validate



# # lgb14 = LGBMClassifier(n_estimators = 1500,

# #                        random_state = 94,

# #                        lambda_l1 = 5,

# #                        feature_fraction = 1,

# #                        bagging_fraction = 0.3,

# #                        importance_type='gain')



# # dist_param = {'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.12, 0.18]}

# # gs_final = RandomizedSearchCV(lgb14,

# #                               cv = 10,

# #                               n_iter = 10,

# #                               param_distributions = dist_param,

# #                               scoring = 'roc_auc',

# #                               return_train_score = False)



# # gs_final.fit(x_train[train_features], y_train)



# # print('gs_final лучшие значения параметров: \n {}'.format(gs_final.best_params_)) 

# # print('gs_final значение скора roc auc: \n {}'.format(gs_final.best_score_)) 



# # # # выяснили что лучший результат достигается при learning_rate: 0.01



# lgb14 = LGBMClassifier(n_estimators = 1500,

#                        random_state = 94,

#                        lambda_l1 = 5,

#                        learning_rate = 0.008,

#                        feature_fraction = 1,

#                        bagging_fraction = 0.3,

#                        importance_type='gain')



# lgb14.fit(x_train[train_features], y_train)



# #############

# # сделаем sub 



# pred = lgb14.predict_proba(df_test[train_features])



# sub = pd.DataFrame()

# sub['ID'] = df_test['ID']

# sub['PredictedProb'] = [i[1] for i in pred]



# current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('date_%Y-%m-%d__time_%H-%M')

# flname_sub = 'submission__' + current_time +'.csv' # submission file name



# sub.to_csv(flname_sub, index = False)



# # скор незначительно улучшился - было / стало :: 0.46956 / 0.46984 / 0.46722 / 0.46728
##############

# пробуем оценить работу признаков на разных уровнях глубины 

# считаем lgbm для всех признаков на разной глубине



fi_depth = []



depth = [1,2,3,4,5]



for i in depth:

    

    lgb15 = LGBMClassifier(n_estimators = 400,

                           max_depth = i,

                           random_state = 94,

                           learning_rate = 0.07,

                           lambda_l1 = 5,

                           feature_fraction = 1,

                           bagging_fraction = 0.3,

                           importance_type='gain')

    

    lgb15 = lgb15.fit(x_train[train_features], y_train)

    fi_depth.append(lgb15.feature_importances_)



fi_depth



# d = ({'features': train_features,

#       'feature_importance ' + str(range(1,5,1)): fi_depth})





    

# делаем дф со значениями объекты = признаки, колонки = важность глуб1/2/3/4/5, важность средняя

fi_depth[0][0:50]
fi_depth_df = pd.DataFrame(index = train_features)

for i in range(len(fi_depth)):

    fi_depth_df['col_' + str(i+1)] = fi_depth[i]

    

fi_depth_df = fi_depth_df.sort_values(list(fi_depth_df.columns), ascending = [False, False,False, False,False])

# fi_depth_df.sort_values(list(fi_depth_df.columns), ascending = [False, False, False, False, False])

fi_depth_df['mean'] = fi_depth_df.mean(axis=1)



fi_depth_df = fi_depth_df.sort_values('mean', ascending = False)

fi_depth_df[60:120]



# почему то есть признаки которые очень важны на 1ом уровне глубины, но менее важны на втором и наоборот! пример 

# v47_C	 на 1ом уровне = 0.000000, а дальше очень важен 5541.829421	6040.218602	7157.042309	7947.199432	5337.257953



# в этом блоке попробуем ORDINAL ENC'



import pandas as pd

import numpy as np



from sklearn.preprocessing import OneHotEncoder    

from sklearn import tree



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier



# импортируем функцию roc_auc_score()

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import PolynomialFeatures, PowerTransformer

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV



from numpy import asarray

from sklearn.preprocessing import OrdinalEncoder





#########



df = pd.DataFrame()

df = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')

df_test = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')





# нам надо сделать по порядку:

# 1. сделать трейн тест

# 2. заменить нан на самое частое значение трейна

# 3. обучить онкод на трейне. сделать ординал енкодинг везде (трейн/валид/тест), но обучим его только на трейне



#########

# берем катег признаки



cat_features = df.dtypes[df.dtypes == 'object'].index



# разбиваем на трейн тест

x_train, x_test, y_train, y_test = train_test_split(

    df.drop(labels = ['target'], axis = 1),

    df['target'],

    test_size = 0.3,

    random_state = 0

)





# заменим все нан на самые частые в трейне

for i in x_train[cat_features].columns:

    x_train[i] = x_train[cat_features][i].fillna(x_train[cat_features][i].value_counts().index[0])

    x_test[i] = x_test[cat_features][i].fillna(x_train[cat_features][i].value_counts().index[0])

    df_test[i] = df_test[cat_features][i].fillna(x_train[cat_features][i].value_counts().index[0])

    

# обучим на трейне

enc = OrdinalEncoder()

enc.fit(x_train[cat_features])



# сделаем OrdinalEncoder для всего

result = enc.fit_transform(x_train[cat_features])

result2 = enc.fit_transform(x_test[cat_features])

result3 = enc.fit_transform(df_test[cat_features])



x_train[cat_features] = result

x_test[cat_features] = result2

df_test[cat_features] = result3



# x_train[[str(i) + '_1' for i in cat_features]] = result

# x_test[str(i) + '_1' for i in cat_features] = result2

# df_test[str(i) + '_1' for i in cat_features] = result3



# print(x_train[['v3', 'v3_1']])



# надо еще проверить, что наны в тест заменились на те же значения что и в ТРЕЙНЕ



train_features = x_train.dtypes[x_train.dtypes != 'object'].index



# заменим нан в количественных признаках

for i in train_features:

    x_train[i].fillna(x_train[i].median(), inplace = True)

    x_test[i].fillna(x_train[i].median(), inplace = True)

    df_test[i].fillna(x_train[i].median(), inplace = True)





x_train[train_features][:20]



# подберем лучшие параметры для LGBM

from sklearn.model_selection import cross_validate

from sklearn.model_selection import RandomizedSearchCV



# нашли лучшие параметры: {'learning_rate': 0.07, 'lambda_l1': 5, 'feature_fraction': 0.5, 'bagging_fraction': 1}



# lgb20 = LGBMClassifier(n_estimators = 400,

#                        random_state = 94,

#                        importance_type='gain')



# dist_param = {'learning_rate': [0.07, 0.1, 0.14, 0.2],

#               'feature_fraction': [0.3, 0.5, 1],

#               'bagging_fraction': [0.3, 0.5, 1],

#               'lambda_l1': [0, 5, 10, 15]

#              }





# rs20 = RandomizedSearchCV(lgb20,

#                         cv = 5,

#                         scoring = 'roc_auc',

#                         param_distributions = dist_param,

#                         return_train_score = True)





# rs20.fit(x_train[train_features], y_train)

# print(rs20.best_params_, rs20.best_score_, rs20.best_estimator_)





# дальше можно посмотреть rs20['estimator'][i].feature_importances_



# {'learning_rate': 0.07, 'lambda_l1': 5, 'feature_fraction': 0.5, 'bagging_fraction': 1} 0.7549020977510434 LGBMClassifier(bagging_fraction=1, feature_fraction=0.5, importance_type='gain',

#                lambda_l1=5, learning_rate=0.07, n_estimators=400,

#                random_state=94)

x_train[train_features].shape

# print(rs20.best_params_, rs20.best_score_, rs20.best_estimator_)

lgb20 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       lambda_l1 = 5,

                       learning_rate = 0.07,

                       feature_fraction = 0.5,

                       bagging_fraction = 1,

                       importance_type='gain')



lgb20.fit(x_train[train_features], y_train)



cv = cross_val_score(lgb20,

                     x_train[train_features],

                     y_train,

                     cv=5,

                     scoring = 'roc_auc'

                    )



print('cross val score BEFORE lgb20 with features removed: ', cv.mean())



# CORR фичи

# удалим фичи, которые имеют корреляцию



##################

# # выберем и удалим наиболее коррелирующие признаки



corr = x_train.corr()

# sns.heatmap(corr, cmap='coolwarm')

# corr.head()

# minmaxscaler (x - xmin / xmax - xmin)



train_features = x_train.dtypes[x_train.dtypes != 'object'].index.values





corr.index = train_features

corr.columns = train_features

corr.head()



##############

# создадим список фич где коэф корреляции > 0,95

list_corr_features = []



for i in corr.index:

    for j in corr.columns:

        if corr.loc[i,j] > 0.95:

            list_corr_features.append([i,j,corr.loc[i,j]])

print('list_corr_features первые 10 штук: ', list_corr_features[1:10])



# переведем список в датафрейм

list_corr_features_1 = []

list_corr_features_2 = []

list_corr_features_3 = []



for i in range(len(list_corr_features)):

    list_corr_features_1.append(list_corr_features[i][0])

    list_corr_features_2.append(list_corr_features[i][1])

    list_corr_features_3.append(list_corr_features[i][2])



df_corr_features = pd.DataFrame(

    {'feature1': list_corr_features_1,

     'feature2': list_corr_features_2,

     'corr_coef': list_corr_features_3,

    })



##############

# найдем дубли

df_corr_features['feat1+feat2'] = df_corr_features['feature1']+' '+df_corr_features['feature2']

df_corr_features['feat2+feat1'] = df_corr_features['feature2']+' '+df_corr_features['feature1']

# print(df_corr_features[df_corr_features['corr_coef'] != 1])



# нам надо найти строки попарно, где feat1+feat2 = feat2+feat1 и разделить их как дубли

    

rows_to_del = []

for i in df_corr_features.index:

    for j in df_corr_features.index:

        if df_corr_features['feat1+feat2'][i] == df_corr_features['feat2+feat1'][j]:

            if i != j:

                rows_to_del.append(j)

rows_to_del



# теперь возьмем все фичи из колонки feature2 (но можно из feature1, это не важно) и фильтр на строки rows_to_del, их хотели удалять

# строки не будем удалять, вместо этого возьмем из них сразу фичи, которые имеют высокую корреляцию с другими, поэтому от них можно избавиться

features_to_del = set(df_corr_features['feature2'].loc[rows_to_del].values)

features_to_del





###############

# # удалим признаки из списка train features чтобы обучить модель

# print('features_to_del after corr(): ', features_to_del, 'len(features_to_del): ', len(features_to_del))



# train_features = list(train_features)

# train_features = [i for i in train_features if i not in features_to_del]

# print('\n', 'длина списка train_features после удаления features_to_del: ', len(train_features),'\n','длина списка features_to_del: ',  len(features_to_del), '\n', train_features, '\n', features_to_del,)





# # удаление коррелирующих дает незначительный прирост в моделях



# # # ################## Mutual Information

# # # чем выше скор, тем сильнее влияет признак на таргет

# # mi = mutual_info_classif(x_train, y_train)

# # mi = pd.Series(mi)

# # print(mi.describe())

# # mi.index = x_train.columns

# # top50mi = mi.sort_values(ascending=False).head(50)

# # mi[top50mi.index.values].sort_values(ascending=False).plot.bar(figsize=(20, 8))

# # mi.describe()

# # mi.head()



# # # использование MI не дает прироста в моделях





# lgb20 = LGBMClassifier(n_estimators = 400,

#                        random_state = 94,

#                        lambda_l1 = 5,

#                        learning_rate = 0.07,

#                        feature_fraction = 0.5,

#                        bagging_fraction = 1,

#                        importance_type='gain')



# lgb20.fit(x_train[train_features], y_train)



# cv = cross_val_score(lgb20,

#                      x_train[train_features],

#                      y_train,

#                      cv=5,

#                      scoring = 'roc_auc'

#                     )



# print('cross val score after lgb20 with features removed: ', cv.mean())





df_corr_features[df_corr_features['corr_coef'] < 1]



df_corr_features[df_corr_features['corr_coef'] < 1]['feature1'].value_counts()

# df_corr_features['feature2'].values_counts()



conc = pd.concat([df_corr_features[df_corr_features['corr_coef'] < 1]['feature1'].value_counts(),

                  df_corr_features[df_corr_features['corr_coef'] < 1]['feature2'].value_counts()],

                  axis = 0)

conc = conc.reset_index(drop = False)



conc.columns = ['feat', 'quantity']

conc_gr = conc.groupby(by = 'feat', axis = 0).sum()

conc_gr.sort_values(by = 'quantity', ascending = False)



corr_feat_to_remove = [i for i in conc_gr.index if conc_gr['quantity'][i] >= 4]

corr_feat_to_remove



train_features = [i for i in train_features if i not in corr_feat_to_remove]



lgb20 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       lambda_l1 = 5,

                       learning_rate = 0.07,

                       feature_fraction = 0.5,

                       bagging_fraction = 1,

                       importance_type='gain')



lgb20.fit(x_train[train_features], y_train)



cv = cross_val_score(lgb20,

                     x_train[train_features],

                     y_train,

                     cv=5,

                     scoring = 'roc_auc'

                    )



print('было 0.7529159048596458, стало после удаления  corr_feat_to_remove: ', cv.mean())







# !pip install rfpimp

# from rfpimp import *



# # вычисляем матрицу взаимозависимостей признаков, значения - это 

# # пермутированные важности признаков,с помощью которых мы пытаемся 

# # предсказать интересующий признак

# D = feature_dependence_matrix(x_train[train_features], sort_by_dependence=True)

# viz = plot_dependence_heatmap(D, figsize=(18, 18))

# viz.view()



# # TypeError: _generate_unsampled_indices() missing 1 required positional argument: 'n_samples_bootstrap'

from lightgbm import LGBMModel,LGBMClassifier





# применяем лучшие параметры к модели

lgb20 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       importance_type = 'gain',

                       learning_rate = 0.07, 

                       lambda_l1 = 5, 

                       feature_fraction = 0.5, 

                       bagging_fraction = 1)



# ищем важности признаков

cv20 = cross_validate(lgb20,

                      x_train[train_features],

                      y_train,

                      cv = 5,

                      return_estimator = True,

                      scoring = 'roc_auc',

                      verbose = 1)



cv20['test_score'].mean()



# записываем важности в датафрейм



fi20 = []



# добавим все feature_importances_ в датафрейм



for i in range(len(cv20['estimator'])):

    fi20.append(cv20['estimator'][i].feature_importances_) 

    

fi20_pd = pd.DataFrame(index = x_train[train_features].columns.values, columns = range(len(fi20)))



for i in fi20_pd.columns:

    fi20_pd[i] = fi20[i]

    

# for i in range(fi20_pd.shape[0]):

#     fi20_pd['mean'][i] = fi20_pd.iloc[i][0:5].mean()

fi20_pd['mean'] = fi20_pd.mean(axis = 1)



pict_data = fi20_pd['mean'].sort_values(ascending = False)



plt.figure(figsize = (15,14))

plt.barh(pict_data.index, pict_data, color='g', height=0.4, align='center', alpha=0.4)

# plt.yticks(fi20_pd['mean'])

# plt.xlabel(lgb21_best_feat.index)



# You can control the size of the figure using plt.figure (e.g., plt.figure(figsize = (6,12)))





# оставим только ТОП300 признаков и из них уже будем удалять плохие

# отсортируем по возрастанию важности признаков, вверху будут самые слабые, их начнем удалять по одному



lgb20_test_feat = fi20_pd['mean'].sort_values(ascending = False).index 



lgb20_test_feat




# # теперь будем удалять по одному признаку и считать модель скор на каждой итерации



# # для этого создаем три списка:

# # 1 - для удаляемых признаков 2 - для скоров 3 - для подсчета разницы скоров



# # 1 - для удаляемых признаков

# feature_to_remove = []

# # 2 - для скоров

# feature_to_remove_auc_score = []

# # 3 - для подсчета разницы скоров

# feature_to_remove_auc_score_diff = []



# # train_features = x_train.dtypes[x_train.dtypes != 'object'].index.values



# count = 1



# auc_score_all = cv20['test_score'].mean()

# feature_to_remove_auc_score.append(auc_score_all)





# # запишем все в датафрейм, для него создадим отдельные списки

# features_for_df = []

# scores_for_df1_after = []

# scores_for_df2_before = []

# diff_for_df = []



# # len(lgb20_test_feat)

# for i in range(2):

    

#     feature_to_remove.append(lgb20_test_feat[i])

    

#     print('\n', 'счет номер = ', count, '\n', 'удаляем признак ', lgb20_test_feat[i], '\n', 'список feature_to_remove пополнили:', feature_to_remove)



#     model = LGBMClassifier(n_estimators = 400,

#                        random_state = 94,

#                        importance_type = 'gain',

#                        learning_rate = 0.07, 

#                        lambda_l1 = 5, 

#                        feature_fraction = 0.5, 

#                        bagging_fraction = 1)



#     auc_score = cross_val_score(model,

#                                 x_train[lgb20_test_feat].drop(feature_to_remove, axis = 1),

#                                 y_train,

#                                 scoring = 'roc_auc',

#                                 verbose = 1)



# # auc_score_mean - аук модели в цикле на каждой итерации считается свой

#     auc_score_mean = auc_score.mean()



#     diff = auc_score_mean - max(feature_to_remove_auc_score)



    

#     # запишем все в датафрейм, для него создадим отдельные списки



#     features_for_df.append(lgb20_test_feat[i])

#     scores_for_df1_after.append(auc_score_mean)

#     scores_for_df2_before.append(max(feature_to_remove_auc_score))

#     diff_for_df. append(diff)

    

# #     all тоже надо пересчитывать!

#     count = count + 1

      

#     print('auc модели после удаления', '\n', 

#           auc_score_mean, '\n',

#           'auc модели со всеми признаками :', '\n', 

#           max(feature_to_remove_auc_score), '\n')

    

#     feature_to_remove_auc_score.append(auc_score_mean)

    

#     if diff > 0:

#         print('видим, что разница скоров > 0.0001, значит стало лучше, от признака можно избавиться')

        

#     else:

#         print('видим, что разница скоров < 0.0001 или отрицательна, значит стало хуже, признак следует оставить')

#         feature_to_remove.remove(lgb20_test_feat[i])

#         feature_to_remove_auc_score.remove(auc_score_mean)



# # создадим датафрейм, в который будем записывать все значения

# df_feat_to_remove = pd.DataFrame({'features': features_for_df,

#                                   'scores before remove': scores_for_df2_before,

#                                   'scores after remove': scores_for_df1_after,

#                                   'diff': diff_for_df})

# df_feat_to_remove

# feature_to_remove



# found_list = ['v21', 'ID', 'v23', 'v16', 'v32', 'v55']



# train_features = [x for x in train_features if x not in feature_to_remove and x not in found_list]

# train_features





# lgb20 = LGBMClassifier(n_estimators = 400,

#                        random_state = 94,

#                        lambda_l1 = 5,

#                        learning_rate = 0.07,

#                        feature_fraction = 0.5,

#                        bagging_fraction = 1,

#                        importance_type='gain')



# lgb20.fit(x_train[train_features], y_train)



# #############

# # сделаем sub 



# pred = lgb20.predict_proba(df_test[train_features])





# sub = pd.DataFrame()

# sub['ID'] = df_test['ID']

# sub['PredictedProb'] = [i[1] for i in pred]



# current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('date_%Y-%m-%d__time_%H-%M')

# flname_sub = 'submission__' + current_time +'.csv' # submission file name



# sub.to_csv(flname_sub, index = False)



# # скор незначительно улучшился - было / стало :: 0.46956 / 0.46984 / 0.46722 / 0.46728





# # ЖАДНЫЙ ОТБОР,  список feature_to_remove пополнили: ['v21', 'ID', 'v23', 'v16', 'v32', 'v55']







# попробуем другой подход

# возьмем ТОП20 фичей по важности и начнем добавлять по одной фиче 



lgb20_test_feat_top = list(fi20_pd['mean'].sort_values(ascending = False)[0:20].index)

lgb20_test_feat_add = list(fi20_pd['mean'].sort_values(ascending = False)[21:].index)





# применяем лучшие параметры к модели

lgb20 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       importance_type = 'gain',

                       learning_rate = 0.07, 

                       lambda_l1 = 5, 

                       feature_fraction = 0.5, 

                       bagging_fraction = 1)



# ищем важности признаков

cv20 = cross_validate(lgb20,

                      x_train[lgb20_test_feat_top],

                      y_train,

                      cv = 5,

                      return_estimator = True,

                      scoring = 'roc_auc',

                      verbose = 1)



cv20['test_score'].mean()





# теперь будем добавлять по одному признаку и считать модель скор на каждой итерации



# для этого создаем три списка:

# 1 - для удаляемых признаков 2 - для скоров 3 - для подсчета разницы скоров



# 1 - для удаляемых признаков

feature_to_remove = []

# 2 - для скоров

feature_to_remove_auc_score = []

# 3 - для подсчета разницы скоров

feature_to_remove_auc_score_diff = []



# train_features = x_train.dtypes[x_train.dtypes != 'object'].index.values



count = 1



auc_score_all = cv20['test_score'].mean()

feature_to_remove_auc_score.append(auc_score_all)





# запишем все в датафрейм, для него создадим отдельные списки

features_for_df = []

scores_for_df1_after = []

scores_for_df2_before = []

diff_for_df = []



# len(lgb20_test_feat)

for i in range(len(lgb20_test_feat_add)):

    

    lgb20_test_feat_top.append(lgb20_test_feat_add[i])

    

    print('\n', 'счет номер = ', count, '\n', 'добавляем признак ', lgb20_test_feat_add[i], '\n')

    

    model = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       importance_type = 'gain',

                       learning_rate = 0.07, 

                       lambda_l1 = 5, 

                       feature_fraction = 0.5, 

                       bagging_fraction = 1)



    auc_score = cross_val_score(model,

                                x_train[lgb20_test_feat_top],

                                y_train,

                                scoring = 'roc_auc',

                                verbose = 1)



# auc_score_mean - аук модели в цикле на каждой итерации считается свой

    auc_score_mean = auc_score.mean()



    diff = auc_score_mean - max(feature_to_remove_auc_score)



    

    # запишем все в датафрейм, для него создадим отдельные списки



    features_for_df.append(lgb20_test_feat_add[i])

    scores_for_df1_after.append(auc_score_mean)

    scores_for_df2_before.append(max(feature_to_remove_auc_score))

    diff_for_df.append(diff)

    

#     all тоже надо пересчитывать!

    count = count + 1

      

    print('auc модели после добавления', '\n', 

          auc_score_mean, '\n',

          'максимальный auc модели со до добавления :', '\n', 

          max(feature_to_remove_auc_score), '\n')

    

    feature_to_remove_auc_score.append(auc_score_mean)

    

    if diff > 0.0003:

        print('видим, что разница скоров > 0.0003, значит стало лучше, от признак можно оставить, финальный список:', lgb20_test_feat_top)

        

    else:

        print('видим, что разница скоров < 0.0003 или отрицательна, значит стало хуже, признак следует убрать')

        lgb20_test_feat_top.remove(lgb20_test_feat_add[i])

        feature_to_remove_auc_score.remove(auc_score_mean)



# создадим датафрейм, в который будем записывать все значения

# df_feat_to_add = pd.DataFrame({'features': lgb20_test_feat_top,

#                                   'scores before remove': scores_for_df2_before,

#                                   'scores after remove': scores_for_df1_after,

#                                   'diff': diff_for_df})







train_features = [x for x in lgb20_test_feat_top]

train_features





lgb20 = LGBMClassifier(n_estimators = 400,

                       random_state = 94,

                       lambda_l1 = 5,

                       learning_rate = 0.07,

                       feature_fraction = 0.5,

                       bagging_fraction = 1,

                       importance_type='gain')



lgb20.fit(x_train[train_features], y_train)



#############

# сделаем sub 



pred = lgb20.predict_proba(df_test[train_features])





sub = pd.DataFrame()

sub['ID'] = df_test['ID']

sub['PredictedProb'] = [i[1] for i in pred]



current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('date_%Y-%m-%d__time_%H-%M')

flname_sub = 'submission__' + current_time +'.csv' # submission file name



sub.to_csv(flname_sub, index = False)



# скор незначительно улучшился - было / стало :: 0.46956 / 0.46984 / 0.46722 / 0.46728





# ЖАДНЫЙ ОТБОР,  список feature_to_remove пополнили: ['v21', 'ID', 'v23', 'v16', 'v32', 'v55']



###################



# лог регрессия + power transform



# logreg = LogisticRegression()

# logreg.fit(x_train_pt, y_train)

# log_pred = logreg.predict(x_test_pt)

# print(accuracy_score(y_test, log_pred)) # 0.777123363559495

# print(roc_auc_score(y_test, log_pred)) #0.5827874126253638





###################

# лог регрессия



# logreg = LogisticRegression(verbose=1)

# logreg.fit(x_train, y_train)

# log_pred = logreg.predict(x_test)

# print(accuracy_score(y_test, log_pred)) #0.7625156719246581

# print(roc_auc_score(y_test, log_pred))



# print(cross_val_score(logreg, df_x, df_y, cv=15))





###################

# дерево решеений



# clf = tree.DecisionTreeClassifier()

# clf.fit(x_train[train_features], y_train)

# tree_pred = clf.predict(x_test[train_features])

# print(accuracy_score(y_test, tree_pred)) # 0.6966206956876695

# print(roc_auc_score(y_test, tree_pred)) #0.5861745101264721

# print(cross_val_score(clf, df_x, df_y, cv=15))





###################

# кнн



# knn = KNeighborsClassifier(n_neighbors=10)

# knn.fit(x_train, y_train)

# knn_pred = knn.predict(x_test)

# print(accuracy_score(y_test, knn_pred)) # 0.730646995364026

# print(roc_auc_score(y_test, knn_pred)) #0.5028577536322103



###################

# кнн + PT



# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(x_train_pt, y_train)

# knn_pred = knn.predict(x_test_pt)

# print(accuracy_score(y_test, knn_pred)) # 0.730646995364026

# print(roc_auc_score(y_test, knn_pred)) #0.4991711748494437



# # >>>>>>>>>>>>>>>>>>



# ###################

# # случайный лес



# clfRF = RandomForestClassifier(max_depth=7, random_state=0, verbose=1)

# clfRF.fit(x_train[train_features], y_train)

# clfRF_pred = clfRF.predict(x_test[train_features])

# print(accuracy_score(y_test, clfRF_pred)) # 0.7625448289937895

# print(roc_auc_score(y_test, clfRF_pred)) #0.5

# # print(cross_val_score(clfRF, df_x, df_y, cv=15))



# # визуализируем важные признаки

# features = train_features

# importances = clfRF.feature_importances_[0:20]

# indices = np.argsort(importances)



# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# pred = clfRF.predict(df_test)



# # собираем sub

# # Lgb

# sub = pd.DataFrame()

# sub['ID'] = df_test['ID']

# sub['PredictedProb'] = pred

# sub.to_csv('submit_baseline6.csv', index = False)



# print('clfRF importances: \n', importances, '\n', 'feat indices: \n', features[indices])



# # print('clfRF features importance: \n', features[indices], '\n', 'VS MI - mi[top50mi.index.values]: \n', mi[top50mi.index.values])



# # >>>>>>>>>>>>>>>>>>





###################

# случайный лес + PT



# clfRF1 = RandomForestClassifier(max_depth=7, random_state=12)

# clfRF1.fit(x_train_pt, y_train)

# clfRF_pred1 = clfRF1.predict(x_test_pt)

# print(accuracy_score(y_test, clfRF_pred1)) # 0.7625156719246581

# print(roc_auc_score(y_test, clfRF_pred1)) #0.5

# print(cross_val_score(clfRF1, x_train_pt, y_train, cv=15))







# ###################

# # бустинг



# Lgb = LGBMClassifier(n_estimators=150, silent=False, random_state =94, max_depth=5,num_leaves=31,objective='binary')

# Lgb.fit(x_train, y_train)

# Lgb_pred = Lgb.predict_proba(x_test)

# Lgb_pred

# # print(accuracy_score(y_test, [i[1] for i in Lgb_pred])) 

# print(roc_auc_score(y_test, [i[1] for i in Lgb_pred])) #0.7515020019901595

# # print(roc_auc_score(y_test, Lgb_pred)) #0.7515020019901595





# # cross_val_score

# # print(cross_val_score(Lgb, x_train, y_train, cv=5))

# # [0.73181719 0.7824965  0.77982855 0.78070329 0.78223408]





###################

# бустинг + PT



# Lgb = LGBMClassifier(n_estimators=90, silent=False, random_state = 94, max_depth = 5, num_leaves = 31, objective = 'binary')

# Lgb.fit(x_train_pt, y_train)

# Lgb_pred = Lgb.predict_proba(x_test_pt)

# Lgb_pred

# # print(accuracy_score(y_test, [i[1] for i in Lgb_pred])) 

# print(roc_auc_score(y_test, [i[1] for i in Lgb_pred])) #0.7504605882482182



# cross_val_score

# df_pt = pd.concat(x_train_pt, x_test_pt, axis = 0) 

# df_target = pd.concat(y_train, y_test, axis = 0)

# print(cross_val_score(Lgb, df_pt, df_target, cv=5))

# [0.73181719 0.7824965  0.77982855 0.78070329 0.78223408]







# pred = Lgb.predict_proba(df_test[train_features])

# import os

# import glob

# import datetime

# import dateutil.tz



# # собираем sub

# # get the current time and add it to the submission filename, helps to keep track of submissions

# current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('date_%Y-%m-%d__time_%H-%M')

# flname_sub = 'submission__' + current_time +'.csv' # submission file name





# # Lgb

# sub = pd.DataFrame()

# sub['ID'] = df_test['ID']

# sub['PredictedProb'] = pred

# sub.to_csv(flname_sub, index = False)



# # # собираем sub

# # # Lgb

# # sub = pd.DataFrame()

# # sub['ID'] = df_test['ID']

# # sub['PredictedProb'] = pred

# # sub.to_csv('clfRF_submit_baseline3.csv', index = False)

# # проверим есть ли дисбаланс и попробуем избавиться от дисбаланса данных и привести их размерности к одному уровню



# # Shuffle the Dataset.

# shuffled_df = df.sample(frac=1,random_state=4)



# print('df : ', df['target'].value_counts())

# print('shuffled_df : ', shuffled_df['target'].value_counts())

# # Put all the fraud class in a separate dataset.

# df_one = shuffled_df.loc[shuffled_df['target'] == 1].sample(n=27300, random_state=42)

# print('df_one : ', df_one['target'].value_counts())



# #Randomly select 492 observations from the non-fraud (majority class)

# df_zero = shuffled_df.loc[shuffled_df['target'] == 0]

# print('df_zero : ', df_zero['target'].value_counts())



# # Concatenate both dataframes again

# normalized_df = pd.concat([df_zero, df_one])



# print('normalized_df : ', normalized_df['target'].value_counts())
    

# # надо взять каждое значение в колонке v22 (это можно сделать через val counts)

# # 1 посмотреть число таких значений --> получим столбец v22_numb_total

# v22_numb_total = pd.DataFrame(x_train['v22'].value_counts().reset_index())

# v22_numb_total.columns = ['v22', 'v22_numb_total_count']

# # print(v22_numb_total)



# # 1.1 записать v22_numb_total в x_train через merge

# # вместо merge лучше использовать .join и .set_index, тогда можно сразу избавиться от дублей

# # пример: df.set_index('key').join(other.set_index('key'))



# x_train = x_train.set_index('v22').join(v22_numb_total.set_index('v22'))





# # x_train[['v22','v22_numb_total_count']].fillna(0)

# print(x_train)

# # x_train.shape, x_test.shape

# # что делает reset index

# # v22 который был индексом останется в датасете, индекс сбросится и перейдет к новому значению

# x_train = x_train.reset_index()



# # что делает set index

# # старый индекс сбросится и удалится и перейдет к столбцу ID. столбец изначльный ID удалится, v22 останется

# x_train = x_train.set_index('ID',drop = False)



# print(x_train.index,

#       x_train.columns[x_train.columns == 'v22'],

#       x_train.columns[x_train.columns == 'ID'],

#       x_train.columns,

#       x_train.shape)



# # v22 должен остаться, он понадобится для следующего сопоставления




# # 2 посмотреть число таких значений при y=1 --> получим столбец v22_numb_y1





# x_train['target'] = y_train



# v22_numb_y1 = pd.DataFrame(x_train['v22'][x_train['target'] == 1].value_counts().reset_index())





# # делаем через .join - пример: df.set_index('key').join(other.set_index('key'))





# v22_numb_y1.columns = ['v22', 'v22_numb_y1']

# v22_numb_y1



# # v22 должен остаться, он понадобится для следующего сопоставления (c test)

# x_train = x_train.set_index('v22', drop = False).join(v22_numb_y1.set_index('v22', drop = True))



# # колонку ID нужно обратно вернуть в индекс



# x_train = x_train.set_index('ID', drop = True)



# # проверяем, что колонки ID нет, колонка v22 есть



# print(x_train.index,

#       x_train.columns[x_train.columns == 'v22'],

#       x_train.columns[x_train.columns == 'ID'],

#       x_train.columns,

#       x_train.shape)





# # теперь надо поделить v22_numb_y1 на v22_numb_total_count, чтобы получить вероятность, получим prob_v22_numb_y1



# x_train['prob_v22_numb_y1'] = x_train['v22_numb_y1'] / x_train['v22_numb_total_count']

# # x_train['prob_v22_numb_y1'] = x_train['prob_v22_numb_y1'].fillna(0)




