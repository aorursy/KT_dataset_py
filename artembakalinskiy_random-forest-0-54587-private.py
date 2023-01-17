import numpy as np 

import pandas as pd



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import f1_score as f1



from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 150)



import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import make_scorer, mean_squared_error

from scipy.stats import ttest_rel

from statsmodels.stats.weightstats import zconfint, _tconfint_generic



# расширенная кросс-валидация + проверка гипотезы равенства средних(ttest_rel)

# чтобы получить достаточную выборку для сравнения средних на кросс-валидации, с учетом того, что малые классы и так слишком малы

# поэтому делаю 3 фолда n раз



def cross_val_ttest(model, X, y, prev_scores = None, k_folds = 3, n = 6):

    if prev_scores is not None: 

        assert len(prev_scores) == k_folds * n

    

    scores_test = np.array([])

    for i in range(n):

        fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=i)

        scorer = make_scorer(f1, average = 'macro')



        scores_test_on_this_split = cross_val_score(estimator = model, X=X, y=y, cv=fold, scoring=scorer)

        scores_test = np.append(scores_test, scores_test_on_this_split)



        print('------ step ', i,' ------')



    if prev_scores is not None: return scores_test, ttest_rel(scores_test, prev_scores)

    else: return scores_test

    

# доверительный интервал на базе распределения Стьюдента, для оценки 

def tconfint(scores, alpha = 0.05):

    mean = scores.mean()

    scores_std = scores.std(ddof=1)/np.sqrt(len(scores))

    

    return _tconfint_generic(mean, scores_std, len(scores) - 1, alpha, 'two-sided')



# быстрый тест для разовой оценки эффекта, либо предсказание для тестовой выборки

def single_test(model, X_train, Y, X_test = None, subm_name = None):



    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y, test_size = 0.3, shuffle = True, random_state = 10, stratify = Y)

    model.fit(x_train, y_train)

    pred_train = model.predict(x_valid)



    if X_test is not None:

        pred = model.predict(X_test)

        return pred

    

    return f1(pred_train, y_valid, average = 'macro')
train = pd.read_csv('/kaggle/input/mf-accelerator/contest_train.csv')

test = pd.read_csv('/kaggle/input/mf-accelerator/contest_test.csv')

data = pd.concat([train, test], ignore_index=True)
# по таким графикам изучил все фичи



cols = data.columns[2:]

def jitter(arr):

    return arr + np.random.uniform(low=-0.25, high=0.25, size=len(arr))



plt.figure(figsize = (10, 5))

    

plt.subplot(121)

plt.scatter(jitter(train.TARGET), jitter(train[cols[55]]),  edgecolors="black")

plt.xlabel(cols[55])

plt.xticks([0,1,2])



plt.subplot(122)

sns.boxplot(y=train[cols[55]])

plt.xlabel('boxplot')
# Визуально определил категориальные признаки, оценил что из себя представляют фичи, "на глаз" выбросил фичи, которые были равномерно распределены между классами 

# (проверяя при этом не упал ли показатель на бейзлайне)
# не буду перегружать всем анализом корреляций и отбором фичей - это в другом ноутбуке

# здесь - группы признаков оставшиеся после анализа статистически значимых корреляций между фичами:

# (проверка корреляций, поправка на множественную проверку гипотез, отбор)



# вещественные, целые, бинарные, категориальные, с пропусками

r_cols = ['FEATURE_21', 'FEATURE_225', 'FEATURE_102', 'FEATURE_110', 'FEATURE_24', 'FEATURE_139', 'FEATURE_83', 'FEATURE_252', 'FEATURE_111', 'FEATURE_90', 'FEATURE_34', 'FEATURE_57', 'FEATURE_147', 'FEATURE_82', 'FEATURE_76', 'FEATURE_97', 'FEATURE_86', 'FEATURE_51', 'FEATURE_77', 'FEATURE_170', 'FEATURE_58', 'FEATURE_23', 'FEATURE_108', 'FEATURE_114', 'FEATURE_7', 'FEATURE_168', 'FEATURE_103', 'FEATURE_169', 'FEATURE_47', 'FEATURE_95', 'FEATURE_141', 'FEATURE_61', 'FEATURE_113', 'FEATURE_46', 'FEATURE_59', 'FEATURE_62', 'FEATURE_67', 'FEATURE_65', 'FEATURE_105', 'FEATURE_89', 'FEATURE_143', 'FEATURE_240', 'FEATURE_99', 'FEATURE_228', 'FEATURE_68', 'FEATURE_50', 'FEATURE_52', 'FEATURE_64', 'FEATURE_244', 'FEATURE_107', 'FEATURE_217', 'FEATURE_101', 'FEATURE_26', 'FEATURE_94', 'FEATURE_45', 'FEATURE_33', 'FEATURE_106', 'FEATURE_38', 'FEATURE_80', 'FEATURE_85', 'FEATURE_96', 'FEATURE_109', 'FEATURE_66', 'FEATURE_160', 'FEATURE_79', 'FEATURE_98', 'FEATURE_177', 'FEATURE_25', 'FEATURE_60', 'FEATURE_91', 'FEATURE_87', 'FEATURE_223', 'FEATURE_53', 'FEATURE_186', 'FEATURE_100', 'FEATURE_112', 'FEATURE_35', 'FEATURE_63', 'FEATURE_236', 'FEATURE_104']

z_cols = ['FEATURE_39', 'FEATURE_122', 'FEATURE_30', 'FEATURE_42', 'FEATURE_22', 'FEATURE_179', 'FEATURE_229', 'FEATURE_242', 'FEATURE_27', 'FEATURE_31', 'FEATURE_178', 'FEATURE_142', 'FEATURE_231', 'FEATURE_234', 'FEATURE_146', 'FEATURE_165', 'FEATURE_248', 'FEATURE_180', 'FEATURE_222', 'FEATURE_70', 'FEATURE_28', 'FEATURE_224', 'FEATURE_115', 'FEATURE_40', 'FEATURE_246', 'FEATURE_120', 'FEATURE_221', 'FEATURE_155', 'FEATURE_215', 'FEATURE_32', 'FEATURE_176', 'FEATURE_227', 'FEATURE_154', 'FEATURE_74', 'FEATURE_167', 'FEATURE_250', 'FEATURE_251', 'FEATURE_121', 'FEATURE_216', 'FEATURE_226', 'FEATURE_20', 'FEATURE_198', 'FEATURE_117', 'FEATURE_44', 'FEATURE_199', 'FEATURE_116', 'FEATURE_145', 'FEATURE_243', 'FEATURE_29', 'FEATURE_75', 'FEATURE_202', 'FEATURE_247', 'FEATURE_185', 'FEATURE_0', 'FEATURE_13', 'FEATURE_241', 'FEATURE_197', 'FEATURE_41', 'FEATURE_171', 'FEATURE_201', 'FEATURE_172', 'FEATURE_43']



bi_cols = ['FEATURE_6', 'FEATURE_16', 'FEATURE_2', 'FEATURE_4', 'FEATURE_11', 'FEATURE_159', 'FEATURE_15', 'FEATURE_19', 'FEATURE_17', 'FEATURE_18', 'FEATURE_5', 'FEATURE_140']

cat_cols = ['FEATURE_9', 'FEATURE_157', 'FEATURE_257', 'FEATURE_259', 'FEATURE_214', 'FEATURE_156', 'FEATURE_218', 'FEATURE_258', 'FEATURE_219', 'FEATURE_220', 'FEATURE_10']



miss_cols = ['FEATURE_73', 'FEATURE_213', 'FEATURE_191', 'FEATURE_175', 'FEATURE_192', 'FEATURE_173', 'FEATURE_188', 'FEATURE_134', 'FEATURE_203', 'FEATURE_153', 'FEATURE_8', 'FEATURE_138', 'FEATURE_149', 'FEATURE_211', 'FEATURE_238', 'FEATURE_195', 'FEATURE_212', 'FEATURE_125', 'FEATURE_152', 'FEATURE_210', 'FEATURE_126', 'FEATURE_150', 'FEATURE_206', 'FEATURE_209', 'FEATURE_193', 'FEATURE_124', 'FEATURE_71', 'FEATURE_189', 'FEATURE_137', 'FEATURE_12', 'FEATURE_183', 'FEATURE_194', 'FEATURE_72', 'FEATURE_136', 'FEATURE_196', 'FEATURE_151', 'FEATURE_174', 'FEATURE_181', 'FEATURE_132', 'FEATURE_187', 'FEATURE_129', 'FEATURE_133', 'FEATURE_162', 'FEATURE_239', 'FEATURE_190', 'FEATURE_128', 'FEATURE_130']



useless_cols = ['FEATURE_69', 'FEATURE_88', 'FEATURE_123', 'FEATURE_127', 'FEATURE_131', 'FEATURE_135', 'FEATURE_148', 'FEATURE_204', 'FEATURE_205', 'FEATURE_207', 'FEATURE_208', 'FEATURE_254', 'FEATURE_255']

const_cols = ['FEATURE_3', 'FEATURE_144', 'FEATURE_249', 'FEATURE_256']



catmiss_cols = ['FEATURE_213', 'FEATURE_203', 'FEATURE_211', 'FEATURE_212', 'FEATURE_210', 'FEATURE_206','FEATURE_209']

nummiss_cols = ['FEATURE_134', 'FEATURE_124', 'FEATURE_137', 'FEATURE_73', 'FEATURE_125', 'FEATURE_175', 'FEATURE_181', 'FEATURE_195', 'FEATURE_71', 'FEATURE_132', 'FEATURE_196', 'FEATURE_136', 'FEATURE_151', 'FEATURE_133', 'FEATURE_129', 'FEATURE_239', 'FEATURE_153', 'FEATURE_192', 'FEATURE_189', 'FEATURE_173', 'FEATURE_12', 'FEATURE_138', 'FEATURE_8', 'FEATURE_193', 'FEATURE_190', 'FEATURE_174', 'FEATURE_162', 'FEATURE_72', 'FEATURE_130', 'FEATURE_183', 'FEATURE_191', 'FEATURE_128', 'FEATURE_149', 'FEATURE_152', 'FEATURE_188', 'FEATURE_194', 'FEATURE_187', 'FEATURE_150', 'FEATURE_126', 'FEATURE_238']
# precomputed baseline score estimation

# зафиксировал значения кросс-валидации на бейзлайне для сравнения с новыми средними через cross_val_ttest



test_baseline = np.array([0.49247369, 0.49354152, 0.48317658, 0.49637811, 0.48853664,

       0.48700242, 0.4849887 , 0.48657228, 0.49926884, 0.49285571,

       0.48597107, 0.49468759, 0.4903521 , 0.49272306, 0.49119276,

       0.49750482, 0.48570297, 0.48399252])

tconfint(test_baseline)
# поскольку пропуски будут восстанавливаться регрессией, логарифмирую и масштабирую данные, т.к. выбросы по признакам черезчур велики

# собираю трансформированный датасет в новый датафрейм



from sklearn.preprocessing import StandardScaler



current_cols = r_cols + z_cols + nummiss_cols

data_scaled = data[current_cols]



# сдвинем минимум по фичам в "1" для удобного логарифмирования

for col in current_cols:

    data_scaled[col] = data_scaled[col]-data_scaled[col].min()+1



data_scaled = np.log(data_scaled[current_cols])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled), columns = current_cols).join(data[bi_cols])



# ohe на категориальные признаки

for col in cat_cols:

    ohe = pd.get_dummies(data[col], prefix = col)

    data_scaled = data_scaled.join(ohe)

    bi_cols = bi_cols + ohe.columns.tolist()



# наконец добавим категориальные признаки, пока что без изменений

data_scaled = data_scaled.join(data[catmiss_cols])
# обычная регрессия для восстановления числовых фичей - грубо, но быстро



from sklearn.linear_model import Ridge



for col in nummiss_cols:

    test_id = data_scaled[col].isna()

    train_id = test_id == False



    model = Ridge(random_state = 1)



    X_train = data_scaled[bi_cols+r_cols+z_cols][train_id]

    Y = data_scaled[col][train_id]



    model.fit(X_train, Y)

    print(mean_squared_error(model.predict(X_train), Y))



    data_scaled[col][test_id] = model.predict(data_scaled[bi_cols+r_cols+z_cols][test_id])
# лес - для категориальных



for col in catmiss_cols:

    test_id = data_scaled[col].isna()

    train_id = test_id == False



    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state = 1, class_weight = 'balanced')



    X_train = data_scaled[bi_cols+r_cols+z_cols+nummiss_cols][train_id]

    Y = data_scaled[col][train_id]



    model.fit(X_train, Y)

    print(f1(model.predict(X_train), Y, average = 'macro'))



    data_scaled[col][test_id] = model.predict(data_scaled[bi_cols+r_cols+z_cols+nummiss_cols][test_id])
# и тоже переводим их в ohe



for col in catmiss_cols:

    ohe = pd.get_dummies(data_scaled[col], prefix = col)

    data_scaled = data_scaled.join(ohe)

    bi_cols = bi_cols + ohe.columns.tolist()

    data_scaled.drop([col], axis = 1, inplace = True)
%%time



preds = np.zeros((10, 3, 9484))



for i in range(10):

    pred = single_test(model = RandomForestClassifier(n_estimators=200, max_depth=11, min_samples_leaf = 6, random_state=i, class_weight = 'balanced'), 

                X_train = data_scaled.iloc[:24521], 

                Y = train['TARGET'], 

                X_test = data_scaled.iloc[24521:])

                

    preds[i][0] = pred == 0

    preds[i][1] = pred == 1

    preds[i][2] = pred == 2

    

    print(i)

    

subm = pd.DataFrame({'ID': test.ID, 'Predicted': np.argmax(preds.sum(axis = 0), axis = 0)})

subm.to_csv('rfc103-11-6_cols_scaled_ohe_nocorr.csv', index=False)
# scores_test = []

# for i in range(10): 

#     scores_test.append(cross_val_ttest(model = RandomForestClassifier(n_estimators=200, max_depth=11, min_samples_leaf = 6, random_state=i, class_weight = 'balanced'), 

#                 X = data_scaled.iloc[:24521], 

#                 y = train['TARGET'],

#                 k_folds = 3, n = 3))
# np.vstack(scores_test).mean(axis = 1)
# single_test(model = RandomForestClassifier(n_estimators=200, max_depth=11, min_samples_leaf = 6, random_state=1, class_weight = 'balanced'), 

#                 X_train = data_scaled.iloc[:24521], 

#                 Y = train['TARGET']) 

#               #  X_test = data_scaled.iloc[24521:])
# from catboost import CatBoostClassifier, Pool

# import xgboost as xgb

# from lightgbm import LGBMClassifier

# from sklearn.utils.class_weight import compute_class_weight
# CLASS_WEIGHTS = compute_class_weight(class_weight='balanced', classes=[0,1,2], y=train['TARGET'])



# pred = single_test(model = CatBoostClassifier(random_state=10, 

#                                        verbose = 0, 

#                                        class_weights=CLASS_WEIGHTS, 

#                                        custom_loss=['TotalF1'],

#                                        loss_function = 'MultiClassOneVsAll'), 

#                 X_train = data_scaled.iloc[:24521], 

#                 Y = train['TARGET'],

#                 X_test = data_scaled.iloc[24521:]) 



# subm = pd.DataFrame({'ID': test.ID, 'Predicted': pred.reshape(-1)})

# subm.to_csv('cat_scaled_ohe.csv', index=False)