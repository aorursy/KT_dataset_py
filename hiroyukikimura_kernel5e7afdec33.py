from sklearn.metrics import mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import GroupKFold, KFold

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMRegressor

from lightgbm import LGBMClassifier
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_columns = None

pd.options.display.max_rows =  50
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
df_train = pd.read_csv('../input/exam-for-students20200129/train.csv')

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv')

df_union = pd.read_csv('../input/exam-for-students20200129/country_info.csv')
# 商品マスタを学習データと結合する

df_train = df_train.merge(df_union, on='Country', how='left')
df_train.ConvertedSalary.hist()
df_train = df_train.drop(df_train[(df_train['GDP ($ per capita)']>50000)].index)

df_train = df_train.drop(df_train[(df_train['ConvertedSalary']>800000)].index)

test_train = df_train



df_train.ConvertedSalary.hist()
test_train = df_train



df_train["ConvertedSalary"] = np.log1p(df_train["ConvertedSalary"])
df_train.columns
df_train.corr()
df_test.describe()
object_cats = []

non_object_cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        object_cats.append(col)        

        print('object_' + col, df_train[col].nunique())

    else:

        non_object_cats.append(col)

        print('non_object_' + col, df_train[col].nunique())
y_train = df_train['ConvertedSalary'].reset_index(drop=True)
df_train.drop(['Respondent'], axis=1, inplace=True)

df_test.drop(['Respondent'], axis=1, inplace=True)
df_train.head()
X_test = df_test

X_train = df_train.drop(['ConvertedSalary'], axis=1)
# カテゴリカル項目のFEをするため、X_trainとX_testを一緒にして後で再分割する

X_train['distinct'] = 'train'

X_test['distinct'] = 'test'

X_train_test = pd.concat([X_train,X_test])

X_train.shape
X_train.head()


# これ何かに使えないか？検討

def cat_reverse(cat):

    if cat == '1.0':

        return 12

    if cat == '2.0':

        return 11

    if cat == '3.0':

        return 10

    if cat == '4.0':

        return 9

    if cat == '5.0':

        return 8

    if cat == '6.0':

        return 7

    if cat == '5.0':

        return 6

    if cat == '7.0':

        return 5

    if cat == '8.0':

        return 4

    if cat == '9.0':

        return 3

    if cat == '10.0':

        return 2

    else:

        return 1

    
X_train_test['AssessJob1'] = X_train_test['AssessJob1'].apply(cat_reverse)

X_train_test['AssessJob2'] = X_train_test['AssessJob2'].apply(cat_reverse)

X_train_test['AssessJob3'] = X_train_test['AssessJob3'].apply(cat_reverse)

X_train_test['AssessJob4'] = X_train_test['AssessJob4'].apply(cat_reverse)

X_train_test['AssessJob5'] = X_train_test['AssessJob5'].apply(cat_reverse)

X_train_test['AssessJob6'] = X_train_test['AssessJob6'].apply(cat_reverse)

X_train_test['AssessJob7'] = X_train_test['AssessJob7'].apply(cat_reverse)

X_train_test['AssessJob8'] = X_train_test['AssessJob8'].apply(cat_reverse)

X_train_test['AssessJob9'] = X_train_test['AssessJob9'].apply(cat_reverse)

X_train_test['AssessJob10'] = X_train_test['AssessJob10'].apply(cat_reverse)
X_train_test
# これ何かに使えないか？検討

def cat_reverse2(cat):

    if cat == '1.0':

        return 13

    if cat == '2.0':

        return 12

    if cat == '3.0':

        return 11

    if cat == '4.0':

        return 10

    if cat == '5.0':

        return 9

    if cat == '6.0':

        return 8

    if cat == '5.0':

        return 7

    if cat == '7.0':

        return 6

    if cat == '8.0':

        return 5

    if cat == '9.0':

        return 4

    if cat == '10.0':

        return 3

    if cat == '11.0':

        return 2

    else:

        return 1
X_train_test['AssessBenefits1'] = X_train_test['AssessBenefits1'].apply(cat_reverse2)

X_train_test['AssessBenefits2'] = X_train_test['AssessBenefits2'].apply(cat_reverse2)

X_train_test['AssessBenefits3'] = X_train_test['AssessBenefits3'].apply(cat_reverse2)

X_train_test['AssessBenefits4'] = X_train_test['AssessBenefits4'].apply(cat_reverse2)

X_train_test['AssessBenefits5'] = X_train_test['AssessBenefits5'].apply(cat_reverse2)

X_train_test['AssessBenefits6'] = X_train_test['AssessBenefits6'].apply(cat_reverse2)

X_train_test['AssessBenefits7'] = X_train_test['AssessBenefits7'].apply(cat_reverse2)

X_train_test['AssessBenefits8'] = X_train_test['AssessBenefits8'].apply(cat_reverse2)

X_train_test['AssessBenefits9'] = X_train_test['AssessBenefits9'].apply(cat_reverse2)

X_train_test['AssessBenefits10'] = X_train_test['AssessBenefits10'].apply(cat_reverse2)

X_train_test['AssessBenefits11'] = X_train_test['AssessBenefits11'].apply(cat_reverse2)
X_train_test['JobContactPriorities1'] = X_train_test['JobContactPriorities1'].apply(cat_reverse2)

X_train_test['JobContactPriorities2'] = X_train_test['JobContactPriorities2'].apply(cat_reverse2)

X_train_test['JobContactPriorities3'] = X_train_test['JobContactPriorities3'].apply(cat_reverse2)

X_train_test['JobContactPriorities4'] = X_train_test['JobContactPriorities4'].apply(cat_reverse2)

X_train_test['JobContactPriorities5'] = X_train_test['JobContactPriorities5'].apply(cat_reverse2)
X_train_test['JobEmailPriorities1'] = X_train_test['JobEmailPriorities1'].apply(cat_reverse2)

X_train_test['JobEmailPriorities2'] = X_train_test['JobEmailPriorities2'].apply(cat_reverse2)

X_train_test['JobEmailPriorities3'] = X_train_test['JobEmailPriorities3'].apply(cat_reverse2)

X_train_test['JobEmailPriorities4'] = X_train_test['JobEmailPriorities4'].apply(cat_reverse2)

X_train_test['JobEmailPriorities5'] = X_train_test['JobEmailPriorities5'].apply(cat_reverse2)

X_train_test['JobEmailPriorities6'] = X_train_test['JobEmailPriorities6'].apply(cat_reverse2)

X_train_test['JobEmailPriorities7'] = X_train_test['JobEmailPriorities7'].apply(cat_reverse2)
X_train_test['AdsPriorities1'] = X_train_test['AdsPriorities1'].apply(cat_reverse2)

X_train_test['AdsPriorities2'] = X_train_test['AdsPriorities2'].apply(cat_reverse2)

X_train_test['AdsPriorities3'] = X_train_test['AdsPriorities3'].apply(cat_reverse2)

X_train_test['AdsPriorities4'] = X_train_test['AdsPriorities4'].apply(cat_reverse2)

X_train_test['AdsPriorities5'] = X_train_test['AdsPriorities5'].apply(cat_reverse2)

X_train_test['AdsPriorities6'] = X_train_test['AdsPriorities6'].apply(cat_reverse2)

X_train_test['AdsPriorities7'] = X_train_test['AdsPriorities7'].apply(cat_reverse2)

X_train_test['count_DevType'] = X_train_test['DevType'].map(X_train_test['DevType'].value_counts())

X_train_test['count_CommunicationTools'] = X_train_test['CommunicationTools'].map(X_train_test['CommunicationTools'].value_counts())

X_train_test['count_FrameworkWorkedWith'] = X_train_test['FrameworkWorkedWith'].map(X_train_test['FrameworkWorkedWith'].value_counts())
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

oe = OrdinalEncoder(cols=object_cats, return_df=False)

X_train_test[object_cats] = oe.fit_transform(X_train_test[object_cats])
X_train_test.head()
X_train_test
# X_train_test をX_trainとX_testに再分割

X_train = X_train_test[X_train_test['distinct'] == 'train']

X_test = X_train_test[X_train_test['distinct'] == 'test']



# featrure importance重要項目での特徴量追加

X_train['sub_a'] = X_train['GDP ($ per capita)'] * X_train['Employment']

X_test['sub_a'] = X_test['GDP ($ per capita)'] * X_test['Employment']



X_train['sub_b'] = X_train['GDP ($ per capita)'] * X_train['LastNewJob']

X_test['sub_b'] = X_test['GDP ($ per capita)'] * X_test['LastNewJob']



X_train['sub_c'] = X_train['Employment'] * X_train['LastNewJob']

X_test['sub_c'] = X_test['Employment'] * X_test['LastNewJob']



X_train['sub_d'] = X_train['GDP ($ per capita)'] * X_train['YearsCodingProf']

X_test['sub_d'] = X_test['GDP ($ per capita)'] * X_test['YearsCodingProf']



X_train['sub_e'] = X_train['GDP ($ per capita)'] * X_train['SalaryType']

X_test['sub_e'] = X_test['GDP ($ per capita)'] * X_test['SalaryType']



X_train['sub_f'] = X_train['LastNewJob'] * X_train['YearsCodingProf']

X_test['sub_f'] = X_test['LastNewJob'] * X_test['YearsCodingProf']



X_train['sub_g'] = X_train['LastNewJob'] * X_train['SalaryType']

X_test['sub_g'] = X_test['LastNewJob'] * X_test['SalaryType']



X_train['sub_h'] = X_train['Employment'] * X_train['YearsCodingProf']

X_test['sub_h'] = X_test['Employment'] * X_test['YearsCodingProf']



X_train['sub_i'] = X_train['Employment'] * X_train['SalaryType']

X_test['sub_i'] = X_test['Employment'] * X_test['SalaryType']



X_train['sub_j'] = X_train['YearsCodingProf'] * X_train['SalaryType']

X_test['sub_j'] = X_test['YearsCodingProf'] * X_test['SalaryType']



X_train['sub_k'] = X_train['GDP ($ per capita)'] * X_train['Employment'] * X_train['LastNewJob']

X_test['sub_k'] = X_test['GDP ($ per capita)'] * X_test['Employment'] * X_test['LastNewJob']
X_test
X_train.drop(['distinct'], axis=1, inplace=True)

X_test.drop(['distinct'], axis=1, inplace=True)
X_train
X_train.head()
X_train
# カテゴリカル変数とユニーク値の少ない数値項目をtarget_encodingしていく

from sklearn.model_selection import GroupKFold, KFold

# cate_list = ['AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5',

#              'AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10',

#              'AssessBenefits1','AssessBenefits2','AssessBenefits3','AssessBenefits4','AssessBenefits5',

#              'AssessBenefits6','AssessBenefits7','AssessBenefits8','AssessBenefits9','AssessBenefits10','AssessBenefits11',

#              'JobContactPriorities1','JobContactPriorities2','JobContactPriorities3','JobContactPriorities4','JobContactPriorities5',

#              'JobEmailPriorities1','JobEmailPriorities2','JobEmailPriorities3','JobEmailPriorities4',

#              'JobEmailPriorities5','JobEmailPriorities6','JobEmailPriorities7',

#              'AdsPriorities1','AdsPriorities2','AdsPriorities3','AdsPriorities4','AdsPriorities5',

#              'AdsPriorities6','AdsPriorities7'

#             ]

cate_list = ['JobContactPriorities1','JobContactPriorities2','JobContactPriorities3','JobContactPriorities4','JobContactPriorities5',

             'JobEmailPriorities1','JobEmailPriorities2','JobEmailPriorities3','JobEmailPriorities4',

             'JobEmailPriorities5','JobEmailPriorities6','JobEmailPriorities7',

             'AdsPriorities1','AdsPriorities2','AdsPriorities3','AdsPriorities4','AdsPriorities5',

             'AdsPriorities6','AdsPriorities7',

             'AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5',

             'AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10',

             'Employment','LastNewJob'

            ]







target = 'ConvertedSalary'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_trainのカテゴリ変数をoofでエンコーディングする

kf = KFold(n_splits=5, random_state=71, shuffle=True)
# # バリデーションのfoldごとにtarget encodingをやり直す

# from pandas import DataFrame, Series

# for col in cate_list:

#     # X_testはX_trainでエンコーディングする

#     summary = X_temp.groupby([col])[target].mean()

#     enc_test = X_test[col].map(summary) 

    

#     enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

#     for i, (train_ix, val_ix) in enumerate((kf.split(X_train, y_train))):

#         X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#         X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        

#         summary = X_train_.groupby([col])[target].mean()

#         enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#     # target_encoding項目追加

#     X_train['target_' + col] = enc_train

#     X_test['target_' + col] = enc_test
X_train.drop(['AssessBenefits1'], axis=1, inplace=True)
X_test.drop(['AssessBenefits1'], axis=1, inplace=True)


X_train.drop(['AssessBenefits2'], axis=1, inplace=True)

X_test.drop(['AssessBenefits2'], axis=1, inplace=True)



X_train.drop(['AssessBenefits3'], axis=1, inplace=True)

X_test.drop(['AssessBenefits3'], axis=1, inplace=True)



X_train.drop(['AssessBenefits4'], axis=1, inplace=True)

X_test.drop(['AssessBenefits4'], axis=1, inplace=True)



X_train.drop(['AssessBenefits5'], axis=1, inplace=True)

X_test.drop(['AssessBenefits5'], axis=1, inplace=True)



X_train.drop(['AssessBenefits6'], axis=1, inplace=True)

X_test.drop(['AssessBenefits6'], axis=1, inplace=True)



X_train.drop(['AssessBenefits7'], axis=1, inplace=True)

X_test.drop(['AssessBenefits7'], axis=1, inplace=True)



X_train.drop(['AssessBenefits8'], axis=1, inplace=True)

X_test.drop(['AssessBenefits8'], axis=1, inplace=True)



X_train.drop(['AssessBenefits9'], axis=1, inplace=True)

X_test.drop(['AssessBenefits9'], axis=1, inplace=True)



X_train.drop(['AssessBenefits10'], axis=1, inplace=True)

X_test.drop(['AssessBenefits10'], axis=1, inplace=True)

X_train.drop(['AssessBenefits11'], axis=1, inplace=True)
X_test.drop(['AssessBenefits11'], axis=1, inplace=True)
# 欠損値補完

X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。



from sklearn import linear_model

scores = []

lgb_y_pred_train = np.zeros(len(X_train))

lgb_y_pred_test = np.zeros(len(X_test))

kf = KFold(n_splits=5, random_state=81, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

    clf = linear_model.LinearRegression()

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict(X_val)

    lgb_y_pred_train[test_ix] = y_pred

    score = np.sqrt(mean_squared_error(y_val, y_pred))

    scores.append(score)

    lgb_y_pred_test += clf.predict(X_test)

    

    print('CV Score of Fold_%d is %f' % (i, score))

lgb_y_pred_test /= 5
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。



from sklearn.ensemble import RandomForestRegressor

scores2 = []

lgb_y_pred_train2 = np.zeros(len(X_train))

lgb_y_pred_test2 = np.zeros(len(X_test))

kf = KFold(n_splits=5, random_state=81, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

    clf = clf = RandomForestRegressor(max_depth=10, random_state=41)

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict(X_val)

    lgb_y_pred_train[test_ix] = y_pred

    score = np.sqrt(mean_squared_error(y_val, y_pred))

    scores.append(score)

    lgb_y_pred_test += clf.predict(X_test)

    

    print('CV Score of Fold_%d is %f' % (i, score))

lgb_y_pred_test2 /= 5
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
print(rmsle(y_train, lgb_y_pred_train))
lgb_pred2 = np.expm1(lgb_y_pred_test2)
lgb_pred2
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = lgb_pred2

submission.to_csv('submission.csv')
from pandas import DataFrame, Series

DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 16))

lgb.plot_importance(clf, max_num_features=100, ax=ax, importance_type='gain')
df_train.ConvertedSalary.hist()
df_train