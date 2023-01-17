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
!pip install -U category_encoders

!pip install patsy

!pip install statsmodels
import category_encoders as ce

from sklearn.model_selection import KFold

from tqdm import tqdm_notebook as tqdm



pd.set_option("display.max_columns", 2000)

np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_rows', None)
df_train = pd.read_csv('/kaggle/input/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/test.csv', index_col=0)



# /kaggle/input/sample_submission.csv

# /kaggle/input/test.csv

# /kaggle/input/country_info.csv

# /kaggle/input/survey_dictionary.csv

# /kaggle/input/train.csv
df_train.head()
y_train = df_train['ConvertedSalary']



# 列をドロップ

# col_drop = ['DevType','CommunicationTools','FrameworkWorkedWith','AdsActions','ErgonomicDevices']

col_drop = []



X_train = df_train.drop(col_drop, axis=1)

X_train = X_train.drop('ConvertedSalary', axis=1)



X_test = df_test.drop(col_drop, axis=1)
# 欠損処理(カテゴリ)

# 'NoData'で埋める

X_train.fillna('NoData',inplace=True)

X_test.fillna('NoData',inplace=True)
# 対数化

y_train = y_train.apply(np.log1p)
y_train_restore = y_train.apply(np.expm1)

print(y_train.head())

print(y_train_restore.head())
# カテゴリ値のエンコーディング

# とりあえず全部同じ方式で行く

# cats = ['Hobby', 'OpenSource', 'Country', 'Student', 'Employment', 'FormalEducation', 'UndergradMajor', 'CompanySize', 'YearsCoding', 'YearsCodingProf', 'JobSatisfaction', 'CareerSatisfaction', 'HopeFiveYears', 'JobSearchStatus', 'LastNewJob', 'AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4', 'AssessJob5', 'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9', 'AssessJob10', 'AssessBenefits1', 'AssessBenefits2', 'AssessBenefits3', 'AssessBenefits4', 'AssessBenefits5', 'AssessBenefits6', 'AssessBenefits7', 'AssessBenefits8', 'AssessBenefits9', 'AssessBenefits10', 'AssessBenefits11', 'JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 'JobContactPriorities4', 'JobContactPriorities5', 'JobEmailPriorities1', 'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7', 'UpdateCV', 'Currency', 'SalaryType', 'CurrencySymbol', 'TimeFullyProductive', 'TimeAfterBootcamp', 'AgreeDisagree1', 'AgreeDisagree2', 'AgreeDisagree3', 'OperatingSystem', 'NumberMonitors', 'CheckInCode', 'AdBlocker', 'AdBlockerDisable', 'AdsAgreeDisagree1', 'AdsAgreeDisagree2', 'AdsAgreeDisagree3', 'AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7', 'AIDangerous', 'AIInteresting', 'AIResponsible', 'AIFuture', 'EthicsChoice', 'EthicsReport', 'EthicsResponsible', 'EthicalImplications', 'StackOverflowRecommend', 'StackOverflowVisit', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowJobs', 'StackOverflowDevStory', 'StackOverflowJobsRecommend', 'StackOverflowConsiderMember', 'HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3', 'HypotheticalTools4', 'HypotheticalTools5', 'WakeTime', 'HoursComputer', 'HoursOutside', 'SkipMeals', 'Exercise', 'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity', 'Age', 'Dependents', 'MilitaryUS', 'SurveyTooLong', 'SurveyEasy']

cats = ['DevType','CommunicationTools','FrameworkWorkedWith','AdsActions','ErgonomicDevices','Hobby', 'OpenSource', 'Country', 'Student', 'Employment', 'FormalEducation', 'UndergradMajor', 'CompanySize', 'YearsCoding', 'YearsCodingProf', 'JobSatisfaction', 'CareerSatisfaction', 'HopeFiveYears', 'JobSearchStatus', 'LastNewJob', 'AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4', 'AssessJob5', 'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9', 'AssessJob10', 'AssessBenefits1', 'AssessBenefits2', 'AssessBenefits3', 'AssessBenefits4', 'AssessBenefits5', 'AssessBenefits6', 'AssessBenefits7', 'AssessBenefits8', 'AssessBenefits9', 'AssessBenefits10', 'AssessBenefits11', 'JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 'JobContactPriorities4', 'JobContactPriorities5', 'JobEmailPriorities1', 'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7', 'UpdateCV', 'Currency', 'SalaryType', 'CurrencySymbol', 'TimeFullyProductive', 'TimeAfterBootcamp', 'AgreeDisagree1', 'AgreeDisagree2', 'AgreeDisagree3', 'OperatingSystem', 'NumberMonitors', 'CheckInCode', 'AdBlocker', 'AdBlockerDisable', 'AdsAgreeDisagree1', 'AdsAgreeDisagree2', 'AdsAgreeDisagree3', 'AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7', 'AIDangerous', 'AIInteresting', 'AIResponsible', 'AIFuture', 'EthicsChoice', 'EthicsReport', 'EthicsResponsible', 'EthicalImplications', 'StackOverflowRecommend', 'StackOverflowVisit', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowJobs', 'StackOverflowDevStory', 'StackOverflowJobsRecommend', 'StackOverflowConsiderMember', 'HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3', 'HypotheticalTools4', 'HypotheticalTools5', 'WakeTime', 'HoursComputer', 'HoursOutside', 'SkipMeals', 'Exercise', 'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity', 'Age', 'Dependents', 'MilitaryUS', 'SurveyTooLong', 'SurveyEasy']



# ターゲットエンコーディング

ce_e = ce.TargetEncoder(cols=cats)



# ラベルエンコ―ディング

# ce_e = ce.OrdinalEncoder(cols=cats)



X_train[cats] = ce_e.fit_transform(X_train[cats], y_train)

X_test[cats] = ce_e.transform(X_test[cats])
# モデリング

# Light GBM

import lightgbm as lgb



learning_rate = 0.1

num_leaves = 31

min_data_in_leaf = 200

feature_fraction = 1.0

num_boost_round = 10000

params = {"objective": 'regression',

          "boosting_type": "gbdt",

          "learning_rate": learning_rate,

          "num_leaves": num_leaves,

          "feature_fraction": feature_fraction,

          "verbosity": 0,

          "drop_rate": 0.1,

          "is_unbalance": False,

          "max_drop": 50,

          "min_child_samples": 20,

          "min_child_weight": 1e-3,

          "min_split_gain": 0.0,

          "subsample": 1.0,

          "metric": "rmse",

          }



NFOLDS = 5

skf = KFold(n_splits=NFOLDS, random_state=71, shuffle=True)



final_cv_train = np.zeros(len(y_train))

final_cv_pred = np.zeros(len(X_test))
num_cv = 3



for s in range(num_cv):

    cv_train = np.zeros(len(y_train))

    cv_pred = np.zeros(len(X_test))



    params['seed'] = s



    kf = skf.split(X_train, y_train)



    best_trees = []

#     fold_scores = []

    

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

        dtrain = lgb.Dataset(X_train_, y_train_)

        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100, early_stopping_rounds=500)

        best_trees.append(bst.best_iteration)

        cv_pred_tmp = bst.predict(X_test, num_iteration=bst.best_iteration)

        cv_pred += cv_pred_tmp

        cv_train[test_ix] += bst.predict(X_val)



#         score = roc_auc_score(y_val, cv_train[test_ix])

#         print(score)

#         fold_scores.append(score)



    cv_pred /= NFOLDS

    final_cv_train += cv_train

    final_cv_pred += cv_pred
print((final_cv_pred / float(num_cv))[:50])
final_cv_pred_restore = np.expm1(final_cv_pred / float(num_cv))



np.set_printoptions(suppress=True)

print(final_cv_pred_restore)
submission = pd.read_csv('/kaggle/input/sample_submission.csv', index_col=0)

submission['ConvertedSalary'] = final_cv_pred_restore

submission.to_csv('submission2.csv')