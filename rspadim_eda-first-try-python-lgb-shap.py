# inspirated by a friend - https://www.kaggle.com/brandao/starting-point-in-r

#

#

# # data.y <- ifelse(data$`SARS-Cov-2 exam result` == "positive", 1, 0);

#

#

#



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option("display.max_rows",500)

pd.set_option("display.max_columns",500)
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

print("Data Frame, shape: ", df.shape)

print("Columns: ", list(df.columns))

df.head()
df.describe().T
null_counts = df.isnull().sum()/len(df)

plt.figure(figsize=(16,8))

plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')

plt.ylabel('fraction of rows with missing data')

plt.bar(np.arange(len(null_counts)),null_counts)

plt.show()
# drop non-feature columns

feature_columns = df.columns.drop([

    'Patient ID', 

    'SARS-Cov-2 exam result', 

    'Patient addmited to regular ward (1=yes, 0=no)', 

    'Patient addmited to semi-intensive unit (1=yes, 0=no)', 

    'Patient addmited to intensive care unit (1=yes, 0=no)'

])

# create multi-index with (count, fraction, number of NaN sequences) per feature column

iterables = [feature_columns,['count','fraction','seq']]

index = pd.MultiIndex.from_product(iterables,names=['feature','stat'])

# use list of IDs as index (only sorted for easier navigation)

ids = df['Patient ID'].unique()

ids.sort()

# create empty data frame

nan_df = pd.DataFrame(data=None,index=df['SARS-Cov-2 exam result'].unique(),columns=index)



from itertools import groupby

# iterate over all asset ID

total_groups = len(df.groupby('SARS-Cov-2 exam result'))

print("Groups:", total_groups)

for i, (name, group) in enumerate(df.groupby('SARS-Cov-2 exam result')):

    print('i:', i, '/', total_groups, ', SARS-Cov-2 exam result:', name)

    # for every feature column

    for c in feature_columns:

        #print('name:', name, ", group:", len(group), ', c:', c)

        # total number of rows with missing data

        nan_count = group[c].isnull().sum()

        # time span this ID was present

        timespan = len(group[c])

        # row indices for missing data

        nan_indices = pd.isnull(group[c]).to_numpy().nonzero()[0]

        # get number of joint time spans of missing values

        nseq = len(list(groupby(enumerate(nan_indices),lambda x:x[0]-x[1])))

        nan_df.loc[name][c,'count'] = nan_count

        nan_df.loc[name][c,'fraction'] = nan_count * 1.0/timespan

        nan_df.loc[name][c,'seq'] = nseq

        

nan_df.head(20).T

fractions = nan_df.xs('fraction',level='stat',axis=1)

fraction_mean = fractions.mean()

fraction_std = fractions.std()

plt.figure(figsize=(16,8))

plt.xticks(np.arange(len(fraction_mean)),fraction_mean.index,rotation='vertical')

plt.errorbar(np.arange(len(fraction_mean)),fraction_mean,yerr=fraction_std,fmt='o')

plt.ylabel('mean fraction of rows with missing data per [SARS-Cov-2 exam result]');

plt.show()
plt.hist(fractions.values.flatten(),bins=50)

plt.xlabel('fraction of rows with missing data per [SARS-Cov-2 exam result]')

plt.legend()

plt.show()
nseq = nan_df.xs('seq',level='stat',axis=1)

nseq_mean = nseq.mean()

nseq_std = nseq.std()

plt.figure(figsize=(16,8))

plt.xticks(np.arange(len(nseq_mean)),nseq_mean.index,rotation='vertical') #todo: check this (we don't have timestamp)

plt.errorbar(np.arange(len(nseq_mean)),nseq_mean,yerr=nseq_std,fmt='o')

plt.ylabel('mean number of connected NaN ranges')

plt.show()
plt.hist(nseq.values.flatten(),bins=50)

plt.xlabel('number of connected time ranges with missing data per [SARS-Cov-2 exam result]');  #todo: check this (we don't have timestamp)

plt.legend()

plt.show()
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.feature_selection import VarianceThreshold

import lightgbm as lgb

import shap
cat_features = [

    i for i in df.columns if str(df[i].dtype) in ['object', 'category']

]

if len(cat_features) > 0:

    df[cat_features] = df[cat_features].astype('category')

print(df.dtypes)



# lgb don't like strings/category, only numbers and boolean

df_lgb = df.copy()

for i in cat_features:

    df_lgb[i] = df[i].cat.codes

# it don't like complex names too..

df_lgb.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_lgb.columns]

print(df_lgb.columns)



y = (df['SARS-Cov-2 exam result'] == 'positive').astype(int)

x = df_lgb.drop(['Patient_ID', 

                 'SARS_Cov_2_exam_result', 

                 'Patient_addmited_to_regular_ward__1_yes__0_no_', 

                 'Patient_addmited_to_semi_intensive_unit__1_yes__0_no_', 

                 'Patient_addmited_to_intensive_care_unit__1_yes__0_no_'

                ], axis=1)

print(y.shape, x.shape, df_lgb.shape)
# x/y train/test

while True:

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    train_weight = 1-train_y.replace(train_y.value_counts()/len(train_y))

    valid_weight = 1-valid_y.replace(valid_y.value_counts()/len(valid_y))

    if len(train_y.unique()) > 1 and len(valid_y.unique()) > 1:

        break

positive_weight = train_weight[train_y==1].values[0]

train_data=lgb.Dataset(train_x,label=train_y, weight=train_weight)

valid_data=lgb.Dataset(valid_x,label=valid_y, weight=valid_weight, reference=train_data)



print("train_y:\n",train_y.value_counts())

print("valid_y:\n",valid_y.value_counts())

print("y:\n", y.value_counts())

#Select Hyper-Parameters

params = {'metric': 'auc',

          'objective':'binary',

          'eta': 0.004,

          'boosting_type': 'gbdt',

          'colsample_bytree': 0.9,

          'max_depth': 9,

          'n_estimators': 1200,

          'subsample': 0.9,

          'num_threads': -1,

          'scale_pos_weight': positive_weight

}

#Train model on selected parameters and number of iterations

lgbm = lgb.train(

    params=params,

    train_set=train_data,

    valid_sets=valid_data,

    early_stopping_rounds=500,

    verbose_eval=100,

#    categorical_feature=cat_features

)
y_hat = lgbm.predict(x)

score = roc_auc_score(y, y_hat)

print("Overall AUC: {:.3f}" .format(score))

plt.figure(figsize=(12,12))

plt.subplot(2,3,1)

plt.hist(lgbm.predict(x[y==1]), label='y=1', bins=50, density=True)

plt.legend()

plt.subplot(2,3,2)

plt.hist(lgbm.predict(train_x[y==1]), label='train y=1', bins=50, density=True)

plt.legend()

plt.subplot(2,3,3)

plt.hist(lgbm.predict(valid_x[y==1]), label='valid y=1', bins=50, density=True)

plt.legend()



plt.subplot(2,3,4)

plt.hist(lgbm.predict(x[y==0]), label='y=0', bins=50, density=True)

plt.legend()

plt.subplot(2,3,5)

plt.hist(lgbm.predict(train_x[y==0]), label='train y=0', bins=50, density=True)

plt.legend()

plt.subplot(2,3,6)

plt.hist(lgbm.predict(valid_x[y==0]), label='valid y=0', bins=50, density=True)

plt.legend()

plt.show()

def create_threshold_chart(y_hat, y, weight):

    # valid threshold

    x, y1, y2, y3, y4 = np.array(range(0,100))/100., [], [], [], []

    for ii, i in enumerate(x):

        predicted_class = y_hat>i

        y1.append(len(y[(predicted_class==1) & (y==0)])/len(y[y==0]))

        y2.append(len(y[(predicted_class==0) & (y==1)])/len(y[y==1]))

        y3.append(

            len(y[(predicted_class==1) & (y==0)]) * weight[0] / 

            (len(y[y==0]) * weight[0] + len(y[y==1]) * weight[1])

        )

        y4.append(

            len(y[(predicted_class==0) & (y==1)]) * weight[0] / 

            (len(y[y==0]) * weight[0] + len(y[y==1]) * weight[1])

        )

    

    plt.figure(figsize=(14,7))

    plt.subplot(1,2,1)

    plt.title("False rates")

    plt.plot(x, y1, label='false positives')

    plt.plot(x, y2, label='false negatives')

    plt.xlabel("Classifier Threshold")

    plt.ylabel("False Rate %")

    plt.legend()

    

    plt.subplot(1,2,2)

    plt.title("False rates, weighted by positive/negative weight\n(0=" + str(weight[0]) + ", 1=" + str(weight[1]) + ")")

    plt.plot(x, y3, label='false positives')

    plt.plot(x, y4, label='false negatives')

    plt.xlabel("Classifier Threshold")

    plt.ylabel("False Rate %")

    plt.legend()

    

    plt.show()

create_threshold_chart(

    y_hat=lgbm.predict(valid_x),

    y=valid_y,

    weight={0:1-positive_weight, 1:positive_weight}

)
%time shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)

print(shap_values[0].shape)
shap.summary_plot(shap_values, valid_x)

print("Total columns to display:", len(valid_x.columns))

for i in valid_x.columns:

    shap.dependence_plot(i, shap_values[0], valid_x)

    plt.show()
# sorted(zip(clf.feature_importances_, X.columns), reverse=True)

feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(),x.columns)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

y = (df['SARS-Cov-2 exam result'] == 'positive').astype(int)

x = df_lgb.drop([

    'Patient_age_quantile',

    'Patient_ID', 

    'SARS_Cov_2_exam_result', 

    'Patient_addmited_to_regular_ward__1_yes__0_no_', 

    'Patient_addmited_to_semi_intensive_unit__1_yes__0_no_', 

    'Patient_addmited_to_intensive_care_unit__1_yes__0_no_'

], axis=1)

print(y.shape, x.shape, df_lgb.shape)



# x/y train/test

# x/y train/test

while True:

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    train_weight = 1-train_y.replace(train_y.value_counts()/len(train_y))

    valid_weight = 1-valid_y.replace(valid_y.value_counts()/len(valid_y))

    if len(train_y.unique()) > 1 and len(valid_y.unique()) > 1:

        break

positive_weight = train_weight[train_y==1].values[0]

train_data=lgb.Dataset(train_x,label=train_y, weight=train_weight)

valid_data=lgb.Dataset(valid_x,label=valid_y, weight=valid_weight, reference=train_data)



print("train_y:\n",train_y.value_counts())

print("valid_y:\n",valid_y.value_counts())

print("y:\n", y.value_counts())



#Select Hyper-Parameters

params = {'metric': 'auc',

          'objective':'binary',

          'eta': 0.004,

          'boosting_type': 'gbdt',

          'colsample_bytree': 0.9,

          'max_depth': 9,

          'n_estimators': 1200,

          'subsample': 0.9,

          'num_threads': -1,

          'scale_pos_weight': positive_weight

}

#Train model on selected parameters and number of iterations

lgbm = lgb.train(

    params=params,

    train_set=train_data,

    valid_sets=valid_data,

    early_stopping_rounds=500,

    verbose_eval=100,

#    categorical_feature=cat_features

)
# sorted(zip(clf.feature_importances_, X.columns), reverse=True)

feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(),x.columns)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

%time shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)

print(shap_values[0].shape)
shap.summary_plot(shap_values, valid_x)

print("Total columns to display:", len(valid_x.columns))

for i in valid_x.columns:

    shap.dependence_plot(i, shap_values[0], valid_x)

    plt.show()
plt.figure(figsize=(12,12))

plt.subplot(2,3,1)

plt.hist(lgbm.predict(x[y==1]), label='y=1', bins=50, density=True)

plt.legend()

plt.subplot(2,3,2)

plt.hist(lgbm.predict(train_x[y==1]), label='train y=1', bins=50, density=True)

plt.legend()

plt.subplot(2,3,3)

plt.hist(lgbm.predict(valid_x[y==1]), label='valid y=1', bins=50, density=True)

plt.legend()



plt.subplot(2,3,4)

plt.hist(lgbm.predict(x[y==0]), label='y=0', bins=50, density=True)

plt.legend()

plt.subplot(2,3,5)

plt.hist(lgbm.predict(train_x[y==0]), label='train y=0', bins=50, density=True)

plt.legend()

plt.subplot(2,3,6)

plt.hist(lgbm.predict(valid_x[y==0]), label='valid y=0', bins=50, density=True)

plt.legend()

plt.show()



create_threshold_chart(

    y_hat=lgbm.predict(valid_x),

    y=valid_y,

    weight={0:1-positive_weight, 1:positive_weight}

)
cat_features = [

    i for i in df.columns if str(df[i].dtype) in ['object', 'category']

]

if len(cat_features) > 0:

    df[cat_features] = df[cat_features].astype('category')



# lgb don't like strings/category, only numbers and boolean

df_lgb = df.copy()

for i in cat_features:

    df_lgb[i] = df[i].cat.codes

# it don't like complex names too..

df_lgb.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_lgb.columns]



for with_without in [0,1]:

    for task in [

         'Patient_addmited_to_regular_ward__1_yes__0_no_', 

         'Patient_addmited_to_semi_intensive_unit__1_yes__0_no_', 

         'Patient_addmited_to_intensive_care_unit__1_yes__0_no_'

    ]:

        print('TASK: ', task, "without age column" if with_without == 1 else '')

        if 'positive' in df_lgb[task].unique():

            y = (df_lgb[task] == 'positive').astype(int)

        else:

            y = (df_lgb[task] == 1).astype(int)

        x = df_lgb.drop(['Patient_ID', 

                         'SARS_Cov_2_exam_result', 

                         'Patient_addmited_to_regular_ward__1_yes__0_no_', 

                         'Patient_addmited_to_semi_intensive_unit__1_yes__0_no_', 

                         'Patient_addmited_to_intensive_care_unit__1_yes__0_no_'

                        ], axis=1)

        if with_without == 1:

            x = x.drop(['Patient_age_quantile'], axis=1)

        # x/y train/test

        error_count = 0

        while True:

            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

            train_weight = 1-train_y.replace(train_y.value_counts()/len(train_y))

            valid_weight = 1-valid_y.replace(valid_y.value_counts()/len(valid_y))

            if len(train_y.unique()) > 1 and len(valid_y.unique()) > 1:

                break

            error_count += 1

            print(

                "error [", error_count,"] spliting dataset, len(train_y)=",

                len(train_y.unique()) ,

                " or len(valid_y)=",

                len(valid_y.unique()),

                " are equal to 1")

            if error_count>10:

                break

        if error_count>10:

            print("Can't split dataset")

            continue



        positive_weight = train_weight[train_y==1].values[0]

        train_data=lgb.Dataset(train_x,label=train_y, weight=train_weight)

        valid_data=lgb.Dataset(valid_x,label=valid_y, weight=valid_weight, reference=train_data)



        #Select Hyper-Parameters

        params = {'metric': 'auc',

                  'objective':'binary',

                  'eta': 0.004,

                  'boosting_type': 'gbdt',

                  'colsample_bytree': 0.9,

                  'max_depth': 9,

                  'n_estimators': 1200,

                  'subsample': 0.9,

                  'num_threads': -1,

                  'scale_pos_weight': positive_weight

        }



        #Train model on selected parameters and number of iterations

        lgbm = lgb.train(

            params=params,

            train_set=train_data,

            valid_sets=valid_data,

            early_stopping_rounds=500,

            verbose_eval=0,

        #    categorical_feature=cat_features

        )

        # sorted(zip(clf.feature_importances_, X.columns), reverse=True)

        feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(),x.columns)), columns=['Value','Feature'])



        plt.figure(figsize=(20, 20))

        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

        plt.title(task + ' - LightGBM Features (avg over folds)')

        plt.tight_layout()

        plt.show()

        

        shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)

        shap.summary_plot(shap_values, valid_x)

        plt.show()

        

        plt.figure(figsize=(12,12))

        plt.subplot(2,3,1)

        plt.hist(lgbm.predict(x[y==1]), label='y=1', bins=50, density=True)

        plt.legend()

        plt.subplot(2,3,2)

        plt.title(task)

        plt.hist(lgbm.predict(train_x[y==1]), label='train y=1', bins=50, density=True)

        plt.legend()

        plt.subplot(2,3,3)

        plt.hist(lgbm.predict(valid_x[y==1]), label='valid y=1', bins=50, density=True)

        plt.legend()



        plt.subplot(2,3,4)

        plt.hist(lgbm.predict(x[y==0]), label='y=0', bins=50, density=True)

        plt.legend()

        plt.subplot(2,3,5)

        plt.hist(lgbm.predict(train_x[y==0]), label='train y=0', bins=50, density=True)

        plt.legend()

        plt.subplot(2,3,6)

        plt.hist(lgbm.predict(valid_x[y==0]), label='valid y=0', bins=50, density=True)

        plt.legend()

        plt.show()

        

        create_threshold_chart(

            y_hat=lgbm.predict(valid_x),

            y=valid_y,

            weight={0:1-positive_weight, 1:positive_weight}

        )