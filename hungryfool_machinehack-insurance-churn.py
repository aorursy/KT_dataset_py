import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel

%matplotlib inline

# Classification

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier



import xgboost as xgb

import lightgbm as lgb

import catboost as cat



# Preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
test = pd.read_csv('../input/insurance-churn-prediction/Test.csv')

train = pd.read_csv('../input/insurance-churn-prediction/Train.csv')
train.head()
sns.heatmap(train.isnull())
df=train.append(test,ignore_index=True)
# #pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10))

# import pandas_profiling

# prof = pandas_profiling.ProfileReport(df)

# prof.to_file(output_file='output.html')
original_cols = test.columns

original_cols
df.info()
df['labels'].value_counts()
#df['feature_3'].head()

#plt.hist(df['feature_3'])

sns.distplot(df['feature_3'])
x3=df['feature_3']

diff = np.diff(np.sort(x3))

diff

np.unique(diff)
x3_shift = ((x3/0.00388311868) - 0.8369211 +260).round()

x3_shift.value_counts()

x3_shift_log = np.log(x3_shift)

df['feature_3_shift_log'] = x3_shift_log

sns.distplot(x3_shift_log)
x2=df['feature_2']

np.diff(np.sort(x2.unique()))

np.diff(np.sort(x2.unique())/0.12015788)

x2_shift = ((df['feature_2']/0.12015788) -0.193581 + 15)

x2_shift = x2_shift.round()

np.sort(x2_shift.unique())
(x2_shift).value_counts()

sns.distplot(x2_shift)

x2_shift_bin = x2_shift.apply(lambda x : 1 if x>15 else 0)

df['feature_2_shift'] =x2_shift

df['feature_2_shift_bin'] =x2_shift_bin
x2_shift.value_counts()
x0 = df['feature_0']

sns.boxplot(x0)
diff = np.diff(np.sort(x0))

np.unique(diff)
x0_shift = (x0/0.09417398 -  0.063788 + 54).round()

sns.distplot(np.log(x0_shift))

#sns.distplot(x0_shift)

#sns.boxplot(x0_shift)

x0_shift.describe()

df['feature_0_shift_log']  = np.log(x0_shift)
x1 = df['feature_1']

sns.boxplot(x1)
diff = np.diff(np.sort(x1))

np.unique(diff)
x1_shift = (x1/0.000328436115 - 0.727934 + 10000).round()

x1_shift_log = np.log(x1_shift)

sns.distplot(x1_shift_log)

x1_shift.value_counts()

#sns.boxplot(x1_shift_log)

df['feature1_shift_log']  = x1_shift_log
x1_shift.value_counts()
x14 = df['feature_14']

#sns.boxplot(x14)

sns.distplot(x14)
x14.value_counts()
x4 = df['feature_4']
sns.distplot(x4)
diff = np.diff(np.sort(x4))

np.unique(diff)
x4_shift = (x4/0.64558058 - 0.11808 + 2).round().value_counts()
sns.distplot(np.log(x4_shift))

df['x4_shift_log'] = np.log(x4_shift)
x5 = df['feature_5']

x5.value_counts()
diff = np.diff(np.sort(x5))

np.unique(diff)
x5_shift = (x5/0.00998725 - 0.802206 +43).round()

x5_shift.value_counts()
(np.log(x5_shift)).value_counts()

df['x5_shift_log'] = np.log(x5_shift)

df['x5_cat'] = df['x5_shift_log'].apply(lambda x : 1 if x==0 else 0)
x6 = df['feature_6']

x6.value_counts()
diff = np.diff(np.sort(x6))

np.unique(diff)
x6_shift = (x6/0.4341379 - 0.419677 + 2).round()

x6_shift.value_counts()

(np.log(x6_shift)).value_counts()

df['x6_shift'] = x6_shift

df['x6_shift_log'] = np.log(x6_shift)

df['x6_cat'] = df['x6_shift_log'].apply(lambda x : 1 if x==0 else 0)

x7 = df['feature_7']

x7.value_counts()
for i in [0,1,2,3,4,5,6,14] :

    col = 'feature_'+str(i)

    print(col)

    df.drop(columns = [col],inplace = True)



new_col = ['feature_3_shift_log',

       'feature_2_shift', 'feature_2_shift_bin', 'feature_0_shift_log',

       'feature1_shift_log', 'x4_shift_log', 'x5_shift_log', 'x5_cat',

       'x6_shift', 'x6_shift_log', 'x6_cat']

new_col
# for i in range(len(new_col)):

#     if(new_col[i]=='labels') :

        

#         continue

    

#     else :

        

#         for j in range(len(new_col)) :

            

#             if(new_col[j]=='labels') :

#                 continue

#             elif i<j :

# #                print(new_col[i],new_col[j])

#                 colm = new_col[i]+"_mul_"+new_col[j]

#                 cols = new_col[i]+"_sum_"+new_col[j]

#                 cold = new_col[i]+"_diff_"+new_col[j]

#                 coldi = new_col[i]+"_div_"+new_col[j]

#                 #print(col)

#                 df[colm] = df[new_col[i]]*df[new_col[j]]

#                 df[cols] = df[new_col[i]]+df[new_col[j]]

#                 #df[cold] = df[new_col[i]]-df[new_col[j]]

#                 #df[coldi] = df[new_col[i]]/df[new_col[j]]

#             else :

#                 continue
labels = df['labels']

df = df.dropna(axis=1)
df['labels']= labels
sns.heatmap(df.isnull())
df = df.replace([np.inf, -np.inf], 0)
feat = df.columns

feat = feat.drop('labels')
feat
target = 'labels'
(train[target].value_counts() / train.shape[0])*100
df_train=df[df['labels'].isnull()==False].copy()
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=42, sampling_strategy='all')

# X_train_ovr, y_train_ovr = sm.fit_sample(df_train[feat], df_train[target])



# print("After Oversampling : {} --> {}".format(X_train_ovr.shape, y_train_ovr.shape))
# train_ovr = pd.DataFrame(X_train_ovr, columns=df_train.columns.tolist())

# train_ovr[target] = y_train_ovr



# train_ovr.shape
# (train_ovr[target].value_counts() / train_ovr.shape[0])*100
def baseliner(X, y, cv=3, metric='f1_macro'):

    print("Baseliner Models\n")

    eval_dict = {}

    models = [lgb.LGBMClassifier(), xgb.XGBClassifier(),

              #GradientBoostingClassifier(),

                  LogisticRegression(), GaussianNB(), RandomForestClassifier(), DecisionTreeClassifier(),

                  ExtraTreeClassifier(), AdaBoostClassifier(), BaggingClassifier(),

              #ExtraTreesClassifier(),

              #SVC(probability=True), KNeighborsClassifier() 

                 ]

    print("Model Name \t |   f1")

    print("--" * 50)



    for index, model in enumerate(models, 0):

        model_name = str(model).split("(")[0]

        eval_dict[model_name] = {}



        results = cross_val_score(model, X, y, cv=cv, scoring=metric)

        eval_dict[model_name]['cv'] = results.mean()



        print("%s \t | %.4f \t" % (

            model_name[:12], eval_dict[model_name]['cv']))
df_train=df[df['labels'].isnull()==False].copy()

df_test=df[df['labels'].isnull()==True].copy()

df_test.drop(columns=['labels'],axis=1, inplace=True)



print(df_train.shape,df_test.shape)
x = df_train.drop('labels',axis=1)

y = df_train['labels']
baseliner(x, y)
from sklearn.metrics import f1_score



def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_hat), True
def lgb_model(train, features, target, ts=False, plot=True):

    evals_result = {}

    trainX, validX, trainY, validY = train_test_split(train[features], train[target], shuffle=False, test_size=0.2, random_state=13)

    print("LGB Model")

    lgb_train_set = lgb.Dataset(trainX, label=trainY)

    lgb_valid_set = lgb.Dataset(validX, label=validY)



    MAX_ROUNDS = 2000

    lgb_params = {

        "boosting": 'gbdt',

        "learning_rate": 0.1,

        "nthread": -1,

        "seed": 13,

        "num_boost_round": MAX_ROUNDS,

        "objective": "binary",

    }



    lgb_model = lgb.train(

        lgb_params,

        train_set=lgb_train_set,

        valid_sets=[lgb_train_set, lgb_valid_set],

        early_stopping_rounds=250,

        verbose_eval=100,

        evals_result=evals_result,

        feval=lgb_f1_score # New metric to be optimised

    )

    if plot:

        lgb.plot_importance(lgb_model, figsize=(24, 24))

        lgb.plot_metric(evals_result, metric='f1')



    return lgb_model, lgb_model.best_score
lgbM, score = lgb_model(df_train, feat, target, True, True)
y_preds = lgbM.predict(df_test[feat])

y_preds
df_lgb = pd.DataFrame({'labels':y_preds})

df_lgb['labels'] = df_lgb['labels'].apply(lambda x : 1 if x>0.5 else 0)
df_lgb['labels'].value_counts()
import time

times = time.strftime("%Y%m%d-%H%M%S")



df_lgb.to_excel('submission-lgb_'+times+'.xlsx',index=False)