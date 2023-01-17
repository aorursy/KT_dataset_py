import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.gpd.read_csv)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

import xgboost as xgb

from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold, KFold

import catboost as cat

print(os.listdir("../input"))

plt.rcParams['figure.figsize'] = (15, 5)
def univariate_num(df):

    print("The length of the dataset is  {}".format(len(df)))

    print("The missing values in the data are as follows")

    print(df.isnull().sum())

    print("The description of the data is a s follows")

    uni = df.describe()

    uni = uni.T

    print(uni)

    print("The various necessary plots of columns in data")

   

    

    if (len(df.columns) == 1):

        print("This is a KDE plot")

        plt.figure(figsize = (14, 5)) 

        sns.kdeplot(df, color ='r', shade =True)

        plt.show()

        print("This is a BOXPLOT")

        sns.boxplot(x = df)

        plt.show()



    else:

        for col in df.columns:

            #sns.distplot(df[col], bins =100, kde=True)

            print("This is a KDE plot")

            plt.figure(figsize = (14, 5))

            sns.kdeplot(df[col], color ='r', shade = True)

            plt.show()

            

            plt.figure(figsize = (14, 5))

            print("This is a BOXPLOT")

            sns.boxplot(x = df[col])

            plt.show()

    





def univariate_char(df):

    print("The number of training examples in the data are {}".format(len(df)))

    print("The description of the data is as follows")

    uni = df.describe()

    uni = uni.T

    print(uni)

    

    if (len(df.columns) == 1):

        print("Barplot for {}".format(df.columns))

        df.value_counts().plot(kind = 'bar')

                        

    for col in df.columns:

        plt.figure(figsize = (20, 5))

        print("The Barplots")

        sns.countplot(df[col])

        plt.show()









def bivariate_num(df):

    print("The corelation matrix for the data is as follows")

    cor = df.corr()

    print(cor)

    

    #tar = target.columns

    

    print("The heatmap for correlation matrix is as follows")

    sns.heatmap(cor)

    plt.show()

    

    Nn = []

    for i in cor:

        for j in cor:

            if (cor[i][j] > 0.4):

                if i!=j:

                    print("{} and {} are highly correalted hence only one out of the two should be kept in the data".format(i, j))



def bivariate_char(df, target):

    for col in df:

        tar = list(target.columns)

        c = tar[0]

        if (len(df) == 1):

            

            ss = pd.crosstab(df, target[c] )

            print("The table between a independent and dependent variable")

            print(ss)

            print("The barplot")

            ss.plot(kind = 'bar')

            plt.show()

        else:

            print("The table between a independent and dependent variable")

            ss = pd.crosstab(df[col], target[c] )

            print(ss)

            print("The barplot")

            ss.plot(kind = 'bar')

            plt.show()
train = pd.read_csv(r"../input/train.csv")

test = pd.read_csv(r"../input/test.csv")
train.head(5)
train['target'].value_counts().plot(kind='bar')
cols = train.columns.values.tolist()[2: ]

predictors = train[cols]

target = train[['target']]

pre_test = test[cols]
predictors.head()
%%time

import lightgbm as lgb



sfl = StratifiedKFold(n_splits = 3, shuffle=True)

pred_test_y = np.zeros((test.shape[0]))

seed = 2019

N = 0

for train_indices, test_indices in sfl.split(predictors, target):

    params = {

        'num_leaves': 15,

        'max_bin': 119,

        'min_data_in_leaf': 11,

        'learning_rate': 0.02,

        'min_sum_hessian_in_leaf': 0.00245,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': 0.05,

        'lambda_l1': 4.972,

        'lambda_l2': 2.276,

        'min_gain_to_split': 0.65,

        'max_depth': 14,

        'save_binary': True,

        'seed': seed,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }

    X_train, X_test = predictors.iloc[train_indices], predictors.iloc[test_indices]

    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]



    #TRAINING LIGHTGBM Model WITH DIFFERENT SEED VALUES

    lgtrain = lgb.Dataset(X_train, label=y_train)

    lgval = lgb.Dataset(X_test, label=y_test)

    evals_result = {}

    model2 = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20, 

                      evals_result=evals_result)

    

    pred_val =     model2.predict(X_test, num_iteration=model2.best_iteration)

    pred_test_y += model2.predict(pre_test, num_iteration=model2.best_iteration)

    

    

    print("Validation score is :", roc_auc_score(y_test, pred_val))

    print(N, "Iteration completed")

    seed+= 2000

    N+=1

pred_test = pred_test_y/N
print("FEATURE IMPORTANCES FROM LIGHTGBM \n")

feature_importance = pd.concat([pd.DataFrame(predictors.columns, columns =['Features']) , 

                                 pd.DataFrame(model2.feature_importance(), columns=['Importances'])], axis = 1)

feature_importance = feature_importance.set_index('Features')



#FEATURE IMPORTANCES FROM LIGHTGBM

feature_importance.sort_values('Importances').plot(kind='bar')

plt.xlabel("FEATURES ", fontsize = 15)

plt.ylabel("IMPORTANCES", fontsize = 15)

plt.show()
feature_importance.sort_values(by='Importances', ascending=False).reset_index()['Features'][feature_importance.sort_values(by='Importances', ascending=False).reset_index()['Importances']>300 ].values.tolist()
important_cols= ['var_33','var_6','var_91','var_170','var_13','var_1','var_190','var_9','var_21','var_127','var_108','var_174','var_173','var_18',

                 'var_146','var_34','var_110','var_92','var_78','var_198','var_22','var_169','var_165','var_121','var_133','var_184','var_12',

                 'var_94','var_75','var_99','var_53','var_154','var_76','var_80','var_191','var_166','var_157','var_122','var_36','var_26',

                 'var_109','var_131','var_107','var_164','var_2','var_115','var_40','var_179','var_130','var_192','var_81','var_32','var_177',

                 'var_147','var_67','var_56','var_123','var_141','var_44','var_197','var_172','var_89','var_118','var_128','var_162','var_149',

                 'var_180', 'var_71', 'var_139', 'var_186']

new_preds = train[important_cols]
new_preds.head()

new_preds_test = test[important_cols]
%%time

import lightgbm as lgb



sfl = StratifiedKFold(n_splits = 3, shuffle=True)

pred_test_y2 = np.zeros((test.shape[0]))

seed = 2019

N = 0

for train_indices, test_indices in sfl.split(predictors, target):

    params = {

        'num_leaves': 15,

        'max_bin': 119,

        'min_data_in_leaf': 11,

        'learning_rate': 0.02,

        'min_sum_hessian_in_leaf': 0.00245,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': 0.05,

        'lambda_l1': 4.972,

        'lambda_l2': 2.276,

        'min_gain_to_split': 0.65,

        'max_depth': 14,

        'save_binary': True,

        'seed': seed,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }

    X_train, X_test = predictors.iloc[train_indices], predictors.iloc[test_indices]

    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]



    #TRAINING LIGHTGBM Model WITH DIFFERENT SEED VALUES

    lgtrain = lgb.Dataset(X_train, label=y_train)

    lgval = lgb.Dataset(X_test, label=y_test)

    evals_result = {}

    model3 = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20, 

                      evals_result=evals_result)

    

    pred_val =     model3.predict(X_test, num_iteration=model3.best_iteration)

    pred_test_y2 += model2.predict(pre_test, num_iteration=model2.best_iteration)

    

    

    print("Validation score is :", roc_auc_score(y_test, pred_val))

    print(N, "Iteration completed")

    seed+= 2000

    N+=1

pred_test2 = pred_test_y2/N
predictions = pred_test*0.5 + pred_test2*0.5
predictions = pd.DataFrame(predictions, columns =['target'])

sub = pd.concat([test[['ID_code']], predictions[['target']]], axis = 1)

sub.to_csv('submission.csv', index=False)
sub.head()