# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

%matplotlib inline

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier,Pool

from IPython.display import display

import matplotlib.patches as patch

import matplotlib.pyplot as plt

from sklearn.svm import NuSVR

from scipy.stats import norm

from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

import warnings

import time

import glob

import sys

import os

import gc
# for get better result chage fold_n to 5

fold_n=5

folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)

%matplotlib inline

%precision 4

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

np.set_printoptions(suppress=True)

pd.set_option("display.precision", 15)
# загрузим обучающую и тестовую выборки

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# посмотрим на заголовок обучающей выборки

train_df.head()
test_df.head()
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
sample_submission.shape
train_df.shape, test_df.shape, sample_submission.shape
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

#снижение объёма памяти

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
#Reducing for train data set

train, NAlist = reduce_mem_usage(train_df)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
#Reducing for test data set

test, NAlist = reduce_mem_usage(test_df)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
#Data set fields

train.columns
print(len(train.columns))
print(train.info())
#numerical values Describe

train.describe()
#Visualization

##hist

train['target'].value_counts().plot.bar();
f,ax=plt.subplots(1,2,figsize=(20,10))

train[train['target']==0].var_0.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('target= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train[train['target']==1].var_0.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('target= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
##Mean Frequency

train[train.columns[2:]].mean().plot('hist');plt.title('Mean Frequency');
##countplot

f,ax=plt.subplots(1,2,figsize=(18,8))

train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target',data=train,ax=ax[1])

ax[1].set_title('target')

plt.show()
##hist If you check histogram for all feature, you will find that most of them are so similar

train["var_0"].hist();
train["var_81"].hist();
train["var_2"].hist();
##distplot The target in data set is imbalance

sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(train['target']);
##violinplot

sns.violinplot(data=train,x="target", y="var_0")
sns.violinplot(data=train,x="target", y="var_81")
#Data Preprocessing
#Check missing data for test & train

def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        # written by MJ Bahmani

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)
check_missing_data(train)
check_missing_data(test)
#Binary Classification

train['target'].unique()
#Is data set imbalance?

#A large part of the data is unbalanced, but how can we solve it?

train['target'].value_counts()
def check_balance(df,target):

    check=[]

    # written by MJ Bahmani for binary target

    print('size of data is:',df.shape[0] )

    for i in [0,1]:

        print('for target  {} ='.format(i))

        print(df[target].value_counts()[i]/df.shape[0]*100,'%')
check_balance(train,'target')
#skewness and kurtosis или ассиметрия и куртосис

print("Skewness: %f" % train['target'].skew())

print("Kurtosis: %f" % train['target'].kurt())
#Permutation Importance

##Prepare our data for our model

cols=["target","ID_code"]

X = train.drop(cols,axis=1)

y = train["target"]
X_test  = test.drop("ID_code",axis=1)
##Create a sample model to calculate which feature are more important.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
#How to calculate and show importances?

##ELI5 is a Python library which allows to visualize and debug various Machine Learning models using unified API. It has built-in support for several ML frameworks and provides a way to explain black-box models.

##Here is how to calculate and show importances with the eli5 library:

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)
#Partial Dependence Plots

##partial dependence plots show how a feature affects predictions. And partial dependence plots are calculated after a model has been fit.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
#For the sake of explanation, I use a Decision Tree which you can see below.

features = [c for c in train.columns if c not in ['ID_code', 'target']]
from sklearn import tree

import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)
graphviz.Source(tree_graph)
#Partial Dependence Plot

##In this section, we see the impact of the main variables discovered in the previous sections by using the pdpbox.

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_81')



# plot it

pdp.pdp_plot(pdp_goals, 'var_81')

plt.show()
#Chart analysis

  ##The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.

  ##A blue shaded area indicates level of confidence
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_82')



# plot it

pdp.pdp_plot(pdp_goals, 'var_82')

plt.show()
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_139')



# plot it

pdp.pdp_plot(pdp_goals, 'var_139')

plt.show()
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_110')



# plot it

pdp.pdp_plot(pdp_goals, 'var_110')

plt.show()
#SHAP Value

##SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details).

##Shap can answer to this qeustion : how the model works for an individual prediction?

row_to_show = 5

data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





rfc_model.predict_proba(data_for_prediction_array);
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(rfc_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
#Model Development

##lightgbm

# params is based on following kernel https://www.kaggle.com/brandenkmurray/nothing-works

params = {'objective' : "binary", 

               'boost':"gbdt",

               'metric':"auc",

               'boost_from_average':"false",

               'num_threads':8,

               'learning_rate' : 0.01,

               'num_leaves' : 13,

               'max_depth':-1,

               'tree_learner' : "serial",

               'feature_fraction' : 0.05,

               'bagging_freq' : 5,

               'bagging_fraction' : 0.4,

               'min_data_in_leaf' : 80,

               'min_sum_hessian_in_leaf' : 10.0,

               'verbosity' : 1}
%%time

y_pred_lgb = np.zeros(len(X_test))

num_round = 1000000

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000

                    valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5
##RandomForestClassifier

y_pred_rfc = rfc_model.predict(X_test)
##DecisionTreeClassifier

y_pred_tree = tree_model.predict(X_test)
##CatBoostClassifier

train_pool = Pool(train_X,train_y)

cat_model = CatBoostClassifier(

                               iterations=3000,# change 25 to 3000 to get best performance 

                               learning_rate=0.03,

                               objective="Logloss",

                               eval_metric='AUC',

                              )

cat_model.fit(train_X,train_y,silent=True)

y_pred_cat = cat_model.predict(X_test)
submission_rfc = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_rfc

    })

submission_rfc.to_csv('submission_rfc.csv', index=False)
submission_tree = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_tree

    })

submission_tree.to_csv('submission_tree.csv', index=False)
submission_cat = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_cat

    })

submission_cat.to_csv('submission_cat.csv', index=False)
# good for submit

submission_lgb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_lgb

    })

submission_lgb.to_csv('submission_lgb.csv', index=False)
#Funny Combine

submission_rfc_cat = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": (y_pred_rfc +y_pred_cat)/2

    })

submission_rfc_cat.to_csv('submission_rfc_cat.csv', index=False)
submission_lgb_cat = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": (y_pred_lgb +y_pred_cat)/2

    })

submission_lgb_cat.to_csv('submission_lgb_cat.csv', index=False)
submission_rfc_lgb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": (y_pred_rfc +y_pred_lgb)/2

    })

submission_rfc_lgb.to_csv('submission_rfc_lgb.csv', index=False)