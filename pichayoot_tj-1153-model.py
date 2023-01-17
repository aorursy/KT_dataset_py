# Presentation : https://drive.google.com/file/d/1a72wPWpKC3NgKsWM5G3ZwONDthNzoz9l/view?usp=sharing
import pandas as pd

import numpy as np 



import os



import pandas_profiling



from IPython.core.interactiveshell import InteractiveShell

from IPython.display import display, HTML 

from multiprocessing import Pool

import datetime

import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta



InteractiveShell.ast_node_interactivity = "all"



pd.options.display.max_rows = 99999

pd.options.display.max_columns = 99999





%matplotlib inline



pd.options.display.float_format = '{:,}'.format

train = pd.read_csv('../TJ_1153/train.csv')

train.shape

train.describe()

train.id.nunique()
train.isnull().sum()
demo = pd.read_csv('../TJ_1153/demo.csv')

demo.shape

# demo.describe(percentiles=[.98,.981,.982,.983,.984,.985,.986,.987,.988,.989,.99])
# demo.nunique()
# demo.isnull().sum()
demo['flg_n1_null']=np.where(demo.n1.isnull(),1,0)
demo =demo.fillna(0.0)
# demo.isnull().sum()
train_df = pd.merge(train,demo,on='id',how='left')

train_df.shape
txn = pd.read_csv('../TJ_1153/txn.csv')

txn.shape

txn.describe()
# txn.isnull().sum()
get_text = ~txn.t0.str.isdigit()&~txn.t0.str.startswith('-')
# txn.head()

# txn.shape


txn['c7_new']= txn.c7.apply(lambda x: "{:02d}".format(int(x)))

txn['c5_new']= txn.c5.apply(lambda x: "{:02d}".format(int(x)))

txn['mcc_code'] = txn.c7_new.astype(str) +txn.c5_new.astype(str)

txn.drop(columns=['c7_new','c5_new'],inplace=True)

txn.head()

txn['flg_t0_text'] = np.where(get_text,1,0)
del get_text
oldcard = txn[['id','old_cc_no','old_cc_label']].copy()

oldcard.drop_duplicates(subset=['id','old_cc_no','old_cc_label'],inplace=True)
oldcard.head()
oldcard = oldcard.pivot_table(index='id',columns='old_cc_label',values='id',aggfunc='size').fillna(0)



oldcard.reset_index(inplace=True)
oldcard.columns = ['old_cc_label_'+str(col) for col in oldcard.columns]

oldcard.rename(columns={'old_cc_label_id':'id'},inplace=True)
oldcard.head()
oldcard.id.nunique()

oldcard.shape
train_df = pd.merge(train_df,oldcard,on='id',how='left')

train_df.shape
del oldcard
txn.describe()
txnbyid=txn.groupby(['id']).agg({

                                            'n3':['nunique','max'],

                                            'n4':['count','sum','max','min','mean','median','std'],

                                            'n5':['count','sum','max','min','mean','median','std'],

                                            

                                            'n6':['count','sum','max','min','mean','median','std'],

                                            'n7':['count','sum','max','mean','min','median','std']              

                                                           })

txnbyid.columns =txnbyid.columns.map("_".join)



# kplus_bymth.rename(columns={'id_':'id','kplus_mth_':'kplus_mth','kplus_mthweek_nunique':'kplus_nweek'},inplace=True)

txnbyid.reset_index(inplace=True)

txnbyid.head()



txnbyid.id.nunique()

txnbyid.shape
train_df = pd.merge(train_df,txnbyid,on='id',how='left')

train_df.shape
del txnbyid
plus_n4_df = txn[txn.n4>=0].copy()

minus_n4_df = txn[txn.n4<0].copy()

minus_n4_df.n4 = minus_n4_df.n4.abs()
plus_n4_df = plus_n4_df.groupby(['id']).agg({ 'n4':['count','sum','max','min','mean','median','std'] })

plus_n4_df.columns =plus_n4_df.columns.map("_plus_".join)

plus_n4_df.reset_index(inplace=True)

plus_n4_df.head()



minus_n4_df = minus_n4_df.groupby(['id']).agg({ 'n4':['count','sum','max','min','mean','median','std'] })

minus_n4_df.columns =minus_n4_df.columns.map("_minus_".join)

minus_n4_df.reset_index(inplace=True)

minus_n4_df.head()
minus_n4_df.id.nunique()

minus_n4_df.shape

plus_n4_df.id.nunique()

plus_n4_df.shape
train_df = pd.merge(train_df,minus_n4_df,on='id',how='left')

train_df.shape

train_df = pd.merge(train_df,plus_n4_df,on='id',how='left')

train_df.shape

del minus_n4_df,plus_n4_df
train_df = train_df.fillna(0)
train_df.head()
txn.head()
train_df.to_csv('prep_001.txt',sep="|",index=False)
train_df = pd.read_csv('prep_001.txt',sep="|")
train_df.set_index('id',inplace=True)
train_df.shape

train_df.head()
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

list_cat = ['c0','c1','c2','c3','c4']

train_df['c0'] = labelencoder_X.fit_transform(train_df['c0'])

train_df['c1'] = labelencoder_X.fit_transform(train_df['c1'])

train_df['c2'] = labelencoder_X.fit_transform(train_df['c2'])

train_df['c3'] = labelencoder_X.fit_transform(train_df['c3'])

train_df['c4'] = labelencoder_X.fit_transform(train_df['c4'])



train_df.head()
# split
seed=12345

from sklearn.model_selection import train_test_split

import numpy as np

x_tmp, x_test, y_tmp, y_test = train_test_split(train_df.drop('label',axis=1), train_df['label'], 

                                                    test_size=0.2, random_state=seed)

test_df = pd.concat([x_test,y_test], axis=1)



x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=0.25, random_state=seed)



x_train.shape

x_test.shape

x_val.shape

y_train.value_counts(normalize=True)*100.0

y_test.value_counts(normalize=True)*100.0

y_val.value_counts(normalize=True)*100.0
y_test = np.array(y_test)

y_train = np.array(y_train)

y_val = np.array(y_val)
import lightgbm as lgb
train_df.label.nunique()
import pandas as pd

from collections import Counter



def get_class_weights(y):

    counter = Counter(y)

    majority = max(counter.values())

    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}



class_weights = get_class_weights(train_df.label.values)

print(class_weights)
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, GroupKFold







# param_grid = {

#           "objective" : "multiclass",

#         "metric": "logloss"

#           "num_class" : 13,

#           "num_leaves" : 60,

#           "max_depth": -1,

#           "learning_rate" : 0.01,

#           "bagging_fraction" : 0.9,  # subsample

#           "feature_fraction" : 0.9,  # colsample_bytree

#           "bagging_freq" : 5,        # subsample_freq

#           "bagging_seed" : 2018,

#           "verbosity" : -1 }



# param_grid = {

#     'n_estimators': [400, 700, 1000],

#     'colsample_bytree': [0.7, 0.8],

#     'max_depth': [15,20,25],

#     'num_leaves': [50, 100, 200],

#     'reg_alpha': [1.1, 1.2, 1.3],

#     'reg_lambda': [1.1, 1.2, 1.3],

#     'min_split_gain': [0.3, 0.4],

#     'subsample': [0.7, 0.8, 0.9],

#     'subsample_freq': [20],

#     'random_state':[4484],

#     'learning_rate':[0.01,0.001],

#     "objective" : ["multiclass"],

#     "num_class" : [13]

# }



# model = lgb.LGBMClassifier(param_grid)

# lgbmodel = model.fit(x_train, y_train,eval_set=[(x_val,y_val)], eval_metric='auc', early_stopping_rounds=200, verbose=20,categorical_feature=lst_cat_col_sel)





model = lgb.LGBMClassifier(boosting_type='gbdt', 

        objective='multiclass',    

                          

#                           num_boost_round=50,

        eval_metric='multi_logloss',

        num_class = 13,

        num_leaves=60, 

        max_depth=-1, 

        learning_rate=0.01, 

        n_estimators=10000, 

      

        random_state=4484,

        silent=False, 

        class_weight=class_weights

        

#         num_leaves =31,

#         reg_alpha= 0.1

                          #importance_type='gain',

#        nthread=-1

                         )







model.fit(x_train, y_train, eval_metric='multi_logloss',

        eval_set=[(x_val, y_val)],

          early_stopping_rounds=200,

          categorical_feature=list_cat,

          

        verbose = True)
from sklearn.externals import joblib

joblib.dump(model, './lgbm_01.pkl')
clf = joblib.load('./lgbm_01.pkl')

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
pred_prob_train = clf.predict_proba(x_train, num_iteration=clf.best_iteration_)

pred_class_train = np.argmax(pred_prob_train, axis=1)



cnf_matrix = confusion_matrix(y_train, pred_class_train)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9','10','11','12'],

                      title='Confusion matrix, without normalization')
pred_prob_val = clf.predict_proba(x_val, num_iteration=clf.best_iteration_)

pred_class_val = np.argmax(pred_prob_val, axis=1)



cnf_matrix = confusion_matrix(y_val, pred_class_val)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9','10','11','12'],

                      title='Confusion matrix, without normalization')
pred_prob_test = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)

pred_class_test = np.argmax(pred_prob_test, axis=1)



cnf_matrix = confusion_matrix(y_test, pred_class_test)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9','10','11','12'],

                      title='Confusion matrix, without normalization')
x_train.head()
backtest = pd.read_csv('./preptest_001.txt',sep="|")
backtest.set_index('id',inplace=True)

backtest.head()
pred_prob_backtest = clf.predict_proba(backtest, num_iteration=clf.best_iteration_)
['class'+str(i) for i in list(range(12))]
np.round(pred_prob_backtest,2)
pd.DataFrame(np.round(pred_prob_backtest,2),columns= ['class'+str(i) for i in list(range(13))])
df2 = pd.DataFrame(pred_prob_backtest,columns= ['class'+str(i) for i in list(range(13))])

dfx = pd.DataFrame(backtest.index,columns=['id'])

result = pd.concat([ dfx, df2], axis=1,ignore_index=True)

result.head()
['id']+['class'+str(i) for i in list(range(13))]
result.columns=['id']+['class'+str(i) for i in list(range(13))]

display(backtest.shape)

display(len(pred_prob_backtest))

display(result.dtypes)

display(result.shape)

result.head()
result2 =result.copy()

result2.set_index('id',inplace=True)

result2.sum(axis=1).sum()
result.to_csv('test_1158_p_1.csv',sep=",",index=False)
pd.read_csv('test_1158_p_1.csv',sep=",").head()
#------END