import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score

import gc



from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
folder_path = '../input/ieee-fraud-detection/'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv', index_col='TransactionID')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv', index_col='TransactionID')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv', index_col='TransactionID')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv', index_col='TransactionID')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

del train_identity, train_transaction, test_identity, test_transaction

gc.collect()
def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]

    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',

                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',

                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',

                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',

                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',

                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',

                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',

                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',

                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',

                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',

                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',

                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',

                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',

                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',

                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',

                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',

                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id']
train = id_split(train)

test = id_split(test)
cols_to_drop = [col for col in train.columns if col not in useful_features]

cols_to_drop.remove('isFraud')

cols_to_drop.remove('TransactionDT')

train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
columns_a = ['TransactionAmt', 'id_02', 'D15']

columns_b = ['card1', 'card4', 'addr1']



for col_a in columns_a:

    for col_b in columns_b:

        for df in [train, test]:

            df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')

            df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')
# New feature - log of transaction amount.

train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])

test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])



# New feature - decimal part of the transaction amount.

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# New feature - day of week in which a transaction happened.

train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)



# New feature - hour of the day in which a transaction happened.

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))
# Encoding - count encoding for both train and test

for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:

    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))



# Encoding - count encoding separately for train and test

for feature in ['id_01', 'id_31', 'id_33', 'id_36']:

    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499



emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']



for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
for col in train.columns:

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))

X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)

y = train.sort_values('TransactionDT')['isFraud']



X_test = test.drop(['TransactionDT'], axis=1)



del train, test

gc.collect()
X = reduce_mem_usage(X)

# y = reduce_mem_usage(y)

X_test = reduce_mem_usage(X_test)
import xgboost as xgb



clf = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

#     missing=-999,

    random_state=51,

    tree_method='gpu_hist'  # THE MAGICAL PARAMETER

)

import xgboost as xgb



clf = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

#     missing=-999,

    random_state=51,

    tree_method='gpu_hist'  # THE MAGICAL PARAMETER

)

clf.fit(X, y)
y_train_pred = clf.predict_proba(X)[:,1]

print('AUC score: ',roc_auc_score(y, y_train_pred))
y_pred = clf.predict_proba(X_test)[:,1]

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

sample_submission['isFraud'] = y_pred

sample_submission.to_csv('submission1.csv')
import sklearn.metrics as metrics



def create_eval_metric_df(y,y_pred,model_name='model'):

    

    total = np.shape(y)[0]

    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()

    tp /= float(total)

    tn /= float(total)

    fp /= float(total)

    fn /= float(total)

    roc_auc = metrics.roc_auc_score(y, y_pred)

    

    columns_list = ['model_name',

    "total_number", 

    "cond_pos", "cond_neg", 

    "true_pos", "true_neg", "false_pos", "false_neg",

    "recall", "specificity", "precision", "neg_pred_val", "miss_rate", "false_pos_rate", "false_disc_rate", "false_omission_rate",

    "accuracy", "f1_score", "matthews_corr_coef", "informedness", "markedness",

    "roc_auc_score"

  ]

    

    eval_metrics = [model_name,

      total,

      tp+fn, tn+fp,

      tp, tn, fp, fn,

      (tp)/(tp + fn), (tn)/(tn + fp), (tp)/(tp + fp), (tn)/(tn + fn), (fn)/(fn + tp), (fp)/(fp + tn), (fp)/(fp + tp), (fn)/(fn + tn),

      (tp + tn), (2*tp)/(2*tp + fp + fn), (tp*tn - fp*fn)/(np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))), (tp)/(tp + fn) + (tn)/(tn + fp) - 1, (tp)/(tp + fp) + (tn)/(tn + fn) - 1,

      roc_auc

    ]

    

    df_eval = pd.DataFrame(eval_metrics)

    df = df_eval.T

    df.columns = columns_list

    return df

y_train_pred = clf.predict(X)
df_eval = create_eval_metric_df(y,y_train_pred,model_name='xgboost1')

print(df_eval)
import matplotlib.gridspec as gridspec

import matplotlib.patches as patches

def plot_confusion_matrix_train_test(df_train, df_test, model_names, display_image=False):

  """

  Plot confusion matrix from given train and test eval metric dataset and list of model.

  df = eval metric dataset from create_eval_metric_df function

  model_names = list of model names

  """

  #model_names = best_models_dict.keys()[:]

  n_plot = np.shape(model_names)[0]



  fig_h = n_plot*8

  fig_w = 14



  plot_h = 8

  plot_w = 10



  plt.close()

  fig = plt.figure(figsize=(fig_w, fig_h))  



  gs = gridspec.GridSpec(plot_h*n_plot, plot_w, wspace=0, hspace=0)



  common_dict = {

    'facecolor':'xkcd:light grey', 

    'edgecolor':'k', 

    'fontsize':10

  }



  gs_coors =[

    [0, 1, 1, -1],

    [2, 3, 1, 2],

    [3, 5, 0, 1],

    [1, 2, 2, 6],

    [2, 3, 2, 4],

    [2, 3, 4, 6],

    [3, 4, 1, 2],

    [4, 5, 1, 2],

    [3, 4, 2, 4],

    [4, 5, 2, 4],

    [3, 4, 4, 6],

    [4, 5, 4, 6],

    [5, 6, 2, 4],

    [5, 6, 4, 6],

    [3, 4, 6, 8],

    [4, 5, 6, 8],

    [2, 3, 6, 8],

    [6, 7, 2, 4],

    [6, 7, 4, 6],

    [3, 4, 8, 10],

    [4, 5, 8, 10],

    [5, 6, 6, 8],

    [6, 7, 6, 8]

  ]



  kwarg_list = [common_dict for l in range(np.shape(gs_coors)[0])]

  kwarg_list[0] = {

      'facecolor':'w', 

      'edgecolor':'w', 

      'fontsize':24

    }



  for i, model_name in enumerate(model_names):

    print(i)



    pd_mod = df_train[df_train["model_name"] == model_name]

    pd_mod_test = df_test[df_test["model_name"] == model_name]



    text_lists = [

      "{}".format(model_name),

      "Total\nTrain:{}\nTest:{}".format(int(pd_mod["total_number"].sum()), int(pd_mod_test["total_number"].sum())),

      "Predicted",

      "True Condition",

      "Condition Positive\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["cond_pos"].sum()), (pd_mod_test["cond_pos"].sum())),

      "Condition Negative\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["cond_neg"].sum()), (pd_mod_test["cond_neg"].sum())),

      "Pred Positive\nTrain:{:.02%}\nTest:{:.02%}".format((pd_mod["true_pos"].sum())+(pd_mod["false_pos"].sum()), (pd_mod_test["true_pos"].sum())+(pd_mod_test["false_pos"].sum())),

      "Pred Negative\nTrain:{:.02%}\nTest:{:.02%}".format((pd_mod["true_neg"].sum())+(pd_mod["false_neg"].sum()), (pd_mod_test["true_neg"].sum())+(pd_mod_test["false_neg"].sum())),

      "True Pos\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["true_pos"].sum()), (pd_mod_test["true_pos"].sum())),

      "False Neg\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["false_neg"].sum()), (pd_mod_test["false_neg"].sum())),

      "False Pos\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["false_pos"].sum()), (pd_mod_test["false_pos"].sum())),

      "True Neg\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["true_neg"].sum()), (pd_mod_test["true_neg"].sum())),

      "Recall\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["recall"].sum()), (pd_mod_test["recall"].sum())),

      "False Positive Rate\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["false_pos_rate"].sum()), (pd_mod_test["false_pos_rate"].sum())),

      "Precision\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["precision"].sum()), (pd_mod_test["precision"].sum())),

      "False omission rate\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["false_omission_rate"].sum()), (pd_mod_test["false_omission_rate"].sum())),

      "Accuracy\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["accuracy"].sum()), (pd_mod_test["accuracy"].sum())),

      "Miss Rate\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["miss_rate"].sum()), (pd_mod_test["miss_rate"].sum())),

      "Specificity\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["specificity"].sum()), (pd_mod_test["specificity"].sum())),

      "False Discovery Rate\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["false_disc_rate"].sum()), (pd_mod_test["false_disc_rate"].sum())),

      "Negative Predictive Value\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["neg_pred_val"].sum()), (pd_mod_test["neg_pred_val"].sum())),

      "F1 Score\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["f1_score"].sum()), (pd_mod_test["f1_score"].sum())),

      "ROC AUC Score\nTrain:{:.02%},\nTest:{:.02%}".format((pd_mod["roc_auc_score"].sum()), (pd_mod_test["roc_auc_score"].sum()))

    ]



    for k in range(np.shape(text_lists)[0]):



      gs_coor = gs_coors[k]

      ax = plt.subplot(gs[(gs_coor[0]+ plot_h*i):(gs_coor[1]+ plot_h*i), gs_coor[2]:gs_coor[3]])

      plot_text(text_lists[k], ax, **kwarg_list[k])



  plt.subplots_adjust(wspace=0, hspace=0)

  plt.tight_layout()

  if display_image:

    display(fig)

  else:

    return fig



def plot_text(text, ax, facecolor='xkcd:light grey', edgecolor='b', fontsize=12):

  """

  Simple function to plot a text in given axis.

  """

  ax.set_axis_off()

  p = patches.Rectangle(

    (-0, -0), 1, 1,

    fill=True, transform=ax.transAxes, clip_on=False, facecolor=facecolor, edgecolor=edgecolor, linestyle='-',linewidth=1

    )



  ax.add_patch(p)

  ax.text(0.5, 0.5, text, fontsize=fontsize, multialignment="center", ha='center', va='center')

  ax.set_xlim(0,1)

  ax.set_ylim(0,1)
model_names = ['xgboost1']

plot_confusion_matrix_train_test(df_eval, df_eval, model_names, display_image=False)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
import xgboost as xgb



clf2 = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

#     missing=-999,

    random_state=51,

    tree_method='gpu_hist'  # THE MAGICAL PARAMETER

)

clf2.fit(X_train,y_train)
y_train_pred = clf2.predict(X_train)

y_val_pred = clf2.predict(X_val)

print('AUC score train: ',roc_auc_score(y_val, y_val_pred))

print('AUC score val: ',roc_auc_score(y_val, y_val_pred))
df_eval_train = create_eval_metric_df(y_train,y_train_pred,model_name='xgboost2')

df_eval_val = create_eval_metric_df(y_val,y_val_pred,model_name='xgboost2')
model_names = ['xgboost2']

plot_confusion_matrix_train_test(df_eval_train, df_eval_val, model_names, display_image=False)
def plot_dist_class(df, model_names, class_selected=1, display_image=True):

  """

  Plot proba distribution from prediction result.

  Specific columns name.

  """

  

  n_plot = np.shape(model_names)[0]

  

  fig_h = n_plot*4

  fig_w = 12

  

  plot_h = 4

  plot_w = 4

  

  plt.close()

  fig = plt.figure(figsize=(fig_w, fig_h))  

  

  for i, model_name in enumerate(model_names):

    

    ax = plt.subplot2grid((plot_h*n_plot, plot_w), (0 + i*plot_h,0), rowspan=4, colspan=4)

    sns.kdeplot(df["prob_c{}_{}".format(class_selected, model_name)], 

                ax=ax, 

                label="Probability of Class {}".format(class_selected),

                clip = [-0.05, 1.05]

               )



    ax.set_xlabel("Probability")

    ax.set_xlim([-0.05, 1.05])

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:.02%}".format(x)))

    ax.set_title("Distribution of Probability of Class {} from {} Model".format(class_selected, model_name))

    

  plt.tight_layout()

    

  if display_image:

    display(fig)

  else:

    return fig
plot_dist_class(df, model_names, class_selected=1, display_image=True)
clfFiveFold = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

#     missing=-999,

    random_state=51,

    tree_method='gpu_hist'  # THE MAGICAL PARAMETER

)

%%time



from sklearn.model_selection import KFold

from tqdm import tqdm_notebook

import time

NFOLDS = 5

folds = KFold(n_splits = NFOLDS)



splits = folds.split(X,y)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = X.columns



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    clfFiveFold.fit(X_train, y_train)

    y_pred_train = clfFiveFold.predict(X_train)

    print(f"Fold {fold_n + 1} | Training AUC: {roc_auc_score(y_train, y_pred_train)}")

    y_pred_valid = clfFiveFold.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | Validation AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clfFiveFold.predict(X_test) / NFOLDS

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
y_train_pred = clfFiveFold.predict(X)

print('AUC score train: ',roc_auc_score(y, y_train_pred))

print('AUC score OOF: ',roc_auc_score(y, y_oof))

df_eval_train = create_eval_metric_df(y,y_train_pred,model_name='xgboost2')

df_eval_oof = create_eval_metric_df(y,y_oof,model_name='xgboost2')
model_names = ['xgboost2']

plot_confusion_matrix_train_test(df_eval_train, df_eval_oof, model_names, display_image=False)
def plot_class_probability_dist(X_train, y_train, X_test, y_test, best_models_dict,hist=True):

  f, ax = plt.subplots(len(list(best_models_dict.keys())), 2, figsize = (20, 10))

  i = 0

  

  

  for key, value in best_models_dict.items():

    if key == "ANN":

      y_train_pred_proba = value.predict(X_train_scaled).ravel()

    else:

      y_train_pred_proba = value.predict_proba(X_train)[:,1]

    df_train = pd.DataFrame({"class" : y_train, "prob" : y_train_pred_proba})

    sns.distplot(df_train.loc[df_train["class"] == 0, "prob"], ax = ax[i,0], label = 0, hist=hist)

    sns.distplot(df_train.loc[df_train["class"] == 1, "prob"], ax = ax[i,0], label = 1, hist=hist)

    ax[i,0].set_title(str(key) + " Train")

    ax[i,0].set_xlim([0,1.01])

    ax[i,0].legend()

    

    if key == "ANN":

      y_test_pred_proba = value.predict(X_test_scaled).ravel()

    else:

      y_test_pred_proba = value.predict_proba(X_test)[:,1]

    df_test = pd.DataFrame({"class" : y_test, "prob" : y_test_pred_proba})

    sns.distplot(df_test.loc[df_test["class"] == 0, "prob"], ax = ax[i,1], label = 0, hist=hist)

    sns.distplot(df_test.loc[df_test["class"] == 1, "prob"], ax = ax[i,1], label = 1, hist=hist)

    ax[i,1].set_title(str(key) + " Test")

    ax[i,1].set_xlim([0,1.01])

    ax[i,1].legend()

    

    i = i+1

    

  plt.tight_layout()

  display()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
best_models_dict = {'no-cv': clf,'cv1': clf2,'cv5': clfFiveFold}

plot_class_probability_dist(X_train, y_train, X_val, y_val, best_models_dict, hist=False)
import eli5
X_val.iloc[0]
eli5.show_weights(clf)
eli5.show_weights(clf2)
eli5.show_weights(clfFiveFold)
# To fix ELI5 incompatibility issue with latest XGB

# https://stackoverflow.com/questions/53783731/eli5-show-prediction-not-showing-probability



from xgboost import XGBClassifier, XGBRegressor

def _check_booster_args(xgb, is_regression=None):

    # type: (Any, bool) -> Tuple[Booster, bool]

    if isinstance(xgb, eli5.xgboost.Booster): # patch (from "xgb, Booster")

        booster = xgb

    else:

        booster = xgb.get_booster() # patch (from "xgb.booster()" where `booster` is now a string)

        _is_regression = isinstance(xgb, XGBRegressor)

        if is_regression is not None and is_regression != _is_regression:

            raise ValueError(

                'Inconsistent is_regression={} passed. '

                'You don\'t have to pass it when using scikit-learn API'

                .format(is_regression))

        is_regression = _is_regression

    return booster, is_regression



eli5.xgboost._check_booster_args = _check_booster_args
colnames = list(X_val.columns)

elist = []

for i in range(10):

    e = eli5.explain_prediction(clf2, X_val.iloc[i], top=10, feature_names=colnames)

    elist.append(e)
for i in range(10):

    etext = eli5.formatters.text.format_as_text(elist[i])

    print(etext)

    print('------')

df_eli = eli5.formatters.as_dataframe.explain_prediction_df(clf2, X_val.iloc[1], top=10, feature_names=colnames)
print(df_eli)
x = eli5.explain_prediction_xgboost(clfFiveFold, X_val.iloc[1])

print(x)