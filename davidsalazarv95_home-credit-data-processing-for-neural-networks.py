from fastai.imports import *
from fastai.structured import *
from fastai.column_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import gc 
%matplotlib inline
%reload_ext autoreload
%autoreload 2
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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
application_train.head()
def find_cats_non_cats(df_, grouper):
    
    df = df_.copy().drop(grouper, axis = 1)
    cats = [col for col in df.columns if df[col].dtype == "object"]
    non_cats = list(set(df.columns) - set(cats))
    return cats, non_cats
def aggregate_heuristics(df, grouper):
    cats_, non_cats_ = find_cats_non_cats(df, grouper)
    cats = cats_[:] + grouper
    non_cats = non_cats_[:] + grouper
    
    if not cats_:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[non_cats].groupby(grouper).mean(),
               left_index = True, right_index = True))
    if not non_cats_:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[cats].groupby(grouper).agg(lambda x: scipy.stats.mode(x)[0]),
               left_index = True, right_index = True))
    else:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[cats].groupby(grouper).agg(lambda x: scipy.stats.mode(x)[0]),
               left_index = True, right_index = True).
               merge(df[non_cats].groupby(grouper).mean(),
                    left_index = True, right_index = True))
# Bureau Data
bureau = reduce_mem_usage(pd.read_csv("../input/bureau.csv"))
bureau_balance = reduce_mem_usage(pd.read_csv("../input/bureau_balance.csv"))

bureau_data = bureau.merge(aggregate_heuristics(bureau_balance, ['SK_ID_BUREAU']), left_on = 'SK_ID_BUREAU', right_index = True)
avg_bureau_data = aggregate_heuristics(bureau_data, ['SK_ID_CURR'])
app_train_bureau = application_train.merge(avg_bureau_data, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
app_test_bureau = application_test.merge(avg_bureau_data, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
del bureau_data, avg_bureau_data
gc.collect()
# Previous credits Data
credit_card_balance = reduce_mem_usage(pd.read_csv("../input/credit_card_balance.csv"))
avg_credi_card = aggregate_heuristics(credit_card_balance, ['SK_ID_PREV', 'SK_ID_CURR'])
del credit_card_balance
gc.collect()
installments_payments = reduce_mem_usage(pd.read_csv("../input/installments_payments.csv"))
avg_installment = aggregate_heuristics(installments_payments, ['SK_ID_PREV', 'SK_ID_CURR'])
del installments_payments
gc.collect()
previous_application = reduce_mem_usage(pd.read_csv("../input/previous_application.csv"))
avg_previous = aggregate_heuristics(previous_application, ['SK_ID_PREV', 'SK_ID_CURR'])
avg_previous.head()
POS_CASH_balance = reduce_mem_usage(pd.read_csv("../input/POS_CASH_balance.csv"))
avg_cash_balance = aggregate_heuristics(POS_CASH_balance, ['SK_ID_PREV', 'SK_ID_CURR'])
del previous_application, POS_CASH_balance
gc.collect()
previous_final = avg_installment.merge(avg_credi_card, left_index = True, right_index = True, how = 'outer', suffixes = ['_installment','_credit']).merge(avg_cash_balance, left_index = True, right_index = True, how = 'outer', suffixes = ['__','_balance']).merge(avg_previous, left_index = True, right_index = True, how = 'outer', suffixes = [':__','_previous'])
train_cats(previous_final)
previous_final.dtypes.value_counts()
previous_final_agg = aggregate_heuristics(previous_final.reset_index(), ['SK_ID_CURR'])
app_bureau_previous = pd.merge(app_train_bureau, previous_final_agg, on = 'SK_ID_CURR', how = 'left')
app_test_bureau_previous = pd.merge(app_test_bureau, previous_final_agg, on = 'SK_ID_CURR', how = 'left')
del previous_final, previous_final_agg
gc.collect()
app_bureau_previous.dtypes.value_counts()
app_test_bureau_previous.dtypes.value_counts()
app_bureau_previous.to_feather('tables_merged_train')
app_test_bureau_previous.to_feather('tables_merged_test')
#train = pd.read_feather('../')
#df, y, nas, mapper = proc_df(app_bureau_previous, 'TARGET', do_scale=True)
#df_test, _, nas, mapper = proc_df(app_test_bureau_previous, na_dict=nas, do_scale = True, mapper = mapper)

