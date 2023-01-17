import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

import seaborn as sns



import category_encoders as ce

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold,TimeSeriesSplit

import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

#display notebook in full width

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



warnings.filterwarnings("ignore", message="categorical_feature in Dataset is overridden")

warnings.filterwarnings("ignore", message="categorical_feature in param dict is overridden")

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples")



random_state = 42



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
def get_class_from_prob(y_prob):

        return  [0  if x < THR else 1 for x in y_prob]

    

def write_predictions(sub, y_prob, filename ):          

#     sub_prob = sub.copy()

    sub[TARGET] = y_prob    

    sub.to_csv( filename, index = False)

    

    y_pred =   get_class_from_prob(y_prob) 

    print('Test prediction:Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))

    return sub





def set_ordinal_encoding(data, cat_cols):       

    for col in [x for x in cat_cols if data[x].dtype == 'object']:

        data[col], uniques = pd.factorize(data[col])

        #the factorize sets null values to -1, so convert them back to null, as we want LGB to handle null values

        data[col] = data[col].replace(-1, np.nan)

    print('Finished: Ordinal Encoding')

    return data



def get_train_test(df, features, ID):

    X_train =  df[df[TARGET].notnull()]

    X_test  =  df[df[TARGET].isnull()]

    y_train = X_train[TARGET]

    sub = pd.DataFrame()

    sub[ID] = X_test[ID]

    X_train = X_train[features]

    X_test = X_test[features]

    return X_train, X_test, y_train, sub



def plot_feature_imp(feature_imp, top_n = 30):

    feature_imp = feature_imp.sort_values(['importance'], ascending = False)

    feature_imp_disp = feature_imp.head(top_n)

    plt.figure(figsize=(10, 12))

    sns.barplot(x="importance", y="feature", data=feature_imp_disp)

    plt.title('LightGBM Features')

    plt.show() 

    





def cv_results(y_valid, y_prob, verbose = True):   

    scores = {}                      

    y_pred_class =  [0  if x < 0.5 else 1 for x in y_prob]

    scores['cv_accuracy']  = accuracy_score(y_valid, y_pred_class)

    scores['cv_auc']       = roc_auc_score(y_valid, y_prob)

    scores['cv_f1']      =   f1_score(y_valid, y_pred_class, average = 'binary')

    if verbose:

        print('CV accuracy {:0.6f}'.format( scores['cv_accuracy'] ))

        print('CV AUC  {:0.6f}'.format( scores['cv_auc']   ))

        print('CV F1 %0.6f' %scores['cv_f1'] )

    return scores  





def run_lgb_with_cv(params, X_train, y_train, X_test,cat_cols, shuffle_split = True, 

                    test_size =0.2, verbose_eval = 100, esr = 300 ):

    

    if shuffle_split: 

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = test_size , 

                                                  random_state = random_state, stratify = y_train)

    else:

        #since data is sorted according to time, the split will be according to time with shuffle = False

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = test_size , 

                                                      random_state = random_state, shuffle = False)

  

    print('Train shape{} Valid Shape{}, Test Shape {}'.format(X_train.shape, X_valid.shape, X_test.shape))

    print('Number of Category Columns {}:'.format(len(cat_cols)))



    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_valid  = lgb.Dataset(X_valid, y_valid)

   

    lgb_results = {}    

   

    

    model = lgb.train(params,

                      lgb_train,

                      num_boost_round = 10000,

                      valid_sets =  [lgb_train,lgb_valid],  #Including train set will do early stopiing for train instead of validation

                      early_stopping_rounds = esr,                      

                      categorical_feature = cat_cols,

                      evals_result = lgb_results,

                      verbose_eval = verbose_eval

                       )

    y_prob_valid = model.predict(X_valid)    

    cv_results(y_valid, y_prob_valid, verbose = True)

  

    feature_imp = pd.DataFrame()

    feature_imp['feature'] = model.feature_name()

    feature_imp['importance']  = model.feature_importance()

    feature_imp = feature_imp.sort_values(by = 'importance', ascending= False )

    return model, feature_imp, lgb_results



def run_lgb_no_cv(params, X_train, y_train, X_test,cat_cols,num_rounds = 100, verbose_eval = 100, ):    

      

    print('Train shape{}  Test Shape {}'.format(X_train.shape, X_test.shape))

    print('Number of Category Columns {}, Number of Rounds {}:'.format(len(cat_cols), num_rounds))

   

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_results = {}    



    warnings.filterwarnings("ignore", message="categorical_feature in Dataset is overridden")

    model = lgb.train(params,

                      lgb_train,

                      num_boost_round = num_rounds,                 

                      categorical_feature = cat_cols,

                      evals_result = lgb_results,

                      valid_sets =  [lgb_train],

                      verbose_eval = verbose_eval

                       )

  

    feature_imp = pd.DataFrame()

    feature_imp['feature'] = model.feature_name()

    feature_imp['importance']  = model.feature_importance()

    feature_imp = feature_imp.sort_values(by = 'importance', ascending= False )

    return model, feature_imp



def plot_results(history):

    tr_auc =history['training']['auc']

    val_auc = history['valid_1']['auc']

   

    n_iter = list(range(1, len(history['training']['auc']) + 1))



    plt.figure(figsize = (12, 6))

#     plt.subplot(1,2,1)

    plt.plot(n_iter , tr_auc, 'b', label = 'Training AUC')

    plt.plot(n_iter , val_auc, 'r', label = 'Validation AUC')

    plt.grid(True)

    plt.legend()

    plt.xlabel('iteration')  



    plt.show()
%%time

# data_path = 'data'

data_path = '/kaggle/input/av-amex-coupon/data/'

TARGET = 'redemption_status'

THR = 0.5



train =  pd.read_csv(os.path.join(data_path, 'train.csv'))

test =  pd.read_csv(os.path.join(data_path, 'test.csv'))

campaign = pd.read_csv(os.path.join(data_path, 'campaign_data.csv'))

coupon  = pd.read_csv(os.path.join(data_path, 'coupon_item_mapping.csv'))

cust_demo = pd.read_csv(os.path.join(data_path, 'customer_demographics.csv'))

cust_trans = pd.read_csv(os.path.join(data_path, 'customer_transaction_data.csv'))

item = pd.read_csv(os.path.join(data_path, 'item_data.csv'))

data = train.append(test, ignore_index = True, sort = False)

print('Train  - rows:', train.shape[0], 'columns:', train.shape[1])

print('Test  - rows:',  test.shape[0], 'columns:', test.shape[1])

print('Campaign  - rows:',  campaign.shape[0], 'columns:', campaign.shape[1])

print('Coupon  - rows:',  coupon.shape[0], 'columns:', coupon.shape[1])

print('Cust_demo   - rows:',  cust_demo.shape[0], 'columns:', cust_demo .shape[1])

print('Cust_trans   - rows:',  cust_trans.shape[0], 'columns:', cust_trans.shape[1])

print('item  - rows:',  item.shape[0], 'columns:', item.shape[1])

print('combined data - rows:',  data.shape[0], 'columns:', data.shape[1])
campaign['start_date'] = pd.to_datetime(campaign['start_date'], format = '%d/%m/%y')

campaign['end_date'] = pd.to_datetime(campaign['end_date'],format = '%d/%m/%y')

cust_trans['date'] = pd.to_datetime(cust_trans['date'])

data = pd.merge(data, campaign, how = 'left', on =['campaign_id'])

data = pd.merge(data, cust_demo, how = 'left', on =['customer_id'])

print(data.shape)

cust_trans = pd.merge(cust_trans, item, how = 'left', on =['item_id'])

cust_trans = cust_trans.drop_duplicates(subset= cust_trans.columns.tolist(), keep= 'first')



# trans_merge = pd.merge(train_merge, cust_trans, how = 'left', on =['customer_id'])

print(cust_trans.shape)

# Number of redemmptions by customer: Overftiing do not use due to target leak

# col ='red_count'

# df = data.groupby(['customer_id']).agg({'redemption_status':'sum'}).reset_index()

# df.columns = ['customer_id', col]

# data = data.merge(df, how = 'left', on = 'customer_id')

# data[data[TARGET] == 0]
# # Number of redemptions before start date of event for customer

# col ='cust_red_before'

# df = data.groupby(['customer_id', 'start_date']).agg({'redemption_status':'sum'})

# df.columns = ['red_count']

# df = df.reset_index()

# df['red_sum'] = df.groupby('customer_id')['red_count'].apply(lambda x : x.cumsum())

# df[col] = df['red_sum'] - df['red_count']





# data = data.merge(df[['customer_id', 'start_date', 'cust_red_before']], on = ['customer_id', 'start_date'], how = 'left')



# # Number of redemptions before start date of event for a coupon

# col ='coup_red_before'

# df = data.groupby(['coupon_id', 'start_date']).agg({'redemption_status':'sum'})

# df.columns = ['red_count']

# df = df.reset_index()

# df['red_sum'] = df.groupby('coupon_id')['red_count'].apply(lambda x : x.cumsum())

# df[col] = df['red_sum'] - df['red_count']

# data = data.merge(df[['coupon_id', 'start_date', col]], on = ['coupon_id', 'start_date'], how = 'left')



# # Redemption Count and or Features

# col = 'red_count_and'

# data[col] =  data['cust_red_before'] *  data['coup_red_before']

# data[col] = data[col].apply(lambda x: 0 if x ==0 else 1)







# col = 'red_count_or'

# data[col] =  data['cust_red_before'] +  data['coup_red_before']

# data[col] =  data[col].apply(lambda x: 0 if x ==0 else 1)







col = 'coup_purc_count'

df = cust_trans.merge(coupon, on = 'item_id', how = 'inner')

df = data.merge(df, on = ['customer_id', 'coupon_id'], how = 'inner')

df = df[df.date < df.start_date]



# agg_map = { 'redemption_status':['mean', 'sum']}

df = df.groupby(['customer_id', 'coupon_id']).agg({'id':'count'})

df.columns = [col]

df = df.reset_index()



data = data.merge(df, on = ['customer_id', 'coupon_id'], how = 'left' )

data[col] = data[col].fillna(0)





# data[data[TARGET] ==1].head()
# Number of redemptions for a coupon by all customers

col = 'coup_disc_count'

df = cust_trans[cust_trans.coupon_discount != 0]

df = df.merge(coupon, on = 'item_id', how = 'inner')



df_train = data.groupby(['coupon_id', 'start_date', 'end_date']).agg({'id':'count'})

df_train = df_train.reset_index()



df = df_train.merge(df, how = 'inner', on = ['coupon_id'])

df = df[df.date < df.start_date]





df = df.groupby('coupon_id').agg({'coupon_id':'count'})

df.columns = [col]

df = df.reset_index()



data = data.merge(df, on = [ 'coupon_id'], how = 'left' )

data[col] = data[col].fillna(0)



data[data[TARGET] ==1].head()

## Magic Feature

col = 'red_magic'

data[col] = data['coup_purc_count'] * data['coup_disc_count']

data[col] = data[col].apply(lambda x: 0 if x ==0 else 1)
# Nunber of trasnactions where coupon redemetion was done before event start date

def find_coupon_counts(df_data):

    col = 'coup_count_b'  



    df = cust_trans[cust_trans.coupon_discount !=0]

    df = df_data.merge(df, how = 'left', on = 'customer_id')

    df  = df[(df.date < df.start_date )]



    #Only keep tranasaction where coupon and item match with scheme

    df = df.merge(coupon, on = ['coupon_id','item_id'], how = 'inner')





    #for a cutomer and coopn find number of transations were coupons was redeemed

    df =  df.groupby(['coupon_id', 'customer_id']).agg({'coupon_id':'count'})

    df.columns = [col]

    df =df.reset_index()



    df_data = pd.merge(df_data, df, how = 'left', on = ['coupon_id', 'customer_id'])

    df_data[col] = df_data[col].fillna(0)

    return df_data



data = find_coupon_counts(data)

# data[data[TARGET] ==1].head()
## Aggregates for customer transaction before event start date

agg_map = {'customer_id': 'count', 'selling_price':'sum' ,'quantity':'sum', 'coupon_discount':'sum', 'other_discount':'sum'}

df = cust_trans.groupby(['date', 'customer_id']).agg(agg_map)



df.columns = ['c_trans_count', 'selling_price_mean', 'quantity_mean', 'coupon_discount_mean', 'other_discount_mean']

df  = df.reset_index()

df_train = data.groupby(['customer_id', 'start_date', 'end_date']).agg({'coupon_id':'count'})

df_train = df_train.reset_index()



df = data.merge(df, how = 'inner', on = ['customer_id'])

df = df[df.date < df.start_date]

df = df.drop_duplicates(subset = ['customer_id', 'date'], keep = 'first')



agg_map = {'c_trans_count':'sum', 'selling_price_mean':'sum' ,'quantity_mean':'sum', 'coupon_discount_mean':'sum', 'other_discount_mean':'sum'}

df = df.groupby(['customer_id']).agg(agg_map)

df = df.reset_index()



for col in ['selling_price_mean' ,'quantity_mean', 'coupon_discount_mean',  'other_discount_mean']:

    df[col] = df[col] / df['c_trans_count']



# df['cd_sp_ratio'] = df['coupon_discount_mean'] / df['selling_price_mean']

# df['cd_od_ratio'] = df['coupon_discount_mean'] / df['other_discount_mean']

# df['od_sp_ratio'] = df['other_discount_mean'] / df['selling_price_mean']

# df['tot_disc_mean'] =  df['coupon_discount_mean']  + df['other_discount_mean']





data = data.merge(df, on = ['customer_id'], how = 'left' )



data[data[TARGET] ==1].head()



## Aggregates for customer transaction with coupon discounts Before start date

# agg_map = {'customer_id': 'count', 'selling_price':'sum' ,'quantity':'sum', 'coupon_discount':'sum', 'other_discount':'sum'}



# df = cust_trans[cust_trans.coupon_discount !=0 ]

# df = df.groupby(['date', 'customer_id']).agg(agg_map)



# df.columns = ['trans_count_d', 'selling_price_mean_d', 'quantity_mean_d', 'coupon_discount_mean_d', 'other_discount_mean_d']

# df  = df.reset_index()

# df_train = data.groupby(['customer_id', 'start_date', 'end_date']).agg({'coupon_id':'count'})

# df_train = df_train.reset_index()



# df = df_train.merge(df, how = 'inner', on = ['customer_id'])

# df = df[df.date < df.start_date]

# df = df.drop_duplicates(subset = ['customer_id', 'date'], keep = 'first')



# agg_map = {'trans_count_d':'sum', 'selling_price_mean_d':'sum' ,'quantity_mean_d':'sum', 'coupon_discount_mean_d':'sum', 'other_discount_mean_d':'sum'}

# df = df.groupby(['customer_id']).agg(agg_map)

# df = df.reset_index()    



# data = data.merge(df, on = ['customer_id'], how = 'left' )





# for col in ['trans_count_d', 'selling_price_mean_d' ,'quantity_mean_d', 'coupon_discount_mean_d',  'other_discount_mean_d']:   

#     data[col] = data[col].fillna(0)     

    

#     if col != 'trans_count_d':

#         data[col] = data[col] / data['trans_count_d']        

#         data[col] = data[col].fillna(0)            





# data['trans_ratio'] = data['trans_count_d'] / data['c_trans_count'] 

# data['disc_ratio'] =  data['coupon_discount_mean_d'] / data['other_discount_mean_d'] 

# data['disc_ratio']  = data['disc_ratio'].fillna( 0)

# data['disc_ratio']  = data['disc_ratio'].replace(-np.inf, 0)

# # data[data[TARGET] ==0].head()



# Nunmber of items per coupon

col = 'coup_item_count'

df = coupon.groupby('coupon_id').agg({'item_id':'count'}).reset_index()

df.columns = ['coupon_id', col]

data = pd.merge(data, df, how = 'left', on = 'coupon_id')



#Frquency Counts

freq_cols = ['campaign_id']

# freq_cols = ['campaign_id','coupon_id', 'customer_id']

for col in freq_cols:

    data[col + '_count'] = data[col].map(data[col].value_counts(dropna=False)) 

    

# Aggregates based on transaction



#Number of coupons per item

col = 'item_count_coup'

df = coupon.groupby('item_id').agg({'coupon_id':'count'}).reset_index()

df.columns = ['item_id', col]

cust_trans  = pd.merge(cust_trans , df, how = 'left', on = 'item_id')

cust_trans[col] = cust_trans[col].fillna(0)



## Aggregate Values

agg_map = {'customer_id': 'count', 'selling_price':'mean' ,'quantity':'mean', 'coupon_discount':'mean', 'other_discount':'mean', 'item_count_coup':'mean'}



df = cust_trans.groupby('customer_id').agg(agg_map)



df.columns = ['trans_count','selling_price','quantity','coupon_discount','other_discount', 'item_count_coup']

df = df.reset_index()

data = pd.merge(data, df, how = 'left', on = 'customer_id')

# data.head()


# Aggregate transaction values where coipon code discount is not zero



agg_map = {'customer_id': 'count', 'selling_price':'mean' ,'quantity':'mean', 'coupon_discount':'mean', 'other_discount':'mean', 'item_count_coup':'mean'}

df = cust_trans[cust_trans.coupon_discount !=0]

df = df.groupby('customer_id').agg(agg_map)

df.columns = ['trans_count_c','selling_price_c','quantity_c','coupon_discount_c','other_discount_c', 'item_count_coup_c']

df = df.reset_index()



data = pd.merge(data, df, how = 'left', on = 'customer_id')



# Ratio of transaction with coupons and total transaction by a customer

col ='trans_ratio'

data[col] =  data['trans_count_c'] / data['trans_count'] 



col ='disc_ratio'

data[col] =  data['coupon_discount_c'] / data['other_discount_c'] 





# data[data[TARGET] == 1].head()
## Aggregates on coupon

col = 'coup_count_all'

df = cust_trans.merge(coupon, on = 'item_id', how = 'inner')

df_grp = df.groupby('coupon_id').agg({'coupon_id':'count'})

df_grp.columns = [col]

df_grp = df_grp.reset_index()

df_grp.head()

# print(df_grp.shape)



col = 'coup_count_disc'

df_disc = df[df.coupon_discount !=0]

df_disc_grp = df_disc.groupby('coupon_id').agg({'coupon_id':'count'})

df_disc_grp.columns = [col]

df_disc_grp = df_disc_grp.reset_index()

df_disc_grp.head()

df_disc_grp



df = df_grp.merge(df_disc_grp, how = 'left', on = 'coupon_id' )

df[col] = df[col].fillna(0)



col = 'coup_count_ratio'

df[col] = df.coup_count_disc / df.coup_count_all

data =  data.merge(df, how = 'left', on = 'coupon_id')



# data[data[TARGET] == 1].head()
all_cols = data.columns.tolist()





cat_cols = ['campaign_type',  'marital_status', 'rented']

delete_col = ['campaign_id','id','start_date','end_date', TARGET,]  

delete_col += ['trans_count'] + ['selling_price','quantity','coupon_discount','other_discount', 'item_count_coup'] + [ 'coup_count_b','coup_item_count','coup_count_all', 'coup_count_disc']

delete_col += ['trans_count_c','selling_price_c','quantity_c','coupon_discount_c','other_discount_c', 'item_count_coup_c']



features = [x for x in all_cols if x not in delete_col]

cat_cols = [x for x in cat_cols  if x in features]



age_map = {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '56-70' : 5, '70+':6}

data['age_range'] = data['age_range'].map(age_map)

data['family_size'] = data['family_size'].replace(to_replace=['5+'],value= 5).astype(float)

data['no_of_children'] = data['no_of_children'].replace(to_replace=['3+'],value= 3).astype(float)



    

data = data.sort_values(by = 'start_date')

data = set_ordinal_encoding(data, cat_cols)



X_train, X_test, y_train, sub = get_train_test(data, features, 'id')

print('Train Shape {}, Test Shape {}'.format(X_train.shape, X_test.shape))

# ### Category Encoding

# col = 'customer_id_cat'

# X_train[col] = X_train['customer_id']

# X_test[col]  = X_test['customer_id']





# # en = ce.TargetEncoder(cols = [col], smoothing= 50, min_samples_leaf= 2)

# en =  ce.CatBoostEncoder(cols = [col])

# X_train = en.fit_transform(X_train, y_train)

# X_test =  en.transform(X_test)

X_train.describe()
X_test.describe()
%%time

PREDICT_TEST = True

verbose_eval = 100

# cat_cols = []

params = {}



params['bagging_fraction'] = 0.8

params['bagging_freq'] = 1

params['feature_fraction'] = 0.7

# params['min_gain_to_split'] = 2

# params['min_sum_hessian_in_leaf'] = 2

params['num_leaves'] =  6

params['lambda_l1'] = 0.8

params['lambda_l2'] = 4

# params['min_data_in_leaf'] = 80



params['learning_rate'] = 0.02

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['seed'] =  random_state

params['metric'] = 'auc'

# params['scale_pos_weight'] = 5





#test size will create validation set based on start_date  = 2013-05-19

model, feature_imp, history =  run_lgb_with_cv(params, X_train, y_train, X_test, cat_cols ,

                                      shuffle_split = False, test_size= 0.28845589, verbose_eval = verbose_eval, esr = 500)



if PREDICT_TEST:

    #Run for whole training set to predict on test data

    print('*' * 100)

    # num_rounds = int(model.best_iteration + 0.20 * model.best_iteration)

    num_rounds = int(model.best_iteration + 0.15 * model.best_iteration)

    model, feature_imp = run_lgb_no_cv(params, X_train, y_train, X_test,cat_cols,

                                       num_rounds = num_rounds, verbose_eval = 100, )

plot_results(history)
print(params)

if PREDICT_TEST:

    y_prob_test = model.predict(X_test)  

    sub = write_predictions(sub, y_prob_test, 'lgb_sub.csv' )

plot_feature_imp(feature_imp, top_n = 30)

print(features)
# ['coup_count_b', 'coup_item_count','coup_count_all', 'coup_count_disc]