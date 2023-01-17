import pandas as pd

import numpy as np

output_file = pd.read_csv('../input/ensemble-dark-magics/submission.csv')

sample_submission = pd.read_csv('../input/m5-forecasting-uncertainty/sample_submission.csv')

train_sales = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')
output_file_validation = output_file[output_file['id'].str.contains("validation")]

output_file_evaluation = output_file[output_file['id'].str.contains("evaluation")]
id_df = train_sales[['id','item_id','dept_id','cat_id','store_id','state_id']]
output_file_validation = pd.merge(output_file_validation, id_df, how = 'left', on = 'id')
train_sales_evaluation = train_sales

train_sales_evaluation['id'] = train_sales_evaluation['id'].str.replace(r'validation$', 'evaluation')

id_df = train_sales_evaluation[['id','item_id','dept_id','cat_id','store_id','state_id']]

output_file_evaluation = pd.merge(output_file_evaluation, id_df, how = 'left', on = 'id')
quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']

days = range(1, 29)

val_eval = ['validation', 'evaluation']

time_series_columns = [f'F{i}' for i in days]

def CreateSales( train_sales,name_list, group):

    '''

    This function returns a dataframe (sales) on the aggregation level given by name list and group

    '''

    rows_ve = [(name + "_X_" + str(q) + "_" + ve, str(q)) for name in name_list for q in quants for ve in val_eval]

    sales = train_sales.groupby(group)[time_series_columns].sum() #would not be necessary for lowest level

    return sales

def createTrainSet(sales_train_s,train_sales, name, group_level, X = False):

    sales_total = CreateSales(train_sales,name, group_level)

    if(X == True):

        sales_total = sales_total.rename(index = lambda s:  s + '_X')

    sales_train_s = sales_train_s.append(sales_total)

    return(sales_train_s)

def get_agg_df(train_sales):

    total = ['Total']

    train_sales['Total'] = 'Total'

    train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id

    train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id

    train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id

    train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id

    train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id

    train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id

    total = ['Total']

    states = ['CA', 'TX', 'WI']

    num_stores = [('CA',4), ('TX',3), ('WI',3)]

    stores = [x[0] + "_" + str(y + 1) for x in num_stores for y in range(x[1])]

    cats = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

    num_depts = [('FOODS',3), ('HOBBIES',2), ('HOUSEHOLD',2)]

    depts = [x[0] + "_" + str(y + 1) for x in num_depts for y in range(x[1])]

    state_cats = [state + "_" + cat for state in states for cat in cats]

    state_depts = [state + "_" + dept for state in states for dept in depts]

    store_cats = [store + "_" + cat for store in stores for cat in cats]

    store_depts = [store + "_" + dept for store in stores for dept in depts]

    prods = list(train_sales.item_id.unique())

    prod_state = [prod + "_" + state for prod in prods for state in states]

    prod_store = [prod + "_" + store for prod in prods for store in stores]

    cols = [i for i in train_sales.columns if i.startswith('F')]

    sales_train_s = train_sales[cols]

    sales_train_s = pd.DataFrame()

    sales_train_s = createTrainSet(sales_train_s,train_sales, total, 'Total', X=True) #1

    sales_train_s = createTrainSet(sales_train_s, train_sales,states, 'state_id', X=True) #2

    sales_train_s = createTrainSet(sales_train_s,train_sales, stores, 'store_id', X=True) #3

    sales_train_s = createTrainSet(sales_train_s,train_sales, cats, 'cat_id', X=True) #4

    sales_train_s = createTrainSet(sales_train_s,train_sales, depts, 'dept_id', X=True) #5

    sales_train_s = createTrainSet(sales_train_s,train_sales, state_cats, 'state_cat') #6

    sales_train_s = createTrainSet(sales_train_s,train_sales, state_depts, 'state_dept') #7

    sales_train_s = createTrainSet(sales_train_s,train_sales, store_cats, 'store_cat') #8

    sales_train_s = createTrainSet(sales_train_s,train_sales, store_depts, 'store_dept') #9

    sales_train_s = createTrainSet(sales_train_s,train_sales, prods, 'item_id', X=True) #10

    sales_train_s = createTrainSet(sales_train_s,train_sales, prod_state, 'state_item') #11

    sales_train_s = createTrainSet(sales_train_s,train_sales, prod_store, 'item_store')

    sales_train_s['id'] = sales_train_s.index

    return(sales_train_s)
from scipy.stats import poisson

op_file = pd.DataFrame()

for i in quants:

    print(i)

    output_file_validation_i = output_file_validation.copy()

    if(i!='0.500'):

        output_file_validation_i[time_series_columns]= output_file_validation_i[time_series_columns].applymap(lambda x: (x%1)*poisson.ppf(float(i), np.ceil(x)) + (1-x%1)*poisson.ppf(float(i), np.floor(x)))

    output_file_validation_i = get_agg_df(output_file_validation_i)

    output_file_validation_i['id'] = output_file_validation_i['id'] + '_' + i + '_validation'

    op_file = pd.concat([op_file, output_file_validation_i], ignore_index = True)
for i in quants:

    output_file_evaluation_i = output_file_evaluation.copy()

    if(i!='0.500'):



        output_file_evaluation_i[time_series_columns]= output_file_evaluation[time_series_columns].applymap(lambda x: (x%1)*poisson.ppf(float(i), np.ceil(x)) + (1-x%1)*poisson.ppf(float(i), np.floor(x)))

    output_file_evaluation_i = get_agg_df(output_file_evaluation_i)

    output_file_evaluation_i['id'] = output_file_evaluation_i['id'] + '_' + i + '_evaluation'

    op_file = pd.concat([op_file, output_file_evaluation_i], ignore_index = True)



sample_submission = sample_submission.id.to_frame()

sample_submission = pd.merge(sample_submission, op_file, on = 'id', how = 'left')

    
sample_submission.to_csv('lgb1_sub.csv',index = False)