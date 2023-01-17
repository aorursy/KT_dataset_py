import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use("dark_background")

import gc

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

# to display all the columns in the dataset

pd.pandas.set_option('display.max_columns', None)
# load data

train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv")

calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

sample = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
train.shape,calendar.shape,sell_prices.shape
train.info()
calendar.info()
train.head()
calendar.head()
sell_prices.head()
train.isnull().sum().sort_values(ascending = False)
calendar.isnull().sum().sort_values(ascending = False)
for i in range(1942,1970):

    col = "d_"+ str(i)

    train[col] = 0
#Downcast in order to save memory

def downcast(df):

    cols = df.dtypes.index.tolist()

    types = df.dtypes.values.tolist()

    for i,t in enumerate(types):

        if 'int' in str(t):

            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:

                df[cols[i]] = df[cols[i]].astype(np.int8)

            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:

                df[cols[i]] = df[cols[i]].astype(np.int16)

            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:

                df[cols[i]] = df[cols[i]].astype(np.int32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.int64)

        elif 'float' in str(t):

            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:

                df[cols[i]] = df[cols[i]].astype(np.float16)

            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:

                df[cols[i]] = df[cols[i]].astype(np.float32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.float64)

        elif t == np.object:

            if cols[i] == 'date':

                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')

            else:

                df[cols[i]] = df[cols[i]].astype('category')

    return df  
train = downcast(train)

sell_prices = downcast(sell_prices)

calendar = downcast(calendar)
grid_df = pd.melt(train, 

                  id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 

                  var_name = 'd', 

                  value_name = "sales")
grid_df.head()
#del train

#gc.collect()
master = pd.merge(grid_df,calendar, on = "d")

master.head()
del calendar,grid_df

gc.collect()
master = pd.merge(master, sell_prices, on=['store_id','item_id','wm_yr_wk'], how='left') 

master.head()
del sell_prices

gc.collect()
# convert numeric variables into categorical variables

conv_id = dict(zip(master.id.cat.codes, master.id))

conv_item_id = dict(zip(master.item_id.cat.codes, master.item_id))

conv_dept_id = dict(zip(master.dept_id.cat.codes, master.dept_id))

conv_cat_id = dict(zip(master.cat_id.cat.codes, master.cat_id))

conv_store_id = dict(zip(master.store_id.cat.codes, master.store_id))

conv_d_state_id = dict(zip(master.state_id.cat.codes, master.state_id))
master.d = master['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)

cols = master.dtypes.index.tolist()

types = master.dtypes.values.tolist()

for i,type in enumerate(types):

    if type.name == 'category':

        master[cols[i]] = master[cols[i]].cat.codes
master.head()
master.drop('date',1,inplace = True)
valid = master[(master['d']>=1914) & (master['d']<1942)][['id','d','sales']]

test = master[master['d']>=1942][['id','d','sales']]

eval_preds = test['sales']

valid_preds = valid['sales']
cats = master.cat_id.astype('category').cat.codes.unique().tolist()

for cat in cats:

    df = master[master['cat_id']==cat]

    

    # split the data into train,validate and test

    X_train, y_train = df[df['d']<1914].drop('sales',axis=1), df[df['d']<1914]['sales']

    X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('sales',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sales']

    X_test = df[df['d']>=1942].drop('sales',axis=1)

    

    #model

    model = LGBMRegressor(

        n_estimators=1000,

        learning_rate=0.3,

        subsample=0.8,

        colsample_bytree=0.8,

        max_depth=8,

        num_leaves=50,

        min_child_weight=300

    )

    print('*****Prediction for Category: {}*****'.format(conv_cat_id[cat]))

    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)],

             eval_metric='rmse', verbose=20, early_stopping_rounds=20)

    valid_preds[X_valid.index] = model.predict(X_valid)

    eval_preds[X_test.index] = model.predict(X_test)

    del model, X_train, y_train, X_valid, y_valid

    gc.collect()
valid['sales'] = valid_preds

validation = valid[['id','d','sales']]

validation = pd.pivot(validation, index='id', columns='d', values='sales').reset_index()

validation.columns=['id'] + ['F' + str(i + 1) for i in range(28)]

validation.id = validation.id.map(conv_id).str.replace('evaluation','validation')



#Get the evaluation results

test['sales'] = eval_preds

evaluation = test[['id','d','sales']]

evaluation = pd.pivot(evaluation, index='id', columns='d', values='sales').reset_index()

evaluation.columns=['id'] + ['F' + str(i + 1) for i in range(28)]

#Remap the category id to their respective categories

evaluation.id = evaluation.id.map(conv_id)



#Prepare the submission

submit = pd.concat([validation,evaluation]).reset_index(drop=True)

submit.to_csv('submission.csv',index=False)