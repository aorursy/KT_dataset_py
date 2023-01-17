import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
import pathlib
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Read files and creating dataframes
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
sell_prices['wm_yr_wk']=sell_prices['wm_yr_wk'].astype(np.int32)

avg_Coming_Event_1=pd.read_csv("../input/average-unit-sold/avg_Coming_Event_1.csv").rename(columns={'Units_sold':'avg_Coming_Event_1'})
avg_Coming_Event_2=pd.read_csv("../input/average-unit-sold/avg_Coming_Event_2.csv").rename(columns={'Units_sold':'avg_Coming_Event_2'})
avg_event_name_1=pd.read_csv("../input/average-unit-sold/avg_event_name_1.csv").rename(columns={'Units_sold':'avg_event_name_1'})
avg_event_name_2=pd.read_csv("../input/average-unit-sold/avg_event_name_2.csv").rename(columns={'Units_sold':'avg_event_name_2'})
avg_wday=pd.read_csv("../input/average-unit-sold/avg_wday.csv").rename(columns={'Units_sold':'avg_wday'})
avg_month=pd.read_csv("../input/average-unit-sold/avg_month.csv").rename(columns={'Units_sold':'avg_month'})
avg_week=pd.read_csv("../input/average-unit-sold/avg_week.csv").rename(columns={'Units_sold':'avg_week'})
avg_dept_id=pd.read_csv("../input/average-unit-sold/avg_dept_id.csv").rename(columns={'Units_sold':'avg_dept_id'})
avg_cat_id=pd.read_csv("../input/average-unit-sold/avg_cat_id.csv").rename(columns={'Units_sold':'avg_cat_id'})
avg_store_id=pd.read_csv("../input/average-unit-sold/avg_store_id.csv").rename(columns={'Units_sold':'avg_store_id'})
avg_state_id=pd.read_csv("../input/average-unit-sold/avg_state_id.csv").rename(columns={'Units_sold':'avg_state_id'})

elasticity=pd.read_csv("../input/elasticity/elasticity.csv", usecols=['id','elasticity_id_wk'])
elasticity_cat_lvl=pd.read_csv("../input/elasticity-cat-lvl/elasticity_cat_lvl.csv", usecols=['cat_id','elasticity_id_wk_cat_lvl'])

calendar = pd.read_csv("../input/calendar/calendar_v1.csv")
calendar['date']=pd.to_datetime(calendar['date'],format="%d/%m/%y")
#usecols = ['date','wm_yr_wk','wday','month','d',
#'event_name_1', 'event_type_1', 'event_name_2',
#'event_type_2'])
train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")


#selecting fraction of data while keeping same composition of catg ids
f=0.2
sales_train_validation=train[train['cat_id']=='FOODS'].sample(frac=f)
sales_train_validation=sales_train_validation.append(train[train['cat_id']=='HOUSEHOLD'].sample(frac=f))
sales_train_validation=sales_train_validation.append(train[train['cat_id']=='HOBBIES'].sample(frac=f))
del train
#train=pd.DataFrame()
sales_train_validation=sales_train_validation.reset_index()
sales_train_validation=sales_train_validation.drop(columns=['index'])
# Converting Sales train data day columns into day variable
sales_train_validation=sales_train_validation.melt(id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'], 
                                                   var_name='day_id',value_name='Units_sold')
sales_train_validation['Units_sold']=sales_train_validation['Units_sold'].astype(np.int16)
sales_train_validation['day_id']=sales_train_validation['day_id'].str[2:].astype(np.int64)
sales_train_validation=sales_train_validation.merge(calendar, left_on=['day_id'], right_on=['day_id'])
sales_train_validation=sales_train_validation.fillna(0)
sales_train_validation['week']=sales_train_validation['wm_yr_wk'].astype(str).str[3:].astype(np.int16)
#column=['Coming_Event_1','Coming_Event_2'
#     ,'event_name_1','event_name_2','wday','month','week','dept_id','cat_id','store_id', 'state_id']
#Create lag variables on units sold for 1 week
sales_train_validation=sales_train_validation[['date','id', 'item_id', 'dept_id', 'cat_id','week',
                                               'store_id', 'state_id', 'day_id', 'wm_yr_wk', 'wday',
                                               'month', 'event_name_1','event_type_1', 'event_name_2',
                                               'event_type_2', 'Coming_Event_1','Coming_Event_Type_1',
                                               'Coming_Event_2','Coming_Event_Type_2','Units_sold']].sort_values(by=['id','date']).reset_index().drop(columns='index')

for i in [7]:
    sales_train_validation_v1=sales_train_validation[['id','date','Units_sold']]
    sales_train_validation_v1=sales_train_validation_v1.set_index(['date','id']
                                                             ).unstack().shift(i).stack(dropna=False
                                                                                        ).reset_index().sort_values(by=['id','date']).rename(columns={'Units_sold':'Units_sold_d-'+str(i)})
    sales_train_validation['Units_sold_d-'+str(i)]=sales_train_validation_v1['Units_sold_d-'+str(i)].reset_index().drop(columns='index')
#Rolling mean for Units Sold

for i in [3,5,8]:
    sales_train_validation_v1=sales_train_validation[['id','date','Units_sold']]
    sales_train_validation_v1=sales_train_validation_v1.set_index(['date','id']
                                                             ).unstack().rolling(i).mean().stack(dropna=False
                                                                                        ).reset_index().sort_values(by=['id','date']).rename(columns={'Units_sold':'Units_sold_mean_'+str(i)})
    sales_train_validation['Units_sold_mean_'+str(i)]=sales_train_validation_v1['Units_sold_mean_'+str(i)].reset_index().drop(columns='index')

sales_train_validation=sales_train_validation.merge(elasticity, how='left', on='id')
sales_train_validation=sales_train_validation.merge(elasticity_cat_lvl, how='left', on='cat_id')
sales_train_validation['elasticity_id_wk'][sales_train_validation['elasticity_id_wk'].isnull()]=sales_train_validation['elasticity_id_wk_cat_lvl']
sales_train_validation=sales_train_validation.drop(columns=['elasticity_id_wk_cat_lvl'])
##Discount Variable
sell_prices['id']=sell_prices['item_id']+'_'+sell_prices['store_id']+'_validation' 
sell_prices=sell_prices.drop(columns=['item_id','store_id'])

sell_prices['sell_price_max']=sell_prices.groupby(['id'])['sell_price'].transform(np.max)
sell_prices['Disc']=((sell_prices['sell_price_max']-sell_prices['sell_price'])/sell_prices['sell_price_max'])

sales_train_validation=sales_train_validation.merge(sell_prices,how='left' ,left_on=['id','wm_yr_wk'],
                                                    right_on=['id','wm_yr_wk'])
sales_train_validation['Disc'][sales_train_validation['Disc'].isnull()]=np.mean(sell_prices['Disc'][sell_prices['Disc']>0])

#sales_train_validation=sales_train_validation[sales_train_validation['sell_price'].notnull()].reset_index()
#sales_train_validation=sales_train_validation.drop(columns=['index'])

sales_train_validation['sell_price_max']=sales_train_validation.groupby(['id'])['sell_price'].transform(np.max)
sales_train_validation['Disc']=(sales_train_validation['sell_price_max'].sub(sales_train_validation['sell_price']))/sales_train_validation['sell_price_max']

#Discount Variable lag for 1 week
for i in [3,5,7]:
    sales_train_validation_v1=sales_train_validation[['id','date','Disc']]
    sales_train_validation_v1=sales_train_validation_v1.set_index(['date','id']
                                                             ).unstack().shift(i).stack(dropna=False
                                                                                        ).reset_index().sort_values(by=['id','date']).rename(columns={'Disc':'Disc_d-'+str(i)})
    sales_train_validation['Disc_d-'+str(i)]=sales_train_validation_v1['Disc_d-'+str(i)].reset_index().drop(columns='index')


sales_train_validation=sales_train_validation.merge(avg_Coming_Event_1, how='left', on=['Coming_Event_1'])
sales_train_validation=sales_train_validation.merge(avg_Coming_Event_2, how='left', on=['Coming_Event_2'])
sales_train_validation=sales_train_validation.merge(avg_event_name_1, how='left', on=['event_name_1'])
sales_train_validation=sales_train_validation.merge(avg_event_name_2, how='left', on=['event_name_2'])
sales_train_validation=sales_train_validation.merge(avg_wday, how='left', on=['wday'])
sales_train_validation=sales_train_validation.merge(avg_month, how='left', on=['month'])
sales_train_validation=sales_train_validation.merge(avg_week, how='left', on=['week'])
sales_train_validation=sales_train_validation.merge(avg_dept_id, how='left', on=['dept_id'])
sales_train_validation=sales_train_validation.merge(avg_cat_id, how='left', on=['cat_id'])
sales_train_validation=sales_train_validation.merge(avg_store_id, how='left', on=['store_id'])
sales_train_validation=sales_train_validation.merge(avg_state_id, how='left', on=['state_id'])

sales_train_validation=sales_train_validation.drop(columns= ['item_id', 'dept_id', 'cat_id', 'week', 'store_id',
       'state_id', 'wm_yr_wk', 'wday', 'month', 'event_name_1',
       'event_type_1', 'event_name_2', 'event_type_2', 'Coming_Event_1',
       'Coming_Event_Type_1', 'Coming_Event_2', 'Coming_Event_Type_2'])
#sales_train_validation=sales_train_validation.fillna(sales_train_validation.mean())
sales_train_validation=sales_train_validation.fillna(0)

columns_9=['Units_sold_d-1', 'Units_sold_d-7', 'Units_sold_mean_3']
sales_train_validation.columns
columns_11=['Units_sold_d-1', 'Units_sold_d-7', 
            'Units_sold_mean_3', 'Units_sold_mean_5_d-1', 'sell_price', 'Disc',
            'Disc_d-1', 'Disc_d-7',
            'avg_Coming_Event_1','avg_Coming_Event_2',
            'avg_event_name_1', 'avg_event_name_2', 'avg_wday', 
            'avg_month', 'avg_week','elasticity_id_wk']

columns_11=['Units_sold_d-7', 
            'Units_sold_mean_8', 'sell_price', 'Disc',
            'Disc_d-7',
            'avg_Coming_Event_1','avg_Coming_Event_2',
            'avg_event_name_1', 'avg_event_name_2', 'avg_wday', 
            'avg_month', 'avg_week','elasticity_id_wk']
columns_11=['Units_sold_d-7', 'Units_sold_mean_8', 'elasticity_id_wk',
           'sell_price', 'Disc', 'Disc_d-7',
           'avg_Coming_Event_1', 'avg_Coming_Event_2', 'avg_event_name_1',
           'avg_event_name_2', 'avg_wday', 'avg_month', 'avg_week', 'avg_dept_id',
           'avg_cat_id', 'avg_store_id', 'avg_state_id']
columns_11=['Units_sold_d-7','Units_sold_mean_3','Units_sold_mean_5', 'Units_sold_mean_8', 'elasticity_id_wk',
           'sell_price', 'Disc', 'Disc_d-7','Disc_d-5','Disc_d-3',
           'avg_Coming_Event_1', 'avg_Coming_Event_2', 'avg_event_name_1',
           'avg_event_name_2', 'avg_wday', 'avg_month', 'avg_week', 'avg_dept_id',
           'avg_cat_id', 'avg_store_id', 'avg_state_id']
#sales_train_validation = sales_train_validation.dropna(how='any',axis=0) 


X=sales_train_validation[columns_11] 
y=sales_train_validation['Units_sold']

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()



EPOCHS = 4

#history = 
model.fit(X, y,epochs=EPOCHS, validation_split = 0.2, verbose=0,
          callbacks=[tfdocs.modeling.EpochDots()])

#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#hist.tail()

y_predict=model.predict(X)

print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_predict)))
print('R^2:', metrics.r2_score(y, y_predict))


predict_val_set=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
predict_val_set=pd.concat([predict_val_set.iloc[:,:predict_val_set.columns.get_loc("state_id")+1],
                          predict_val_set.iloc[:,predict_val_set.columns.get_loc("d_1906"):]], axis=1)
for r in np.arange(1914,1942):
    predict_val_set_v1=predict_val_set.copy()
    predict_val_set_v1['decile']=np.arange(1,len(predict_val_set_v1)+1)
    predict_val_set_v1['decile']=np.ceil((predict_val_set_v1['decile']/(len(predict_val_set_v1)+1))*10)
    
    temp=pd.DataFrame()
    for d in np.arange(1,11):
        predict_validation=predict_val_set_v1[predict_val_set_v1['decile']==d].reset_index().drop(columns=['decile','index'])
        predict_validation['d_'+str(r)]=np.nan

        #melting dataframe
        predict_validation=predict_validation.melt(id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'], 
                                                       var_name='day_id',value_name='Units_sold')
           
        
        #predict_validation['Units_sold']=predict_validation['Units_sold'].astype(np.int16)
        predict_validation['day_id']=predict_validation['day_id'].str[2:].astype(np.int64)
        predict_validation=predict_validation[predict_validation['day_id']>(r-9)]
        
        
        predict_validation=predict_validation.merge(calendar, left_on=['day_id'], right_on=['day_id'])
        predict_validation=predict_validation.fillna(0)
        predict_validation['week']=predict_validation['wm_yr_wk'].astype(str).str[3:].astype(np.int16)
        
        
        #Create lag variables on units sold for 1 day, 1 month, 1 week, and 1 year
        predict_validation=predict_validation[['date','id', 'item_id', 'dept_id', 'cat_id','week',
                                                   'store_id', 'state_id', 'day_id', 'wm_yr_wk', 'wday',
                                                   'month', 'event_name_1','event_type_1', 'event_name_2',
                                                   'event_type_2', 'Coming_Event_1','Coming_Event_Type_1',
                                                   'Coming_Event_2','Coming_Event_Type_2','Units_sold']].sort_values(by=['date','id']).reset_index().drop(columns=['index'])
        for i in [7]:
            predict_validation_v1=predict_validation[['id','date','Units_sold']]
            predict_validation_v1=predict_validation_v1.set_index(['date','id']
                                                                     ).unstack().shift(i).stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Units_sold':'Units_sold_d-'+str(i)})
            
            predict_validation['Units_sold_d-'+str(i)]=predict_validation_v1['Units_sold_d-'+str(i)].reset_index().drop(columns='index')
            print(str(i)+' : run')


        #Rolling mean for Units Sold
        for i in [3,5,8]:
            predict_validation_v1=predict_validation[['id','date','Units_sold']]
            predict_validation_v1=predict_validation_v1.set_index(['date','id']
                                                                     ).unstack().rolling(window=i,min_periods=1).mean().stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Units_sold':'Units_sold_mean_'+str(i)})
            predict_validation['Units_sold_mean_'+str(i)]=predict_validation_v1['Units_sold_mean_'+str(i)].reset_index().drop(columns='index')
            
            print(str(i)+' : run')
        
        ##Discount Variable
        sell_prices['id']=sell_prices['id'].str[:17]+'validation'
        predict_validation=predict_validation.merge(sell_prices,how='left' ,left_on=['id','wm_yr_wk'],
                                                            right_on=['id','wm_yr_wk'])
        predict_validation['Disc'][predict_validation['Disc'].isnull()]=np.mean(sell_prices['Disc'][sell_prices['Disc']>0])

        #predict_validation=predict_validation[predict_validation['sell_price'].notnull()].reset_index()
        #predict_validation=predict_validation.drop(columns=['index'])
        predict_validation=predict_validation.drop(columns=['sell_price_max'])
       
        #Discount Variable lag for 1 week
        for i in [3,5,7]:
            predict_validation_v1=predict_validation[['id','date','Disc']]
            predict_validation_v1=predict_validation_v1.set_index(['date','id']
                                                                     ).unstack().shift(i).stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Disc':'Disc_d-'+str(i)})
            predict_validation['Disc_d-'+str(i)]=predict_validation_v1['Disc_d-'+str(i)].reset_index().drop(columns='index')
            print(str(i)+' : run')

        #filter only for days we need static variables for --
        condition_v1=(predict_validation['day_id']>=r)
                      #&(predict_validation['day_id']<=r+6))
        predict_validation=predict_validation[condition_v1].reset_index().drop(columns=['index'])
            
        ##average sale of product in festival_1
        predict_validation['Coming_Event_1']=predict_validation['Coming_Event_1'].astype(str)
        predict_validation=predict_validation.merge(avg_Coming_Event_1, how='left', on=['Coming_Event_1'])
        predict_validation['Coming_Event_2']=predict_validation['Coming_Event_2'].astype(str)
        predict_validation=predict_validation.merge(avg_Coming_Event_2, how='left', on=['Coming_Event_2'])
        predict_validation['event_name_1']=predict_validation['event_name_1'].astype(str)
        predict_validation=predict_validation.merge(avg_event_name_1, how='left', on=['event_name_1'])
        predict_validation['event_name_2']=predict_validation['event_name_2'].astype(str)
        predict_validation=predict_validation.merge(avg_event_name_2, how='left', on=['event_name_2'])
        predict_validation=predict_validation.merge(avg_wday, how='left', on=['wday'])
        predict_validation=predict_validation.merge(avg_month, how='left', on=['month'])
        predict_validation=predict_validation.merge(avg_week, how='left', on=['week'])
        predict_validation=predict_validation.merge(avg_dept_id, how='left', on=['dept_id'])
        predict_validation=predict_validation.merge(avg_cat_id, how='left', on=['cat_id'])
        predict_validation=predict_validation.merge(avg_store_id, how='left', on=['store_id'])
        predict_validation=predict_validation.merge(avg_state_id, how='left', on=['state_id'])
        
        predict_validation=predict_validation.merge(elasticity, how='left', on='id')
        predict_validation=predict_validation.merge(elasticity_cat_lvl, how='left', on='cat_id')
        predict_validation['elasticity_id_wk'][predict_validation['elasticity_id_wk'].isnull()]=predict_validation['elasticity_id_wk_cat_lvl']
        predict_validation=predict_validation.drop(columns=['elasticity_id_wk_cat_lvl'])

        
        print("merge done !")
        predict_validation=predict_validation.drop(columns= ['item_id', 'dept_id', 'cat_id', 'week', 'store_id',
       'state_id', 'wm_yr_wk', 'wday', 'month', 'event_name_1',
       'event_type_1', 'event_name_2', 'event_type_2', 'Coming_Event_1',
       'Coming_Event_Type_1', 'Coming_Event_2', 'Coming_Event_Type_2'])

#        predict_validation=predict_validation.fillna(predict_validation.mean())
        predict_validation=predict_validation.fillna(0)
                
        predict_validation['Units_sold']=np.ceil(model.predict(predict_validation[columns_11]))
        predict_validation=predict_validation[['id','day_id','Units_sold']]
        predict_validation['day_id']='d_'+predict_validation['day_id'].astype(str)
        temp=temp.append(predict_validation)

    temp=temp.pivot(index='id', columns='day_id', values='Units_sold').reset_index()
    predict_val_set=predict_val_set.merge(temp, how='left', on='id')
#   predict_val_set=predict_val_set.rename(columns={str(r):'d_'+str(r)})
#   predict_val_set['d_'+str(r)]=np.ceil(predict_val_set['d_'+str(r)])
    print(str(r)+' : run')
predict_val_set_v1=predict_val_set[['id','d_1914','d_1915','d_1916','d_1917','d_1918','d_1919','d_1920','d_1921','d_1922','d_1923','d_1924','d_1925','d_1926','d_1927','d_1928','d_1929','d_1930','d_1931','d_1932','d_1933','d_1934','d_1935','d_1936','d_1937','d_1938','d_1939','d_1940','d_1941']]
predict_val_set_v1.to_csv("/kaggle/working/validation_8.csv")









predict_eva_set=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv")
predict_eva_set=pd.concat([predict_eva_set.iloc[:,:predict_eva_set.columns.get_loc("state_id")+1],
                          predict_eva_set.iloc[:,predict_eva_set.columns.get_loc("d_1934"):]], axis=1)
for r in np.arange(1942,1970):
    predict_eva_set_v1=predict_eva_set.copy()
    predict_eva_set_v1['decile']=np.arange(1,len(predict_eva_set_v1)+1)
    predict_eva_set_v1['decile']=np.ceil((predict_eva_set_v1['decile']/(len(predict_eva_set_v1)+1))*10)
    
    temp=pd.DataFrame()
    for d in np.arange(1,11):
        predict_evaluation=predict_eva_set_v1[predict_eva_set_v1['decile']==d].reset_index().drop(columns=['decile','index'])
        predict_evaluation['d_'+str(r)]=np.nan

        #melting dataframe
        predict_evaluation=predict_evaluation.melt(id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'], 
                                                       var_name='day_id',value_name='Units_sold')
           
        
        #predict_evaluation['Units_sold']=predict_evaluation['Units_sold'].astype(np.int16)
        predict_evaluation['day_id']=predict_evaluation['day_id'].str[2:].astype(np.int64)
        predict_evaluation=predict_evaluation[predict_evaluation['day_id']>(r-9)]
        
        
        predict_evaluation=predict_evaluation.merge(calendar, left_on=['day_id'], right_on=['day_id'])
        predict_evaluation=predict_evaluation.fillna(0)
        predict_evaluation['week']=predict_evaluation['wm_yr_wk'].astype(str).str[3:].astype(np.int16)
        
        
        #Create lag variables on units sold for 1 day, 1 month, 1 week, and 1 year
        predict_evaluation=predict_evaluation[['date','id', 'item_id', 'dept_id', 'cat_id','week',
                                                   'store_id', 'state_id', 'day_id', 'wm_yr_wk', 'wday',
                                                   'month', 'event_name_1','event_type_1', 'event_name_2',
                                                   'event_type_2', 'Coming_Event_1','Coming_Event_Type_1',
                                                   'Coming_Event_2','Coming_Event_Type_2','Units_sold']].sort_values(by=['date','id']).reset_index().drop(columns=['index'])
        for i in [7]:
            predict_evaluation_v1=predict_evaluation[['id','date','Units_sold']]
            predict_evaluation_v1=predict_evaluation_v1.set_index(['date','id']
                                                                     ).unstack().shift(i).stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Units_sold':'Units_sold_d-'+str(i)})
            
            predict_evaluation['Units_sold_d-'+str(i)]=predict_evaluation_v1['Units_sold_d-'+str(i)].reset_index().drop(columns='index')
            print(str(i)+' : run')


        #Rolling mean for Units Sold
        for i in [3,5,8]:
            predict_evaluation_v1=predict_evaluation[['id','date','Units_sold']]
            predict_evaluation_v1=predict_evaluation_v1.set_index(['date','id']
                                                                     ).unstack().rolling(window=i,min_periods=1).mean().stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Units_sold':'Units_sold_mean_'+str(i)})
            predict_evaluation['Units_sold_mean_'+str(i)]=predict_evaluation_v1['Units_sold_mean_'+str(i)].reset_index().drop(columns='index')
            
            print(str(i)+' : run')
        
        ##Discount Variable
        sell_prices['id']=sell_prices['id'].str[:17]+'evaluation'
        predict_evaluation=predict_evaluation.merge(sell_prices,how='left' ,left_on=['id','wm_yr_wk'],
                                                            right_on=['id','wm_yr_wk'])
        predict_evaluation['Disc'][predict_evaluation['Disc'].isnull()]=np.mean(sell_prices['Disc'][sell_prices['Disc']>0])

        #predict_evaluation=predict_evaluation[predict_evaluation['sell_price'].notnull()].reset_index()
        #predict_evaluation=predict_evaluation.drop(columns=['index'])
        predict_evaluation=predict_evaluation.drop(columns=['sell_price_max'])
       
        #Discount Variable lag for 1 week
        for i in [3,5,7]:
            predict_evaluation_v1=predict_evaluation[['id','date','Disc']]
            predict_evaluation_v1=predict_evaluation_v1.set_index(['date','id']
                                                                     ).unstack().shift(i).stack(dropna=False
                                                                                                ).reset_index().rename(columns={'Disc':'Disc_d-'+str(i)})
            predict_evaluation['Disc_d-'+str(i)]=predict_evaluation_v1['Disc_d-'+str(i)].reset_index().drop(columns='index')
            print(str(i)+' : run')

        #filter only for days we need static variables for --
        condition_v1=(predict_evaluation['day_id']>=r)
                      #&(predict_evaluation['day_id']<=r+6))
        predict_evaluation=predict_evaluation[condition_v1].reset_index().drop(columns=['index'])
            
        ##average sale of product in festival_1
        predict_evaluation['Coming_Event_1']=predict_evaluation['Coming_Event_1'].astype(str)
        predict_evaluation=predict_evaluation.merge(avg_Coming_Event_1, how='left', on=['Coming_Event_1'])
        predict_evaluation['Coming_Event_2']=predict_evaluation['Coming_Event_2'].astype(str)
        predict_evaluation=predict_evaluation.merge(avg_Coming_Event_2, how='left', on=['Coming_Event_2'])
        predict_evaluation['event_name_1']=predict_evaluation['event_name_1'].astype(str)
        predict_evaluation=predict_evaluation.merge(avg_event_name_1, how='left', on=['event_name_1'])
        predict_evaluation['event_name_2']=predict_evaluation['event_name_2'].astype(str)
        predict_evaluation=predict_evaluation.merge(avg_event_name_2, how='left', on=['event_name_2'])
        predict_evaluation=predict_evaluation.merge(avg_wday, how='left', on=['wday'])
        predict_evaluation=predict_evaluation.merge(avg_month, how='left', on=['month'])
        predict_evaluation=predict_evaluation.merge(avg_week, how='left', on=['week'])
        predict_evaluation=predict_evaluation.merge(avg_dept_id, how='left', on=['dept_id'])
        predict_evaluation=predict_evaluation.merge(avg_cat_id, how='left', on=['cat_id'])
        predict_evaluation=predict_evaluation.merge(avg_store_id, how='left', on=['store_id'])
        predict_evaluation=predict_evaluation.merge(avg_state_id, how='left', on=['state_id'])
        
        predict_evaluation=predict_evaluation.merge(elasticity, how='left', on='id')
        predict_evaluation=predict_evaluation.merge(elasticity_cat_lvl, how='left', on='cat_id')
        predict_evaluation['elasticity_id_wk'][predict_evaluation['elasticity_id_wk'].isnull()]=predict_evaluation['elasticity_id_wk_cat_lvl']
        predict_evaluation=predict_evaluation.drop(columns=['elasticity_id_wk_cat_lvl'])

        
        print("merge done !")
        predict_evaluation=predict_evaluation.drop(columns= ['item_id', 'dept_id', 'cat_id', 'week', 'store_id',
       'state_id', 'wm_yr_wk', 'wday', 'month', 'event_name_1',
       'event_type_1', 'event_name_2', 'event_type_2', 'Coming_Event_1',
       'Coming_Event_Type_1', 'Coming_Event_2', 'Coming_Event_Type_2'])

#        predict_evaluation=predict_evaluation.fillna(predict_evaluation.mean())
        predict_evaluation=predict_evaluation.fillna(0)
                
        predict_evaluation['Units_sold']=np.ceil(model.predict(predict_evaluation[columns_11]))
        predict_evaluation=predict_evaluation[['id','day_id','Units_sold']]
        predict_evaluation['day_id']='d_'+predict_evaluation['day_id'].astype(str)
        temp=temp.append(predict_evaluation)

    temp=temp.pivot(index='id', columns='day_id', values='Units_sold').reset_index()
    predict_eva_set=predict_eva_set.merge(temp, how='left', on='id')
#   predict_eva_set=predict_eva_set.rename(columns={str(r):'d_'+str(r)})
#   predict_eva_set['d_'+str(r)]=np.ceil(predict_eva_set['d_'+str(r)])
    print(str(r)+' : run')
predict_eva_set_v1=predict_eva_set[['id','d_1942','d_1943','d_1944','d_1945','d_1946','d_1947','d_1948','d_1949','d_1950','d_1951','d_1952','d_1953','d_1954','d_1955','d_1956','d_1957','d_1958','d_1959','d_1960','d_1961','d_1962','d_1963','d_1964','d_1965','d_1966','d_1967','d_1968','d_1969']]
predict_eva_set_v1.to_csv("/kaggle/working/evaluation_8.csv")

