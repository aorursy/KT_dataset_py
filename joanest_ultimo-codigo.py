import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')
#entrena_1=pd.read_csv('entrena_1.csv',parse_dates=[0])

sales_train_eva=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
calendar=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv',parse_dates=[0])
sales_train_eva['id_num']=np.arange(1,len(sales_train_eva)+1,1)
variables_dias=[a for a in sales_train_eva.columns if a.startswith('d_')]
calendar['dias_num']=calendar['d'].apply(lambda x :int(x[2:]))
calendario=calendar[calendar.dias_num<1942]
from datetime import datetime
from datetime import timedelta 
tiempos=[]
for i in range(1,29):
    tiempos.append(calendario['date'][len(calendario)-1]+timedelta(i))  
series=pd.DataFrame()
series['tiempo28']=tiempos
tiempos_slice=[]
for i in tqdm(series.tiempo28,total=len(series),position=1,desc='tiempos'):
    h=i
    primeros=[]
    while h>=calendario['date'][0]:
        h=h-timedelta(28)
        if(h>calendario['date'][0]):
            primeros.append(h)            
    tiempos_slice.append(primeros)
tiempos_completos=dict(zip(list(series['tiempo28']),tiempos_slice))
#datos_1=calendario[calendario.date.isin(tiempos_completos[list(tiempos_completos.keys())[0]])]

def prepro(data,col1,name_col,cambio):
    data.rename(columns=dict(zip(cambio.d,cambio.date)),inplace=True)
    sales_train3=data.melt(id_vars=col1,var_name='dias',value_name=name_col)
#    sales_train3['dias']=sales_train3['dias'].apply(lambda x:int(x[2:]))
    
    sales_train3=sales_train3.sort_values(['dias','id_num'])
    return sales_train3
def create_sales_lags_diff(df, gpby_cols, target_col, lags):
    
    df=df.sort_values(['id_num','dias']).reset_index(drop=True)
#    gpby = df.groupby(gpby_cols)
    for w in lags:
        df['_'.join([target_col, 'diff', str(w)])] =\
            df.groupby(gpby_cols)[target_col].shift(w).diff(w)
        
    df=df.sort_values(['dias','id_num'])
#    df=df.dropna()
    df=df.reset_index(drop=True)
        
    return df
def create_sales_shift_roll_iter(df, gpby_cols, target_col, windows,
                             shift, win_type=None):
    df=df.sort_values(['id_num','dias']).reset_index(drop=True)
#    gpby = df.groupby(gpby_cols)

    for shift in shift:
        for w in windows:   
            df['_'.join([target_col,'shift'+str(shift) ,'roll_mean', str(w)])] = \
            df.groupby(gpby_cols)[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=2,
                                                  win_type=win_type).mean()
            
    for shift in shift:
        for w in windows:            
            df['_'.join([target_col,'shift_'+str(shift) ,'roll_sum', str(w)])] = \
            df.groupby(gpby_cols)[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=2,
                                                  win_type=win_type).sum()        
            
#    for w in tqdm(windows,total=len(windows)):
#        df['_'.join([target_col, 'rolli_sum', str(w)])] = \
#            df.groupby(gpby_cols)[target_col].shift(w).rolling(window=w+1, 
#                                               
#                                                  win_type=win_type).sum()
    
    df=df.sort_values(['dias','id_num'])
#    df=df.dropna()
    df=df.reset_index(drop=True)
        
    return df

def create_sales_rolling_feats1(df, gpby_cols, target_col, windows,
                             shift=1, win_type=None):
    df=df.sort_values(['id_num','dias']).reset_index(drop=True)
#    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rolli', str(w)])] = \
            df.groupby(gpby_cols)[target_col].shift(w).rolling(window=w+1, 
                                                  min_periods=2,
                                                  win_type=win_type).mean()
    for w in windows:
        df['_'.join([target_col, 'rolli_sum', str(w)])] = \
            df.groupby(gpby_cols)[target_col].shift(w).rolling(window=w+1, 
                                               
                                                  win_type=win_type).sum()
    
    df=df.sort_values(['dias','id_num'])
#    df=df.dropna()
    df=df.reset_index(drop=True)
        
    return df

from lightgbm import LGBMRegressor

def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values 
    return df
pre_test1=pd.DataFrame()
%%time
for f,i_ in enumerate(tiempos_completos.keys()):
    datos_1=calendario[calendario.date.isin(tiempos_completos[i_])]
    variables_1=[]
    variables_1=['id_num']+list(datos_1['d'])
    
    sales_train_eva2=sales_train_eva[variables_1]
#    sell_tf2=sell_tf1[variables_1]
    
    sales_train_eva3=prepro(sales_train_eva2,'id_num','sales',datos_1)
    sales_train_eva3['particion']=np.nan
    fechas=len(sales_train_eva3.dias.unique())-1
    sales_train_eva3.loc[0:30490*fechas,'particion']='train'
    sales_train_eva3.loc[30490*fechas:,'particion']='validation'
    
    para_test=pd.DataFrame({'id_num':sales_train_eva2.id_num})
    para_test['dias']=i_
    
    para_test['sales']=0
    para_test['particion']='evaluation'
    
    concat1=pd.concat([sales_train_eva3,para_test],axis=0)
#    print(concat1.dtypes)
    concat1['dayofmonth'] = concat1.dias.dt.day
    concat1['dayofyear'] = concat1.dias.dt.dayofyear
    concat1['dayofweek'] = concat1.dias.dt.dayofweek
    concat1['month'] = concat1.dias.dt.month
    concat1['year'] = concat1.dias.dt.year
    concat1['weekofyear'] = concat1.dias.dt.weekofyear
    concat1['is_month_start'] = (concat1.dias.dt.is_month_start).astype(int)
    concat1['is_month_end'] = (concat1.dias.dt.is_month_end).astype(int)
    
    join1=pd.merge(concat1,sales_train_eva[['id_num','store_id','state_id','dept_id','cat_id']],on='id_num',how='inner')
    join1=join1.sort_values(['dias','id_num']).reset_index(drop=True)
    
    df=pd.get_dummies(join1,columns=['store_id','state_id','dept_id','cat_id'])
    
    
    df1=df.copy()
    df1['sales']=df1.sales.astype(int)
    
    print('*'*10,'f_'+str(f),'*'*10)
    train_df = create_sales_lag_feats(df1, gpby_cols=['id_num'], target_col='sales', 
                               lags=[1,2,3,4,5,6,7])
    train_df=create_sales_rolling_feats1(train_df,['id_num'],'sales',[1,2,3,4])
    train_df=create_sales_lags_diff(train_df,['id_num'],'sales',[1,2,3,4])
    
    entrena=train_df.dropna()
    entrena=entrena.reset_index(drop=True)
    print('*'*10,'tuberia finalizada','*'*10)
    
    print('*'*10,'creacion de train y test','*'*10)

    features=[a for a in entrena.columns if a not in ['id_num','dias']]
    #entrena=train_df
    train,test=entrena[entrena.particion=='train'][features],entrena[entrena.particion=='validation'][features]
    
    x,y,x_test,y_test=train[train.columns[~train.columns.isin(['particion','sales'])]],train['sales'],test[test.columns[~test.columns.isin(['particion','sales'])]],test['sales']
    print('*'*10,'MODELAMOS','*'*10)
    mod1=LGBMRegressor(n_estimators=500, learning_rate=0.01,colsample_bytree=0.8)
    mod1.fit(x,y)   
    pre_test=mod1.predict(x_test)
    #error1=rmsle(y_test,pre_test)
    error=np.sqrt(sum((y_test-pre_test)**2)/len(pre_test))
    print('error :',error)
    
    ######################
    para_mod2=entrena[entrena.particion!='evaluation'][features]
    x1,y1=para_mod2[para_mod2.columns[~para_mod2.columns.isin(['particion','sales'])]],para_mod2['sales']
    ##############################
    mod2=LGBMRegressor(n_estimators=500, learning_rate=0.01,colsample_bytree=0.8)
    mod2.fit(x1,y1)   
    prediccion=entrena[entrena.particion=='evaluation'][features]
    pre_test1[i_]=mod2.predict(prediccion[prediccion.columns[~prediccion.columns.isin(['particion','sales'])]])
    
    print('*'*10,'se predijo','*'*10)
    
    

pre_test2=pre_test1.copy()
pre_test2.columns=['F'+str(x) for x in range(1,29)]
predict=pd.DataFrame({'id':sales_train_eva.id})
predict1=pd.concat([predict.id,pre_test2],axis=1)
predict1.shape
predict1.head()
predict2=predict1.copy()
predict2['id']=predict2['id'].str.replace("evaluation$",'validation')
sub=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
predict4=pd.concat([predict2,predict1],axis=0,sort=False)
predict4=predict4.reset_index(drop=True)
predict4.tail()
predict4.to_csv('predict4.csv',index=None,header=True)
sub.head()
