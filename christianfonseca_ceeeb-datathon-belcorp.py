import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import lightgbm as lgb
def cols_tipos(df, exclude = [], Print = False):

    # Tipo de variable

    cols = [x for x in df.columns if x not in exclude]

    cols_cat = [x for x in list(df.select_dtypes(include=['object'])) if x not in exclude]

    cols_num = [x for x in list(df.select_dtypes(exclude=['object'])) if x not in exclude]



    if Print:

        print ('Categóricas:\n', cols_cat)

        print ('\nNuméricas:\n', cols_num)

    

    return cols, cols_cat, cols_num



pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',1000)
df_status = pd.read_csv('../input/datathon-belcorp-prueba/campana_consultora.csv',encoding='latin1')

df_ventas = pd.read_csv('../input/datathon-belcorp-prueba/dtt_fvta_cl.csv',encoding='latin1')

df_producto = pd.read_csv('../input/datathon-belcorp-prueba/maestro_producto.csv',encoding='latin1')

df_consultora = pd.read_csv('../input/datathon-belcorp-prueba/maestro_consultora.csv',encoding='latin1')

df_predict = pd.read_csv('../input/datathon-belcorp-prueba/predict_submission.csv',encoding='latin1')
df_status = df_status.sort_values('campana')

x = df_status['campana'].unique()



z = {}

for i in range(0,len(x)):

    z[i] = df_status[df_status['campana']==x[i]][['IdConsultora','campana','Flagpasopedido']]

    if i>0:

        df0 = df_status[df_status['campana']<x[i]][['IdConsultora']].drop_duplicates()

        z[i] = pd.merge(df0,z[i],on='IdConsultora',how='outer')        

        z[i]['campana'] = z[i]['campana'].fillna(x[i])

        z[i]['Flagpasopedido'] = z[i]['Flagpasopedido'].fillna(0)

        print(i)



df_predict.rename(columns={'idconsultora':'IdConsultora','flagpasopedido':'Flagpasopedido'},inplace=True)

df_predict['campana'] = 201907

z[18] = df_predict



x = list(x) + [201907]
df_producto = df_producto[['idproducto','unidadnegocio','marca','categoria']]

df_ventas = pd.merge(df_ventas,df_producto,on='idproducto',how='left')

df_ventas.loc[df_ventas['categoria']=='HOGAR','categoria'] = 'HOGAR_2'



ventas_cat = ['categoria','canalingresoproducto','unidadnegocio',

              'marca','grupooferta']

for col in ventas_cat:

    df_ventas = df_ventas.join(pd.get_dummies(df_ventas[col]))

df_ventas.drop(columns={'canalingresoproducto','unidadnegocio','marca','categoria',

                        'palancapersonalizacion','grupooferta','idproducto'},inplace=True)

    

df_ventas = df_ventas.groupby(['campana','idconsultora']).agg('sum').reset_index()

df_status = pd.merge(df_status,df_ventas,left_on=['campana','IdConsultora'],right_on=['campana','idconsultora'],how='left')



cons_cols = ['IdConsultora','estadocivil','flagsupervisor','flagconsultoradigital',

             'flagcorreovalidad','edad','flagcelularvalidado','campanaultimopedido']

df_status = pd.merge(df_status,df_consultora[cons_cols],on='IdConsultora',how='left')
features_din = ['Flagpasopedido','realuuanuladas','flagpasopedidotratamientocorporal',

                'DIG','VARIOS','realuuvendidas','cantidadlogueos','flagdispositivo',

                'realvtamncatalogo','ahorro','realvtamnneto','flagactiva','WEB',

                'realuudevueltas']



features = ['IdConsultora'] + features_din



for i in range(1,len(x)):

    p = {}

    p[1] = df_status[df_status['campana']==x[i-1]]

    p[1].drop(columns={'campana'},inplace=True)

    for j in range(2,13):

        p[j] = df_status[df_status['campana']==x[i-j]][features]

        

    for j in range(1,13):    

        for col in features_din:

            p[j].rename(columns={col:'{0}_{1}'.format(col,j)},inplace=True)

        z[i] = pd.merge(z[i],p[j],on='IdConsultora',how='left')

    

    print(i)
params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': { 'AUC' },

    'num_leaves': 30,

    'max_depth': -1,

    'min_data_in_leaf': 30,

    'bagging_freq': 1,

    'feature_fraction': 0.8,

    'verbose': 1,

    'is_unbalance':False,

    'learning_rate': 0.01,

    'bagging_fraction': 0.9,

}
target = 'Flagpasopedido'



for i in range(1,len(x)):

    for m in features_din:

        try:        

            z[i]['{}_diff1'.format(m)] = (z[i]['{}_1'.format(m)].fillna(0) - 

                                         (z[i]['{}_1'.format(m)].fillna(0) + 

                                          z[i]['{}_2'.format(m)].fillna(0) + 

                                          z[i]['{}_3'.format(m)].fillna(0))/3)

        except:

            z[i]['{}_diff1'.format(m)] = np.nan

        

        try:  

            z[i]['{}_diff2'.format(m)] = (z[i]['{}_1'.format(m)].fillna(0) - 

                                         (z[i]['{}_1'.format(m)].fillna(0) + 

                                          z[i]['{}_2'.format(m)].fillna(0) + 

                                          z[i]['{}_3'.format(m)].fillna(0) + 

                                          z[i]['{}_4'.format(m)].fillna(0) + 

                                          z[i]['{}_5'.format(m)].fillna(0) + 

                                          z[i]['{}_6'.format(m)].fillna(0))/6)

        except:

            z[i]['{}_diff2'.format(m)] = np.nan



        try:  

            z[i]['{}_diff3'.format(m)] = (z[i]['{}_1'.format(m)].fillna(0) - 

                                         (z[i]['{}_1'.format(m)].fillna(0) + 

                                          z[i]['{}_2'.format(m)].fillna(0) + 

                                          z[i]['{}_3'.format(m)].fillna(0) + 

                                          z[i]['{}_4'.format(m)].fillna(0) + 

                                          z[i]['{}_5'.format(m)].fillna(0) + 

                                          z[i]['{}_6'.format(m)].fillna(0) + 

                                          z[i]['{}_7'.format(m)].fillna(0) + 

                                          z[i]['{}_8'.format(m)].fillna(0) + 

                                          z[i]['{}_9'.format(m)].fillna(0) + 

                                          z[i]['{}_10'.format(m)].fillna(0) + 

                                          z[i]['{}_11'.format(m)].fillna(0) + 

                                          z[i]['{}_12'.format(m)].fillna(0))/12)   

        except:

            z[i]['{}_diff3'.format(m)] = np.nan

        

        if m!=target:

            try:

                z[i].drop(columns={'{}_2'.format(m),'{}_3'.format(m)},inplace=True)

            except:    

                pass

            try:

                z[i].drop(columns={'{}_4'.format(m),'{}_5'.format(m),'{}_6'.format(m)},inplace=True)

            except:    

                pass

            try:

                z[i].drop(columns={'{}_7'.format(m),'{}_8'.format(m),'{}_9'.format(m),

                               '{}_10'.format(m),'{}_11'.format(m),'{}_12'.format(m),},inplace=True)

            except:    

                pass



exclude = ['Unnamed: 0','campana','IdConsultora','Flagpasopedido']

cols, cols_cat, cols_num = cols_tipos(z[1],exclude, Print = True)

index_categorical=[cols.index(x) for x in cols_cat]



c = {}

for l in cols_cat:

    df_status[l]=df_status[l].map(str)

    df_status[l]=df_status[l].fillna('NULL')

    a=list(df_status[l])

    a=a+['nan']

    le = preprocessing.LabelEncoder()

    le.fit(a)

    c[l] = le

    print(l)



for i in range(1,len(x)):

    for l in cols_cat:

        z[i][l]=z[i][l].map(str)

        z[i][l]=z[i][l].fillna('NULL')

        z[i][l]=c[l].transform(z[i][l])

        

for i in range(1,len(x)-1):

    

    if i>1: 

        for j in range(i,len(x)):

            z[j]['predict_{}'.format(i)] = model.predict(z[j][cols], num_iteration=model.best_iteration)

        

        cols.append('predict_{}'.format(i))  

    

    X = z[i].loc[z[i][target].notnull(), cols]

    y = z[i].loc[z[i][target].notnull(), target]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    

    train_set = lgb.Dataset(X_train, y_train)

    validation_sets = lgb.Dataset(X_test, y_test, reference=train_set)



    model = lgb.train(

            params,

            train_set,

            num_boost_round=2500,

            valid_sets=validation_sets,

            categorical_feature=index_categorical,

            early_stopping_rounds=500,

            verbose_eval=50,

            )

    

    print('TERMINADO',i)
z[18]['predict'] = model.predict(z[18][cols], num_iteration=model.best_iteration)
dfz = pd.merge(z[18][['IdConsultora','predict']],df_consultora[['IdConsultora','campanaultimopedido']],on='IdConsultora',how='left')

dfz.loc[(dfz['campanaultimopedido']==201907)&(dfz['campanaultimopedido'].notnull()),'predict']=1

dfz.loc[(dfz['campanaultimopedido']<201907)&(dfz['campanaultimopedido'].notnull()),'predict']=0

dfz = dfz[['IdConsultora','predict']]

dfz.rename(columns={'IdConsultora':'idconsultora','predict':'flagpasopedido'},inplace=True)

dfz.to_csv('predict_7.csv',index=False)