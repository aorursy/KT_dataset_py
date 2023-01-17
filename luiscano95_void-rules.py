import pandas as pd

import numpy as np

import gc



df_dtt = pd.read_csv('../input/datathon-belcorp-prueba/dtt_fvta_cl.csv', index_col=0)



df_campana = pd.read_csv('../input/datathon-belcorp-prueba/campana_consultora.csv', index_col=0)



df_merge = pd.merge(df_dtt, df_campana, left_on = ['idconsultora', 'campana'], right_on= ['IdConsultora', 'campana'], how='left')



del df_dtt, df_campana

gc.collect()



df_consultora = pd.read_csv('../input/datathon-belcorp-prueba/maestro_consultora.csv', index_col=0)



df_merge = df_merge.merge(df_consultora, left_on = ['idconsultora'], right_on= ['IdConsultora'], how='left')



del df_consultora

gc.collect()



df_merge.sort_values(by = ['idconsultora', 'campana'], inplace= True)



df_merge.head()



df_merge['campana_num'] = (df_merge['campana']//100-2018)*18+(df_merge['campana']%100)



var_inutil =['codigocanalorigen', 'IdConsultora_x', 'codigofactura', 'IdConsultora_y']



df_merge.drop(columns=var_inutil, inplace = True)



list(df_merge)





df_merge['flagcorreovalidad'].count()



df_merge['flagcorreovalidad'].isnull().count()



df_merge[['preciocatalogo', 'ahorro', 'descuento', 'realanulmnneto', 'realdevmnneto','realvtamnneto', 'realvtamnfaltneto']]







## Desde aquí



#flags dejar como flags o suma/total (ratio)

#categoricas1 necesitan label e



#1. variables que necesitan label encoding

var_categoricas1 = ['canalingresoproducto', 'grupooferta', 'geografia', 'estadocivil', 'flagcorreovalidad']

# flagcorreovalidad -> 1, 0 y NaN



#2. variables que necesitan encoding manual

var_categoricas2 = ['palancapersonalizacion', 'segmentacion']



#3. variables numericas

var_num = ['descuento', 'ahorro', 'preciocatalogo', 'realanulmnneto', 'realdevmnneto', 'realvtamnneto', 

           'realvtamnfaltneto', 'realvtamncatalogo', 'realvtamnfaltcatalogo', 'cantidadlogueos', 'edad']



#4. flags

var_flags = ['flagactiva', 'flagdispositivo', 'flagofertadigital', 'flagsuscripcion', 'flagcelularvalidado']



#5. otras variables

# 'evaluacion_nuevas' -> 'ratio_evaluacion'

var_creadas = ['evaluacion_nuevas2']



def try_to_encode_eval(string_eval):

    try:

        return float(string_eval[-3])/float(string_eval[-1])

    except:

        return 0.5



#5. otras variables

df_merge['evaluacion_nuevas2'] = df_merge['evaluacion_nuevas'].apply(lambda x: try_to_encode_eval(x))

df_merge['evaluacion_nuevas2'] = np.where(df_merge['evaluacion_nuevas'] == 'C_1d1', 0.5, df_merge['evaluacion_nuevas2'])

# queda probar si los 'Est' deben tener un valor mayor a 1 o dejarlo en 0.5

df_merge['evaluacion_nuevas2'] = np.where(df_merge['evaluacion_nuevas'] == 'Est', 2, df_merge['evaluacion_nuevas2'])



dict_segmentacion ={

    'Nuevas' : 1,

    'Nivel2' : 2,

    'Nivel3' : 3,

    'Nivel4' : 4,

    'Nivel5' : 5,

    'Nivel6' : 6,

    'Nivel7' : 7,

    'Tops' : 8

}



df_merge['segmentacion2'] = df_merge['segmentacion'].map(dict_segmentacion)



df_merge['segmentacion2'].value_counts()



df_merge['palancapersonalizacion2'] = df_merge['palancapersonalizacion'].str.split(' ')[0]



dict_ppersonalizacion = {

    'App' : 1, 

    'Desktop' : 2,

    'Mobile' : 3,

    'Ofertas' : 4,

    'Showroom' : 5,

    'Oferta' : 4,

    'Favoritos' : 5

}



df_merge['palancapersonalizacion2'] = df_merge['palancapersonalizacion'].str.split(' ').str[0].map(dict_ppersonalizacion)

















### Var nuevas sobre poblacion en zonas de chile



zonas_dict = {'11 NORTE GRANDE': 'NORTE_GRANDE' ,                

'14 SANTIAGO / SUR CHICO': 'ZONA_SUR',         

'15 SUR GRANDE': 'ZONA_SUR',                   

'12 SANTIAGO - NORTE CHICO': 'NORTE_CHICO',       

'13 STGO. / VI?A DEL MAR / VA': 'STGO_VM_VA',   

'17 SANTIAGO PONIENTE': 'ZONA_PONIENTE',           

'16 SUR AUSTRAL': 'ZONA_AUSTRAL',                 

'00 ADMINISTRATIVO': 'OTROS'}



df_merge['geografia'] = df_merge['geografia'].map(zonas_dict)



santiago = 7112808

valparaiso = 1815902

biobio = 1556805

maule = 1044950

araucania = 957224

higgins = 914555

lagos = 828708

coquimbo = 757586

antofagasta = 607534

ñuble = 480609

rios = 384837

tarapaca = 330558

atacama = 286168

arica = 226068

iquique = 196562

magallanes = 166533

aysen = 103158

copiapo = 175162

serena = 205635

rancagua = 225563

talca = 203873

concepcion = 220746

temuco = 221375

valdivia = 143207

montt = 213119

coyhaique = 7290

arenas = 124169

viña_mar = 326759

cerrillos = 71906

maipu = 521627

estacion_central = 147041

quinta_normal = 110026

pudahuel = 230293

prado_navia = 132622 + 96249



NORTE_GRANDE = arica+iquique+antofagasta

NORTE_CHICO = copiapo+serena

ZONA_CENTRAL = valparaiso+rancagua+talca+concepcion+santiago

ZONA_SUR = temuco+valdivia+montt

ZONA_AUSTRAL = coyhaique+arenas

ZONA_PONIENTE =  cerrillos+maipu+estacion_central+quinta_normal+pudahuel+prado_navia

STGO_VM_VA = santiago+valparaiso+viña_mar



pob_dict = {'NORTE_GRANDE': NORTE_GRANDE ,                

'ZONA_SUR': ZONA_SUR,                            

'NORTE_CHICO': NORTE_CHICO,

'STGO_VM_VA': STGO_VM_VA,   

'ZONA_PONIENTE': ZONA_PONIENTE,           

'ZONA_AUSTRAL': ZONA_AUSTRAL,                 

'OTROS': np.nan}



df_merge['POB_2017'] = df_merge['geografia'].map(pob_dict)



### Rango edad propuesto por ex consultora



#rango = 18-25, 25-35, 35-45, 45+



bins = [18, 25, 35, 45, 100]

names = [1, 2, 3, 4]



df_merge['rango_edad'] = pd.cut(df_merge['edad'], bins=bins, labels=names)



df_merge.drop(columns='edad', inplace=True)



#df_merge.to_csv('df_merge.csv', index=False)



# Agregar df producto



df_producto = pd.read_csv('../input/datathon-belcorp-prueba/maestro_producto.csv', index_col=0)



df_producto = df_producto[['idproducto', 'codigounidadnegocio', 'codigomarca', 'codigocategoria', 'codigotipo']]



df_merge = pd.merge(df_merge, df_producto, left_on = 'idproducto', right_on= 'idproducto', how='left')



del df_producto

gc.collect()





















## Hasta aquí



df_merge['palancapersonalizacion2'].value_counts()



df_merge['cant_dcto'] = df_merge['descuento']*df_merge['preciocatalogo']/100



df_merge['cant_interac'] = 1



df_merge['canalingresoproducto'].fillna('ND', inplace = True)



df_merge['canal_WEB'] = np.where(df_merge['canalingresoproducto'] == 'WEB', 1, 0)

df_merge['canal_APP'] = np.where(df_merge['canalingresoproducto'] == 'APP', 1, 0)

df_merge['canal_ND'] = np.where(df_merge['canalingresoproducto'] == 'ND', 1, 0)

df_merge['canal_DD'] = np.where(df_merge['canalingresoproducto'] == 'DD', 1, 0)

df_merge['canal_DIG'] = np.where(df_merge['canalingresoproducto'] == 'DIG', 1, 0)

df_merge['canal_MIX'] = np.where(df_merge['canalingresoproducto'] == 'MIX', 1, 0)

df_merge['canal_OCR'] = np.where(df_merge['canalingresoproducto'] == 'OCR', 1, 0)



df_merge.dropna(subset = ['segmentacion'], inplace = True)



df_merge['Flagpasopedido'].fillna(0, inplace = True)

df_merge['grupooferta'].fillna('ND', inplace = True)

df_merge['flagactiva'].fillna(0, inplace = True)

df_merge['flagpedidoanulado'].fillna(0, inplace = True)

df_merge['cantidadlogueos'].fillna(0, inplace = True)

df_merge['estadocivil'].fillna('Otros', inplace = True)

df_merge['flagcorreovalidad'].fillna(0, inplace = True)

df_merge['palancapersonalizacion2'].fillna(6.0, inplace = True)

df_merge['rango_edad'].fillna(4.0, inplace = True)

df_merge['POB_2017'].fillna(df_merge['POB_2017'].mean(), inplace = True)



df_merge['mean_ahorro'] = df_merge['ahorro']

df_merge['mean_preciocatalogo'] = df_merge['preciocatalogo']

df_merge['mean_realanulmnneto'] = df_merge['realanulmnneto']

df_merge['mean_realdevmnneto'] = df_merge['realdevmnneto']

df_merge['mean_realvtamnneto'] = df_merge['realvtamnneto']

df_merge['mean_realvtamnfaltneto'] = df_merge['realvtamnfaltneto']

df_merge['mean_realvtamncatalogo'] = df_merge['realvtamncatalogo']

df_merge['mean_cant_dcto'] = df_merge['cant_dcto']



df_merge['std_ahorro'] = df_merge['ahorro']

df_merge['std_preciocatalogo'] = df_merge['preciocatalogo']

df_merge['std_realanulmnneto'] = df_merge['realanulmnneto']

df_merge['std_realdevmnneto'] = df_merge['realdevmnneto']

df_merge['std_realvtamnneto'] = df_merge['realvtamnneto']

df_merge['std_realvtamnfaltneto'] = df_merge['realvtamnfaltneto']

df_merge['std_realvtamncatalogo'] = df_merge['realvtamncatalogo']

df_merge['std_cant_dcto'] = df_merge['cant_dcto']











#df_merge.to_csv('df_merge_29_10.csv', index = False)



#!pip install -q --upgrade pip

#!pip install -q joblib

#!pip install -q s3io

#!pip install -q lightgbm==2.2.2

#!pip install -q fastparquet

#!pip install -q pyarrow



import pandas as pd

import numpy as np

import gc



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import lightgbm as lgb



from sklearn.metrics import roc_auc_score, confusion_matrix, auc, accuracy_score, precision_score, recall_score



def cols_tipos(df, exclude = [], cols_ord = [], Print = False):

    # Tipo de variable

    cols = [x for x in df.columns if x not in exclude]

    cols_cat = [x for x in list(df.select_dtypes(include=['object'])) if x not in exclude]

    cols_num = [x for x in list(df.select_dtypes(exclude=['object'])) if x not in exclude]



    # Categorías nominales y ordinales

    cols_nom = [x for x in cols_cat if x not in cols_ord]



    if Print:

        print ('Categóricas:\n', cols_cat)

        print ('\nCategóricas Ordinal:\n', cols_ord)

        print ('\nCategóricas Nominal:\n', cols_cat)

        print ('\nNuméricas:\n', cols_num)

    

    return cols, cols_cat, cols_num, cols_nom



def impxgb(valores,variables):

    dictimp={variables[a]:valores[a] for a in range(0,len(variables)) }

    xgimp=sorted(dictimp.items(), key=lambda x: x[1],reverse=True)



    return xgimp



def filter_threshold(probabilities, threshold):

    return [1 if f >= threshold else 0 for f in probabilities]



def get_threshold_measures_df(probabilities, observed, steps=[x / 100.0 for x in range(0, 100, 5)]):

    df = pd.DataFrame(columns=['Punto de corte', 'Recall', 'Accuracy', 'Precision'])



    for i in range(len(steps)):

        estimated_threshold = filter_threshold(probabilities, steps[i])

        row = [

            steps[i],

            recall_score(observed, estimated_threshold),

            accuracy_score(observed, estimated_threshold),

            precision_score(observed, estimated_threshold),

            #auc(observed, estimated_threshold),

            #auc(observed, estimated_threshold)*2 - 1

        ]

        df.loc[i] = row



    return df



def preparing_data(df):

    

    df['rango_edad'] = pd.DataFrame(pd.cut(df['edad'],10))

    df.drop(columns=['edad', 'idconsultora'], inplace=True)

    

    df['campanaultimopedido'] = df['campanaultimopedido'].astype(str)

    

    cols, cols_cat, cols_num, cols_nom = cols_tipos(df, exclude = [], cols_ord = [], Print = False)

    

    c = {}

    for l in cols_cat:

        df[l]=df[l].map(str)

        df[l]=df[l].fillna('NULL')

        le = preprocessing.LabelEncoder()

        le.fit(list(df[l]))

        df[l]=le.transform(df[l])

        c[l] = le

        

    cols.remove('TARGET_1')

    

    return cols, cols_cat, cols_num, cols_nom, c, df



def read_data(pc, path_file, pkl = False, **kwargs):

    bucket, key = path_file.split('/', maxsplit=1)

    if pkl == True:

        if pc == 's3':

            with s3io.open('s3://{0}/{1}'.format(bucket, key), mode='r') as s3_file:

                obj = joblib.load(s3_file)

        else:

            obj = joblib.load(s3_file)

    else:

        s3_bool = False

        if pc == 's3':

            s3_bool = True

        obj = raex.readCSV(path_file, s3=s3_bool, print_info = False, **kwargs)

    return obj



df = df_merge.copy()



df.head()



df.shape



df_unos = df[df['campanaultimopedido']==201907]



df_unos.shape



df_zeros = df[df['campanaultimopedido']<201907]



df_zeros.shape



df['codigocombinadoproducto'] = df['codigocombinadoproducto'] = df['codigomarca'].astype(str) + df['codigocategoria'].astype(str) + df['codigotipo'].astype(str)



df[['codigomarca', 'codigocategoria', 'codigotipo']] = df[['codigomarca', 'codigocategoria', 'codigotipo']].astype(str)



f = {'canalingresoproducto': lambda x: pd.Series.mode(x)[0], #pd.Series.mode,

     'grupooferta': lambda x: pd.Series.mode(x)[0], #pd.Series.mode,

     'geografia': 'first',

     'estadocivil': 'last',

     'flagcorreovalidad': 'max',

     'palancapersonalizacion2': lambda x: pd.Series.mode(x)[0], #pd.Series.mode,

     'segmentacion2': lambda x: pd.Series.mode(x)[0], #pd.Series.mode,

     'descuento': 'mean', #sum

     

     'campanaultimopedido':'first',

     'campanaingreso':'first',

     'campanaprimerpedido':'first',

     'codigomarca': lambda x: pd.Series.mode(x)[0],

     'codigocategoria': lambda x: pd.Series.mode(x)[0],

     'codigotipo': lambda x: pd.Series.mode(x)[0],     

     

     'ahorro': 'sum',

     'preciocatalogo': 'sum',

     'realanulmnneto': 'sum',

     'realdevmnneto': 'sum',

     'realvtamnneto': 'sum',

     'realvtamnfaltneto': 'sum',

     'realvtamncatalogo': 'sum',

     'cant_dcto': 'sum',

     

     'mean_ahorro': 'mean',

     'mean_preciocatalogo': 'mean',

     'mean_realanulmnneto': 'mean',

     'mean_realdevmnneto': 'mean',

     'mean_realvtamnneto': 'mean',

     'mean_realvtamnfaltneto': 'mean',

     'mean_realvtamncatalogo': 'mean',

     'mean_cant_dcto': 'mean',

     

     'std_ahorro': 'std',

     'std_preciocatalogo': 'std',

     'std_realanulmnneto': 'std',

     'std_realdevmnneto': 'std',

     'std_realvtamnneto': 'std',

     'std_realvtamnfaltneto': 'std',

     'std_realvtamncatalogo': 'std',

     'std_cant_dcto': 'std',

     

     'cantidadlogueos': 'sum',

     

     'rango_edad': 'last',

     'flagactiva': 'max',

     'flagdispositivo': 'max',

     'flagofertadigital': 'max',

     'flagsuscripcion': 'max',

     'flagcelularvalidado': 'max',

     'evaluacion_nuevas2': 'last',

     

     'cant_interac': 'sum',

     'canal_WEB': 'sum',

     'canal_APP': 'sum',

     'canal_ND': 'sum',

     'canal_DD': 'sum',

     'canal_DIG': 'sum',

     'canal_MIX': 'sum',

     'canal_OCR': 'sum',

         



     'POB_2017': 'mean',

     'codigounidadnegocio':lambda x: pd.Series.mode(x)[0], #pd.Series.mode,

     'codigocombinadoproducto': lambda x: pd.Series.mode(x)[0], #pd.Series.mode, #moda doble

     'Flagpasopedido':'max'}



df_grouped = df.groupby(by='idconsultora', as_index=False).agg(f)



df_grouped['std_ahorro'].fillna(0, inplace = True)

df_grouped['std_preciocatalogo'].fillna(0, inplace = True)

df_grouped['std_realanulmnneto'].fillna(0, inplace = True)

df_grouped['std_realdevmnneto'].fillna(0, inplace = True)

df_grouped['std_realvtamnneto'].fillna(0, inplace = True)

df_grouped['std_realvtamnfaltneto'].fillna(0, inplace = True)

df_grouped['std_realvtamncatalogo'].fillna(0, inplace = True)

df_grouped['std_cant_dcto'].fillna(0, inplace = True)



df_pivot_sum = pd.pivot_table(df, index= 'idconsultora',columns= "campana", values= "Flagpasopedido", aggfunc="sum").reset_index()



df_pivot_sum.fillna(0, inplace=True)



df_pivot = pd.pivot_table(df, index= 'idconsultora',columns= "campana", values= "Flagpasopedido", aggfunc="max").reset_index()



df_pivot.fillna(0, inplace=True)



df_pivot['ultimas_seis_campanas_pedidas'] = df_pivot[[

 201901,

 201902,

 201903,

 201904,

 201905,

 201906]].sum(axis=1)



df_pivot['antiguas_campañas_pedidas'] = df_pivot[[

 201807,

 201808,

 201809,

 201810,

 201811,

 201812]].sum(axis=1)



df_pivot_sum['cantidad_campanas_pedidas'] = df_pivot_sum[[201807,

 201808,

 201809,

 201810,

 201811,

 201812,

 201813,

 201814,

 201815,

 201816,

 201817,

 201818,

 201901,

 201902,

 201903,

 201904,

 201905,

 201906]].sum(axis=1)



df_pivot_sum.rename(columns={

    201807:'201807_sum',

 201808:'201808_sum',

 201809:'201809_sum',

 201810:'201810_sum',

 201811:'201811_sum',

 201812:'201812_sum',

 201813:'201813_sum',

 201814:'201814_sum',

 201815:'201815_sum',

 201816:'201816_sum',

 201817:'201817_sum',

 201818:'201816_sum',

 201901:'201901_sum',

 201902:'201902_sum',

 201903:'201903_sum',

 201904:'201904_sum',

 201905:'201905_sum',

 201906:'201906_sum',

    'cantidad_campanas_pedidas': 'cantidad_campanas_pedidas_sum',

    'ultimas_seis_campanas_pedidas': 'ultimas_seis_campanas_pedidas_sum',

    'antiguas_campañas_pedidas': 'antiguas_campañas_pedidas_sum'

}, inplace=True)



df_grouped = pd.merge(df_grouped, df_pivot_sum, how='left', left_on='idconsultora', right_on='idconsultora')



df_grouped = pd.merge(df_grouped, df_pivot, how='left', left_on='idconsultora', right_on='idconsultora')



df_grouped.shape



df_grouped[['campanaingreso', 'campanaultimopedido', 'campanaprimerpedido', 'rango_edad']] = df_grouped[['campanaingreso', 'campanaultimopedido', 'campanaprimerpedido', 'rango_edad']].astype(str)



cols, cols_cat, cols_num, cols_nom = cols_tipos(df_grouped, exclude = [], cols_ord = [], Print = False)



c = {}

for l in cols_cat:

    df_grouped[l]=df_grouped[l].map(str)

    df_grouped[l]=df_grouped[l].fillna('ND')

    le = preprocessing.LabelEncoder()

    le.fit(list(df_grouped[l]))

    df_grouped[l]=le.transform(df_grouped[l])

    c[l] = le



tmp = df[['idconsultora', 'campanaultimopedido']]



tmp = tmp.groupby(by='idconsultora', as_index=False).agg({'campanaultimopedido':'first'})



df_grouped = pd.merge(df_grouped, tmp, how='left', left_on='idconsultora', right_on='idconsultora')



df_model = df_grouped[df_grouped['campanaultimopedido_y']<=201907.0]



df_predict = df_grouped[df_grouped['campanaultimopedido_y']>201907.0]



df_model['TARGET'] = np.where(df_model['campanaultimopedido_y']==201907.0, 1, 0)



df_model.head()



df_grouped.drop(columns='campanaultimopedido_y', inplace=True)

df_model.drop(columns='campanaultimopedido_y', inplace=True)

df_predict.drop(columns='campanaultimopedido_y', inplace=True)



round((df_model['TARGET'].value_counts() / df_model.shape[0])*100, 2)



df_grouped.shape



# Juntar vars extras



df_model = pd.merge(df_model, df_pivot_sum, how='left', left_on='idconsultora', right_on='idconsultora')



df_model = pd.merge(df_model, df_pivot, how='left', left_on='idconsultora', right_on='idconsultora')



df_model.shape



df_predict = pd.merge(df_predict, df_pivot_sum, how='left', left_on='idconsultora', right_on='idconsultora')



df_predict = pd.merge(df_predict, df_pivot, how='left', left_on='idconsultora', right_on='idconsultora')



df_predict.shape











# LabelEncoding



# MODELO



df_model['rand'] = np.random.RandomState(24).randn(*df_model['TARGET'].shape)

df_model = df_model.sort_values(by='rand')



df_model.reset_index(drop=True, inplace=True)



df_model.head()



df_train = df_model[df_model['rand']<0.7]

df_test = df_model[(df_model['rand']>=0.7)]



cols.remove('campanaultimopedido')



cols_cat.remove('campanaultimopedido')



cols.remove('idconsultora')



# Aquí cambio para prospectar con toda la data

df_train = df_model



d={}

e={}

mylist = list(range(1,6))



X = df_train[cols].values

y = df_train['TARGET'].ravel()



index_categorical=[cols.index(x) for x in cols_cat]



index_categorical



for k in mylist:

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=k*64)



    train_set = lgb.Dataset(X_train, y_train)

    validation_sets = lgb.Dataset(X_validation, y_validation, reference=train_set)



    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'AUC' },

        'num_leaves': 20, 

        'max_depth': 4, 

        'min_data_in_leaf': 200, 

        'feature_fraction': 0.5,

        'bagging_freq': 1,

        'bagging_fraction': 0.9,

        'verbose': 1,

        'is_unbalance':True

    }

   

    model = lgb.train(

        params,

        train_set,

        num_boost_round=2000,

        valid_sets=validation_sets,

        early_stopping_rounds=100,

        categorical_feature=index_categorical

        )

    

    d[k]=model

    test=model.predict(X_validation, num_iteration=model.best_iteration)

    e[k]=roc_auc_score(y_validation, test)



a = []

for k in mylist:

    a.append(e[k])

b = np.array((np.mean(a),np.std(a)))

print(b)



max(a), min(a)



d2 = {k:v for k,v in d.items() if a[k-1]>0}



p = []

for k in d2.keys():

    df_test['PROB_{}'.format(k)]=d2[k].predict(df_test[cols], num_iteration=d2[k].best_iteration)

    p.append('PROB_{}'.format(k))

df_test['PROB'] = df_test[p].mean(axis=1)

print(roc_auc_score(df_test['TARGET'],df_test['PROB']))



i = {}

x = []

mylist = list(range(1,6))

for k in mylist:

    i[k] = d[k].feature_importance(importance_type="gain")

    x.append(i[k]) 

dx = pd.DataFrame(x)

ixg=impxgb(dx.mean(axis=0),cols)

ixg





p = []

for k in d2.keys():

    df_predict['PROB_{}'.format(k)]=d2[k].predict(df_predict[cols], num_iteration=d2[k].best_iteration)

    p.append('PROB_{}'.format(k))

df_predict['flagpasopedido'] = df_predict[p].mean(axis=1)



tmp1 = df_predict[['idconsultora', 'flagpasopedido']]



df_submit_file = pd.read_csv('Data/predict_submission.csv')



tmp2 = pd.merge(df_submit_file, tmp1, how='left', left_on='idconsultora', right_on='idconsultora')



tmp2.head()



tmp2['aa']= np.where(tmp2['flagpasopedido_y']!=np.nan, tmp2['flagpasopedido_y'], tmp2['flagpasopedido_x'])



aaa = df_model[['TARGET', 'idconsultora']]



tmp2 = pd.merge(tmp2, aaa, how='left', left_on='idconsultora', right_on='idconsultora')



tmp2.head()



tmp2.isna().sum()



tmp2['TARGET']>=0



tmp2['flagpasopedido']= np.where(tmp2['TARGET']>=0, tmp2['TARGET'], tmp2['aa'])



final = tmp2[['idconsultora', 'flagpasopedido']]



final.isna().sum()



final.fillna(0, inplace=True)



final.to_csv('finalcano.csv', index=False)



final.shape