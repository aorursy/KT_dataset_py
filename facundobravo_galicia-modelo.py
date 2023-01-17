import numpy as np

import pandas as pd

from datetime import timedelta

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_colwidth = 100



diasSemana = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

mesesAnio = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']





def get_categoria(serie_descripciones):

    r = []

    for descripcion in serie_descripciones:

        string = descripcion.strip()

        if(string[0] == '/'):

            string = string.strip('/').split('/')[0]

        else:

            string = string.split(':')[0]

        r.append(string.strip().upper())

    return r



def poner_nombre_features(lista_features, dict_pages, dict_category, dict_category_bottom, dict_category_top, dict_site):

    nombre_features = []

    for feature in lista_features:

        if(len(feature.split('_')) > 1):

            feature_class = '_'.join(feature.split('_')[:-1])

            feature_number = feature.split('_')[-1]

            if(feature_number.isnumeric()):

                feature_number = int(feature_number)

            if(feature_class == "page"):

                nombre_features.append("page_"+dict_pages[feature_number])

            elif(feature_class == "content_category"):

                nombre_features.append("cat_"+dict_category[feature_number])

            elif(feature_class == "content_category_bottom"):

                nombre_features.append("catbot_"+dict_category_bottom[feature_number])

            elif(feature_class == "content_category_top"):

                nombre_features.append("cattop_"+dict_category_top[feature_number])

            elif(feature_class == "site_id"):

                nombre_features.append("site_"+dict_site[feature_number])

            else:

                nombre_features.append(feature)

        else:

            nombre_features.append(feature)

    return nombre_features



def get_dict_jerarquias(serie_descripciones, serie_id):

    jerarquias = {}

    for descripcion, id_num in zip(serie_descripciones, serie_id):

        string = descripcion.strip()

        if(string[0] == '/'):

            strings = string.strip('/').split('/')

        else:

            strings = string.strip(':').split(':')

            strings = [s.strip() for s in strings]

            

        dict_actual = jerarquias

        for s in strings:

            if s not in dict_actual:

                dict_actual[s] = {}

            dict_actual = dict_actual[s]

        dict_actual['ids'] = [id_num]

    return jerarquias
data = pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"])

data.columns = [s.lower() for s in data.columns]



data_complemento = pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"], dtype="int")

data_complemento.columns = [s.lower() for s in data_complemento.columns]

data_complemento = data_complemento[data.columns]



data = data.append(data_complemento, ignore_index=True)



#cant_cada_pagina = data['page'].value_counts()

#paginas_menores = list(cant_cada_pagina[cant_cada_pagina < 8].index)

#data.loc[data['page'].isin(paginas_menores)]['page'] = 10000



page = pd.read_csv("../input/page/PAGE.csv")

page.columns = [s.lower() for s in page.columns]

page = page.set_index('page').reindex(data['page'].unique()).reset_index()

page["categoria"] = get_categoria(page["page_descripcion"])

categorias_pages_dict = page.drop(['page_descripcion'], axis=1).set_index("page").to_dict()['categoria']

data['categoria_pag'] = [categorias_pages_dict[k] for k in data['page']]



convs = pd.read_csv("../input/conversiones/conversiones.csv", dtype="int")

convs.columns = [s.lower() for s in convs.columns]

#convs["trimestre"] = ((convs["mes"] - 1) // 3) + 1

#conversiones_trimestrales = pd.crosstab(convs["user_id"], convs["trimestre"])

#conversiones_trimestrales.columns = ["convs_" + str(c) + "Q" for c in conversiones_trimestrales.columns]



content = pd.read_csv("../input/content_category/CONTENT_CATEGORY.csv")

content.columns = [s.lower() for s in content.columns]



content_top = pd.read_csv("../input/content_category_top/CONTENT_CATEGORY_TOP.csv")

content_top.columns = [s.lower() for s in content_top.columns]



content_bottom = pd.read_csv("../input/content_category_bottom/CONTENT_CATEGORY_BOTTOM.csv")

content_bottom.columns = [s.lower() for s in content_bottom.columns]



device = pd.read_csv("../input/device_data/device_data.csv", parse_dates=["FEC_EVENT"])

device.columns = [s.lower() for s in device.columns]



site_id = pd.read_csv("../input/site_id/SITE_ID.csv")

site_id.columns = [s.lower() for s in site_id.columns]



dict_pages = page.set_index("page").to_dict()['page_descripcion']

dict_pages[10000] = "pagina_menor"

dict_category = content.set_index("content_category").to_dict()['content_category_descripcion']

dict_category_bottom = content_bottom.set_index("content_category_bottom").to_dict()['content_category_bottom_descripcion']

dict_category_top = content_top.set_index("content_category_top").to_dict()['content_category_top_descripcion']

dict_site = site_id.set_index("site_id").to_dict()['site_id_descripcion']



jerarquiasPage = get_dict_jerarquias(page['page_descripcion'], page['page'])



data.head(5)
logins = [1809, 1808,    2, 1820, 1358, 1362, 1773, 1763, 1758, 1213, 1382,

           1361, 1761, 1743, 1749, 1762,  947, 1456, 1214, 1734, 1174, 1457,

           1835, 1774, 1041, 1221,  536,  168,  739,  596,  597,  897,    9,

            518,   63,  366,  389,  368,  258,  308,  253,  421,   17,  312,

            313,  478,  824,   14,  519,   74,  238,  497,  593,  323,  200,

            758,   71,  388,   44,  231,  124,  358,  423,  492,  720, 1305,

            945, 1026, 1832, 1032, 1447,  951,  711, 1027, 1085, 1314,  833]

inicios = [3]



errores = [1188, 1815, 1001,   57, 1757,   99,  167,  612,    5, 1822, 1422,

           1811, 1200, 1179, 1121, 1277, 1384, 1175, 1166,  337,  422,   18,

             11,   12,  192,  193,  194,  701,  958,  355,  309,  592,   63,

            366,  389,  368,  258,  308,  253,  421,   17,  594,  706,  104,

            414,  101,  134,  216,  753,  106,  459,  105,  265,  266,  267,

            608,  529, 1341,  538, 1267,   74,  238,  497,  593,  323,  200,

            758,   71,  388,   44,  534,  606,   91,  641,  583,  302,  927,

           1053, 1010,  532, 1007,  450, 1022, 1023, 1054, 1052, 1069,  996,

            720, 1305,  945, 1026, 1832, 1032, 1447,  951,  711,  871, 1076,

            722,  852,  728, 1354, 1040,  122,  973, 1074,  756, 1111, 1169,

           1110,  983,  757]



sobrantes = logins + inicios + errores



primeros_nueve_meses = data.loc[~data['page'].isin(sobrantes)]

primeros_nueve_meses = primeros_nueve_meses.loc[primeros_nueve_meses['fec_event'].dt.month < 10]

pages_prop = pd.crosstab(primeros_nueve_meses["user_id"], primeros_nueve_meses['page'])

pages_prop = pages_prop.div(pages_prop.sum(axis=1), axis=0)



y_anio = pd.read_csv("../input/conversiones/conversiones.csv")

y_anio.columns = [s.lower() for s in y_anio.columns]

pages_prop['target'] = 0

idx = set(y_anio.loc[y_anio['mes'] >= 10]["user_id"].unique()).intersection(set(pages_prop.index))

pages_prop.loc[list(idx), ['target']] = 1
fi = []

i = 0

scores_val = []

scores_train = []



for train_idx, valid_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(pages_prop, pages_prop['target']):

    i += 1

    

    Xyt = pages_prop.iloc[train_idx]

    Xt = Xyt.drop('target', axis = 1)

    yt = Xyt['target']



    Xyv = pages_prop.iloc[valid_idx]

    Xv = Xyv.drop('target', axis = 1)

    yv = Xyv['target']

    

    learner = LGBMClassifier(n_estimators=800,

                             max_depth=3,

                             num_leaves=5,

                             learning_rate=0.01,

                             objective='binary',

                             random_state=42

                            )

    

    learner.fit(Xt, yt,  early_stopping_rounds=200, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv, yv)], verbose=0)

    

    scores_val.append(learner.best_score_['valid_1']['auc'])

    scores_train.append(learner.best_score_['training']['auc'])

    

    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns)) 



fi = pd.concat(fi, axis=1).mean(axis=1)
print("Score promedio training: " + str(np.around(np.mean(scores_train), decimals=4)))

print("Score promedio validacion: " + str(np.around(np.mean(scores_val), decimals=4)))
num_paginas = 100

paginas_importantes = fi[fi != 0].sort_values(ascending=False)

paginas_importantes = list(paginas_importantes[:num_paginas].index)

fi_nombres = fi[fi != 0].sort_values(ascending=False)[:num_paginas].copy()

fi_nombres.index = [dict_pages[k] for k in list(fi_nombres.index)]

fi_nombres[:25]
def generar_features(data_periodo, device_periodo, convs_periodo):

    ultimo_dia = data_periodo['fec_event'].dt.date.max()    

    data_periodo = data_periodo.loc[~data_periodo['page'].isin(sobrantes)]

    

    def generar_cantidad_eventos_por_franja(data_periodo_temp, nombre_evento):

        X = pd.DataFrame(index=sorted(data['user_id'].unique()))

        

        # CANTIDAD DE EVENTOS POR TRIMESTRE

        

        cantidad_ocurrencias_evento = data_periodo_temp.copy()

        cantidad_ocurrencias_evento['fec_event_hora'] = cantidad_ocurrencias_evento['fec_event'].dt.floor('H')

        cantidad_ocurrencias_evento['fec_event_trimestre'] = (cantidad_ocurrencias_evento['fec_event'].dt.month - 1) // 3

        cantidad_ocurrencias_evento = cantidad_ocurrencias_evento.drop_duplicates(subset=['fec_event_hora', 'user_id'])

        ocurrencias_por_trimestre = pd.crosstab(cantidad_ocurrencias_evento['user_id'], cantidad_ocurrencias_evento['fec_event_trimestre'])

        ocurrencias_por_trimestre = ocurrencias_por_trimestre.loc[:, sorted(ocurrencias_por_trimestre.columns)]

        ocurrencias_por_trimestre.columns = ["ocurrencias_" + nombre_evento + '_trimestre_' + str(i) for i in range(1, len(ocurrencias_por_trimestre.columns)+1)]

        X = X.join(ocurrencias_por_trimestre, how = "left").fillna(0)

        

        # CANTIDAD DE EVENTOS ULTIMO MES

        

        cantidad_ocurrencias_evento_ult_mes = data_periodo_temp.loc[data_periodo_temp['fec_event'].dt.month == data_periodo_temp['fec_event'].dt.month.max()].copy()

        cantidad_ocurrencias_evento_ult_mes['fec_event_hora'] = cantidad_ocurrencias_evento_ult_mes['fec_event'].dt.floor('H')

        cantidad_ocurrencias_evento_ult_mes = cantidad_ocurrencias_evento_ult_mes.drop_duplicates(subset=['fec_event_hora', 'user_id'])['user_id'].value_counts().to_frame()

        cantidad_ocurrencias_evento_ult_mes.columns = [nombre_evento + '_ultimo_mes']

        X = X.join(cantidad_ocurrencias_evento_ult_mes, how = "left").fillna(0)

        

        # CANTIDAD DE EVENTOS ULTIMOS QUINCE DIAS

        

        inicio_ult_15_dias = ultimo_dia - timedelta(days=15)

        cantidad_ocurrencias_evento_ult_15_dias = data_periodo_temp.loc[data_periodo_temp['fec_event'].dt.date > inicio_ult_15_dias].copy()

        cantidad_ocurrencias_evento_ult_15_dias['fec_event_hora'] = cantidad_ocurrencias_evento_ult_15_dias['fec_event'].dt.floor('H')

        cantidad_ocurrencias_evento_ult_15_dias = cantidad_ocurrencias_evento_ult_15_dias.drop_duplicates(subset=['fec_event_hora', 'user_id'])['user_id'].value_counts().to_frame()

        cantidad_ocurrencias_evento_ult_15_dias.columns = [nombre_evento + '_ultimos_15_dias']

        X = X.join(cantidad_ocurrencias_evento_ult_15_dias, how = "left").fillna(0)

        

        return X

        

    def generar_features_temporales(data_periodo_temp, nombre_evento):

        X = pd.DataFrame(index=sorted(data['user_id'].unique()))



        # CANTIDAD DE DIAS EN CADA TRIMESTRE QUE HUBO AL MENOS UNA OCURRENCIA DEL EVENTO PARA CADA USUARIO



        dias_ocurrencia_evento = data_periodo_temp.copy()

        dias_ocurrencia_evento['fec_event_dia'] = dias_ocurrencia_evento['fec_event'].dt.floor('D')

        dias_ocurrencia_evento['fec_event_trimestre'] = (dias_ocurrencia_evento['fec_event'].dt.month - 1) // 3

        dias_ocurrencia_evento = dias_ocurrencia_evento.drop_duplicates(subset=['fec_event_dia', 'user_id'])

        dias_por_trimestre_ocurrencia_evento = pd.crosstab(dias_ocurrencia_evento['user_id'], dias_ocurrencia_evento['fec_event_trimestre'])

        dias_por_trimestre_ocurrencia_evento = dias_por_trimestre_ocurrencia_evento.loc[:, sorted(dias_por_trimestre_ocurrencia_evento.columns)]

        dias_por_trimestre_ocurrencia_evento.columns = ["dias_actividad_" + nombre_evento + '_trimestre_' + str(i) for i in range(1, len(dias_por_trimestre_ocurrencia_evento.columns)+1)]

        X = X.join(dias_por_trimestre_ocurrencia_evento, how = "left").fillna(0)

        

        cantidades = generar_cantidad_eventos_por_franja(data_periodo_temp, nombre_evento)

        cantidades_totales = generar_cantidad_eventos_por_franja(data_periodo, nombre_evento)

        proporciones_por_periodo = cantidades.div(cantidades_totales)

        proporciones_por_periodo.columns = [col.replace("ocurrencias", "proporcion") for col in proporciones_por_periodo.columns]

        proporciones_cada_trimestre = cantidades.filter(like="trimestre")

        proporciones_cada_trimestre = proporciones_cada_trimestre.div(proporciones_cada_trimestre.sum(axis=1), axis=0)

        

        X = X.join(proporciones_por_periodo, how = "left").fillna(0)

        X = X.join(proporciones_cada_trimestre, how = "left").fillna(0)

        

        # SEMANAS DESDE ULTIMA ACTIVIDAD



        data_ordenada_fecha = data_periodo_temp.sort_values(['fec_event'], ascending=True)



        nombre_feature = 'semanas_desde_ultima_actividad_' + nombre_evento

        semanas_desde = data_ordenada_fecha.drop_duplicates(subset=['user_id'], keep='last').copy()

        semanas_desde[nombre_feature] = (data_periodo_temp['fec_event'].dt.dayofyear.max() - semanas_desde['fec_event'].dt.dayofyear) // 7

        semanas_desde = semanas_desde.loc[:,['user_id', nombre_feature]].set_index('user_id')

        X = X.join(semanas_desde, how = "left").fillna(1000)



        # SEMANAS DESDE PRIMER ACTIVIDAD



        nombre_feature = 'semanas_desde_primer_actividad_' + nombre_evento

        semanas_desde = data_ordenada_fecha.drop_duplicates(subset=['user_id'], keep='first').copy()

        semanas_desde[nombre_feature] = (data_periodo_temp['fec_event'].dt.dayofyear.max() - semanas_desde['fec_event'].dt.dayofyear) // 7

        semanas_desde = semanas_desde.loc[:,['user_id', nombre_feature]].set_index('user_id')

        X = X.join(semanas_desde, how = "left").fillna(1000)



        # CANTIDAD TOTAL TODO EL PERIODO

        

        cantidad_ocurrencias_totales = data_periodo_temp.copy()

        cantidad_ocurrencias_totales['fec_event_hora'] = cantidad_ocurrencias_totales['fec_event'].dt.floor('H')

        cantidad_ocurrencias_totales = cantidad_ocurrencias_totales.drop_duplicates(subset=['fec_event_hora', 'user_id'])['user_id'].value_counts().to_frame()

        cantidad_ocurrencias_totales.columns = [nombre_evento + '_totales']

        X = X.join(cantidad_ocurrencias_totales, how = "left").fillna(0)

        

        return X

    

    

    # CREACION DEL DATAFRAME

    

    X = pd.DataFrame(index=sorted(data['user_id'].unique()))

    

    #

    # FEATURES DE PROPORCION DE ACCESO A CADA PAGE Y CATEGORIA 

    #

    

    prop_importantes = data_periodo.loc[data_periodo['page'].isin(paginas_importantes)]

    prop_importantes = pd.crosstab(prop_importantes["user_id"], prop_importantes['page'])

    prop_importantes = prop_importantes.div(prop_importantes.sum(axis=1), axis=0)

    X = X.join(prop_importantes, how='left')

    

    X.columns = ["page_" + str(i) for i in X.columns]

    

    prop_categorias = pd.crosstab(data_periodo["user_id"], data_periodo['page'])

    prop_categorias = prop_categorias.div(prop_categorias.sum(axis=1), axis=0)

    page_periodo = page.set_index('page').reindex(data_periodo['page'].unique()).reset_index()

    

    pags_prestamos = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("prestamo", case=False, regex=False)]['page'].values)

    pags_hipotecario = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("hipotecario", case=False, regex=False)]['page'].values)

    pags_tarjetas = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("tarjeta", case=False, regex=False)]['page'].values)

    pags_moneda = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("moneda", case=False, regex=False)]['page'].values)

    pags_inversiones = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("inversiones", case=False, regex=False)]['page'].values)

    pags_plazo = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("fijo", case=False, regex=False)]['page'].values)

    pags_fima = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("fima", case=False, regex=False)]['page'].values)

    pags_transferencias = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("tranferencia", case=False, regex=False)]['page'].values)

    pags_agro = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("agro", case=False, regex=False)]['page'].values)

    pags_pyme = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("pyme", case=False, regex=False)]['page'].values)

    pags_haberes = list(page_periodo.loc[page_periodo['page_descripcion'].str.contains("haberes", case=False, regex=False)]['page'].values)

    

    prop_categorias_suma = pd.DataFrame(index=prop_categorias.index)

    

    prop_categorias_suma['prestamos_prop'] =  prop_categorias.loc[:,pags_prestamos].sum(axis=1)

    prop_categorias_suma['hipotecario_prop'] =  prop_categorias.loc[:,pags_hipotecario].sum(axis=1)

    prop_categorias_suma['tarjetas_prop'] =  prop_categorias.loc[:,pags_tarjetas].sum(axis=1)

    prop_categorias_suma['moneda_prop'] =  prop_categorias.loc[:,pags_moneda].sum(axis=1)

    prop_categorias_suma['inversiones_prop'] =  prop_categorias.loc[:,pags_inversiones].sum(axis=1)

    prop_categorias_suma['plazo_fijo_prop'] =  prop_categorias.loc[:,pags_plazo].sum(axis=1)

    prop_categorias_suma['fima_prop'] =  prop_categorias.loc[:,pags_fima].sum(axis=1)

    prop_categorias_suma['transferencias_prop'] =  prop_categorias.loc[:,pags_transferencias].sum(axis=1)

    prop_categorias_suma['agro_prop'] =  prop_categorias.loc[:,pags_agro].sum(axis=1)

    prop_categorias_suma['pyme_prop'] =  prop_categorias.loc[:,pags_pyme].sum(axis=1)

    prop_categorias_suma['haberes_prop'] =  prop_categorias.loc[:,pags_haberes].sum(axis=1)

    del prop_categorias

    

    X = X.join(prop_categorias_suma, how='left')

    

    #

    # FEATURES TEMPORALES

    #

    

    # TODOS LOS EVENTOS

    

    temp = generar_features_temporales(data_periodo, "general")

    X = X.join(temp, how='left')

    

    # PRESTAMOS

    

    ## TODOS

    

    pags = page.loc[(page['page_descripcion'].str.contains("prestamo", case=False, regex=False)) | (page['page_descripcion'].str.contains("pp", case=False, regex=False))]['page'].values

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "prestamos")

    X = X.join(temp, how='left')

    

    ## HIPOTECARIOS

    

    pags = page.loc[page['page_descripcion'].str.contains('HIPOTECARIO', case=False, regex=False)]['page'].values

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "hipotecario")

    X = X.join(temp, how='left')

    

    ## PERSONALES EXITO

    

    temp = data_periodo.loc[data_periodo['page'] == 345]

    temp = generar_features_temporales(temp, "personales_exito")

    X = X.join(temp, how='left')

    

    ## HIPOTECARIO EXITO

    

    temp = data_periodo.loc[data_periodo['page'] == 1297]

    temp = generar_features_temporales(temp, "hipotecarios_exito")

    X = X.join(temp, how='left')

    

    # AUMENTO DE LIMITE TARJETAS

    

    temp = data_periodo.loc[data_periodo['page'].isin([159, 496, 830])]

    temp = generar_features_temporales(temp, "aumento_limite_tarj")

    X = X.join(temp, how='left')

    

    # TRANSFERENCIAS

    

    pags = page.loc[page['page_descripcion'].str.contains('TRANSFERENCIA', case=False, regex=False)]['page'].values

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "transferencias")

    X = X.join(temp, how='left')

    

    # TARJETAS

    

    ## PAGINAS DE RESUMENES Y CONSUMOS

    pags = [1461,  328,  327, 1083,  506, 1224,  113,  235,  329,  330,  535, 1239,   23,   69,  409,  650, 65,  64, 213, 651]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "tarjetas")

    X = X.join(temp, how='left')

    

    # DOLARES

    

    ## COMPRA EXITO

    

    temp = data_periodo.loc[data_periodo['page'] == 173]

    temp = generar_features_temporales(temp, "compra_dolares_exito")

    X = X.join(temp, how='left')

    

    ## VENTA EXITO

    

    temp = data_periodo.loc[data_periodo['page'] == 318]

    temp = generar_features_temporales(temp, "venta_dolares_exito")

    X = X.join(temp, how='left')

    

    # BONOS Y ACCIONES

    pags = [440,  442,  379,  377,  375,  376,  378, 1521, 1523, 1522,  737, 735,  733,  734,  736,  422,  753, 1010,  426,  513]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "inversiones")

    X = X.join(temp, how='left')

    

    # PLAZO FIJO

    

    ## TODOS

    

    pags = page.loc[page['page_descripcion'].str.contains('FIJO', case=False, regex=False)]['page'].values

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "plazo_fijo")

    X = X.join(temp, how='left')

    

    ## CONFIRMACION

    

    temp = data_periodo.loc[data_periodo['page'].isin([38, 172, 212, 491])]

    temp = generar_features_temporales(temp, "plazo_fijo_exito")

    X = X.join(temp, how='left')

    

    # FIMA

    

    ## TODOS

    

    pags = page.loc[page['page_descripcion'].str.contains('fima', case=False, regex=False)]['page'].values

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "fima")

    X = X.join(temp, how='left')

    

    ## RESCATE

    

    pags = [416,  417,  419,  418,  420, 1237, 1431,  309,  196,  538,  352, 996,  743]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "fima_rescate")

    X = X.join(temp, how='left')

    

    ## SUSCRIPCION

    

    pags = [284,  429,  343,  341,  431, 1081, 1072,  342,  430,  592,  307, 1267,  275,  714]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "fima_constitucion")

    X = X.join(temp, how='left')

    

    # HABERES

    

    pags = [603, 600, 846, 601, 507, 602, 599, 818]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "pago_haberes")

    X = X.join(temp, how='left')

    

    # AGRO

    

    pags = [1115,  900, 1332, 1132, 1681,  868, 1194, 1167,  243, 1410, 1796, 1767, 1016,  876]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "agro")

    X = X.join(temp, how='left')

    

    # PYME

    

    pags = [1051,  699,  398,  783,  840, 1304,  823,  779, 1791, 1282, 1364,1289, 1300, 1786, 1197, 1663,  203]

    temp = data_periodo.loc[data_periodo['page'].isin(pags)]

    temp = generar_features_temporales(temp, "pyme")

    X = X.join(temp, how='left')

    

    # DETENER DEBITO AUTOMATICO

    

    ## TODOS

    

    temp = data_periodo.loc[data_periodo['categoria_pag'].isin([1338, 1340, 1339])]

    temp = generar_features_temporales(temp, "detener_debito")

    X = X.join(temp, how='left')

    

    ## CONFIRMACION

    

    temp = data_periodo.loc[data_periodo['categoria_pag'] == 1340]

    temp = generar_features_temporales(temp, "detener_debito_confirmacion")

    X = X.join(temp, how='left')

    

    # TARJETAS

    

    #

    # OTRAS FEATURES

    #

    

    # DEBITO AUTOMATICO

    

    ## DESADHESIONES

    

    ### TARJETAS

    

    #debito_automatico = data.loc[data['page'].isin([141, 486, 1340])].sort_values(by=['fec_event'], ascending=True).drop_duplicates(subset=['user_id'], keep='last')

    

    

    ### PAGO

    

    

    # CANTIDAD DE TARJETAS

    

    def generar_cantidad_tarjetas(df_X, arr_pages_tarjetas, nombre_feature):

        df_X[nombre_feature] = 0

        i = 1

        for num_page in arr_pages_tarjetas:

            usuarios_n_tarjetas = data.loc[data['page'] == num_page]['user_id'].unique()

            df_X.loc[usuarios_n_tarjetas, [nombre_feature]] = i

            i += 1

    

    ## CREDITO

    

    generar_cantidad_tarjetas(X, [23,  69, 409], 'cantidad_tarjetas_credito')

    

    ## DEBITO

    

    generar_cantidad_tarjetas(X, [113,  235,  329,  330,  535, 1239], 'cantidad_tarjetas_debito')

    

    ## ADICIONALES

    

    generar_cantidad_tarjetas(X, [328,  327], 'cantidad_tarjetas_adicionales')

    

    # CATEGORIA USUARIO

    

    paginas_web = [1,40,202,203,243,268,370,1492]

    dict_pag_cat_usuario = {jerarquiasPage['WEB'][pag]['ids'][0]: pag for pag in jerarquiasPage['WEB']}

    cat_usuarios = data.loc[data['page'].isin(paginas_web)].copy()

    cat_usuarios['cat_usuario'] = [dict_pag_cat_usuario[k] for k in cat_usuarios['page']]

    cat_usuarios = pd.crosstab(cat_usuarios["user_id"], cat_usuarios['cat_usuario'])

    cat_usuarios = cat_usuarios.div(cat_usuarios.sum(axis=1), axis=0)

    X = X.join(cat_usuarios, how = "left").fillna(0)



    # CONVERSIONES POR MES

    

    convs_por_mes = pd.crosstab(convs_periodo['user_id'], convs_periodo['mes'])

    convs_por_mes = convs_por_mes.loc[:, sorted(convs_por_mes.columns)]

    convs_por_mes.columns = ['convs_mes_' + str(i) for i in range(1, len(convs_por_mes.columns) + 1)]

    X = X.join(convs_por_mes, how = "left").fillna(0)



    # PORCENTAJE DE ENTRADAS POR MOVIL Y COMPUTADORA

    

    mobile = pd.crosstab(device_periodo['user_id'], device_periodo['is_mobile_device'])

    mobile.columns = ['mobile_' + str(s) for s in mobile.columns]

    mobile = mobile.div(mobile.sum(axis=1), axis=0)

    X = X.join(mobile, how = "left").fillna(0)



    # PORCENTAJE DE ENTRADA POR CADA CATEGORIA DE CONECTION SPEED

    

    conection_speed = pd.crosstab(device_periodo['user_id'], device_periodo['connection_speed'])

    conection_speed.columns = ['con_speed_' + str(s) for s in conection_speed.columns]

    conection_speed = conection_speed.div(conection_speed.sum(axis=1), axis=0)

    X = X.join(conection_speed, how = "left").fillna(0)



    # CANTIDAD DE SESIONES POR MES

    

    sesiones_por_mes = pd.crosstab(device_periodo['user_id'], device_periodo['fec_event'].dt.month)

    sesiones_por_mes = sesiones_por_mes.loc[:, sorted(sesiones_por_mes.columns)]

    sesiones_por_mes.columns = ["sesiones_mes_" + str(i) for i in range(1, len(sesiones_por_mes.columns)+1)]

    X = X.join(sesiones_por_mes, how = "left").fillna(0)



    # CANTIDAD DE SESIONES TOTALES DEL PERIODO

    

    sesiones_totales_periodo = device_periodo['user_id'].value_counts().to_frame()

    sesiones_totales_periodo.columns = ['sesiones_totales_periodo']

    X = X.join(sesiones_totales_periodo, how = "left").fillna(0)

    

    return X
CANT_MESES_PERIODO = 9
# Generar test



mes_inicio_test = 12 - CANT_MESES_PERIODO

data_test = data.loc[data["fec_event"].dt.month > mes_inicio_test]

device_test = device.loc[device["fec_event"].dt.month > mes_inicio_test]

convs_test = convs.loc[convs["mes"] > mes_inicio_test]



X_test = generar_features(data_test, device_test, convs_test)
# Generar train

X_train = []

y_anio = pd.read_csv("../input/conversiones/conversiones.csv")

y_anio.columns = [s.lower() for s in y_anio.columns]



mes_fin_train = 1 + CANT_MESES_PERIODO



data_train = data.loc[data["fec_event"].dt.month < mes_fin_train]

device_train = device.loc[device["fec_event"].dt.month < mes_fin_train]

convs_train = convs.loc[convs["mes"] < mes_fin_train]



X_train = generar_features(data_train, device_train, convs_train)

y_train = pd.Series(0, index=X_train.index)

idx = set(y_anio.loc[y_anio['mes'] >= mes_fin_train]["user_id"].unique()).intersection(set(X_train.index))

y_train.loc[list(idx)] = 1
features = list(set(X_train.columns).intersection(set(X_test.columns)))

X_train = X_train[features]

X_test = X_test[features]
submission = pd.read_csv("../input/samplesubmission/sampleSubmission.csv", index_col=0, header=0, names=["user_id", "score_ej"])

X_test = X_test.reindex(submission.index).fillna({'semanas_desde_ultima_actividad':1000, 'semanas_desde_primer_actividad':1000, 'semanas_desde_ultima_actividad_prestamos':1000, 'semanas_desde_primer_actividad_prestamos':1000}).fillna(0)



fi = []

test_probs = []

i = 0

scores_val = []

scores_train = []



for train_idx, valid_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X_train, y_train):

    i += 1

    Xt = X_train.iloc[train_idx]

    yt = y_train.loc[X_train.index].iloc[train_idx]



    Xv = X_train.iloc[valid_idx]

    yv = y_train.loc[X_train.index].iloc[valid_idx]



    learner = LGBMClassifier(n_estimators=10000,

                             max_depth=3,

                             num_leaves=14,

                             #boosting_type='dart',

                             #bagging_freq=1, bagging_fraction=0.05

                             #is_unbalance = True,

                             learning_rate=0.01,

                             #objective='binary',

                             random_state=42

                            )

    

    learner.fit(Xt, yt,  early_stopping_rounds=200, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv, yv)], verbose=10000)

    

    scores_val.append(learner.best_score_['valid_1']['auc'])

    scores_train.append(learner.best_score_['training']['auc'])

    

    test_probs.append(pd.Series(learner.predict_proba(X_test)[:, -1],

                                index=X_test.index, name="fold_" + str(i)))

    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns))



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

test_probs.index.name="USER_ID"

test_probs.name="SCORE"

test_probs.to_csv("preds.csv", header=True)

fi = pd.concat(fi, axis=1).mean(axis=1)
print("Score promedio training: " + str(np.around(np.mean(scores_train), decimals=4)))

print("Score promedio validacion: " + str(np.around(np.mean(scores_val), decimals=4)))
num_features = 50



features_importantes = fi[fi != 0].sort_values(ascending=False)

features_importantes = list(features_importantes[:num_features].index)

fi_nombres = fi.copy()

fi_nombres.index = poner_nombre_features(lista_features = fi_nombres.index,

                                         dict_pages = dict_pages,

                                         dict_category = dict_category,

                                         dict_category_bottom = dict_category_bottom,

                                         dict_category_top = dict_category_top,

                                         dict_site = dict_site)

fi_nombres[fi_nombres != 0].sort_values(ascending=False)[:num_features]