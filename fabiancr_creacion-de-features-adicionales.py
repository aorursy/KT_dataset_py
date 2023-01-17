# Se importan las librerias básicas para el manejo de los DataFrame

import pandas as pd

import numpy as np

seed = 99
# cargar pageviews

pageviews = pd.concat([ pd.read_csv('../input/banco-galicia-dataton-2019/pageviews.zip', parse_dates=["FEC_EVENT"]),

                      pd.read_csv('../input/banco-galicia-dataton-2019/pageviews_complemento.zip', parse_dates=["FEC_EVENT"]) ], sort=False)



pageviews['mes'] = pageviews['FEC_EVENT'].apply( lambda x : x.month ) # crea la columna mes

pageviews['diasemana'] = pageviews['FEC_EVENT'].apply( lambda x : x.weekday() ) # crea la columna que indica el dia de la semana que hizo la operacion.



print('Datos iniciales del archivo pageviews:')

pageviews.head()
# cargar content category

content = pd.read_csv('../input/banco-galicia-dataton-2019/CONTENT_CATEGORY.zip')



print('Datos iniciales del archivo content category:')

print(content.head())



# Realizar reemplazo para alinear la data

content['CONTENT_CATEGORY_descripcion'] = content['CONTENT_CATEGORY_descripcion'].str.replace('ERRORES', 'ERROR')

content['CONTENT_CATEGORY_descripcion'] = content['CONTENT_CATEGORY_descripcion'].str.replace('GALICIA', '')

content['CONTENT_CATEGORY_descripcion'] = content['CONTENT_CATEGORY_descripcion'].str.replace('NO CATEGORY ASSIGNED > UPGRADE DE PAQUETE : CONFIRMACION',

                                                                                              'NO CATEGORY ASSIGNED > HB:NCA:NCA:NCA:UPGRADEDEPAQUETE : CONFIRMACION')

content['CONTENT_CATEGORY_descripcion'] = content['CONTENT_CATEGORY_descripcion'].str.replace('WEB:RRHH', 'WEB:INSTITUCIONAL:RRHH')



# Realiza split en columnas de la descipcion de content category                                                                                              

content['split'] = content['CONTENT_CATEGORY_descripcion'].str.split(' > :| : | > |:')



# Organizar la data para que quede en una estructura homogénea

for i in content.index:

    if content['split'][i][0] != 'NO CATEGORY ASSIGNED':

        content['split'][i][0:0] = ['CATEGORY ASSIGNED']



for i in content.index:

    if len(content['split'][i]) > 2:

        if content['split'][i][2] == 'PERSONAS':

            content['split'][i].insert(2,'ONLINE')

            content['split'][i].insert(3,'WEB')

        elif content['split'][i][2] == 'PP':

            content['split'][i].pop(-1)

        elif content['split'][i][2] == 'TC':

            content['split'][i].pop(2)

        elif content['split'][i][2] == 'UPGRADE':

            content['split'][i].pop(-1)

        

for i in content.index:

    if len(content['split'][i]) > 5:

        if content['split'][i][5] == 'ERROR':

            content['split'][i].pop()

            content['split'][i][2] = 'ERROR'

            

for i in content.index:

    try:

        if content['split'][i][5].find('BENEFICIOSYPROMO') >= 0:

            content['split'][i].pop(6)

        if content['split'][i][5].find('PRODUCTOSYS') >= 0:

            content['split'][i].pop(6)

        if content['split'][i][5].find('FICHAS') >= 0:

            content['split'][i].pop(5)

        if content['split'][i][5].find('TARJETARURAL') >= 0:

            content['split'][i].pop(-1)

    except IndexError:

        pass



# Separar los datos de content en columnas            

content[['a','b','c','d','e','f','g']] = pd.DataFrame(content['split'].values.tolist(), index= content.index)



# Una vez tenemos los datos organizados, existen varias filas que son idénticas, esta información la podemos agrupar para tener en la tabla original menos valores únicos

# Crea el diccionario repcontent con el content category para reemplazar en la data



cols_t = ['CONTENT_CATEGORY_descripcion', 'CONTENT_CATEGORY', 'split']

repcontent = dict()

i = 0

for npage in list(content[content.drop(cols_t, axis=1).duplicated(keep=False)]['CONTENT_CATEGORY']):

    i += 1

    for npage2 in list(content[content.drop(cols_t, axis=1).duplicated(keep=False)]['CONTENT_CATEGORY'])[i:]:

        n = content[content['CONTENT_CATEGORY'] == npage].drop(cols_t,axis=1).merge(content[content['CONTENT_CATEGORY'] == npage2].drop(cols_t,axis=1)).shape[0]

        if n == 1:           

            if (npage == 4) | (npage == 16):

                continue

            if npage2 in repcontent:

                continue

            else:

                repcontent[npage2] = npage



content.drop(['CONTENT_CATEGORY_descripcion', 'a', 'b', 'c', 'd', 'e', 'f', 'g'],axis=1, inplace=True) # se borran columnas que ya no se necesitan



print('Datos finales del archivo content category:')

print(content.head())
# cargar archivo page

page = pd.read_csv('../input/banco-galicia-dataton-2019/PAGE.zip')



print('Datos iniciales del archivo page:')

print(page.head())



# Crear diccionario para remplazar valores con el objetivo de generalizar y limpiar los datos

replace_dict = {

                'HB :  :':'HB:NCA:NCA:NCA:', 'HB :':'HB:NCA:NCA:', 'HC :':'HC:NCA:NCA:NCA:', '^/':'NCA/NCA/NCA/', '^0':'', ' ':'', '(':'',  ')':'', 

                '-':'_', '2018':'', '%C3%A9MINENT':'EMINENT', '%20/%20%3CBR%':'', '%20/%20%3CBR':'', 'A%20PAGAR:%20':'PAGAR','VISAHOME':'TARJETAS', 

                '_VALIDACION':'/VALIDACION',  '/SELECCIONAR_':'/SELECCIONAR/',  'USUARIO_':'USUARIO/',  'UPGRADE_':'UPGRADE/',   '_AVISO':'/AVISO', 

                'TYC_BAJAPRODUCTO':'BAJAPRODUCTO/TYC','_TYC':'TYC','TITULAR_':'TITULAR/','TARJETASAS':'TARJETAS','TADEC':'TAC','2017':'', '15$':'',

                '_SOLICITAR':'/SOLICITAR',  '/PREGUNTAS_':'/PREGUNTAS/',  '_PREGUNTA_':'/PREGUNTAS_',  'CONFIRMACION':'CONFIRMAR', '_AFIP':'/AFIP',

                'PRE_CONFIRMAR':'PRECONFIRMAR','OPRECONFIRMAR':'O/PRECONFIRMAR', '_PRECONFIRMACI$':'/PRECONFIRMAR','_PRECONFIRMA$':'/PRECONFIRMAR', 

                '_PRECONFIRMAR':'/PRECONFIRMAR', '_NUMERO':'/NUMERO',  '_NUEVO':'/NUEVO', '_MOTIVO':'/MOTIVO', 'O_MONTO':'O/MONTO', ':HOME':':HOM',

                'R_MONTO$':'R/MONTO','A_MONTO$':'A/MONTO','_MONEDA':'/MONEDA','_METODO':'/METODO','_INGRESO$':'/INGRESO','_INGRESAR_':'/INGRESAR_',

                '_IDENTIFI':'/IDENTIFI', 'HIPOTECARIO_':'HIPOTECARIO/', 'HACETEMOVE':'HACETECLIENTE','HACETEGALICIA':'HACETECLIENTE','%20Y%20':'Y',

                '_FIRMA':'/FIRMA', '_FIMA_':'_FIMA/', '_EXITO':'/EXITO', '_ELECCION':'/ELECCION',  '_DELIVERY':'/DELIVERY',  '_DATOS_P':'/DATOS_P',

                '_CONTACTO':'/CONTACTO', '_CONFIRMACIO$':'/CONFIRMAR', '_CONFIRMAR':'/CONFIRMAR', '_CONFIR$':'/CONFIRMAR', 'MOVI':'MOV',  'BUP':'',

                '_CONCEPTO':'/CONCEPTO',  '_CODIGO':'/CODIGO',  'S_CANTIDAD':'S/CANTIDAD',  '_CAJEROS':'/CAJEROS',   '_ADICIONALES':'/ADICIONALES',

                '/BENEFICIOS':'/BENEFICIOSYPROMOCIONES', '_BENEFICIARIO':'/BENEFICIARIO',  'BANCO':'',  '_AUTORIZADOS':'/AUTORIZADOS',  '/1521':'',

                '_ALERTA':'/ALERTA', '_ADHERIR':'/ADHERIR', '/PREFER':'/PREFERPREFER', 'WEB:PREFER':'WEB:PREFERPREFER','PROYECTO':'PROYEC', '_':'',

                'WEB:PERSONAS':'WEB:PERSONASPERSONAS', 'PEDITUSPRODUCTOS':'PEDIDOSHOM', 'OFFICEBANKIN':'', 'WEB:MOVE':'WEB:MOVEMOVE', 'GALICI$':'',

                'WEB:EMINENT':'WEB:EMINENTEMINENT',  'UNHANDLEDERROR':'ERROR', 'TOMAIL':'TO/MAIL', 'TARJETASR':'TARJETAS/R',  'PAQUETES':'PAQUETE',

                'FIRMATERMINOSYCONDICIONE':'TYC', 'TERMINOSYCONDICIONES':'TYC','TERMINOSYCONDICIONE':'TYC','TERMINOSYCONDICION':'TYC','/TYC':'TYC', 

                'TYC':'/TYC', 'SW$':'', 'SUCURSAL99':'SUCURSALES','SUCURSAL$':'SUCURSALES','SPOTIF':'SPOTIFY','SOLICITARHIPOTECARIO':'HIPOTECARIO',

                '/RROR/':'/ERROR/', 'RENOVACIONINICIO':'RENOVACION', 'PRIMER/INGRESO':'PRIMERINGRESO', 'PREFER$':'', 'PERSONAS$':'','MOVIMI':'MOV',

                'SUCURSALESYCAJEROS':'SUCURSAL', 'PROYECTO':'PROYEC', 'PRECONFIRMARREPOSICION':'REPOSICION',  'CONFIRMARREPOSICION' : 'REPOSICION',

                'PRECONFIRMAR/NUEVO':'PRECONFIRMAR', 'FONDOS':'FONDO',  'PEDIDOPRESTAMOSPERSONALES':'PRESTAMOS',  'PEDIDOS:':'PEDIDOS', 'MOVE$':'',

                'OPERACIONES:':'OPERACIONES',  'OBJETIVO$':'OBJETIV','MAS$':'', 'NOMINALES$':'NOMINALE',  'NOLATENES':'NOLARECORDAS',  'GALIC$':'',

                'NOLARECUERDA':'NOLARECORDAS', 'MONTOOBJETIV$':'MONTO','MOMENTONO':'ERRO', 'MESSAGE':'MENSAJE', 'LOGBB':'LOGIN', 'ERRORES':'ERROR',

                'JAVASCRIPT:VOID':'JAVASCRIPT', 'INVERSFSIONES':'INVERSIONES','GALI$':'',  'INVERSIONES$':'INVERSION',  'INICIO:RE$':'INICIO:RECU',

                'INGRESARDOMICILI$':'INGRESARDOMICILIO', 'INICIO:R$':'INICIO:RECU', 'GAL$':'', 'CUENTGALICIA':'CUENTAS', 'FUERADESERVICIO':'ERROR',

                'GALICIA':'',  'INICIONUEVOCONTACTOC' : 'NUEVOCONTACTO/C',  'FILE/DOWNLOAD':'DOWNLOADFILE',   'FECHANACIMIENTO':'/FECHANACIMIENTO',

                'FECHAOBJETIV':'/FECHAOBJETIV', 'DECRE':'D',  'ERORIN':'ERROR',  'EGOORROR':'ERROR',  'EMINENT2':'', 'EMINENT$' : '', 'EMINEN$':'',

                'EMINE$':'', 'EMINET$':'', 'EMI$':'', 'EMAIL':'/EMAIL', 'DETAL$':'DETALLE',  'DETA$':'DETALLE',  'DESCRIPCIONSEGURO':'DESCRIPCION',  

                'DESCRIPCIONPRESTAMO':'DESCRIPCION','ALERTAS':'ALERTA', 'CONVENIOSRURAL':'CONVENIOS', 'TOCONFIRMAR':'TO/CONFIRMAR', 'CARREFOUR':'', 

                'CONFIRMAR/ALERTA':'CONFIRMAR',  'CARGARMONTOCUENTASPROPIAS':'CUENTASPROPIAS/MONTO',  'BONOSACCIONES':'BONOSYACCIONES',   '\.$':'',

                'APERTURADEC':'APERTURAD','APERTURADE':'APERTURAD' #  

               }



# Realizar reemplazo para arreglar los datos

for old, new in replace_dict.items():

    page['PAGE_descripcion'] = page['PAGE_descripcion'].str.replace(old, new)

    

# crea diccionario que corrige el content en la data

fixpage = dict()

for i in page.index:

    if ('HB' in page['PAGE_descripcion'][i]) and ('ERRO' in page['PAGE_descripcion'][i]):

        fixpage[i+1] = 5

    elif ('HB' in page['PAGE_descripcion'][i]) and ('PAQUETE' in page['PAGE_descripcion'][i]):

        continue

    elif ('HB' in page['PAGE_descripcion'][i]) and ('PRESTAMOS' in page['PAGE_descripcion'][i]):

        continue

    elif 'HB' in page['PAGE_descripcion'][i]:

        fixpage[i+1] = 4



# realiza split1 en columnas de la descipcion de page

page['split1'] = page['PAGE_descripcion'].str.split('://|/|:') # check 982



# Organiza los datos de page

for i in page.index:

    

    if (page['split1'][i][0] == 'WEB' or page['split1'][i][0] == 'RURAL' or page['split1'][i][0] == 'RRHH' or page['split1'][i][0] == 'PREFER' or

        page['split1'][i][0] == 'PERSONAS' or page['split1'][i][0] == 'PAGINA' or page['split1'][i][0] == 'NEGOCIOSYPYMES' or 

        page['split1'][i][0] == 'MOVE' or page['split1'][i][0] == 'INSTITUCIONAL' or page['split1'][i][0] == 'HB' or page['split1'][i][0] == 

        'EMPRESAS' or page['split1'][i][0] == 'EMINENT' or page['split1'][i][0] == 'CORPORATIVAS' or page['split1'][i][0] == 

        'BUSCADORDECAJEROSYSUCURSALES' or page['split1'][i][0] == 'PRODUCTOSYSERVICIOS' or page['split1'][i][0] == 'BENEFICIOSYPROMOCIONES'):

        page['split1'][i].pop(0)

    

    if page['split1'][i][-1] == '':

        page['split1'][i].pop(-1)

    

    try:

            

        if page['split1'][i][0] == 'HC':

            page['split1'][i].pop(0)

            if page['split1'][i][0].startswith('STEP') == True:

                page['split1'][i].pop(0)

            if page['split1'][i][0] == 'ERROR':

                page['split1'][i].insert(0,'NCA')

            if page['split1'][i][4].startswith('PASO') == True:

                page['split1'][i].pop(4)

            if page['split1'][i][4] == 'ERROR':

                page['split1'][i].pop(5) #*# borra las explicaciones del error

                

        if page['split1'][i][3].find('INICI') >= 0:

            page['split1'][i][3] = 'INICIO'

            

        if page['split1'][i][3].find('OGIN') >= 0:

            page['split1'][i][3] = 'LOGIN'

            

        if page['split1'][i][3].find('EMPRESA') >= 0:

            page['split1'][i].pop(0)    

        if page['split1'][i][3].find('USER') >= 0:

            page['split1'][i][3] = 'USERS'

            page['split1'][i].pop(0)

            if page['split1'][i][3].find('LOGI') >= 0:

                page['split1'][i][3] = 'LOGIN'

        if page['split1'][i][3].find('PREFER') >= 0:

            page['split1'][i].pop(0)  

            

        if page['split1'][i][3].find('ERR') >= 0:

            page['split1'][i][3] = 'ERROR' 

            if page['split1'][i][2] == 'NCA':

                page['split1'][i][0] = 'ERROR'

                page['split1'][i].insert(3,'NCA')

                page['split1'][i].pop(5) #*# borra las explicaciones del error

            else:

                page['split1'][i].pop(0)

                page['split1'][i].insert(2,'NCA')

        

        if (page['split1'][i][4].startswith('ERRO') == True) and (page['split1'][i][0] == 'NCA'):

            page['split1'][i][4] = 'ERROR' 

            page['split1'][i].pop(0)

            page['split1'][i].pop(4) #*# borra las explicaciones del error   

            

        if page['split1'][i][4].startswith('PASO') == True:

            page['split1'][i].pop(5) #*# borra las explicaciones del paso

            

        if page['split1'][i][4].startswith('ALERT') == True:

            page['split1'][i].pop(5) #*# borra las explicaciones de alertas



        if page['split1'][i][4].find('INICIO') >= 0:

            page['split1'][i][4] = 'INICIO'

       

        if (page['split1'][i][4].startswith('PRECONFIRMAR') == True) and (page['split1'][i][4] != 'PRECONFIRMAR'):

            page['split1'][i][4] = page['split1'][i][4][12:]

            page['split1'][i].append('PRECONFIRMAR')

        

        if (page['split1'][i][4].startswith('CONFIRMAR') == True) and (page['split1'][i][4] != 'CONFIRMAR'):

            page['split1'][i][4] = page['split1'][i][4][9:]

            page['split1'][i].append('CONFIRMAR')

      

        if page['split1'][i][5].find('PAGAR') >= 0:

            page['split1'][i][5] = 'PAGAR'



        if page['split1'][i][-1].isdigit():

            page['split1'][i].pop(-1)

            

        if page['split1'][i][-1] == '':

            page['split1'][i].pop(-1)



    except IndexError:

        pass



# Separar los datos de page en columnas    

page[['a','b','c','d','e','f']] = pd.DataFrame(page['split1'].values.tolist(), index= page.index)



#crea el diccionario reppage con el page para reemplazar en la data valores repetidos

cols_t = ['PAGE_descripcion', 'PAGE', 'split1'] 

reppage = dict()

i = 0

for npage in list(page[page.drop(cols_t, axis=1).duplicated(keep=False)]['PAGE']):

    i += 1

    for npage2 in list(page[page.drop(cols_t, axis=1).duplicated(keep=False)]['PAGE'])[i:]:

        n = page[page['PAGE'] == npage].drop(cols_t,axis=1).merge(page[page['PAGE'] == npage2].drop(cols_t,axis=1)).shape[0]

        if n == 1:           

            if npage2 in reppage:

                continue

            else:

                reppage[npage2] = npage



page.drop(['PAGE_descripcion', 'a', 'b', 'c', 'd', 'e', 'f',],axis=1, inplace=True) # se borran columnas que ya no se necesitan



print('Datos finales del archivo page:')

print(page.head())
# Aplicamos los diccionarios que creamos para unificar la data repetida que surgio al organizar la data

for i in fixpage:

    pageviews.loc[pageviews['PAGE'] == i, 'CONTENT_CATEGORY'] = fixpage[i]

    pageviews.loc[pageviews['PAGE'] == i, 'CONTENT_CATEGORY_BOTTOM'] = fixpage[i]



for i in repcontent:

    pageviews.loc[pageviews['CONTENT_CATEGORY'] == i, 'CONTENT_CATEGORY_BOTTOM'] = repcontent[i]

    pageviews.loc[pageviews['CONTENT_CATEGORY'] == i, 'CONTENT_CATEGORY'] = repcontent[i]

    

for i in reppage:

    pageviews.loc[pageviews['PAGE'] == i, 'PAGE'] = reppage[i]
pageviews = pd.merge(pageviews,content) # se une la columna split a pageviews

pageviews = pd.merge(pageviews,page) # se une la columna split1 a pageviews

pageviews['s_total'] = pageviews['split'] + pageviews['split1'] # se unen la columnas split y split1 en una sola

# La siguiente línea separa en columnas cada dato de la lista de s_total, aquí se entiende el objetivo de la 

# organización que se realizó, quedando primero si tiene o no categoría definida, luego su tipo (banca, HB, HC, …),

# luego si fue un error o una acción online, luego esta web que es una constante por lo que más adelante no la 

# tendremos en cuenta, TyPEr define el tipo de usuario (persona, empresa, negocios, …), ACT específica la acción del 

# usuario (login, consulta, pedido de préstamo, …), A1 y A2 especifica información adicional de la acción del usuario.   

pageviews[['Cat','TyCat','on_err','web','TyPer','Act','A1','A2']] = pd.DataFrame(pageviews['s_total'].values.tolist(), index= pageviews.index)

pageviews.drop(['split1','split','web'], axis=1, inplace=True) # Se borran las columnas que ya no se necesitan

pageviews['on_err'].replace(r'^NCA$', 'ONLINE', inplace=True, regex=True)
# cargar archivo devicedata

devicedata = pd.read_csv('../input/banco-galicia-dataton-2019/device_data.zip', parse_dates=["FEC_EVENT"])



data = pageviews.merge( devicedata, how="outer") # unimos la data de pageviews y devicedata a una nueva variable

del(pageviews) # borramos esta variable para liberar RAM en este kernel

# Existen algunos datos donde un usuario registra evento en un dispositivo pero no en pageviews lo que genera muchas filas con datos nulos que se necesitan borrar

data.dropna(axis='rows', thresh=7, inplace=True) 

data.sort_values(['USER_ID','FEC_EVENT'], inplace=True)

data.reset_index(drop=True, inplace=True)

# Al organizar por fecha podemos agrupar un poco más los datos, en este caso el TyPer donde podemos quitar dos tipos que son muy pequeños y llenar valores faltantes 

data['TyPer'].replace(r'^NCA$', np.NaN, inplace=True, regex=True)

data['TyPer'].replace(r'^MASIVO$', np.NaN, inplace=True, regex=True)

data['TyPer'].replace(r'^USERS$', np.NaN, inplace=True, regex=True)

data['TyPer'].fillna(method='ffill', inplace=True)

data['TyPer'].fillna('EMINENT', inplace = True)

data['CONNECTION_SPEED'].fillna(method='ffill', inplace=True)

data['IS_MOBILE_DEVICE'].fillna(method='ffill', inplace=True)

data['CONNECTION_SPEED'].fillna(1, inplace=True)

data['IS_MOBILE_DEVICE'].fillna(1, inplace=True)

data.fillna('NCA', inplace=True)



data.head()
data['s_total'] = data['s_total'].apply(', '.join) # pasamos de lista a string



# Transformar los strings a datos categoricos

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Cat'] = le.fit_transform(data['Cat'])

data['TyCat'] = le.fit_transform(data['TyCat'])

data['on_err'] = le.fit_transform(data['on_err'])

data['TyPer'] = le.fit_transform(data['TyPer'])

data['Act'] = le.fit_transform(data['Act'])

data['A1'] = le.fit_transform(data['A1'])

data['A2'] = le.fit_transform(data['A2'])

data['s_total'] = le.fit_transform(data['s_total'])



data.head()
# cargar archivo conversiones

conversiones = pd.read_csv('../input/banco-galicia-dataton-2019/conversiones.zip')

columnas = list(data.drop(['FEC_EVENT','CONTENT_CATEGORY_BOTTOM','USER_ID','mes'], axis=1).columns) # Genera lista de las columnas con las que vamos a trabajar

# Se descarta la columna CONTENT_CATEGORY_BOTTOM ya que es igual a la columna CONTENT_CATEGORY, esto se puede probar con la siguente linea

#data['CONTENT_CATEGORY'].equals(data['CONTENT_CATEGORY_BOTTOM'])
meses = [[1,9,4,12]] # Define los meses para seleccionar la data de entrenamiento [mes de inicio para X, mes de fin para X, mes de inicio para Y, mes de fin para Y]

# Se puede definir un solo periodo o varios, ej. meses = [[1,6,4,9], [4,9,7,12]]



# Librerias para el entrenamiento

from sklearn import model_selection

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score



# Organizamos toda la data para realizar la predicción

alldata = []

for c in columnas:

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    alldata.append(temp.apply(lambda x: x / x.sum(), axis=1))

alldata = pd.concat(alldata, axis=1)

allcolumns = list(alldata.columns)



test_probs = [] # Se guardan los resultados de cada predicción



# Genera la data de entrenamiento

for i in range(len(meses)):

    data2 = data[(data['mes'] >= meses[i][0]) & (data['mes'] <= meses[i][1])]

    y_train = pd.Series(1, conversiones[(conversiones['mes']>=meses[i][2])&(conversiones['mes']<=meses[i][3])]['USER_ID'].sort_values().unique())

    y_train = y_train.reindex(range(11676),fill_value=0)

    X_train = []

    for c in columnas:

        temp = pd.crosstab(data.USER_ID, data2[c])

        temp.columns = [c + "_" + str(v) for v in temp.columns]

        X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))

    X_train = pd.concat(X_train, axis=1)

    X_train = X_train.reindex(range(11676),columns=allcolumns,fill_value=0)

    

    j = 0

    

    # Se realiza split sobre la data para tener un modelo mas robusto

    for train_idx, valid_idx in model_selection.KFold(n_splits=10, shuffle=True, random_state=seed).split(X_train):

        j += 1

        Xt = X_train.iloc[train_idx]

        yt = y_train.iloc[train_idx]

        

        Xv = X_train.iloc[valid_idx]

        yv = y_train.iloc[valid_idx]

        

        learner = LGBMClassifier(n_estimators=10000, objective='binary', random_state=seed)



        learner.fit(Xt, yt,  early_stopping_rounds=20, eval_metric="auc", eval_set=[(Xt, yt), (Xv, yv)], verbose=0)



        test_probs.append(pd.Series(learner.predict_proba(alldata)[:, -1], index=alldata.index, name="fold_" + str(j)))



test_probs = pd.concat(test_probs, axis=1).mean(axis=1) # guardamos el promedio de las predicciones.

test_probs.index.name="USER_ID"

test_probs.name="SCORE"

test_probs.to_csv("resultado.csv", header=True)
# Resultados

test_probs