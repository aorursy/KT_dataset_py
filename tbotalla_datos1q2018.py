import numpy as np 
import pandas as pd
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import re # regex
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS # Wordcloud: pip install wordcloud

%matplotlib inline
avisos_detalle = pd.read_csv('../input/fiuba_6_avisos_detalle.csv')
avisos_detalle.info()
avisos_detalle.columns
avisos_detalle['idpais'].value_counts() 
avisos_detalle = avisos_detalle.drop(['idpais'], axis=1)
avisos_detalle.columns
avisos_detalle['mapacalle'].value_counts()
print("El", round(100 * avisos_detalle['mapacalle'].isnull().sum()/len(avisos_detalle), 2), "% de los datos de la columna mapacalle son nulos")
avisos_detalle = avisos_detalle.drop(['mapacalle'], axis=1)
avisos_detalle.columns
avisos_detalle['ciudad'].isnull().sum()
print("El", round(100 * avisos_detalle['ciudad'].isnull().sum()/len(avisos_detalle), 2), "% de los datos de la columna ciudad son nulos")
avisos_detalle['ciudad'].value_counts()
avisos_detalle = avisos_detalle.drop(['ciudad'], axis=1)
avisos_detalle.columns
avisos_detalle['nivel_laboral'].isnull().sum()
avisos_detalle['nivel_laboral'].value_counts()
seniorities = ['Senior / Semi-Senior','Junior','Otro','Jefe / Supervisor / Responsable','Gerencia / Alta Gerencia / Dirección']
num_items=len(seniorities)
plt.figure(figsize=(18,8))
margin = 0.05
width = 4*(1.-1*margin)/num_items
plt.title('Seniority de los avisos', fontsize=18)
plt.xlabel('Seniority', fontsize=16)
plt.ylabel('Cantidad(%)', fontsize=16)
plt.bar(seniorities, 100 * avisos_detalle.nivel_laboral.value_counts()/len(avisos_detalle), width, color="cadetblue")
avisos_detalle['tipo_de_trabajo'].isnull().sum()
avisos_detalle['tipo_de_trabajo'].value_counts()
modalidades_trabajo = ['Full-time','Part-time','Teletrabajo','Pasantia','Por Horas','Temporario','Por Contrato','Fines de Semana','Primer empleo']
num_items=len(modalidades_trabajo)
plt.figure(figsize=(18,8))
margin = 0.05
width = 7*(1.-1*margin)/num_items
plt.title('Modalidad de trabajo de los avisos', fontsize=18)
plt.xlabel('Modalidad', fontsize=16)
plt.ylabel('Cantidad(%)', fontsize=16)
plt.bar(modalidades_trabajo, 100 * avisos_detalle.tipo_de_trabajo.value_counts()/len(avisos_detalle), width, color="cadetblue")

avisos_detalle['nombre_zona'].isnull().sum()
avisos_detalle['nombre_zona'].value_counts()
zona_avisos = ['Buenos Aires (fuera de GBA)','Capital Federal','GBA Oeste','Gran Buenos Aires']
num_items=len(zona_avisos)
plt.figure(figsize=(18,8))
margin = 0.5
width = 5*(1.-1*margin)/num_items
plt.title('Zona de los avisos', fontsize=18)
plt.xlabel('Zona', fontsize=16)
plt.ylabel('Cantidad', fontsize=16)
plt.bar(zona_avisos, avisos_detalle.nombre_zona.value_counts(), width, color="cadetblue")
avisos_detalle['nombre_area'].isnull().sum()
avisos_detalle['nombre_area'].value_counts()
# TOP 15 de las areas mas buscadas
grouped_by_area_avisos = avisos_detalle.groupby('nombre_area')['nombre_area']\
    .agg(['count']).sort_values(by='count', ascending=False).head(15)
grouped_by_area_avisos = grouped_by_area_avisos.apply(lambda row: 100 * row['count'] / len(avisos_detalle), axis=1)
grouped_by_area_avisos
grouped_by_area_avisos.plot.bar(rot=0, figsize=(18,8), color='cadetblue', fontsize=10);
plt.ylabel('Proporción(%)', fontsize=20)
plt.xlabel('Área', fontsize=16)
plt.title('Áreas más buscadas', fontsize=17)
plt.legend('')
plt.xticks(rotation=90)
plt.show()
# TOP 20 empresas con más avisos
empresas_top = avisos_detalle.groupby('denominacion_empresa')['denominacion_empresa']\
    .agg(['count']).sort_values(by='count', ascending=False).head(20)
empresas_top = empresas_top.apply(lambda row: 100 * row['count'] / len(avisos_detalle), axis=1)
empresas_top
empresas_top.plot.bar(rot=0, figsize=(18,8), color='cadetblue', fontsize=10);
plt.ylabel('Proporción(%)', fontsize=20)
plt.xlabel('Empresa', fontsize=16)
plt.title('Empresas con más avisos', fontsize=17)
plt.legend('')
plt.xticks(rotation=90)
plt.show()
def process_words(row, col, col_new):  
    row[col_new] = row[col]
    words = row[col_new].split()
    
    # Se filtran las stop words y los simbolos
    valid_words = []
    for word in words:
        for w in stopwords:
            if (word.lower() == w):
                word = word.lower().replace(w, '')
        for inv in invalid_characters:
            word = word.lower().replace(inv, '')
        if (word != ''):
            valid_words.append(word)
        
    row[col_new] = set(valid_words) 
    return row
stopwords = ['a', 'al', 'ante', 'aquel', 'aires', 'bien', 'buenos', 'como', 'con', 'conseguir', 'cual', 'de', 'del', 
             'desde', 'donde', 'e', 'el', 'ella', 'ello', 'en', 'es', 'esa', 'encima', 'entonces', 'entre', 'encontramos', 'encuentra', 'era', 'esta', 'está',
             'estás', 'estas' 'estan', 'están', 'etc', 'fe', 'fue', 'ha', 'hacen', 'hacemos', 'hacer', 'hasta', 'incluso', 'ir', 'jamas', 
             'jamás', 'la', 'las', 'lo', 'los', 'más', 'me', 'menos', 'mi', 'mis', 'misma', 'mismo', 'mucha', 'muchas', 
             'mucho', 'muchos', 'muy', 'ni', 'no', 'nos', 'nosotros', 'nuestra', 'o', 'para', 'por', 'puesta', 'que', 'qué', 'sabe', 'santa', 'saber', 'se', 
             'según', 'ser', 'serán', 'seran', 'si', 'sí', 'siendo', 'sin', 'sobre', 'solo', 'solicita', 'somos', 'su', 'sus', 'te', 'tiene', 'tus', 'tu', 'uso', 
             'un', 'una', 'vaya', 'y']

invalid_characters = [',', ':', '.', ';', '', '?', '¿', '!', '¡', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      '<', '>', '-', '_', '/', '*']

def clean_html(string):
    clean_html_regex = re.compile('<.*?>')
    string = re.sub(clean_html_regex, '', string)
    return string

def process_top_words(row, col, col_new):  
    row[col] = clean_html(row[col])
    row[col_new] = row[col]
    # Se conservan los primeros 130 caracteres que son los que se ven en la vista previa
    # Descarto este enfoque porque se pierden algunas palabras que quedan partidas en la mitad
    #row['descripcion'] = row['descripcion'][0:130] 
    
    # En cambio nos quedamos con las primeras n palabras
    words = row[col_new].split()
    words = words[:25]
    
    # Se filtran las stop words y los simbolos
    valid_words = []
    for word in words:
        for w in stopwords:
            if (word.lower() == w):
                word = word.lower().replace(w, '')
        for inv in invalid_characters:
            word = word.lower().replace(inv, '')
        if (word != ''):
            valid_words.append(word)
        
    row[col_new] = set(valid_words) 
    return row
avisos_detalle = avisos_detalle.apply(lambda row: process_top_words(row, 'descripcion', 'descripcion_top_words_small'), axis=1)
avisos_detalle['descripcion_top_words_small'][10]
avisos_detalle.head(2)
lista_palabras_frecuentes_descripcion_small = []

def add_word(row, columna):
    words = row[columna]
    for word in words:
        lista_palabras_frecuentes_descripcion_small.append(word)

avisos_detalle.apply(lambda row: add_word(row, 'descripcion_top_words_small'), axis=1)
avisos_detalle.head(1)
text = ''
for word in lista_palabras_frecuentes_descripcion_small:
    text += ' ' + word
wordcloud = WordCloud(relative_scaling = 1,
                      stopwords = stopwords, width=800, height=600,
                      background_color="white"
                      ).generate(text)
wordcloud.to_file('./wc1.png')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
avisos_detalle = avisos_detalle.apply(lambda row: process_words(row, 'titulo', 'titulo_top_words'), axis=1)
avisos_detalle.head(2)
lista_palabras_frecuentes_titulos = []

def add_word(row, columna):
    words = row[columna]
    for word in words:
        lista_palabras_frecuentes_titulos.append(word)

avisos_detalle.apply(lambda row: add_word(row, 'titulo_top_words'), axis=1)
avisos_detalle.head(1)
# TOP 20 palabras mas frecuentes en el titulo
from collections import Counter
words_to_count = (word for word in lista_palabras_frecuentes_titulos if word[:1])
c = Counter(words_to_count)
print (c.most_common(20))
count_palabras_frecuentes_titulos = c.most_common(20)
text = ''
for word in lista_palabras_frecuentes_titulos:
    text += ' ' + word
wordcloud = WordCloud(relative_scaling = 1,
                      stopwords = stopwords, width=800, height=600,
                      background_color="white"
                      ).generate(text)
wordcloud.to_file('./wc2.png')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
avisos_detalle = avisos_detalle.apply(lambda row: process_words(row, 'descripcion', 'descripcion_top_words'), axis=1)
lista_palabras_frecuentes_descripcion = []

def add_word(row, columna):
    words = row[columna]
    for word in words:
        lista_palabras_frecuentes_descripcion.append(word)

avisos_detalle.apply(lambda row: add_word(row, 'descripcion_top_words'), axis=1)
avisos_detalle.head(1)
# TOP 25 palabras mas frecuentes en la descripción
from collections import Counter
words_to_count = (word for word in lista_palabras_frecuentes_descripcion if word[:1])
c = Counter(words_to_count)
print (c.most_common(25))
count_palabras_frecuentes_descripcion = c.most_common(25)
text = ''
for word in lista_palabras_frecuentes_descripcion:
    text += ' ' + word
wordcloud = WordCloud(relative_scaling = 1,
                      stopwords = stopwords, width=800, height=600,
                      background_color='white'
                      ).generate(text)
wordcloud.to_file('wc3.png')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from datetime import datetime, date

sns.set_style('darkgrid')
%matplotlib inline
Postulaciones_Edad = pd.read_csv('../input/fiuba_2_postulantes_genero_y_edad.csv')
Postulaciones_Estudios= pd.read_csv('../input/fiuba_1_postulantes_educacion.csv')
print ("Existen ", len(Postulaciones_Estudios), " registros con estudios")
print ("Existen ", len(Postulaciones_Edad), " registros con sexo y genero")

print("El", round(100 * Postulaciones_Estudios['idpostulante'].isnull().sum()/len(Postulaciones_Estudios), 2), "% de los datos de la columna idpostulante del set de estudios son nulos")
print("El", round(100 * Postulaciones_Edad['idpostulante'].isnull().sum()/len(Postulaciones_Edad), 2), "% de los datos de la columna idpostulante del set de edades son nulos")
Postulaciones_Estudios['idpostulante'].value_counts()
Postulaciones_Estudios.loc[Postulaciones_Estudios['idpostulante'] == 'YlMLGD']
def cuantificar_estudios(row):
    if (row['nombre'] == 'Doctorado'):
        row['nombre'] = 7
    if (row['nombre'] == 'Master'):    
        row['nombre'] = 6
    if (row['nombre'] == 'Posgrado'):    
        row['nombre'] = 5
    if (row['nombre'] == 'Universitario'):    
        row['nombre'] = 4
    if (row['nombre'] == 'Terciario/Técnico'):    
        row['nombre'] = 3
    if (row['nombre'] == 'Secundario'):    
        row['nombre'] = 2
    if (row['nombre'] == 'Otro'):    
        row['nombre'] = 1    
        
    if (row['estado'] == 'Graduado'):
        row['estado'] = 3
    if (row['estado'] == 'En Curso'):    
        row['estado'] = 2
    if (row['estado'] == 'Abandonado'):    
        row['estado'] = 1
        
    return row    

def descuantificar_estudios(row):
    if (row['nombre'] == 7):
        row['nombre'] = 'Doctorado'
    if (row['nombre'] == 6):    
        row['nombre'] = 'Master'
    if (row['nombre'] == 5):    
        row['nombre'] = 'Posgrado'
    if (row['nombre'] == 4):    
        row['nombre'] = 'Universitario'
    if (row['nombre'] == 3):    
        row['nombre'] = 'Terciario/Técnico'
    if (row['nombre'] == 2):    
        row['nombre'] = 'Secundario'
    if (row['nombre'] == 1):    
        row['nombre'] = 'Otro'    
        
    if (row['estado'] == 3):
        row['estado'] = 'Graduado'
    if (row['estado'] == 2):    
        row['estado'] = 'En Curso'
    if (row['estado'] == 1):    
        row['estado'] = 'Abandonado'
        
    return row    
Postulaciones_Estudios.apply(lambda row: cuantificar_estudios(row), axis=1)
Postulaciones_Estudios.head(1)
grouped_postulantes = Postulaciones_Estudios.groupby('idpostulante')
print('Existen ', len(grouped_postulantes), ' postulantes diferentes con estudios registrados')
grouped_postulantes = Postulaciones_Estudios.groupby('idpostulante').apply(lambda g: g.sort_index(by='nombre', ascending=False).head(1))
grouped_postulantes.reset_index(drop=True, inplace=True)
Postulaciones_Estudios = grouped_postulantes
Postulaciones_Estudios.apply(lambda row: descuantificar_estudios(row), axis=1)
Postulaciones_Estudios.loc[grouped_postulantes['idpostulante'] == 'YlMLGD']
postulantes = pd.merge(Postulaciones_Estudios, Postulaciones_Edad, on =['idpostulante'])
postulantes.head(1)
del Postulaciones_Estudios
del Postulaciones_Edad
postulantes['sexo'].value_counts()
postulantes['fechanacimiento'].isnull().sum()
postulantes = postulantes.dropna().reset_index()
del postulantes['index']
postulantes['fechanacimiento'].isnull().sum()
postulantes.head(1)
postulantes['fechanacimiento']= pd.to_datetime(postulantes['fechanacimiento'],errors = 'coerce', format='%Y-%m-%d')
postulantes = postulantes.dropna().reset_index()
postulantes['sexo'].value_counts()
def calculate_age(year, month, day):
    today = date.today()
    return today.year - year - ((today.month, today.day) < (month, day))

postulantes['edad'] = postulantes['fechanacimiento'].map(lambda x: calculate_age(x.year,x.month,x.day))
postulantes.dtypes
postulantes['edad'].isnull().sum()
postulantes = postulantes.dropna().reset_index()
postulantes['edad'] = postulantes['edad'].astype(int)
del postulantes['index']
postulantes.dtypes
edad_jubilacion_fem=60
edad_jubilacion_masc=65
edad_legal=18
cantidad_postulantes_fem=len(postulantes.loc[postulantes['sexo'] == 'FEM'])
cantidad_postulantes_masc=len(postulantes.loc[postulantes['sexo'] == 'MASC'])
postulantes_jubilables_fem = postulantes.loc[postulantes['edad'] > edad_jubilacion_fem]
postulantes_jubilables_fem = postulantes_jubilables_fem.loc[postulantes_jubilables_fem['sexo'] == 'FEM']
postulantes_jubilables_fem.head(3)
print('El ', round(100 * len(postulantes_jubilables_fem)/cantidad_postulantes_fem, 2), '% de las postulantes mujeres están en edad de jubilación')
postulantes_jubilables_masc = postulantes.loc[postulantes['edad'] > edad_jubilacion_masc]
postulantes_jubilables_masc = postulantes_jubilables_masc.loc[postulantes_jubilables_masc['sexo'] == 'MASC']
postulantes_jubilables_masc.head(3)
print('El ', round(100 * len(postulantes_jubilables_masc)/cantidad_postulantes_masc, 2), '% de los postulantes hombres están en edad de jubilación')
postulantes_menores = postulantes.loc[postulantes['edad'] < edad_legal]
postulantes_menores.head(3)
print('El ', round(100 * len(postulantes_menores)/len(postulantes), 2), '% de los postulantes son menores de edad')
print('Representan el ', round(100 * (len(postulantes_menores) + len(postulantes_jubilables_fem) + len(postulantes_jubilables_masc))/len(postulantes), 2), '% del total de postulantes')
postulantes['edad'].value_counts()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Edad postulantes: Histograma', fontsize=16, fontweight='bold')
ax.set_xlabel('Edad')
ax.set_xlim([16,80])
postulantes['edad'].plot.hist(figsize=[10,10], bins=80)
P_Masculino_Edad = postulantes[postulantes['sexo']=='MASC']
len(P_Masculino_Edad)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Edad postulantes Masculinos: Histograma', fontsize=16, fontweight='bold')
ax.set_xlabel('Edad')
ax.set_xlim([16,80])
P_Masculino_Edad['edad'].plot.hist(figsize=[10,10], bins=80)
P_Femenino_Edad = postulantes[postulantes['sexo']=='FEM']
len(P_Femenino_Edad)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Edad postulantes Femeninas: Histograma', fontsize=16, fontweight='bold')
ax.set_xlabel('Edad')
ax.set_xlim([16,80])
P_Femenino_Edad['edad'].plot.hist(figsize=[10,10], bins=80)
postulantes.head(3)
postulantes.info()
postulantes['nombre'].value_counts()
postulantes['estado'].value_counts()
Postulaciones_Graduados = postulantes
Postulaciones_Graduados['cantidad']=0
Postulaciones_Graduados= postulantes.loc[(postulantes['estado']=='Graduado')].groupby('nombre').count()
del Postulaciones_Graduados['idpostulante']
del Postulaciones_Graduados['estado']
del Postulaciones_Graduados['fechanacimiento']
del Postulaciones_Graduados['sexo']
del Postulaciones_Graduados['edad']
del Postulaciones_Graduados['level_0']
Postulaciones_Graduados
plot_postulantes_graduados = 100 * Postulaciones_Graduados['cantidad']/len(postulantes.loc[(postulantes['estado']=='Graduado')])
plot_postulantes_graduados = pd.DataFrame(plot_postulantes_graduados).reset_index()
plt.figure(figsize=(14,5))
plt.title('Distribución de postulantes graduados', fontsize=16, fontweight='bold')
sns.barplot(x='nombre', y='cantidad', data=plot_postulantes_graduados)
postulantes_en_curso = postulantes
postulantes_en_curso['cantidad'] = 0
postulantes_en_curso= postulantes.loc[(postulantes['estado']=='En Curso')].groupby('nombre').count()
del postulantes_en_curso['idpostulante']
del postulantes_en_curso['estado']
del postulantes_en_curso['fechanacimiento']
del postulantes_en_curso['sexo']
del postulantes_en_curso['level_0']
del postulantes_en_curso['edad']
plot_postulantes_en_curso = 100 * postulantes_en_curso['cantidad']/len(postulantes.loc[(postulantes['estado']=='En Curso')])
plot_postulantes_en_curso = pd.DataFrame(plot_postulantes_en_curso).reset_index()
plt.figure(figsize=(14,5))
plt.title('Distribución de postulantes con estudios en curso', fontsize=16, fontweight='bold')
sns.barplot(x='nombre', y='cantidad', data=plot_postulantes_en_curso)
postulantes_abandonado = postulantes
postulantes_abandonado['cantidad'] = 0
postulantes_abandonado= postulantes.loc[(postulantes['estado']=='Abandonado')].groupby('nombre').count()
del postulantes_abandonado['idpostulante']
del postulantes_abandonado['estado']
del postulantes_abandonado['fechanacimiento']
del postulantes_abandonado['sexo']
del postulantes_abandonado['level_0']
del postulantes_abandonado['edad']
postulantes_abandonado = 100 * postulantes_abandonado['cantidad']/len(postulantes.loc[(postulantes['estado']=='En Curso')])
postulantes_abandonado = pd.DataFrame(postulantes_abandonado).reset_index()
plt.figure(figsize=(14,5))
plt.title('Distribución de postulantes con estudios abandonados', fontsize=16, fontweight='bold')
sns.barplot(x='nombre', y='cantidad', data=postulantes_abandonado)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
vistas_raw = pd.read_csv('../input/fiuba_3_vistas.csv')
postulaciones_raw = pd.read_csv('../input/fiuba_4_postulaciones.csv')
avisos_raw = pd.read_csv('../input/fiuba_6_avisos_detalle.csv')
generos_raw = pd.read_csv('../input/fiuba_2_postulantes_genero_y_edad.csv')
educacion_raw = pd.read_csv('../input/fiuba_1_postulantes_educacion.csv')
print(postulaciones_raw.info())
print('---------------------------------')
postulaciones_raw.head(1)
postulaciones_raw.isna().any()
postulaciones = postulaciones_raw.rename(columns={'fechapostulacion':'fecha'})
postulaciones['fecha'] = pd.to_datetime(postulaciones.fecha)
postulaciones.dtypes
postulaciones.shape
# TOP avisos con más postulaciones
data = pd.merge(postulaciones,avisos_raw[['idaviso','titulo','nombre_area']], on='idaviso')
print(data.isna().any())
data.head(1)
top_post = data.idaviso.value_counts().reset_index().rename(columns={'idaviso':'count','index':'idaviso'})
data.idaviso.value_counts().head(3)
top_post.head(3)
avisos_raw[avisos_raw.idaviso == 1112334791]
seleccion = avisos_raw.idaviso.isin(top_post.idaviso)
top_post['titulo'] = avisos_raw.loc[seleccion,'titulo'].values
top_post['area'] = avisos_raw.loc[seleccion,'nombre_area'].values
top_post.head(3)
plot_data = top_post[['titulo','count']].head(50).iloc[::-1]
plt.figure(figsize=(7,14))
plt.title('Top 50 avisos según postulaciones')
plt.barh(plot_data['titulo'], plot_data['count'])
plt.xlabel('cantidad de postulaciones')
plt.ylabel('título del aviso')
plt.show()
data = postulaciones.merge(avisos_raw[['idaviso','titulo','nombre_area']], on='idaviso',how='left')\
                    .merge(educacion_raw, on='idpostulante',how='left')\
                    .merge(generos_raw,on='idpostulante',how='left')\
                    .rename(columns={'nombre_area':'area','nombre':'nivel_educ'})
print(data.info())
print(data.isna().any())
data.head(1)
plt.figure(figsize=(16,5))
plt.title('Cantidad de postulaciones según nivel educativo')
sns.countplot(x=data['nivel_educ'],data=data)
plt.xlabel('nivel educativo')
plt.ylabel('postulaciones')
plt.show()
plt.figure(figsize=(16,5))
plt.title('Cantidad de postulaciones según nivel educativo y género')
sns.countplot(x=data['nivel_educ'],data=data, hue=data['sexo'])
plt.xlabel('nivel educativo')
plt.ylabel('postulaciones')
plt.show()
data_to_pivot = data
data_to_pivot['count'] = 1
data_pivoted = data_to_pivot[['nivel_educ','sexo','count']].groupby(['sexo','nivel_educ']).sum().pivot_table(values='count',index='sexo',columns='nivel_educ')
data_pivoted
plt.figure(figsize=(10,6))
plt.title('Relación género/nivel educativo según cantidad de postuaciones')
sns.heatmap(data_pivoted, cmap='GnBu')
plt.xlabel('nivel educativo')
plt.ylabel('género')
plt.show()
plot_data = data.area.value_counts().head(25).iloc[::-1]
plt.figure(figsize=(6,10))
plt.title('Top 25 áreas según cantidad de postulaciones')
plt.barh(plot_data.index,plot_data.values)
plt.xlabel('cantidad de postulaciones')
plt.ylabel('área')
plt.autoscale(tight=True, axis='y')
plt.gca().invert_xaxis()
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.show()
plot_data = avisos_raw.nombre_area.value_counts().head(25).iloc[::-1]
plt.figure(figsize=(6,10))
plt.title('Top 25 áreas según cantidad de avisos')
plt.barh(plot_data.index,plot_data.values)
plt.xlabel('cantidad de avisos')
plt.ylabel('área')
plt.autoscale(tight=True, axis='y')
#plt.gca().yaxis.tick_right()
#plt.gca().yaxis.set_label_position("right")
plt.show()




vistas_cont = vistas_raw.idAviso.value_counts().reset_index().rename(columns={'idAviso':'vistas','index':'idaviso'})
postulaciones_cont = postulaciones_raw.idaviso.value_counts().reset_index().rename(columns={'idaviso':'postulaciones','index':'idaviso'})
vistas_cont.head(1)
postulaciones_cont.head(1)
vistas_y_post = pd.merge(vistas_cont, postulaciones_cont, on='idaviso')
vistas_y_post.head(1)
# Aproximación lineal por cuadrados mínimos
x = vistas_y_post['vistas']
y = vistas_y_post['postulaciones']
fit = np.polyfit(x, y, deg=1)
least_squares_aprox = fit[0] * x + fit[1]
plt.figure(figsize=(14,8))
plt.title('Relación vistas/postulaciones')
plt.scatter(x=vistas_y_post.vistas,y=vistas_y_post.postulaciones, alpha=.5)
plt.plot(x, least_squares_aprox, color='red', label='aproximación')
plt.legend(['aproximación'])
plt.xlabel('vistas')
plt.ylabel('postulaciones')
plt.show()
vistas_y_post_filtered = vistas_y_post[vistas_y_post['vistas']<2000]
# Aproximación por cuadrados mínimos
x = vistas_y_post_filtered['vistas']
y = vistas_y_post_filtered['postulaciones']
fit = np.polyfit(x, y, deg=1)
least_squares_aprox = fit[0] * x + fit[1]
plt.figure(figsize=(14,8))
plt.title('Relación vistas/postulaciones')
plt.scatter(x=vistas_y_post_filtered.vistas,y=vistas_y_post_filtered.postulaciones, alpha=.5)
plt.plot(x, least_squares_aprox, color='red', label='aproximación')
plt.legend(['aproximación'])
plt.xlabel('vistas')
plt.ylabel('postulaciones')
plt.show()
postulaciones.shape
post_cont = postulaciones.idpostulante.value_counts()
print('Promedio de postulaciones por usuario:',post_cont.values.mean())
print('Cantidad máxima de postulaciones de un usuario:', post_cont.values.max())
print('Cantidad mínima de postulaciones de un usuario:', post_cont.values.min())
post_cont.describe()
max_user_post = postulaciones.idpostulante.value_counts().head(1)
max_user_post
postulaciones.idpostulante.value_counts().head(1).index
data = postulaciones.merge(avisos_raw, on='idaviso', how='inner').rename(columns={'nombre_area':'area'})
data.shape
max_user_post_areas = data.loc[data.idpostulante == max_user_post.index[0],['area']].area.value_counts()
max_user_post_areas.sum()
max_user_post_areas.describe()
print('Promedio de postulaciones por área:',max_user_post_areas.values.mean())
print('fecha inicial:', data.fecha.min())
print('fecha final:', data.fecha.max())
print('tiempo transcurrido: ',data.fecha.max()-data.fecha.min())
print('primedio de postulaciones por día:',max_user_post_areas.sum()/44)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
vistas_raw = pd.read_csv('../input/fiuba_3_vistas.csv')
postulaciones_raw = pd.read_csv('../input/fiuba_4_postulaciones.csv')
avisos_raw = pd.read_csv('../input/fiuba_6_avisos_detalle.csv')
generos_raw = pd.read_csv('../input/fiuba_2_postulantes_genero_y_edad.csv')
educacion_raw = pd.read_csv('../input/fiuba_1_postulantes_educacion.csv')
vistas_raw.head(1)
postulaciones_raw.head(1)
avisos_raw.head(1)
generos_raw.head(1)
educacion_raw.head(1)
vistas = vistas_raw
postulaciones = postulaciones_raw
vistas['timestamp'] = pd.to_datetime(vistas.timestamp)
vistas['fecha'] = pd.to_datetime(vistas.timestamp.dt.date)
vistas['hora'] = vistas.timestamp.dt.hour
vistas['min'] = vistas.timestamp.dt.minute
vistas['seg'] = vistas.timestamp.dt.second
#vistas.rename(columns={'timestamp':'fecha'}, inplace=True)

vistas.head(1)
vistas.dtypes
plt.figure(figsize=(14,4))
plt.title('Vistas por fecha')
sns.countplot(vistas.fecha.dt.date)
plt.show()
data = pd.merge(vistas, generos_raw, on='idpostulante')
data.head(1)
plt.figure(figsize=(14,5))
plt.title('Vistas por fecha según género')
sns.countplot(x=data.fecha.dt.date,hue='sexo',data=data)
plt.show()
dias = ['lunes','martes','miércoles','jueve','viernes','sábado','domingo']
plt.figure(figsize=(14,5))
plt.title('Vistas por día de la semana según género')
sns.countplot(data.fecha.dt.dayofweek, data=data)
plt.xticks(data.fecha.dt.dayofweek,dias)
plt.show()

postulaciones.rename(columns={'fechapostulacion':'fecha'}, inplace=True)
vistas.info()
vistas.head(1)
# Veo que NO faltan datos pero debo transofrmar la columna timestamp en datetime
vistas.fecha = pd.to_datetime(vistas.fecha)
vistas['hora'] = vistas.fecha.dt.time
vistas['fecha'] = vistas.fecha.dt.date
vistas.dtypes
vistas.head(3)
postulaciones.info()
postulaciones.head(1)
postulaciones.fecha = pd.to_datetime(postulaciones.fecha)
postulaciones.dtypes
postulaciones.idpostulante.isna().any()
data = vistas.idAviso.value_counts().value_counts()
print('cant:',data.count(),'| min:',data.min(),'| max:',data.max())
data.plot(kind='hist',
          bins=60, 
          logy=True,
          figsize=(14,4), 
          title='Distribución de vistas de publicaciones', 
          grid=True,
          xticks=[x for x in range(0,1200,20)],
          rot=75)
plt.xlabel('Cantidad de vistas')
plt.ylabel('Frecuencia')
plt.show()
data[data < 20].plot(kind='hist',
          bins=18,
          figsize=(14,4), 
          title='Distribución de vistas de publicaciones (menos de 20 vistas)', 
          grid=True,
          xticks=[x for x in range(0,20)])
plt.xlabel('Cantidad de vistas')
plt.ylabel('Frecuencia')
plt.show()
data.plot(kind='kde', figsize=(14,4))
plt.show()

vistas.idAviso.value_counts().head(50).plot(kind='bar', figsize=(12,5))
plt.show()
data = vistas.idAviso.value_counts()
(data.max()-data.min())
data = vistas.idAviso.value_counts()
plt.figure(figsize=(14,6))
plt.title('Distribución')
plt.hist(data.head(5000), bins=20, density=True)
plt.xticks([x for x in range(0,4000,200)])
plt.grid(True)
plt.show()

avisos = avisos_raw.rename(columns={'idaviso':'idAviso'})
avisos.head(1)
top_vistas = vistas.idAviso.value_counts().head(10).reset_index().rename(columns={'idAviso':'count','index':'idAviso'})
top_vistas
data = top_vistas.merge(avisos[['idAviso','titulo','nombre_area']], on='idAviso')
data