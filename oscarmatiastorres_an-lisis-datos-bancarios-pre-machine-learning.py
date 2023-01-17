import pandas as pd
import numpy as np
import altair as alt
from sklearn import metrics

alt.data_transformers.disable_max_rows()
# Cargamos los datos del fichero .csv

credit_g = pd.read_csv('../input/datos-credito-bancario-alemania-csv/credit-g.csv')
# Información general del contenido de los datos

credit_g.info()
credit_g.describe()
credit_g.describe(include = 'object')
# Cambiamos las columnas con pocos valores posibles a tipo categórico

columnas_num = ['duration', 'credit_amount', 'age']
columnas_cat = credit_g.select_dtypes(include = 'object').columns.to_list() + \
               ['installment_commitment', 'residence_since', 'existing_credits', 'num_dependents']

for field in columnas_cat:
    credit_g[field] = credit_g[field].astype('category')
credit_g.info()
# Podemos hacer unos boxplot para ver la distribución de valores de las variables continuas para
# los casos de buen crédito y mal crédito.
# Como se puede ver en los gráficos, los prestamos de mayor duración y de mayor cantidad tienen un
# riesgo mayor (¡en este caso!), y cuanto menor edad, más probabilidad de mal crédito

alt.Chart(credit_g).mark_boxplot().encode(y = alt.X('class:N', title = None), 
                                         x = alt.Y(alt.repeat("row"), type='quantitative'),
                                         color = alt.Color('class:N', legend = None))\
                                  .repeat(row=columnas_num)\
                                  .resolve_scale(x = 'independent')
# Para las variables categóricas, una opción es mostrar el porcentaje de mal crédito para cada uno
# de los valores posibles de cada una de esas variables
# Como se ve en los gráficos, no hay ninguna variable individual con gran capacidad predictiva,
# aunque algunas parecen dar buenas pistas

def get_data_chunk(field):
    return credit_g.groupby(field)['class'].apply(lambda x: (x == 'bad').mean())\
                   .rename('percentage bad')\
                   .reset_index()\
                   .rename(columns = {field : 'value'})\
                   .assign(var = field)

mal_credito_por_valor = pd.concat([get_data_chunk(field) for field in columnas_cat if field != 'class'])

alt.Chart(mal_credito_por_valor, title = 'Porcentaje de casos con mal crédito').mark_bar()\
                        .encode( x = 'percentage bad:Q', y = alt.Y('value:N', title = None), row = 'var:N')\
                        .resolve_scale(y = 'independent')
# La variable 'foreign_worker' parece tener bastante capadidad predictiva, pero si hacemos un gráfico para sus distintos
# valores, vemos lo que pasa: es cierto que los casos que no corresponden a 'foreing_worker' son, muy mayoritariamente, de
# buen crédigo, pero es un porcentaje muy pequeño del total de casos.

alt.Chart(credit_g).mark_bar().encode(x = alt.X('class:N', title = None), 
                                         y = 'count(foreign_worker):Q', 
                                         color = alt.Color('class:N', legend = None),
                                         column = 'foreign_worker:O')
# Otras variables paracen más interesantes para predecir la calidad del crédito en muchos casos...

alt.Chart(credit_g).mark_bar().encode(x = alt.X('class:N', title = None), 
                                         y = 'count(checking_status):Q', 
                                         color = alt.Color('class:N', legend = None),
                                         column = 'checking_status:O')
alt.Chart(credit_g).mark_bar().encode(x = alt.X('class:N', title = None), 
                                         y = 'count(credit_history):Q', 
                                         color = alt.Color('class:N', legend = None),
                                         column = 'credit_history:O')
# A partir de las observaciones del apartado anterior, podemos crear diversos modelos predictivos sencillos
# y calcular cómo de bien funcionan. Por ejemplo, un modelo podría consistir en hacer la media de los porcentajes
# de mal crédito que corresponden a los valores de las variables para un caso
def modelo1(caso):
    porcentajes = [mal_credito_por_valor[(mal_credito_por_valor['var'] == field) & 
                                         (mal_credito_por_valor['value'] == caso[field])]['percentage bad'].tolist()[0] \
                   for field in columnas_cat if field != 'class']
    return np.mean(porcentajes)
score1 = credit_g.apply(lambda x: modelo1(x), axis = 1)
# Podemos comprobar si este modelo tiene algún poder predictivo calculando el AUC ROC

from sklearn.metrics import roc_auc_score

roc_auc_score((credit_g['class'] == 'bad'), score1)
# Un modelo más sencillo sería, por ejemplo, utilizar una cascada de criterios específicos
# (que es un método que, sorprendentemente, se utiliza en muchísimas ocasiones)
def modelo2(caso):
    if caso['duration'] >= 50 or caso['checking_status'] in(['\'<0\'', '\'0<=X<200\'']) or caso['credit_history'] in(['\'all paid\'', '\'no credits/all paid\'']):
        return 1
    else:
        return 0
score2 = credit_g.apply(lambda x: modelo2(x), axis = 1)
# Calculamos el poder predictivo de un modelo de este tipo y vemos que es más limitado

roc_auc_score((credit_g['class'] == 'bad'), score2)
# En el caso del modelo2, como el resultado es binario, simplemente se puede hacer una tabla de contingencia

tc_2 = pd.crosstab(score2, credit_g['class'])
tc_2
# Podemos hacer una tabla con el coste de cada tipo de casos, para multiplicarla por la tabla de contigencia...

coste = pd.DataFrame([[5, 0], [0, 1]], columns=tc_2.columns, index=tc_2.index)
coste
# El coste de usar el modelo 2 sería...

(tc_2 * coste).sum().sum()
# Para el modelo 1, como nos da una escala de certeza de la predicción, hay que seleccionar un umbral a partir del cual
# consideraríamos el caso como 'malo'

def tc_1(umbral):
    return pd.crosstab(score1 > umbral, credit_g['class'])

tc_1(0.30)
def coste_1(umbral):
    return (tc_1(umbral) * coste).sum().sum()

coste_1(0.4)
# Podemos generar una gráfica del coste que tendrían los errores de predicción, en función del umbral elegido...

x = np.linspace(0.2, 0.5, 100)
data = pd.DataFrame({
  'umbral': x,
  'coste': np.vectorize(coste_1)(x)
})

alt.Chart(data).mark_line().encode(x = 'umbral:Q', y = 'coste:Q')
# Se aprecia como el mínimo está alrededor de 0.29 y, naturalmente, como el modelo tiene mejor capacidad predictiva,
# el coste por errores de predicción es menor que en el caso del otro modelo.

coste_1(0.29)