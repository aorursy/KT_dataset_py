import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error


plt.rcParams['figure.figsize'] = (16,8)
df = pd.read_csv("../input/delitos.csv", parse_dates=['fecha'])
df
df.info()
df.drop(columns=["lugar", "origen_dato"], inplace=True)
df
# Cantidad de valores 
df.uso_arma.value_counts()
# Cantidad de valores
df.uso_moto.value_counts()
# Cantidad de valores 
df.cantidad_vehiculos.value_counts()
# Cantidad de valores 
df.cantidad_victimas.value_counts()
# Se elimina la feature
df.drop(columns=["cantidad_vehiculos"], inplace=True)
# Contabilizamos los valores nulos
df.barrio.isnull().sum()
df.comuna.isnull().sum()
df[(df['comuna'].isna()) & (df['barrio'].isna())]
# Eliminamos las filas que tienen registros nulos en la comuna, por ende en barrio, latitud y longitud
df.drop(df[df['comuna'].isna()].index)
df.comuna.value_counts()
sns.barplot(x=df.comuna.value_counts().index, y=df.comuna.value_counts())
df.barrio.value_counts()
# Veamos un gráfico del top 10 de barrios con mayor cantidad de crimenes:
sns.barplot(x=df.barrio.value_counts().head(10).index, y=df.barrio.value_counts().head(10))
sns.boxplot(df.barrio.value_counts())
# En porcentaje
df.tipo_delito.value_counts()
# Pie chart
labels = df.tipo_delito.value_counts().index
sizes = df.tipo_delito.value_counts(normalize=True).values.round(6)*100
 
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
plt.rcParams['figure.figsize'] = (14,8)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=30)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()
pd.crosstab(index=df.barrio, columns=df.tipo_delito, margins=True)
plt.rcParams['figure.figsize'] = (16,8)
sns.countplot(y=df['comuna'], hue=df['fecha'].dt.year)
df_comuna_1 = df[df["comuna"]=="Comuna 1"]
plt.rcParams['figure.figsize'] = (16,8)
sns.countplot(y=df_comuna_1['barrio'], hue=df_comuna_1['fecha'].dt.year).set_title("Cantidad anual de delitos en la Comuna 1")
df_comuna_3 = df[df["comuna"]=="Comuna 3"]
plt.rcParams['figure.figsize'] = (16,8)
sns.countplot(y=df_comuna_3['barrio'], hue=df_comuna_3['fecha'].dt.year).set_title("Cantidad anual de delitos en la Comuna 3")
df_comuna_4 = df[df["comuna"]=="Comuna 4"]
plt.rcParams['figure.figsize'] = (16,8)
sns.countplot(y=df_comuna_4['barrio'], hue=df_comuna_4['fecha'].dt.year).set_title("Cantidad anual de delitos en la Comuna 4")
plt.rcParams['figure.figsize'] = (16,8)
sns.countplot(x=df['tipo_delito'], hue=df['fecha'].dt.year).set_title("Distribucion absoluta de los delitos según el año")
df_robos = df[df.tipo_delito == "Robo (Con violencia)"]
df_hurtos = df[(df.tipo_delito == "Hurto (Sin violencia)") | (df.tipo_delito == "Hurto Automotor")]
df_les = df[df.tipo_delito == "Lesiones Seg Vial"]
df_robo_auto = df[df.tipo_delito == "Robo Automotor"]             
df_homicidio = df[(df.tipo_delito == "Homicidio Doloso") | (df.tipo_delito == "Homicidio Seg Vial")]           
# Seteamos las configuraciones generales del gráfico
fig = plt.figure(figsize=(20,6))

# Agrupamos los valores y le agregamos la cantidad de delito (en este caso robos) y gráficamos la línea
df_group_robos = df_robos.groupby(df_robos.fecha).agg({'tipo_delito': 'count'})
sns.lineplot(x=df_group_robos.index, y=df_group_robos['tipo_delito'], label="Robo (Con violencia)").set_title("Comparación de las cantidades de tipo de delitos a través del tiempo")

# Agrupamos los valores y le agregamos la cantidad de delito (en este caso hurtos) y gráficamos la línea
df_group_hurtos = df_hurtos.groupby(df_hurtos.fecha).agg({'tipo_delito': 'count'})
sns.lineplot(x=df_group_hurtos.index, y=df_group_hurtos['tipo_delito'], label="Hurtos")

# Agrupamos los valores y le agregamos la cantidad de delito (en este caso lesiones) y gráficamos la línea
df__group_lesiones = df_les.groupby(df_les.fecha).agg({'tipo_delito': 'count'})
sns.lineplot(x=df__group_lesiones.index, y=df__group_lesiones['tipo_delito'], label="Lesiones Seg Vial")

# Agrupamos los valores y le agregamos la cantidad de delito (en este caso robo automotor) y gráficamos la línea
df_group_roboauto = df_robo_auto.groupby(df_robo_auto.fecha).agg({'tipo_delito': 'count'})
sns.lineplot(x=df_group_roboauto.index, y=df_group_roboauto['tipo_delito'], label="Robo Automotor")

# Agrupamos los valores y le agregamos la cantidad de delito (en este caso homicidios) y gráficamos la línea
df_group_homicidios = df_homicidio.groupby(df_homicidio.fecha).agg({'tipo_delito': 'count'})
sns.lineplot(x=df_group_homicidios.index, y=df_group_homicidios['tipo_delito'], label="Homicidios")

plt.show()
df_2_filtered = df_group_robos.copy()
df_2_filtered

df_2_filtered["mes"] = df_2_filtered.index.get_level_values(0).strftime('%m')
df_2_filtered["anio"] = df_2_filtered.index.get_level_values(0).strftime('%Y')
df_2_filtered_my = df_2_filtered.groupby(["mes","anio"]).agg({"tipo_delito":"sum"}).rename(columns={"tipo_delito":"robos"}).reset_index()
sns.relplot(x="mes", y="robos", col="anio", data=df_2_filtered_my, kind="line")
df_3_filtered = df_group_hurtos.copy()
df_3_filtered

df_3_filtered["mes"] = df_3_filtered.index.get_level_values(0).strftime('%m')
df_3_filtered["anio"] = df_3_filtered.index.get_level_values(0).strftime('%Y')
df_3_filtered_my = df_3_filtered.groupby(["mes","anio"]).agg({"tipo_delito":"sum"}).rename(columns={"tipo_delito":"hurtos"}).reset_index()
sns.relplot(x="mes", y="hurtos", col="anio", data=df_3_filtered_my, kind="line")
sns.pairplot(df)
#retornamos la correlacion del data frame
corr = df.corr() 

# Con esa variable removemos las variables superiores ya que estan repetidas
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)

#Dibujamos el heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='RdYlGn');

fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
# Hacemos una copia del dataframe antes de impactar el cambio directamente
df_engineering = df.copy()
# MUY IMPORTANTE, completar los valores nulos, ya que sin esto, en este caso, no funciona la funcion "lambda"
df_engineering.hora.fillna('00:00:00', inplace=True)

# Armamos los bins (rangos)
bins = [0, 6, 12, 20, 23]
# Esta variable nos va a servir para poder identificar cada rango
names = ['Madrugada', 'Mañana', 'Tarde', 'Noche']

# Asignamos a esta variable la serie con las nuevas columnas en base a los bins y label
# La funcion lambda se encarga de cortar la hora ya que caso contrario no se puede binarizar al ser string
category = pd.cut(df_engineering.hora.map(lambda x: int(x.split(':')[0])), bins, labels = names)
# Concatenamos al data frame a partir de una matriz dummie la variable category
df_engineering = pd.concat([df_engineering, pd.get_dummies(category)], axis=1)
df_engineering
parte_del_dia = df_engineering.loc[:,['Madrugada', 'Mañana', 'Tarde', 'Noche']].sum().sort_values(ascending=False)
sns.barplot(x=parte_del_dia.index, y=parte_del_dia.values)
parte_del_dia_porc = df_engineering.loc[:,['Madrugada', 'Mañana', 'Tarde', 'Noche']].mean()*100
# Pie chart
labels = parte_del_dia_porc.index
sizes = parte_del_dia_porc.values
 
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
plt.rcParams['figure.figsize'] = (12,6)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=30)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()
# Importamos Scickit Learn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse = False)

# Definimos la variable a realizar el onehotencoder
delito = df_engineering.tipo_delito.values.reshape(-1,1)

onehot_encoder.fit(delito)

delito_encoded = onehot_encoder.transform(delito)
# Mostramos las categorias que quedaron
onehot_encoder.categories_
# Lo agregamos al data set
# Primero armamos un data frame pequeño, con lo que queremos unir
delito_encoded_df = pd.DataFrame(delito_encoded, dtype=int,columns=(['Homicidio Doloso','Homicidio Seg Vial', 'Hurto (Sin violencia)','Hurto Automotor', 'Lesiones Seg Vial', 'Robo (Con violencia)','Robo Automotor']))

#Le ponemos el mismo indice que el data frame donde queremos agregarlo
delito_encoded_df = delito_encoded_df.set_index(df_engineering.index)
df_engineering = pd.concat([df_engineering, delito_encoded_df], axis=1)
df_engineering
#retornamos la correlacion del data frame
corr = df_engineering.corr() 

# Con esa variable removemos las variables superiores ya que estan repetidas
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)

#Dibujamos el heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='RdYlGn');

fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
df_ml = df.copy()
df_ml.drop(columns=["id","comuna","fecha","uso_arma","uso_moto","cantidad_victimas"], inplace=True)
df_ml.drop(columns=["latitud","longitud"], inplace=True)
df_ml.info()
df_ml.isnull().sum()
df_ml.dropna(inplace=True)
df_ml.isnull().sum()
hora = df_ml.hora.str.split(":",n=1,expand=True)
df_ml["hora"] = hora[0].astype(int)
from sklearn.preprocessing import LabelEncoder

df_encoder = df_ml.barrio.values

le = LabelEncoder()
le.fit(df_encoder)
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
barrio_encoded = df_ml.barrio.values.reshape(len(df_ml.barrio.values), 1)
#print(integer_encoded)

onehot_encoded_barrio = onehot_encoder.fit_transform(barrio_encoded)
# probar con df_ml.barrio.unique()
df_barrio = pd.DataFrame(onehot_encoded_barrio, dtype=int, columns=le.classes_)
df_barrio = df_barrio.set_index(df_ml.index)

df_ml = pd.concat([df_ml, df_barrio], axis=1)
df_ml
df_ml["tipo_delito_encode"] = df_ml.tipo_delito.map({'Homicidio Doloso': '1','Robo (Con violencia)': '2','Hurto Automotor': '3','Hurto (Sin violencia)': '4','Robo Automotor': '5'
           ,'Homicidio Seg Vial': '6','Lesiones Seg Vial': '7'})
df_ml["tipo_delito_encode"] = df_ml["tipo_delito_encode"].astype(int)
df_ml.info()
#retornamos la correlacion del data frame
corr = df_ml.corr() 

# Con esa variable removemos las variables superiores ya que estan repetidas
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)

#Dibujamos el heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='RdYlGn');

fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
X = df_ml.drop(columns=["barrio", "tipo_delito", "tipo_delito_encode"])
y= df_ml["tipo_delito_encode"].values
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

train_score = []
test_score = []
profundidad = np.arange(1,40,5)

for max_depth in profundidad:
    rfc = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=42)
    rfc.fit(X_train, y_train)
    
    y_pred_train = rfc.predict(X_train)
    y_pred_test = rfc.predict(X_test)
    
    train_score.append(rfc.score(X_train, y_train))
    test_score.append(rfc.score(X_test, y_test))
    
plt.plot(profundidad, train_score, label="Training score")
plt.plot(profundidad, test_score, label="Testing score")
plt.legend()
plt.ylabel('Score')
plt.xlabel('Profundiad')
plt.show()
df_ml_2 = df.copy()
df_ml_2.drop(columns=["id","comuna","hora","uso_arma","uso_moto","cantidad_victimas"], inplace=True)
df_ml_2.isna().sum()
df_ml_2.dropna(inplace=True)
df_ml_2.isna().sum()
df_ml_2["dia_semana"] = df_ml_2["fecha"].dt.dayofweek.values
df_ml_2
from sklearn.preprocessing import LabelEncoder

df_encoder = df_ml.barrio.values

le = LabelEncoder()
le.fit(df_encoder)
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
barrio_encoded = df_ml_2.barrio.values.reshape(len(df_ml_2.barrio.values), 1)
#print(integer_encoded)

onehot_encoded_barrio = onehot_encoder.fit_transform(barrio_encoded)
# probar con df_ml.barrio.unique()
df_barrio = pd.DataFrame(onehot_encoded_barrio, dtype=int, columns=le.classes_)
df_barrio = df_barrio.set_index(df_ml_2.index)

df_ml_2 = pd.concat([df_ml_2, df_barrio], axis=1)

df_ml_2["tipo_delito_encode"] = df_ml_2.tipo_delito.map({'Homicidio Doloso': '1','Robo (Con violencia)': '2','Hurto Automotor': '3','Hurto (Sin violencia)': '4','Robo Automotor': '5'
           ,'Homicidio Seg Vial': '6','Lesiones Seg Vial': '7'})
df_ml_2["tipo_delito_encode"] = df_ml_2["tipo_delito_encode"].astype(int)
#retornamos la correlacion del data frame
corr = df_ml_2.corr() 

# Con esa variable removemos las variables superiores ya que estan repetidas
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)

#Dibujamos el heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='RdYlGn');

fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
X = df_ml_2.drop(columns=["barrio", "fecha", "tipo_delito", "tipo_delito_encode"])
y = df_ml_2.tipo_delito_encode.values
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_score = []
test_score = []
profundidad = np.arange(1,40,5)

for max_depth in profundidad:
    rfc = RandomForestClassifier(max_depth=max_depth, criterion='gini', n_jobs=-1, random_state=42)
    rfc.fit(X_train, y_train)
    
    y_pred_train = rfc.predict(X_train)
    y_pred_test = rfc.predict(X_test)
    
    train_score.append(rfc.score(X_train, y_train))
    test_score.append(rfc.score(X_test, y_test))    
plt.figure(figsize=(12,10))

feat_imp_df = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), X_train.columns), reverse=True))

mapping = {feat_imp_df.columns[0]:'Importancia', feat_imp_df.columns[1]: 'Variable'}
feat_imp_df = feat_imp_df.rename(columns=mapping)
sns.barplot(x=feat_imp_df['Importancia'],y=feat_imp_df['Variable'], palette="Greens_d")

#Otra forma
#n_features = X.shape[1]
#plt.barh(range(n_features),rfc.feature_importances_)
#plt.yticks(np.arange(n_features),train_data.columns[1:])
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_train, y_pred_train)
balanced_accuracy_score(y_test, y_pred_test)
plt.plot(profundidad, train_score, label="Training score")
plt.plot(profundidad, test_score, label="Testing score")
plt.legend()
plt.ylabel('Score')
plt.xlabel('Profundiad')
plt.show()
np.argmax(test_score)
test_score
rdm = RandomForestClassifier(max_depth=np.argmax(test_score)*5+1, n_jobs=-1, n_estimators=42)
rdm.fit(X_train, y_train)

y_pred_train = rdm.predict(X_train)
y_pred_test = rdm.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)
cm
pd.crosstab(y_test, y_pred_test, rownames=['True'], colnames=['Predicted'], margins=True)
labels = ['Homicidio Doloso','Robo (Con violencia)','Hurto Automotor','Hurto (Sin violencia)','Robo Automotor','Homicidio Seg Vial','Lesiones Seg Vial']
sns.heatmap(cm, annot=True, xticklabels= labels, yticklabels= labels, fmt='.2f')
plt.show()
print (classification_report(y_test, y_pred_test))
from sklearn.model_selection import cross_validate
tree_train_scores_mean = []
tree_train_scores_std = []
tree_test_scores_mean = []
tree_test_scores_std = []

profundidades = np.arange(1,40,5)

for profundidad in profundidades:
    rdm = RandomForestClassifier(max_depth=profundidad, criterion='gini', random_state=42, n_jobs=-1)
    cv_result = cross_validate(rdm, X_train, y_train, cv=10, return_train_score=True)
    
    tree_train_scores_mean.append(cv_result['train_score'].mean())
    tree_train_scores_std.append(cv_result['train_score'].std())
    
    tree_test_scores_mean.append(cv_result['test_score'].mean())
    tree_test_scores_std.append(cv_result['test_score'].std())
    
tree_train_scores_mean = np.array(tree_train_scores_mean)
tree_train_scores_std = np.array(tree_train_scores_std)
tree_test_scores_mean = np.array(tree_test_scores_mean)
tree_test_scores_std = np.array(tree_test_scores_std)
plt.fill_between(profundidades, tree_train_scores_mean - tree_train_scores_std,
                 tree_train_scores_mean + tree_train_scores_std, alpha=0.1,
                 color="r")

plt.fill_between(profundidades, tree_test_scores_mean - tree_test_scores_std,
                 tree_test_scores_mean + tree_test_scores_std, alpha=0.1, color="g")

plt.plot(profundidades, tree_train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(profundidades, tree_test_scores_mean, 'o-', color="g",
         label="Test score")


plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Profundidad Arbol de Decision')
plt.show()
cv_result
#Hacemos una copia del data frame utilizado en la clasificación anterior ya que nos sirven casi todas sus features.
df_ml_3 = df_ml_2.copy()
df_ml_3.tail(3)
dia_semana = pd.get_dummies(df_ml_3["dia_semana"], prefix="dia_semana")
df_ml_3 = pd.concat([df_ml_3, dia_semana], axis=1)
#retornamos la correlacion del data frame
corr = df_ml_3.corr() 

# Con esa variable removemos las variables superiores ya que estan repetidas
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)

#Dibujamos el heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='RdYlGn');

fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
X = df_ml_3.drop(columns=["barrio", "fecha", "tipo_delito", "tipo_delito_encode", "dia_semana"])
y = df_ml_3.tipo_delito_encode.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)
train_score = []
test_score = []
profundidad = np.arange(1,40,5)

for max_depth in profundidad:
    rfc = RandomForestClassifier(max_depth=max_depth, criterion='gini', n_jobs=-1, random_state=42)
    rfc.fit(X_train, y_train)
    
    train_score.append(rfc.score(X_train, y_train))
    test_score.append(rfc.score(X_test, y_test))    
plt.figure(figsize=(12,10))

feat_imp_df = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), X_train.columns), reverse=True))

mapping = {feat_imp_df.columns[0]:'Importancia', feat_imp_df.columns[1]: 'Variable'}
feat_imp_df = feat_imp_df.rename(columns=mapping)
sns.barplot(x=feat_imp_df['Importancia'],y=feat_imp_df['Variable'], palette="Greens_d")

#Otra forma
#n_features = X.shape[1]
#plt.barh(range(n_features),rfc.feature_importances_)
#plt.yticks(np.arange(n_features),train_data.columns[1:])
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_train, y_pred_train)
balanced_accuracy_score(y_test, y_pred_test)
plt.plot(profundidad, train_score, label="Training score")
plt.plot(profundidad, test_score, label="Testing score")
plt.legend()
plt.ylabel('Score')
plt.xlabel('Profundiad')
plt.show()
test_score
