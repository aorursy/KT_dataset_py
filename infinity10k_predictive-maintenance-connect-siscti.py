# Creación de gráficos.

import matplotlib.pyplot as plt

import seaborn as sns

import scikitplot as skplt

# Manipulación de datos / álgebra lineal.

import numpy as np

import pandas as pd

# Utilidades.

from sklearn.preprocessing import scale

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

# Algoritmos

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier as xgb

# Otros

import warnings
# Algunas configuraciones.

%matplotlib inline

sns.set_style("darkgrid")

warnings.filterwarnings("ignore")

plt.rc("font", family="serif", size=15)
telemetry = pd.read_csv("../input/PdM_telemetry.csv", error_bad_lines=False)

errors = pd.read_csv("../input/PdM_errors.csv", error_bad_lines=False)

maint = pd.read_csv("../input/PdM_maint.csv", error_bad_lines=False)

failures = pd.read_csv("../input/PdM_failures.csv", error_bad_lines=False)

machines = pd.read_csv("../input/PdM_machines.csv", error_bad_lines=False)
telemetry.head()
telemetry.tail()
telemetry.info()
telemetry.dtypes
telemetry["machineID"].nunique()
# Cambiamos el formato de datetime ya que viene como string.

telemetry["datetime"] = pd.to_datetime(telemetry["datetime"], format="%Y-%m-%d %H:%M:%S")

telemetry.head()
# el dtype de esta serie es objeto porque tiene tipos mixtos

telemetry.dtypes
# Kernel Density function -> PDF (Función de Densidad de Probabilidad)



"""

Una distribución normal nos habla de una muestra representativa (Teorema del Limite Central).

Nos habla de un comportamiento natural.

Muestra un único grupo para trabajar (de lo contrario hacer clustering).

Los modelos paramétricos esperan distribuciones normales porque utilizan la media, la std, etc.

Los modelos no paramétricos e.g. CART (Classification and Regression Trees),

son insensibles respecto a la distribución de los datos,

lo único que importa es crear nodos que maximicen la separación de clases.

En otros casos, aplicar a los datos transformaciones logarítmicas, regresión cuantílica, normalización.

Paramétrico: Regresión Lineal

No-paramétrico: Árbol de Decisión

¿Hay o no cambios a medida que el dataset crece?

"""



telemetry["pressure"].plot(kind="kde")

plt.title("Distribución de la Presión")

plt.ylabel("Densidad")

plt.xlabel("Presión")

plt.show()
# Confirmamos integridad; totales, promedio, dsviasión estándar, mínimo, máximo, y cuantiles.

telemetry.describe()
# Mostramos un gráfico de ejemplo de los valores de voltaje para la máquina 1 durante los primeros 6 meses del 2015.



plot_df = telemetry.loc[

    (telemetry["machineID"] == 1)

    & (telemetry["datetime"] > pd.to_datetime("2015-01-01"))

    & (telemetry["datetime"] < pd.to_datetime("2015-06-01")), ["datetime", "volt"]

]



plt.figure(figsize=(12, 6))

plt.plot(plot_df["datetime"], plot_df["volt"])

plt.title("Variación del Voltaje en la Máquina 1")

plt.ylabel("Voltaje")



# Hacemos legibles las etiquetas.

adf = plt.gca().get_xaxis().get_major_formatter()

adf.scaled[1.0] = "%m-%d"

plt.xlabel("Tiempo")

plt.show()
errors.head()
errors.tail()
errors.info()
# Formateo del campo de fecha y hora que viene como string.

# Las categorías permiten la comparación entre valores, ordenamiento automático, graficado más sencillo y otras funciones.

# También menos memoria (similar a "factor" en R).

errors["datetime"] = pd.to_datetime(errors["datetime"], format="%Y-%m-%d %H:%M:%S")

errors["errorID"] = errors["errorID"].astype("category")



errors.head()
sns.set_style("darkgrid")

plt.figure(figsize=(8, 4))

errors["errorID"].value_counts().plot(kind="bar", rot=0)

plt.title("Distribución de los Tipos de Error")

plt.ylabel("Count")

plt.xlabel("Error Type")

plt.show()
maint.head()
maint.tail()
maint.info()
# Formateo del campo de fecha y hora que viene como string.

maint["datetime"] = pd.to_datetime(maint["datetime"], format="%Y-%m-%d %H:%M:%S")

maint["comp"] = maint["comp"].astype("category")

maint.dtypes
sns.set_style("darkgrid")

plt.figure(figsize=(8, 4))

maint["comp"].value_counts().plot(kind="bar", rot=0)

plt.title("Distribución de Reemplazos de Componentes")

plt.ylabel("Cantidad")

plt.xlabel("Componente")

plt.show()
machines.head()
machines.tail()
machines.shape
machines.dtypes
# Revisamos si existen varias colinas ya que puede sugerir dos grupos diferentes.

machines["age"].plot(kind="kde")

plt.title("Distribución de Edades de Máquinas")

plt.xlabel("Edad")

plt.ylabel("Densidad")

plt.show()
# Aplicamos logaritmo natural para normalizar.

np.log(machines[machines["age"] != 0].iloc[:, 0]).plot(kind="kde")

plt.show()
machines["model"] = machines["model"].astype("category")

machines.dtypes
plt.figure(figsize=(8, 6))

_, bins, _ = plt.hist([

    machines.loc[machines["model"] == "model1", "age"],

    machines.loc[machines["model"] == "model2", "age"],

    machines.loc[machines["model"] == "model3", "age"],

    machines.loc[machines["model"] == "model4", "age"]],

    20, stacked=True, label=["model1", "model2", "model3", "model4"

])

plt.title("Distribución de Edades por Modelo")

plt.xlabel("Edad (años)")

plt.ylabel("Cantidad")

plt.legend()

plt.show()
failures.head()
failures.tail()
failures.info()
# Formateamos el datetime que viene como string

failures["datetime"] = pd.to_datetime(failures["datetime"], format="%Y-%m-%d %H:%M:%S")

failures["failure"] = failures["failure"].astype("category")

failures.dtypes
failures.describe(include="all")
plt.figure(figsize=(8, 4))

failures["failure"].value_counts().plot(kind="bar", rot=0)

plt.title("Distribución de Fallas de Componentes")

plt.ylabel("Cantidad")

plt.xlabel("Componentes")

plt.show()
# Calculamos valores promedio para características de telemetría



temp = []

fields = ["volt", "rotate", "pressure", "vibration"]



# pivotamos porque necesitamos el datetime como índice para para que "resample" funcione

# resample crea el lagging

# closed = 'right' => (6:00, 9:00] o 6:00 < x <= 9:00

# closed='left'  => [6:00, 9:00) o 6:00 <= x < 9:00

# no pueden ser ambos

# unstack: devuelve df al formato original

# tenemos 100 máquinas, 4 sensores = 400 columnas

# unstack muestra un dataseries el las columnas como índice y regresa serie (si hay varios índices se reacomodan).

# Cada dataframe en temp tiene los valores del campo que le corresponde en ese momento.



temp = [

    pd.pivot_table(

        telemetry,

        index="datetime",

        columns="machineID",

        values=col).resample("3H", closed="left", label="right").mean().unstack()

    for col in fields

]

temp[0].head()
telemetry_mean_3h = pd.concat(temp, axis=1) # Unimos las series.

telemetry_mean_3h.columns = [col + "mean_3h" for col in fields] # Asignamos nombres de columnas.

telemetry_mean_3h.reset_index(inplace=True) # Aplanamos el frame.

telemetry_mean_3h.head()
# Repetimos para la desviación estándar.

temp = [

    pd.pivot_table(

        telemetry,

        index="datetime",

        columns="machineID",

        values=col).resample("3H", closed="left", label="right").std().unstack()

    for col in fields

]

temp[0].head()
telemetry_sd_3h = pd.concat(temp, axis=1)

telemetry_sd_3h.columns = [i + "sd_3h" for i in fields]

telemetry_sd_3h.reset_index(inplace=True)

telemetry_sd_3h.head()
# Para capturar un efecto a mayor plazo, las funciones de lagging de 24 horas también se calculan.

# Creamos nuevos valores con promedios de 24 horas, y luego seleccionamos el primer resultado cada 3 horas.

# De esta manera podremos unir los resultados con las características de lagging anteriores (calculadas a 3 horas).



temp = []

fields = ["volt", "rotate", "pressure", "vibration"]



temp = [

    pd.pivot_table(

        telemetry,

        index="datetime",

        columns="machineID",

        values=col).rolling(window=24).mean().resample("3H", closed="left", label="right").first().unstack()

    for col in fields

]

temp[0].head()
telemetry_mean_24h = pd.concat(temp, axis=1)

telemetry_mean_24h.columns = [i + "mean_24h" for i in fields]

telemetry_mean_24h.reset_index(inplace=True)

# Debido al método de la media móvil, los primeros 23 registros son nulos; hay que eliminarlos.

# No ocurre al final del frame porque rolling topa al final.

# Terminamos con un frame de menos datos que el original de telemetría así como el anterior de 3H.

telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h["voltmean_24h"].isnull()]
telemetry_mean_24h.head()
# Repetimos para la desviación estándar

temp = []

fields = ["volt", "rotate", "pressure", "vibration"]



temp = [

    pd.pivot_table(

        telemetry,

        index="datetime",

        columns="machineID",

        values=col).rolling(window=24).std().resample("3H", closed="left", label="right").first().unstack(level=-1)

    for col in fields

]

temp[0].head()
telemetry_sd_24h = pd.concat(temp, axis=1)

telemetry_sd_24h.columns = [i + "sd_24h" for i in fields]

telemetry_sd_24h.reset_index(inplace=True)

telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h["voltsd_24h"].isnull()]
telemetry_sd_24h.head(10)
# Combinamos las características creadas hasta ahora.

# Tomamos los valores 2:6 para evitar ID y fechas duplicadas.

# axis=0 nos movemos en dirección de las filas, axis=1, nos movemos en dirección de las columnas.

telemetry_feat = pd.concat([

    telemetry_mean_3h,

    telemetry_sd_3h.iloc[:, 2:6],

    telemetry_mean_24h.iloc[:, 2:6],

    telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()

telemetry_feat.head()
telemetry_feat.describe()
"""

Comenzamos por reformatear los datos de error para tener una entrada por máquina por tiempo

cuando ocurrió al menos un error.

Creamos una columna para cada tipo de error.

"""

error_count = pd.get_dummies(errors) # Ponemos un 1 si el error aparece para esa máquina, 0 de lo contrario.

error_count.columns = ["datetime", "machineID", "error1", "error2", "error3", "error4", "error5"]

error_count.head(15)
"""

Las fechas del dataframe se repiten, así que agrupamos por fecha.

Combinamos errores para una máquina dada en una hora específica.

Hacemos suma en caso de que existan múltiples erroes del mismo tipo al mismo tiempo, pero no esperado.

"""

error_count_grouped = error_count.groupby(["machineID", "datetime"]).sum().reset_index()

error_count_grouped.head(15)
"""

Revisamos que los errores registrados existan en las máquinas disponibles llenano con 0

las no coincidencias por eso solo buscamos coincidencia con datetime y machineID.

"""

error_count_filtered = telemetry[["datetime", "machineID"]].merge(

    error_count_grouped,

    on=["machineID", "datetime"],

    how="left"

).fillna(0.0)



error_count_filtered.head()
# Revisamos que no existan anomalías.

error_count_filtered.describe()
# Calculamos la cantidad total de errores para cada tipo de error durante lapsos de 24 horas. 

# Tomaremos puntos cada 3 horas.



temp = []

fields = [

    "error%d" % i

    for i in range(1,6)

]



temp = [

    pd.pivot_table(

        error_count_filtered,

        index="datetime",

        columns="machineID",

        values=col).rolling(window=24).sum().resample("3H", closed="left", label="right").first().unstack()

    for col in fields

]

temp[0].head(10)
error_count_total = pd.concat(temp, axis=1)

error_count_total.columns = [i + "count" for i in fields]

error_count_total.reset_index(inplace=True)

error_count_total = error_count_total.dropna()

error_count_total.head()
error_count_total["error5count"].unique()
error_count_total.describe()
maint.head()
# creamos una columna para cada tipo de error

comp_rep = pd.get_dummies(maint)

comp_rep.columns = ["datetime", "machineID", "comp1", "comp2", "comp3", "comp4"]

comp_rep.head()
# Combinamos reparaciones para una cierta máquina en cierto momento.

# Si no agrupamos por fecha podemos ver otra perspectiva.

# Encontramos qué componenetes fallan juntos, ya que agrupamos por fecha.

comp_rep = comp_rep.groupby(["machineID", "datetime"]).sum().reset_index()

comp_rep.head()
# hay que agregar los timepos donde no hubo reemplazos

comp_rep = telemetry[["datetime", "machineID"]].merge(

    comp_rep,

    on=["datetime", "machineID"],

    how="outer").fillna(0).sort_values(by=["machineID", "datetime"]

)

comp_rep.head()
components = ["comp1", "comp2", "comp3", "comp4"]

for comp in components:

    # Queremos obtener la fecha del cambio del componente más reciente.

    comp_rep.loc[comp_rep[comp] < 1, comp] = None # Llenamos con nulo las muestras sin reemplazo.

    # las fechas de las entradas que sí tienen reemplazos.

    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), "datetime"]

    # Hacemos un forward-fill de las fechas más recientes de un cambio de componente.

    # Llenamos con el último valor válido encontrado top-bottom.

    comp_rep[comp] = pd.to_datetime(comp_rep[comp].fillna(method="ffill"))



# eliminamos muestras del 2014, podrían tener nulos, los manenimientos comenzaron ese año.

comp_rep = comp_rep.loc[comp_rep["datetime"] > pd.to_datetime("2015-01-01")]

comp_rep.head(50)
# Reemplazamos las fechas más recientes de cambios por la cantidad de días desde el cambio más reciente.

for comp in components: comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D")

comp_rep.head()
comp_rep.describe()
# Finalmente unimos todas las características creadas.

final_feat = telemetry_feat.merge(error_count_total, on=["datetime", "machineID"], how="left")

final_feat = final_feat.merge(comp_rep, on=["datetime", "machineID"], how="left")

final_feat = final_feat.merge(machines, on=["machineID"], how="left")

final_feat.head()
final_feat.describe()
final_feat.head()
"""

Le estamos diciendo al modelo que cualquier valor similar a los que se encuentran dentro de la ventana de 24 horas

es una falla de ese componente, por eso que las máquinas se repiten.

Usamos limit=7 porque tenemos separaciones de 3 horas; 8 * 3 = las 24 horas

pero tenemos en cuenta el primer valor no nulo, por lo que es 7.

"""

labeled_features = final_feat.merge(failures, on=["datetime", "machineID"], how="left")

# Aplicamos un backward-fill de hasta 24h.

# fillna no funciona con tipos categóricos por el momento (¿cómo encajaría la categoría nueva? tal vez).

# Pasamos a object o string, aplicamos la operación, y regresamos a categoría.

labeled_features["failure"] = labeled_features["failure"].astype(object).fillna(method="bfill", limit=7)

labeled_features["failure"] = labeled_features["failure"].fillna("none")

labeled_features["failure"] = labeled_features["failure"].astype("category")

labeled_features.head()
model_dummies = pd.get_dummies(labeled_features["model"])

labeled_features = pd.concat([labeled_features, model_dummies], axis=1)

labeled_features.drop("model", axis=1, inplace=True)
labeled_features.head()
### Análisis de Correlación
# Es necesario eliminar las variables con alta correlación (sólo una), considerar > 70%.

f, ax = plt.subplots(figsize=(10, 8))

corr = labeled_features.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.title("Correlación Entre Variables")

plt.show()
# Guardamos para aplicar optimización de hiper-parámetros.

#labeled_features.to_pickle("final_datset.pickle")
# Establecemos los tiempos correspondientes a los registros que se utilizarán para entrenamiento y pruebas.

threshold_dates = [

    pd.to_datetime("2015-09-30 01:00:00"), pd.to_datetime("2015-10-01 01:00:00")

]
test_results = []

models = []

total = len(threshold_dates)



# Hacemos la partición de fechas separadas.

last_train_date = threshold_dates[0]

first_test_date = threshold_dates[1]
# Típicamente se utiliza entre el 20 y el 30% de los datos.

ntraining = labeled_features.loc[labeled_features["datetime"] < last_train_date]

ntesting = labeled_features.loc[labeled_features["datetime"] > first_test_date]

print(f"{ntraining.shape[0]} registros para entrenamiento.")

print(f"{ntesting.shape[0]} registros para pruebas.")

print(f"{ntesting.shape[0] / ntraining.shape[0] * 100:0.1f}% de los datos se usarán para pruebas.")
fails_train = ntraining[ntraining["failure"] != "none"].shape[0]

no_fails_train = ntraining[ntraining["failure"] == "none"].shape[0]

fails_test = ntesting[ntesting["failure"] != "none"].shape[0]

no_fails_test = ntesting[ntesting["failure"] == "none"].shape[0]



print(f"{fails_train / no_fails_train * 100:0.1f}% de los casos son fallas en set de entrenamiento.")

print(f"{fails_test / no_fails_test * 100:0.1f}% de los casos son fallas en set de pruebas.")
# Asignamos los valores correspondientes a entrenamiento y pruebas.

train_y = labeled_features.loc[labeled_features["datetime"] < last_train_date, "failure"]

train_X = labeled_features.loc[labeled_features["datetime"] < last_train_date].drop(["datetime",

                                                                                    "machineID",

                                                                                    "failure"], axis=1)

test_y = labeled_features.loc[labeled_features["datetime"] > first_test_date, "failure"]

test_X = labeled_features.loc[labeled_features["datetime"] > first_test_date].drop(["datetime",

                                                                                   "machineID",

                                                                                   "failure"], axis=1)
# %%timeit

# Entrenamiento del modelo.

# model = GradientBoostingClassifier(random_state=42)

model = xgb(n_jobs=-1)

model.fit(train_X, train_y)
# Obtenemos resultados sobre el set de pruebas.

test_result = pd.DataFrame(labeled_features.loc[labeled_features["datetime"] > first_test_date])

test_result["predicted_failure"] = model.predict(test_X)

test_results.append(test_result)

models.append(model)
# Below, we plot the feature importances in the (first) trained model

plt.figure(figsize=(10, 10))

labels, importances = zip(*sorted(zip(test_X.columns, models[0].feature_importances_), reverse=False, key=lambda x: x[1]))

plt.yticks(range(len(labels)), labels)

_, labels = plt.xticks()

plt.setp(labels, rotation=0)

plt.barh(range(len(importances)), importances)

plt.ylabel("Признаки")

plt.xlabel("Значимость (%)")

plt.title("Значимость признаков")

plt.show()
# Hay un desbalance esperado.

plt.figure(figsize=(8, 4))

labeled_features["failure"].value_counts().plot(kind="bar", rot=0)

plt.title("Distribución de Causas de Fallos")

plt.xlabel("Componente")

plt.ylabel("Cantidad")

plt.show()
def Evaluate(predicted, actual, labels):

    output_labels = []

    output = []

    

    # Calculate and display confusion matrix

    cm = confusion_matrix(actual, predicted, labels=labels)

    #print("Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels")

    #print(cm)

    

    # Calculate precision, recall, and F1 score

    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))

    precision = precision_score(actual, predicted, average=None, labels=labels)

    recall = recall_score(actual, predicted, average=None, labels=labels)

    f1 = 2 * precision * recall / (precision + recall)

    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])

    output_labels.extend(["accuracy", "precision", "recall", "F1"])

    

    # Calculate the macro versions of these metrics

    output.extend([[np.mean(precision)] * len(labels),

                   [np.mean(recall)] * len(labels),

                   [np.mean(f1)] * len(labels)])

    output_labels.extend(["macro precision", "macro recall", "macro F1"])

    

    # Find the one-vs.-all confusion matrix

    cm_row_sums = cm.sum(axis = 1)

    cm_col_sums = cm.sum(axis = 0)

    s = np.zeros((2, 2))

    for i in range(len(labels)):

        v = np.array([[cm[i, i],

                       cm_row_sums[i] - cm[i, i]],

                      [cm_col_sums[i] - cm[i, i],

                       np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])

        s += v

    s_row_sums = s.sum(axis = 1)

    

    # Add average accuracy and micro-averaged  precision/recall/F1

    avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)

    micro_prf = [float(s[0,0]) / s_row_sums[0]] * len(labels)

    output.extend([avg_accuracy, micro_prf])

    output_labels.extend(["average accuracy",

                          "micro-averaged precision/recall/F1"])

    

    # Compute metrics for the majority classifier

    mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]

    cm_row_dist = cm_row_sums / float(np.sum(cm))

    mc_accuracy = 0 * cm_row_dist; mc_accuracy[mc_index] = cm_row_dist[mc_index]

    mc_recall = 0 * cm_row_dist; mc_recall[mc_index] = 1

    mc_precision = 0 * cm_row_dist

    mc_precision[mc_index] = cm_row_dist[mc_index]

    mc_F1 = 0 * cm_row_dist;

    mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)

    output.extend([mc_accuracy.tolist(), mc_recall.tolist(),

                   mc_precision.tolist(), mc_F1.tolist()])

    output_labels.extend(["majority class accuracy", "majority class recall",

                          "majority class precision", "majority class F1"])

        

    # Random accuracy and kappa

    cm_col_dist = cm_col_sums / float(np.sum(cm))

    exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))

    kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)

    output.extend([exp_accuracy.tolist(), kappa.tolist()])

    output_labels.extend(["expected accuracy", "kappa"])

    



    # Random guess

    rg_accuracy = np.ones(len(labels)) / float(len(labels))

    rg_precision = cm_row_dist

    rg_recall = np.ones(len(labels)) / float(len(labels))

    rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)

    output.extend([rg_accuracy.tolist(), rg_precision.tolist(),

                   rg_recall.tolist(), rg_F1.tolist()])

    output_labels.extend(["random guess accuracy", "random guess precision",

                          "random guess recall", "random guess F1"])

    

    # Random weighted guess

    rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist**2)

    rwg_precision = cm_row_dist

    rwg_recall = cm_row_dist

    rwg_F1 = cm_row_dist

    output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),

                   rwg_recall.tolist(), rwg_F1.tolist()])

    output_labels.extend(["random weighted guess accuracy",

                          "random weighted guess precision",

                          "random weighted guess recall",

                          "random weighted guess F1"])



    output_df = pd.DataFrame(output, columns=labels)

    output_df.index = output_labels

                  

    return output_df
evaluation_results = []

test_result = test_results[0]

evaluation_result = Evaluate(actual = test_result["failure"],

                             predicted = test_result["predicted_failure"],

                             labels = ["none", "comp1", "comp2", "comp3", "comp4"])

skplt.metrics.plot_confusion_matrix(

    test_result["failure"],

    test_result["predicted_failure"],

    normalize=False,

    title="Матрица ошибок"

)

plt.ylabel('Истинные классы', fontsize=14)

plt.xlabel('Предсказанные классы', fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title("Матрица ошибок", fontsize=15)



skplt.metrics.plot_confusion_matrix(

    test_result["failure"],

    test_result["predicted_failure"],

    normalize=True,

)

plt.ylabel('Истинные классы', fontsize=14)

plt.xlabel('Предсказанные классы', fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title("Матрица ошибок (нормированная)", fontsize=15)



plt.show()



evaluation_results.append(evaluation_result)

evaluation_results[0]  # show full results for first split only
evaluation_results[0].mean(axis=1)[0:4]
# Para un problema de clasificación binaria por lo general se utiliza la curva ROC-AUC.

# Para este caso multi-clase utilizaremos precisión vs sensivilidad.

skplt.metrics.plot_precision_recall_curve(

    test_y,

    model.predict_proba(test_X),

    title="ROC - кривая",

    figsize=(10,10)

)

plt.show()
evaluation_results[0].loc["recall"].values
recall_df = pd.DataFrame([evaluation_results[0].loc["recall"].values],

                         columns=["none", "comp1", "comp2", "comp3", "comp4"],

                         index=["Sensibilidad por Componente"])

recall_df.T
test_values = train_X.iloc[0].values

test_values
# XGBoost acepta únicamente matrices de 2 dimensiones.

single_test = pd.DataFrame([test_values], columns=test_X.columns, index=[0])

single_test
probas = model.predict_proba(single_test)

prediction = model.predict(single_test)

ordered_classes = np.unique(np.array(test_y))
gr_test = pd.DataFrame(test_X.values, columns=test_X.columns)



probas = model.predict_proba(gr_test)

prediction = model.predict(gr_test)

ordered_classes = np.unique(np.array(test_y))
results = pd.DataFrame(probas,

                       columns=ordered_classes)

print(f"Predicción: {prediction}")

results
np.unique(prediction, return_counts = True)
for i, j in zip(prediction, range(len(prediction))):

    if i != 'none': print(j)
for i, j in zip(prediction, range(len(prediction))):

    if i != 'none':

        print(prediction[j], 1-results.none[j])
test_X.head()