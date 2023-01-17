#                                      CONSTANTES
# =========================================================================================

#Semilla:
SEED = 333

# Validacion Cruzada Stratificada(n_splits=5):
from sklearn.model_selection import StratifiedKFold
CV = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_variants = pd.read_csv("../input/data-c/training_variants",index_col='ID', engine='python')
df_variants.sample(10, random_state=SEED)
# Breve descripcion de las variables 
print("Hay un total de ",df_variants.shape[0], "casos,")
print(len(df_variants.Gene.unique()), "valores posbiles de genes,")
print(len(df_variants.Variation.unique()), "valores posibles de variaciones y ")
print(len(df_variants.Class.unique()),  "clases")
# Eliminamos Variation pues no aporta mucha informacion (hay casi un valor para cada caso)
df_variants = df_variants.drop(["Variation"], axis=1)
# Distribucion de cada clase
import plotly.express as px
fig = px.pie(df_variants,names='Class',title='Distribucion de cada Clase')
fig.update_traces(textposition='inside', textinfo='value+percent+text')
fig.show()
# Habilita que se pueda graficar directamente desde el dataframe
import cufflinks as cf

# Agrupa por clase y tipo de gen (descartando el resto de atributos)
grouped_df = df_variants.groupby(["Class", "Gene"]).Class.agg("count")

# Normaliza para obtener la frecuencia relativa
grouped_df /= grouped_df.groupby(level=0).sum(axis=0)

# Desapila para obtener un formato que sea fácil de plotear
grouped_df = grouped_df.unstack(level=1).fillna(0)

grouped_df.iplot(kind="bar", barmode="stack", asFigure=True,title="Frecuencia Relativa de Genes agrupados por Clase")
# Separacion variables predictivas 'X' y la clase 'y'
X = df_variants.drop(columns=['Class'])
y = df_variants['Class']
from sklearn.preprocessing import LabelEncoder
# Numerizamos 
enc = LabelEncoder()
X = pd.DataFrame(enc.fit_transform(X))
X.sample(5,random_state = SEED)
# Para crear pipelines de manera sencilla
from imblearn.pipeline import make_pipeline as make_pipeline

# Seleccion de Variables y Estimador
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

# Binarizado
from sklearn.preprocessing import OneHotEncoder

# Aumento
from imblearn.combine import SMOTETomek

# Transformadores
from sklearn.compose import make_column_transformer,make_column_selector

# Estimador para la seleccion de variables
from sklearn.tree import DecisionTreeClassifier

# Pipeline
def create_pipeline(estimator,binarizado = False,seleccion = False,aumento = False,cv=CV):
    if binarizado:
        # Binarizado-numerizado. Ignoramos las categorias que quedaron en el conjunto de validacion y no se pudieron reconocer 
        one_hot = OneHotEncoder(sparse=False, handle_unknown='ignore')
        if seleccion:
            # Selector de variables con arbol de decision por ser de los mas rapidos y basados en ganancias de informacion
            feature_selection = RFECV(estimator=DecisionTreeClassifier(random_state=SEED),cv=cv)
            if aumento:
                pipeline = make_pipeline(one_hot,feature_selection,SMOTETomek(random_state=SEED),estimator)
            else:
                pipeline = make_pipeline(one_hot,feature_selection,estimator)
        else:
            if aumento:
                pipeline = make_pipeline(one_hot,SMOTETomek(random_state=SEED),estimator)
            else:
                pipeline = make_pipeline(one_hot,estimator)
    else:
        if aumento:
             pipeline = make_pipeline(SMOTETomek(random_state=SEED),estimator)
        else:
             pipeline = make_pipeline(estimator)
    return pipeline
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML
import base64

# Generacion del GridSearch
def generate_gs(pipe,grid_params,cv_g = CV,scoring =["neg_log_loss","accuracy","f1_macro","roc_auc_ovr"]):
    gs = GridSearchCV(
        pipe,
        grid_params,
        verbose=1,
        cv =cv_g,
        n_jobs = -1,
        scoring = scoring,
        refit = "neg_log_loss"
    )
    return gs

## Columnas de guardado para los algortimos 
COLUMNS = ['mean_fit_time','std_fit_time',
           'mean_test_neg_log_loss','std_test_neg_log_loss','rank_test_neg_log_loss',
           'mean_test_accuracy','std_test_accuracy','rank_test_accuracy',
           'mean_test_f1_macro','std_test_f1_macro','rank_test_f1_macro',
           'mean_test_roc_auc_ovr','std_test_roc_auc_ovr','rank_test_roc_auc_ovr']

# Funcion de guardado de resultados que es un subconjunto de cv_results. 
# Guarda los resultados de los parametros del algoritmo y las metricas que le pasamos como parametro.
def save_results(gs,params_to_evaluate,columns=COLUMNS):
    aux = pd.DataFrame(gs.cv_results_)
    gs_res = pd.DataFrame()
    for col in params_to_evaluate:
        gs_res[col] = aux[col]
    for col in columns:
        gs_res[col] = aux[col]
    return gs_res

# Funcion para la representacion grafica de resultados
def plot_scores(param_to_evaluate,scores,df_results,postive=1):
    scores_title = ""
    fig = go.Figure()
    for score in scores:
        fig.add_trace(go.Scatter(x = df_results[param_to_evaluate], y = postive*df_results[score], name=score,
                        line_shape='linear'))
        scores_title = scores_title + score + ';'
    # Edit the layout
    fig.update_layout(title=' Scores: '+ scores_title, xaxis_title = param_to_evaluate)
    fig.show()

# Funcion para exportar el dataframe a un link
# Solo funciona si el archivo es menor que 2MB
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
# Creamos nuestro clasificador en su forma primitiva
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

#Creamos un diccionario con los parametros de interes de nuestro estimador y los valores posibles de estos
knn_params = {
    # Consideramos los vecinos mas cercanos hasta sqrt(3321)= 57.628 (redondeamos a 60)
    'kneighborsclassifier__n_neighbors': tuple(range(1,60,2))
}

# Creamos el pipeline con las opciones de preprocesado a evaluar
pipe_knn = create_pipeline(knn,binarizado = True)

# Creamos el GSCV
gs_knn = generate_gs(pipe_knn,knn_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 143.5 min - 300 ; 207.3 - 442; 
gs_knn.fit(X,y)
# Guardamos los parametros del clasificador que nos interesa evaluar
params_to_evaluate_knn = ['param_kneighborsclassifier__n_neighbors']

# Extraemos de cv_results los resultados a mostrar
knn_gs_res = save_results(gs_knn,params_to_evaluate_knn)

# Mostramos los 10 mejores resultados
knn_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
# Exportamos los resultados
create_download_link(knn_gs_res,filename='knn_bin.csv')
# Creamos el pipeline con las opciones de preprocesado a evaluar
pipe_knn = create_pipeline(knn,binarizado = True,seleccion = True)

# Creamos el GSCV
gs_knn = generate_gs(pipe_knn,knn_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 143.5 min - 300 ; 207.3 - 442; 
gs_knn.fit(X,y)
knn_gs_res = save_results(gs_knn,params_to_evaluate_knn)
knn_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(knn_gs_res,filename='knn_bin+sel.csv')
pipe_knn = create_pipeline(knn,binarizado = True,aumento = True)
gs_knn = generate_gs(pipe_knn,knn_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 143.5 min - 300 ; 207.3 - 442; 
gs_knn.fit(X,y)
knn_gs_res = save_results(gs_knn,params_to_evaluate_knn)
knn_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(knn_gs_res,filename='knn_bin+aum.csv')
pipe_knn = create_pipeline(knn,binarizado = True,seleccion = True,aumento = True)
gs_knn = generate_gs(pipe_knn,knn_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 143.5 min - 300 ; 207.3 - 442; 
gs_knn.fit(X,y)
knn_gs_res = save_results(gs_knn,params_to_evaluate_knn)
knn_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(knn_gs_res,filename='knn_bin+sel+aum.csv')
# Debemos usar el naive bayes categorico que aprenda las probabilidades a priori de las clases
from sklearn.naive_bayes import CategoricalNB
cat_NB = CategoricalNB(fit_prior=True)

#Creamos un diccionario con los parametros de nuestro estimador y los valores posibles de estos
NB_grid_params = {
    # Probamos con y sin suavizado de Laplace
    'categoricalnb__alpha': (0.0,1.0)
}
# Creamos el pipeline y el gridsearch con las opciones de preprocesado y los parametros del clasificador a evaluar
pipe_NB = create_pipeline(cat_NB)
gs_NB = generate_gs(pipe_NB,NB_grid_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 207.3 - 442; 
gs_NB.fit(X,y)
# Mostramos los resultados del entrenamiento
params_to_evaluate_NB = ['param_categoricalnb__alpha']
NB_gs_res = save_results(gs_NB,params_to_evaluate_NB)
NB_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)
# Exportamos resultados 
create_download_link(NB_gs_res,filename='NB_results.csv')
# Ahora no tiene sentido aprendes las probabilidades a priori pues todas las clases tienen la misma ocurrencia
cat_NB = CategoricalNB(fit_prior=False)

pipe_NB = create_pipeline(cat_NB,aumento = True)
gs_NB = generate_gs(pipe_NB,NB_grid_params)
# 19.9 - 42 ; 89.3 - 192 ; 115 min - 245 task; 207.3 - 442; 
gs_NB.fit(X,y)
# Mostramos los resultados del entrenamiento
NB_gs_res = save_results(gs_NB,params_to_evaluate_NB)
NB_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
# Exportamos resultados 
create_download_link(NB_gs_res,filename='NB_results_aum.csv')
# Creamos el clasificador en su forma primitiva
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=SEED,ccp_alpha = 0.01)

#Creamos un diccionario con los parametros de nuestro estimador y los valores posibles de estos
tree_grid_params = {
    'decisiontreeclassifier__criterion': ['gini','entropy']
}

# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree,binarizado = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
params_to_evaluate_tree = ['param_decisiontreeclassifier__criterion']
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin.csv')
# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree,binarizado = True,aumento = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin+aum.csv')
# Creamos el arbol modificado para que haga un balanceo
tree_balance = DecisionTreeClassifier(random_state = SEED,class_weight = 'balanced',ccp_alpha = 0.01)

# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree_balance,binarizado = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin+balan.csv')
# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree,binarizado = True,seleccion = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin+sel.csv')
# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree,binarizado = True,seleccion = True,aumento = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin+sel+aum.csv')
# Creamos el pipeline y la busqueda exhaustiva
pipe_tree = create_pipeline(tree_balance,binarizado = True,seleccion = True)
gs_tree = generate_gs(pipe_tree,tree_grid_params)
gs_tree.fit(X,y)
Tree_gs_res = save_results(gs_tree,params_to_evaluate_tree)
Tree_gs_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(10)
create_download_link(Tree_gs_res,filename='Tree_bin+sel+bal.csv')
