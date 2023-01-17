#Biblioteca necessárias -- descomente as linhas abaixo para instala-las
# !pip install --upgrade pandas
# !pip install --upgrade numpy
# !pip install --upgrade gcsfs
# !pip install --upgrade sklearn
# !pip install --upgrade missingno
# !pip install --upgrade dask
# !pip install --upgrade toolz
# !pip install --upgrade plotly
# !pip install --upgrade Pillow
# !pip install --upgrade xgboost
# !pip install --upgrade yellowbrick
# !pip install --upgrade statsmodels
# !pip install --upgrade imblearn
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import missingno as msno
import gc
#import gcsfs
from PIL import  Image
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
pd.options.display.float_format = '{:.2f}'.format
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [15, 7], 'axes.labelsize': 14,\
   'axes.titlesize': 14, 'font.size': 14, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 14,\
   'xtick.labelsize': 12, 'ytick.labelsize': 12}

sns.set(style='dark',rc=rc)
default_color = '#56B4E9'
colormap = plt.cm.cool
# fs = gcsfs.GCSFileSystem(project='My First Project')
# with fs.open('spark-madson/user-status-after.csv') as f:
#     data_user = pd.read_csv(f)
path = "../input/"
data_user = pd.read_csv(path+"user-status-after.csv")
data_week = pd.read_csv(path+"weekly-infos-before.csv", low_memory=False)
data_user.info()
data_user.head()
print ("\nMissing values :  ", data_user.isnull().sum().values.sum())
print ("\nUnique values :  \n",data_user.nunique())
def add_percentbar(ax,df):
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 1,
                '{:1.2f}%'.format(height*100/float(len(df))),
                ha="center")
ax = sns.countplot(x="status",data=data_user)
add_percentbar(ax,data_user)

data_week.info()
data_week.head()
print ("\nMissing values :  ", data_week.isnull().sum().values.sum())
print ("\nUnique values :  \n",data_week.nunique())
#criar coluna id com valor numerico para substituir coluna user
data_user['id'] = np.arange(len(data_user))
#juntar os 2 dataset
baseline = data_user.join(data_week.set_index('user'), on= 'user',how='left')
#remover coluna "user"
baseline.drop('user',axis=1,inplace=True)
baseline.head()
print ("Linhas   : " ,baseline.shape[0])
print ("Colunas  : " ,baseline.shape[1])
def get_meta(df):
    data = []
    for col in df.columns:
        # Defining the role
        if col == 'status':
            role = 'target'
        elif col == 'id':
            role = 'id'
        else:
            role = 'input'

        # Defining the level
        level=None
        if len(df[col].unique()) == 2 or col == 'status':
            level = 'binary'
        elif df[col].dtype == np.object:
            level = 'nominal'
        elif df[col].dtype == np.float64:
            level = 'interval'
        elif df[col].dtype == np.int64:
            level = 'ordinal'


        # Defining the data type 
        dtype = df[col].dtype

        # Creating a Dict that contains all the metadata for the variable
        col_dict = {
            'varname': col,
            'role'   : role,
            'level'  : level,
            'dtype'  : dtype
        }
        data.append(col_dict)
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta
meta = get_meta(baseline)
meta
print("Contagem de variaveis através dos tipos de dados")
meta_counts = meta.groupby(['role', 'level']).agg({'dtype': lambda x: x.count()}).reset_index()
meta_counts
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=meta_counts[(meta_counts.role != 'target') & (meta_counts.role != 'id') ],x="level",y="dtype",ax=ax,color=default_color)
ax.set(xlabel='Tipo de variável', ylabel='Count',title="Contagem de variaveis através dos tipos de dados")
#separar colunas por tipo de dados
col_ordinal   = meta[(meta.level == 'ordinal' ) & (meta.role != 'id') ].index
col_nominal   = meta[(meta.level == 'nominal') ].index
col_interval = meta[(meta.level == 'interval') ].index
missing_col = baseline.columns[baseline.isna().any()].tolist()
print("Colunas com valores Missing:")
print(missing_col)
print("")
print("Total de colunas com Missing: ",len(missing_col))
#Usuários que utilizaram o serviço em mais de uma semana
df_tmp = baseline.groupby('id').size().reset_index(name='count')
id_repetido =  df_tmp[df_tmp['count'] > 1]
id_repetido.head()
print("Exemplo de usuário que utilizou o produto por mais de uma semana")
pd.set_option('display.max_columns', 50)
baseline[baseline.id == 2]
#colunas sem id ou status
col_without_idORstatus = [c for c in baseline.columns if c != "id" and c !="status"]
ag = {}
for c in col_without_idORstatus:
    ag[c] = 'nunique'
print("Quantidade de valores diferentes das colunas agrupado por usuário")
df_tmp = baseline.groupby('id').agg(ag).reset_index()
df_tmp[df_tmp >1].count()
#Usuaris que mudaram a forma de pagamento
cobranca_diff = df_tmp[df_tmp.tipo_de_cobranca >1].tipo_de_cobranca.index
#status dos usuários que mudaram o tipo de pagamento
baseline.loc[cobranca_diff].status.unique()
#colunas que os valores se repetem
col_valor_rep = ["age_without_access","total_dependents", "total_active_dependents", 
                 "total_played_for_dependents", "total_cancels", "month_subs",
                 "assinatura_age", "sexo", "idade", "cidade" , "estado", "tipo_de_cobranca"]
#colunas que os valores variam
col_valor_diff = np.setdiff1d(np.array([col_without_idORstatus]),np.array(col_valor_rep))
#calunas que variam
col_valor_diff
#Dicionário de agregação
def create_agg_dict(colunas):
    agg = {}
    for c in colunas: # Week será agregado pelo total de semanas que
        if c == "week":
            agg[c] = "count"
        elif c in col_valor_rep or c =="status": 
            agg[c] = "first" # colunas que os volores se repetem serão agregadas pela primeira ocorrência
        else:
            agg[c] = "sum" # demais colunas serão agregadas pelo somatório
            
    return agg
#colunas sem o id dos usuários
col_without_id = [c for c in baseline.columns if c != "id"]
print("Dicionário de agregação")
aggregation = create_agg_dict(col_without_id)
aggregation

#dataset agregado
agregado = baseline.groupby('id')[col_without_id].agg(aggregation).reset_index()
agregado.info()
pd.set_option('display.max_columns', 50)
agregado.head()

msno.bar(agregado[missing_col],figsize=(20,8),color=default_color,fontsize=18,labels=True)
msno.heatmap(agregado[missing_col])
msno.dendrogram(agregado[missing_col],figsize=(20,8))
sorted_data = msno.nullity_sort(agregado[missing_col], sort='descending') # or sort='ascending'
msno.matrix(sorted_data,figsize=(20,8),fontsize=14)
#separar usuarios por status(churn)
churn = agregado[agregado.status == "cancelou"]
no_churn = agregado[agregado.status == "assinante"]
print("Matriz de nulidade dos Missing com churn")
sorted_data = msno.nullity_sort(churn[missing_col], sort='descending') # or sort='ascending'
ax = msno.matrix(sorted_data,figsize=(15,6),fontsize=14)

print("Matriz de nulidade Missing sem churn")
sorted_data = msno.nullity_sort(no_churn[missing_col], sort='descending') # or sort='ascending'
msno.matrix(sorted_data,figsize=(15,6),fontsize=14)
print("Mapa de calor dos missing com churn")
msno.heatmap(churn[missing_col])
print("Mapa de calor dos missing sem churn")
msno.heatmap(churn[missing_col])
#novo dataset agregado sem missing
agregado_sem_missing = agregado.dropna()
print("Contagem das entradas das colunas nominal")
#colunas nominal
for c in col_nominal:
    if c in ["cidade","estado"]:
        display(agregado_sem_missing.groupby(c)[c].count().nlargest(5))
    else:
        display(agregado_sem_missing.groupby(c)[c].count())
        
agregado_sem_missing[agregado_sem_missing.sexo == 'N']['status']
# remover linhas com sexo N
agregado_sem_missing = agregado_sem_missing[agregado_sem_missing.sexo != 'N'].reset_index()
grupo_cobranca = agregado_sem_missing.groupby('tipo_de_cobranca')['tipo_de_cobranca'].count().sort_values( ascending=False)
ax = grupo_cobranca.plot(kind='bar',figsize=(15,6))
#colunas do tipo ordinal
col_ordinal
#total_days
display(agregado_sem_missing.total_days.describe())
agregado_sem_missing.total_days.plot(kind="box")

def total_days_grupo(data):
    
    if (data["total_days"] < 5) :
        return "total_days_0-4"
    elif (data["total_days"] >= 5) & (data["total_days"] <= 15 ):
        return "total_days_5-15"
    elif (data["total_days"] > 15) & (data["total_days"] <= 30) :
        return "total_days_16-30"
    elif (data["total_days"] > 30) & (data["total_days"] <= 80) :
        return "total_days_31-80"
    elif data["total_days"] > 80 :
        return "total_days_gt_80"
    
agregado_sem_missing["total_days_grupo"] = agregado_sem_missing.apply(lambda agregado_sem_missing:total_days_grupo(agregado_sem_missing),axis = 1)
#age_without_access
display(agregado_sem_missing.age_without_access.describe())
agregado_sem_missing.age_without_access.plot(kind="box")
def age_without_access_grupo(data):
    
    if (data["age_without_access"] < -200) :
        return "age_without_access_lt_-200"
    elif (data["age_without_access"] >= -200) & (data["age_without_access"] <= -100 ):
        return "age_without_access_-200_-100"
    elif (data["age_without_access"] > -100) & (data["age_without_access"] <0) :
        return "age_without_access_-100_-1"
    elif (data["age_without_access"] >= 0) & (data["age_without_access"] <= 20) :
        return "age_without_access_0-20"
    elif data["age_without_access"] > 20 :
        return "age_without_access_gt_20"
    
agregado_sem_missing["age_without_access_grupo"] = agregado_sem_missing.apply(lambda agregado_sem_missing:age_without_access_grupo(agregado_sem_missing),axis = 1)
agregado_sem_missing.age_without_access.plot(kind="hist")
#week
agregado_sem_missing.week.plot(kind="box")
#total_sessions
display(agregado_sem_missing.total_sessions.describe())
agregado_sem_missing.total_sessions.plot(kind="box")
def total_sessions_grupo(data):
    
    if (data["total_sessions"] <= 5) :
        return "total_sessions_0-5"
    elif (data["total_sessions"] > 5) & (data["total_sessions"] <= 20 ):
        return "total_sessions_6-20"
    elif (data["total_sessions"] > 15) & (data["total_sessions"] <= 60) :
        return "total_sessions_16-60"
    elif data["total_sessions"] > 60 :
        return "total_sessions_gt_80"

    
agregado_sem_missing["total_sessions_grupo"] = agregado_sem_missing.apply(lambda agregado_sem_missing:total_sessions_grupo(agregado_sem_missing),axis = 1)
#total_dependents
display(agregado_sem_missing.total_dependents.describe())
agregado_sem_missing.total_dependents.plot(kind='hist')
agregado_sem_missing[agregado_sem_missing.total_dependents >= 1].groupby(['status'])['status'].count()
agregado_sem_missing[agregado_sem_missing.total_dependents < 1].groupby(['status'])['status'].count()
#total_active_dependents
display(agregado_sem_missing.total_active_dependents.describe())
agregado_sem_missing.total_active_dependents.plot(kind="hist")
agregado_sem_missing[agregado_sem_missing.total_active_dependents >= 1].groupby(['status'])['status'].count()
agregado_sem_missing[agregado_sem_missing.total_active_dependents < 1].groupby(['status'])['status'].count()
#total_cancels
display(agregado_sem_missing.total_cancels.describe())
agregado_sem_missing.total_cancels.plot(kind='box')
agregado_sem_missing.total_cancels.plot(kind='hist')
agregado_sem_missing[agregado_sem_missing.total_cancels > 0].groupby(['status'])['status'].count()
agregado_sem_missing[agregado_sem_missing.total_cancels == 0].groupby(['status'])['status'].count()
#month_subs
agregado_sem_missing.month_subs.plot(kind="box")
#converter entradas para valores inteiros
for c in col_interval:
    agregado_sem_missing[c] = agregado_sem_missing[c].astype(int)
    
agregado_sem_missing['idade'].plot(kind='box')
agregado_sem_missing['idade'].plot(kind='hist',bins=100)
agregado_sem_missing.assinatura_age.plot(kind='box')
agregado_sem_missing.assinatura_age.describe()
agregado_sem_missing[agregado_sem_missing.assinatura_age < 0].assinatura_age.count()
col_plataforma = ['android_app_time','ios_app_time', 'tv_app_time', 'mobile_web_time', 'desktop_web_time']
col_conteudo = ['time_spent_on_news', 'time_spent_on_humor', 'time_spent_on_series','time_spent_on_novelas', 'time_spent_on_special','time_spent_on_varieties', 'time_spent_on_sports','time_spent_on_realities']
col_acesso = ['time_spent_on_archived', 'time_spent_on_subscribed_content','time_spent_on_free_content', 'time_spent_on_grade']
col_formato = ['video_info_excerpt_time', 'video_info_extra_time','video_info_episode_time']
col_tamanho = ['video_info_time_spent_0_5','video_info_time_spent_5_15', 'video_info_time_spent_15_30','video_info_time_spent_30_60', 'video_info_time_spent_60mais']
col_tempo_total =  ["total_played_for_dependents",'total_played', 'max_played_time']
agregado_sem_missing[col_plataforma].describe()
agregado_sem_missing[agregado_sem_missing[col_plataforma] >= 1][col_plataforma].describe()
agregado_sem_missing[agregado_sem_missing[col_conteudo] >= 1][col_conteudo].describe()
agregado_sem_missing[agregado_sem_missing[col_tamanho] >= 1][col_tamanho].describe()
agregado_sem_missing[agregado_sem_missing[col_formato] >= 1][col_formato].describe()
agregado_sem_missing[agregado_sem_missing[col_tempo_total] >= 1][col_tempo_total].describe()
## Novas colunas que agrupam intervalos
def idade_grupo(data):
    
    if (data["idade"] < 18) :
        return "idade_0-17"
    elif (data["idade"] >= 18) & (data["idade"] <= 30 ):
        return "idade_18-30"
    elif (data["idade"] > 30) & (data["idade"] <= 50) :
        return "idade_31-50"
    elif (data["idade"] > 50) & (data["idade"] <= 80) :
        return "idade_51-80"
    elif data["idade"] > 80 :
        return "idade_gt_80"
    
agregado_sem_missing["idade_grupo"] = agregado_sem_missing.apply(lambda agregado_sem_missing:idade_grupo(agregado_sem_missing),axis = 1)

def assinatura_age_grupo(data):
    
    if data["assinatura_age"] < 0 :
        return "age_lt-0"
    elif (data["assinatura_age"] >= 0) & (data["assinatura_age"] <= 100 ):
        return "age_0-100"
    elif (data["assinatura_age"] > 100) & (data["assinatura_age"] <= 500) :
        return "age_101-500"
    elif (data["assinatura_age"] > 500) & (data["assinatura_age"] <= 1000) :
        return "age_501-1000"
    elif (data["assinatura_age"] > 1000) & (data["assinatura_age"] <= 5000) :
        return "age_1001-5000"
    elif data["assinatura_age"] > 5000 :
        return "age_gt_5000"
    
agregado_sem_missing["assinatura_age_grupo"] = agregado_sem_missing.apply(lambda agregado_sem_missing:assinatura_age_grupo(agregado_sem_missing), axis = 1)

# def tempo_grupo(data,c):
    
#     if data[c] < 1 :
#         return "tempo_lt-1"
#     elif (data[c] >= 1) & (data[c] <= 120 ):
#         return "tempo_0-120"
#     elif (data[c] > 120) & (data[c] <= 1000) :
#         return "tempo_121-1000"
#     elif (data[c] > 1000) & (data[c] <= 7200) :
#         return "tempo_1001-7200"
#     elif data[c] > 7200 :
#         return "tempo_gt_7200"
# col_produto = col_acesso + col_tamanho + col_conteudo + col_formato + col_tempo_total +col_plataforma+col_tempo_total
# for c in col_produto :
#     agregado_sem_missing["{}_grupo".format(c)] = agregado_sem_missing.apply(lambda agregado_sem_missing:tempo_grupo(agregado_sem_missing,c), axis = 1)
agregado_sem_missing.sample(5)
churn = agregado_sem_missing[agregado_sem_missing.status == "cancelou"]
no_churn = agregado_sem_missing[agregado_sem_missing.status == "assinante"]
gc.collect()
# def plot_pie(column) :
    
#     trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
#                     labels  = churn[column].value_counts().keys().tolist(),
#                     hoverinfo = "label+percent+name",
#                     domain  = dict(x = [0,.48]),
#                     name    = "Cancelou",
#                     marker  = dict(line = dict(width = 2,
#                                                color = "rgb(243,243,243)")
#                                   ),
#                     hole    = .6
#                    )
#     trace2 = go.Pie(values  = no_churn[column].value_counts().values.tolist(),
#                     labels  = no_churn[column].value_counts().keys().tolist(),
#                     hoverinfo = "label+percent+name",
#                     marker  = dict(line = dict(width = 2,
#                                                color = "rgb(243,243,243)")
#                                   ),
#                     domain  = dict(x = [.52,1]),
#                     hole    = .6,
#                     name    = "Assinante" 
#                    )


#     layout = go.Layout(dict(title = "Distribuição do status por " +column ,
#                             plot_bgcolor  = "rgb(243,243,243)",
#                             paper_bgcolor = "rgb(243,243,243)",
#                             annotations = [dict(text = "Cancelou",
#                                                 font = dict(size = 13),
#                                                 showarrow = False,
#                                                 x = .15, y = .5),
#                                            dict(text = "Assinante",
#                                                 font = dict(size = 13),
#                                                 showarrow = False,
#                                                 x = .88,y = .5
#                                                )
#                                           ]
#                            )
#                       )
#     data = [trace1,trace2]
#     fig  = go.Figure(data = data,layout = layout)
#     py.iplot(fig)
def histogram(column) :
    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Cancelou",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = no_churn[column],
                          histnorm = "percent",
                          name = "Assinante",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title ="Distribuição do status por " +column,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)
sexo_churn = churn.groupby('sexo')['sexo'].count().reset_index(name='count')
sexo_no_churn = no_churn.groupby('sexo')['sexo'].count().reset_index(name='count')
plt.subplot(the_grid[0, 1],  title='cancelou')
sns.barplot(x='sexo',y='count', data=sexo_churn, palette='Spectral')
plt.subplot(the_grid[0, 0], title='assinante')
sns.barplot(x='sexo',y='count', data=sexo_no_churn, palette='Spectral')

plt.suptitle('Status dos usuarios por sexo', fontsize=16)

histogram('sexo')
grupo_estado = agregado_sem_missing.groupby('estado')['estado'].count().sort_values( ascending=False)
ax = grupo_estado.plot(kind='bar',figsize=(15,6))
print("porcentagem de usuários nas top 3 estados: {:.2f}%".format((grupo_estado.nlargest(3).sum()*100 )/float(grupo_estado.sum()) ) )
plt.figure(2, figsize=(20,17))
the_grid = GridSpec(2, 2)
estado_churn = churn.groupby('estado')['estado'].count().sort_values( ascending=False)
estado_no_churn = no_churn.groupby('estado')['estado'].count().sort_values( ascending=False)
plt.subplot(the_grid[1, 0],  title='cancelou')
estado_churn.plot.barh()
plt.subplot(the_grid[0, 0], title='assinante')
estado_no_churn.plot.barh()

plt.suptitle('Status dos usuarios por estado', fontsize=16)
gc.collect()
grupo_cidades = agregado_sem_missing.groupby('cidade')['cidade'].count().sort_values( ascending=False)
ax = grupo_cidades.nlargest(50).plot.barh(figsize=(15,15))

print("porcentagem de usuários nas top 10 cidades: {:.2f}%".format((grupo_cidades.nlargest(10).sum()*100 )/float(grupo_cidades.sum()) ) )
plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)
cidade_churn = churn.groupby('cidade')['cidade'].count().sort_values( ascending=False)
cidade_no_churn = no_churn.groupby('cidade')['cidade'].count().sort_values( ascending=False)
plt.subplot(the_grid[0, 1],  title='cancelou')
cidade_churn.nlargest(15).plot.barh()
plt.subplot(the_grid[0, 0], title='assinante')
cidade_no_churn.nlargest(15).plot.barh()

plt.suptitle('Status dos usuarios por cidade', fontsize=16)
#idade
histogram('idade_grupo')
histogram("idade")
histogram('tipo_de_cobranca')
histogram("month_subs")
histogram('week')
histogram("assinatura_age")
histogram("total_dependents")
histogram('total_cancels')
histogram("total_days")

col_originais = [c for c in agregado_sem_missing.columns if "grupo" not in c ]
#novo dataset que será utililzado pelos modelos de Machine Learning
data = agregado_sem_missing[col_originais]
data.drop(['index'],axis=1,inplace=True)
data.head()
col_nominal
### codificar features com valores não numericos em valores numericos
le = LabelEncoder()
data['sexo'] = le.fit_transform(data['sexo'])
le = LabelEncoder()
data['status'] = le.fit_transform(data['status'])
for c in col_nominal[1:]:
    data[c] = le.fit_transform(data[c])
## codificar age_age_without_access
def age_without_access_encode(data):
    
    if (data["age_without_access"] < 0) :
        return 0
    elif (data["age_without_access"] >= 0) & (data["age_without_access"] <= 20) :
        return 1
    elif data["age_without_access"] > 20 :
        return 2
    
data["age_without_access"] = data.apply(lambda data:age_without_access_encode(data),axis = 1)
get_meta(data)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=25)
rf.fit(data.drop(['status'],axis=1), data.status)
features = data.drop(['status'],axis=1).columns.values
print("----- Training Done -----")
def get_feature_importance_df(feature_importances, 
                              column_names, 
                              top_n=25):
    """Get feature importance data frame.
 
    Parameters
    ----------
    feature_importances : numpy ndarray
        Feature importances computed by an ensemble 
            model like random forest or boosting
    column_names : array-like
        Names of the columns in the same order as feature 
            importances
    top_n : integer
        Number of top features
 
    Returns
    -------
    df : a Pandas data frame
 
    """
     
    imp_dict = dict(zip(column_names, 
                        feature_importances))
    top_features = sorted(imp_dict, 
                          key=imp_dict.get, 
                          reverse=True)[0:top_n]
    top_importances = [imp_dict[feature] for feature 
                          in top_features]
    df = pd.DataFrame(data={'feature': top_features, 
                            'importance': top_importances})
    return df
feature_importance = get_feature_importance_df(rf.feature_importances_, features)

feature_importance
features_importantes = feature_importance.feature.iloc[:9].tolist()
features_importantes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score
from yellowbrick.classifier import DiscriminationThreshold
from xgboost import XGBClassifier

#dividir conjunto de treinamento e teste
train,test = train_test_split(data,test_size = .25 ,random_state = 25)
##separar variaveis dependentes e independentes

train_X = train.drop(['status'],axis=1)
train_Y = train['status']
test_X  = test.drop(['status'],axis=1)
test_Y  = test['status']

#função de predição
 
def churn_prediction(model,training_x,testing_x, training_y,testing_y) :
    #model
    model.fit(training_x,training_y)
    predictions   = model.predict(testing_x)
    probabilities = model.predict_proba(testing_x)

    print (model)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
    #confusion matrix
    cm = confusion_matrix(testing_y,predictions)
    display( pd.crosstab(testing_y ,predictions, rownames=['True label'], margins=True,colnames=['Predict label']) )
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions)
    print ("Area under curve : ",model_roc_auc,"\n")
       

logr_params = {}
logr_params['C'] = 1.0
logr_params['class_weight'] = None
logr_params['dual'] = False
logr_params['fit_intercept'] = True
logr_params['intercept_scaling'] = 1
logr_params['max_iter'] =100
logr_params['multi_class'] ='ovr'
logr_params['n_jobs'] =1
logr_params['penalty'] ='l2'
logr_params['random_state']=None
logr_params['solver']='liblinear'
logr_params['tol']=0.0001
logr_params['verbose'] =0
logr_params['warm_start'] = False

logr  = LogisticRegression(**logr_params)

churn_prediction(logr,train_X,test_X,train_Y,test_Y)
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30
rf_model = RandomForestClassifier(**rf_params)
churn_prediction(rf_model,train_X,test_X,train_Y,test_Y)
xgb_params = {}
xgb_params['learning_rate'] = 0.02
xgb_params['n_estimators'] = 1000
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9
xgb_model = XGBClassifier(**xgb_params)
churn_prediction(xgb_model,train_X,test_X,train_Y,test_Y)
from imblearn.over_sampling import SMOTE

cols    = [i for i in data.columns if i != 'status']

smote_X = data[cols]
smote_Y = data['status']

#separar em conjunto de treino e teste
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,
                                                                        test_size = .25 ,
                                                                        random_state = 25)

#super amostragem usando smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=['status'])
###
churn_prediction(logr,os_smote_X,test_X,os_smote_Y,test_Y)
churn_prediction(rf_model,os_smote_X,test_X,os_smote_Y,test_Y)
churn_prediction(xgb_model,os_smote_X,test_X,os_smote_Y,test_Y)
base = test_X.join(data_user.set_index('id'), on= 'id',how='left')
# sub = pd.DataFrame()
# sub['user'] = base['user']
# sub['status'] = predictions
sub.to_csv('predictons.csv', index=False)
sub
#rf_model.fit(os_smote_X,os_smote_Y)
predictions   = rf_model.predict(test_X)
