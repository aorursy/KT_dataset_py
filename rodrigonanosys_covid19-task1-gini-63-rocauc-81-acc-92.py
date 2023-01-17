# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

import math

pd.options.display.max_columns = 999

pd.options.display.max_rows = 112

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sn
#Loading dataset

#Carregando o dataset

df00= pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
#Doing some changes

#Modificando o nome de algumas colunas e aproveitando apra deixar o target binário

df00.reset_index(drop=True, inplace=True)

df00.rename(columns={"Patient ID":"id","SARS-Cov-2 exam result":"target"},inplace=True)

df00['target'].replace("negative",0,inplace=True)

df00['target'].replace("positive",1,inplace=True)
df00.columns.tolist()
#Missings
msno.matrix(df00)
#a lot of missings values

# Primeiro vamos verificar a quantidade de missings em todas variáveis

df01_missing = df00.isnull().sum()

df01_missing
# Salvando quantidade de linhas da tabela em variável

# number of rows

qt_rows = df00.shape[0]



#Creating a missing count df

# Gerando data frame com quantidade de missings por variavel

df_pct_missing = pd.DataFrame(df01_missing,columns=['qt_missing'])

df_pct_missing = pd.DataFrame(df01_missing,columns=['qt_missing'])

df_pct_missing['Features'] = df_pct_missing.index

df_pct_missing['pc_miss'] = (100*df_pct_missing['qt_missing'].divide(qt_rows)).astype(int)

df_pct_missing['qt_rows'] = qt_rows

df_pct_missing.reset_index(drop = True, inplace = True)

df_pct_missing = df_pct_missing[["Features","qt_rows","qt_missing","pc_miss"]]

df_pct_missing
#Lets verify how many patients id are usefull for our future model..

#Se todos os exames de um id forem missing, não vejo motivo para manter esses Ids no dataset.. vamos verificar!
df01 = df00.copy()
#drop uselles features for now

df01.drop(['id', 'Patient age quantile', 'target',

       'Patient addmited to regular ward (1=yes, 0=no)',

       'Patient addmited to semi-intensive unit (1=yes, 0=no)',

       'Patient addmited to intensive care unit (1=yes, 0=no)'],axis=1,inplace=True)
df01['soma'] = df01.count(axis=1)
len(df01[df01['soma']<=1])
#queremos apenas as linhas que possuem informação

#lets drop de useless rows

df01 = df01[df01['soma']>1]
df01.reset_index(inplace=True,drop=False)
df01 = df01['index'].to_frame()
df00.reset_index(drop=False, inplace=True)
df00 = pd.merge(df01,df00, how="left", on="index")
df00.set_index("index", inplace=True)
df00.head()
#ok a little bit better 5644 - 2055 = 3589 rows droped

#sobraram 2055 linhas

msno.matrix(df00)
#This function is not perfect for this but will help our analisys

#vamos verificar os metadados com o auxílio dessa função

def AjusteMetadados(dataframe): 



    train = dataframe

    # Verifica os tipos de variáveis presentes na tabela de treino

    t = []

    for i in train.columns:

            t.append(train[i].dtype)



    n = []

    for i in train.columns:

            n.append(i)



    aux_t = pd.DataFrame(data=t,columns=["Tipos"])

    aux_n = pd.DataFrame(data=n,columns=["Features"])

    df_tipovars = pd.concat([aux_n, aux_t], axis=1, join_axes=[aux_n.index])



    data = []

    for f in train.columns:



        # Definindo o papel das variáveis:

        if f == 'target':

            role = 'target'

        elif f == 'id':

            role = 'id'

        else:

            role = 'input'

        # Definindo o tipo das variáveis: nominal, ordinal, binary ou interval

        if f == 'target':

            level = 'binary'

        elif train[f].dtype == 'object' or f == 'id': 

            level = 'nominal'

        elif train[f].dtype in ['float','float64'] :

            level = 'interval'

        elif train[f].dtype in ['int','int64'] :

            level = 'ordinal'



        # Todas variáveis são incializadas com keep exceto o id

        keep = True

        if f == 'id':

            keep = False



        # Definindo o tipo das variáveis da tabela de entrada

        dtype = train[f].dtype



        # Criando a lista com todo metadados

        f_dict = {

            'Features': f,

            'Role': role,

            'Level': level,

            'Keep': keep,

            'Tipo': dtype

        }

        data.append(f_dict)



    meta = pd.DataFrame(data, columns=['Features', 'Role', 'Level', 'Keep', 'Tipo'])



    # Quantidade de domínios distintos para cada cariável do tipo ordinal e nominal

    card = []



    v = train.columns

    for f in v:

        dist_values = train[f].value_counts().shape[0]

        f_dict = {

                'Features': f,

                'Cardinality': dist_values

            }

        card.append(f_dict)



    card = pd.DataFrame(card, columns=['Features', 'Cardinality'])



    metadados_train = pd.merge(meta, card, on='Features')



    return metadados_train 
metadados = AjusteMetadados(df00)
#Cardinality = 0 are useless

#Variáveis com cardinalidade zero devem ser dropadas.. pois não possuem informação



metadados = metadados[metadados['Cardinality']>0]
metadados
#verificando e tratando as variaveis nominais
vars_nominais = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input')]
vars_nominais
#Vamos verificar como estão os valores únicos de cada feature

#Verifying unique values
vars_nominais_df = df00[vars_nominais['Features']]
#DATAFRAME + NUMERO DE LINHAS

#DEVOLVE UM DF COM OS VALORES ÚNICOS POR FEATURE EM UM DF

def unique_df(df,x):    

    rows = np.arange(x).tolist()

    unique = pd.DataFrame(rows,columns={"index"})

    for i in df.columns:

        u = df[i].unique()

        a = pd.DataFrame(u,columns=[i])

        try:

            a.sort_values(i,inplace=True,ascending=True)

            a.reset_index(drop=True,inplace=True)

        except:

            a = a

        unique=unique.merge(a,how="left",left_index=True,right_index=True)

    unique.fillna("",inplace=True)

    unique.drop("index",axis=1,inplace=True)

    return unique
unique_df(df00[vars_nominais_df.columns],5)
#We have some problems here.. lets fix

#vamos consertar algumas variáveis
df00['Urine - Esterase'].replace('not_done', np.nan, inplace=True)
#numerica 

df00['Urine - pH'].replace('Não Realizado', np.nan, inplace=True)
df00['Urine - Hemoglobin'].replace('not_done', np.nan, inplace=True)
df00['Urine - Bile pigments'].replace('not_done', np.nan, inplace=True)
df00['Urine - Ketone Bodies'].replace('not_done', np.nan, inplace=True)
df00['Urine - Nitrite'].replace('not_done', np.nan, inplace=True)
df00['Urine - Urobilinogen'].replace('not_done', np.nan, inplace=True)
df00['Urine - Protein'].replace('not_done', np.nan, inplace=True)
#numerica

df00['Urine - Leukocytes'].replace("<1000", "900", inplace=True)
#results

#Resultado

unique_df(df00[vars_nominais_df.columns],3)
#Urine=PH and Urine-Leucocytes are float, so lets do it

#Notamos que duas variáveis não são de fato nominais..vamos adicionar o dtype correto a elas
df00['Urine - pH'] = df00['Urine - pH'].astype("float64")
df00['Urine - Leukocytes'] = df00['Urine - Leukocytes'].astype("float64")
metadados = AjusteMetadados(df00)
vars_nominais = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input')]
vars_nominais
#get dummies

#podemos verificar que todas tem baixa cardinalidade, vamos dummieficar

nom = df00[vars_nominais['Features'].tolist()]
df_nom = pd.get_dummies(

                        nom,

                        prefix=nom.columns,

                        prefix_sep='_',

                        dummy_na=False,

                        columns=nom.columns,

                        sparse=False,

                        drop_first=False,

                        dtype=None,

                        )
df_nom.head()
#ok, next step...ordinals

#tudo certo com as nominais, vamos para as ordinais!!
vars_ordinais = metadados[(metadados.Level  == 'ordinal') & (metadados.Role == 'input')]
vars_ordinais
#somente uma coluna nos interessa aqui

#1 feature, 3 craps

ord_ = df00['Patient age quantile'].to_frame()
#low cardinality interval, its strange but its happening here

#vamos investigar as variáveis interval de baixa cardinalidade

vars_interval = metadados[(metadados.Level  == 'interval') & (metadados.Role == 'input')]
vars_interval_bc = vars_interval[vars_interval['Cardinality']<10]
interval_bc=df00[vars_interval_bc['Features'].tolist()]
interval_bc
unique_df(interval_bc,15)
#ok some columns, adios..

#Dropando colunas com apenas um valor distinto

interval_bc.drop(["Fio2 (venous blood gas analysis)","Myeloblasts"],axis=1,inplace=True)
#All clinical data were standardized to have a mean of zero and a unit standard deviation.

#vou substituir os nan por zero
interval_bc.fillna(0,inplace=True)
interval_bc.head()
#interval high cardinality

#Variáveis interval de alta cardinalidade
vars_interval_ac = vars_interval[vars_interval['Cardinality']>=10]
interval_ac=df00[vars_interval_ac['Features'].tolist()]
interval_ac.fillna(0,inplace=True)
interval_ac.head()
#Lets MinMaxScaler - interval_ac,interval_bc,ord_

#vamos normalizar - interval_ac,interval_bc,ord_
normalizar_ = pd.merge(ord_,interval_ac,right_index=True,left_index=True)
normalizar = pd.merge(normalizar_,interval_bc,right_index=True,left_index=True)
normalizar
from sklearn.preprocessing import MinMaxScaler



# Classe responável pela normalização

sc = MinMaxScaler(feature_range=(0, 1))



scaled_features = sc.fit_transform(normalizar)

num_normalizadas = pd.DataFrame(scaled_features, columns=normalizar.columns)
#dataframe stuff

#Fazendo alguns ajustes no dataframe

df_nom.reset_index(drop=True,inplace=True)

df00.reset_index(inplace=True,drop=True)

id_=df00['id'].to_frame()

target=df00['target'].to_frame()
#

##criando a abt
abt_ = pd.merge(df_nom,num_normalizadas, left_index=True,right_index=True)
abt__ = pd.merge(id_,abt_, left_index=True,right_index=True)
abt = pd.merge(abt__,target,left_index=True,right_index=True)
abt.head()
df00 = abt.copy()
#Change some simbols (-,N),(+,P)

#Esses símbolos podem interferir em alguns algorítimos, vamos fazer uma pequena mudança

df00.rename(columns={"Urine - Crystals_Oxalato de Cálcio +++":"Urine_Crystals_Oxalato_de_Calcio_PPP",

                     "Urine - Crystals_Oxalato de Cálcio -++":"Urine_Crystals_Oxalatode_Calcio_NPP",

                    "Urine - Crystals_Urato Amorfo +++":"Urine - Crystals_Urato_Amorfo_PPP",

                    "Urine - Crystals_Urato Amorfo --+":"Urine - Crystals_Urato_Amorfo_NNP"},inplace=True)
df00.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df00.columns]
#Ok we have a huge problem here..few rows..a lot of columns.. we have to be carefull

#Temos poucas linhas e muitas colunas, oque pode facilmente gerar singularidades matriciais

#Ou seja o processo de seleção de variaveis deve ser impecável, mas não vou me atentar a isso nesta versão!
df00.drop("id",inplace=True,axis=1)
#poucos casos positivos!!

#few positive cases

sn.countplot(df00['target'])
from sklearn.model_selection import train_test_split



explicativas = df00.drop(['target'], axis=1)

resposta = df00["target"]



x_train, x_test, y_train, y_test = train_test_split(explicativas, resposta, test_size = 0.2, random_state = 0)
#Lets use a lgbm for feature importancess first!! We just wanna know how many features are importants!

#Vamos rodar um lgbm somente para verificar a importância das variáveis, neste momento,vamos focar na quantidade!
from lightgbm import LGBMClassifier

lgbm=LGBMClassifier(boosting_type='gbdt',

                        num_leaves=10,

                        max_depth=1,

                        learning_rate=0.05,

                        n_estimators=200,

                        subsample_for_bin=200000,

                        objective='binary',

                        class_weight=None,

                        is_unbalance = True,

                        min_split_gain=1,

                        min_child_weight=0.0001,

                        min_child_samples=1,

                        subsample=1.0, 

                        subsample_freq=0,

                        colsample_bytree=1.0, 

                        reg_alpha=0.0,

                        reg_lambda=0.0, 

                        random_state=37,

                        n_jobs=3,

                        silent=True,

                        importance_type='gain'

                        )



lgbm.fit(x_train, y_train)
fi = pd.DataFrame(lgbm.feature_importances_, columns={"importances"})
#normalizar a importância das variáveis

from sklearn.preprocessing import MinMaxScaler



# Classe responável pela normalização

scf = MinMaxScaler(feature_range=(0, 1))



scaled_features = scf.fit_transform(fi)

fi = pd.DataFrame(scaled_features, columns=fi.columns)
feat = pd.DataFrame(df00.columns,columns={"Features"})
Feat_imp = pd.merge(feat,fi,left_index=True,right_index=True)
Feat_imp=Feat_imp.sort_values("importances",ascending=False)
#17 features

len(Feat_imp[Feat_imp['importances']>0.001])
#We are talking about tests here .. it would not be possible to perform all tests for all patients, in this case,

# we will try to build a model with few variables, only the 20 most important ones in addition to the 135 we created at abt



#Estamos falando de exames aqui.. não seria possível realizar todos para todos os pacientes, neste caso, 

#vamos tentar montar um modelo com poucas variáveis, apenas as 20 mais importantes ao invés das 135 que criamos na abt

#RFE pode nos ajudar!

#just a example for grid search!

#podemos procurar por melhores parâmetros, deixo aqui um exemplo!

from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier



parametros = {'num_leaves':[20,30,40],

              'learning_rate':[0.1],

              'n_estimators':[150],

              'n_jobs':[4],

              'max_depth':[1],

              'verbose':[0]}



GS_LGBM = GridSearchCV(estimator=LGBMClassifier(),

             param_grid=parametros, scoring='roc_auc', verbose=0)



GS_LGBM.fit(x_train,y_train)

#RFE will help us to feature selection 20 top features for our model

#Vamos usar o RFE para selecionar as top 20 variáveis e rodar o modelo com os parâmetros do gridsearch
from lightgbm import LGBMClassifier

from sklearn.feature_selection import RFE



#vamos utilizar os parâmetros encontrados pelo Grid Search

#GridSearch best parameter

lgbm2=GS_LGBM.best_estimator_



rfe = RFE(lgbm2, 20, step=1)

rfe = rfe.fit(x_train, y_train)



# Treino

y_pred_lgbm_train = rfe.predict(x_train)

y_score_lgbm_train = rfe.predict_proba(x_train)[:,1]



# Teste

y_pred_lgbm_test = rfe.predict(x_test)

y_score_lgbm_test = rfe.predict_proba(x_test)[:,1]
# 1) Cálculo da acurácia

from sklearn.metrics import accuracy_score



#Treino

acc_lgbm_train = round(accuracy_score(y_pred_lgbm_train, y_train) * 100, 2)



#Teste

acc_lgbm_test = round(accuracy_score(y_pred_lgbm_test, y_test) * 100, 2)



# 2) Cálculo da área sob curva ROC e Gini

from sklearn.metrics import roc_curve, auc



# Treino

fpr_lgbm_train, tpr_lgbm_train, thresholds = roc_curve(y_train, y_score_lgbm_train)

roc_auc_lgbm_train = 100*auc(fpr_lgbm_train, tpr_lgbm_train)

gini_lgbm_train = 100*round((2*roc_auc_lgbm_train/100 - 1), 2)



# Teste

fpr_lgbm_test, tpr_lgbm_test, thresholds = roc_curve(y_test, y_score_lgbm_test)

roc_auc_lgbm_test = 100*auc(fpr_lgbm_test, tpr_lgbm_test)

gini_lgbm_test = 100*round((2*roc_auc_lgbm_test/100 - 1), 2)





# 3) Gráfico da curva ROC

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(12,6))



lw = 2



plt.plot(fpr_lgbm_train, tpr_lgbm_train, color='blue',lw=lw, label='ROC (Treino = %0.0f)' % roc_auc_lgbm_train)

plt.plot(fpr_lgbm_test, tpr_lgbm_test, color='darkorange',lw=lw, label='ROC (Teste = %0.0f)' % roc_auc_lgbm_test)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Falso Positivo', fontsize=15)

plt.ylabel('Verdadeiro Positivo', fontsize=15)

plt.legend(loc="lower right")

plt.legend(fontsize=20) 

plt.title('COVID-19-Previsão de Testes Positivos - Baseados em Exames Clínicos', fontsize=20)

plt.show()



print('Acurácia, Gini e Área Curva ROC (Base de Treino): ',acc_lgbm_train, gini_lgbm_train, roc_auc_lgbm_train)

print('Acurácia, Gini e Área Curva ROC (Base de Teste): ',acc_lgbm_test, gini_lgbm_test, roc_auc_lgbm_test)
#We have to look for better parameter but is not a bad one

#Não é aquela coisa que se diga,"nossa que modelo", mas é o que temos pra hoje!
fi_2 = pd.DataFrame(lgbm2.feature_importances_, columns={"importances"})



#normalizar a importância das variáveis

from sklearn.preprocessing import MinMaxScaler



# Classe responável pela normalização

scf2 = MinMaxScaler(feature_range=(0, 1))



scaled_features = scf2.fit_transform(fi)

fi_2 = pd.DataFrame(scaled_features, columns=fi_2.columns)



feat2 = pd.DataFrame(df00.columns,columns={"Features"})



Feat_imp2 = pd.merge(feat2,fi_2,left_index=True,right_index=True)



Feat_imp2=Feat_imp.sort_values("importances",ascending=False)
import seaborn as sns

sns.set(style="whitegrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

ax = sns.barplot(x="importances", y="Features", data=Feat_imp2.head(20),errwidth=True,)
#Obrigado!

#Thx!