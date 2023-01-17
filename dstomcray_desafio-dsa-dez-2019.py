import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from inspect import signature



import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import *

from sklearn import model_selection

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
# Carregando o DataSet

df_treino = pd.read_csv("../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv", index_col = 0)

df_teste  = pd.read_csv("../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv", index_col = 0)



#Visualizando as 10 primeiras linhas do dataset

df_treino.head(10)
# examinando os tipos de dados e estatísticas descritivas 



print (df_treino.info ()) 

print (df_treino.describe ())
# funcao para avaliar distribuição dos dados missings em cada atributo

def checa_missing(df):

    for col in df.columns:

        if df[col].isnull().sum() > 0:

            print(col, df[col].isnull().sum())
checa_missing(df_treino)
# função para Avaliar a distribuição dos dados em cada atributo

def checa_distribuicao(df):

    for col in df.columns:

        if df[col].dtype == object:

            print(df.groupby([col])[col].count())

            print('')
checa_distribuicao(df_treino)
df_teste.head()
df_treino.shape
# Visualizando a distribuição de classes



print("Class Counts")

print(df_treino["target"].value_counts(), end="\n\n")

print("Class Proportions")

print(df_treino["target"].value_counts()/len(df_treino["target"]))
sns.set(style="whitegrid")



#Usando um gráfico de barras para mostrar a distribuição das classes: ativado e não-ativado

bp = sns.countplot(x=df_treino["target"])

plt.title("Distribuição de classe do conjunto de dados")

bp.set_xticklabels(["não ativado","ativado"])

plt.show()
# funcao para transformar os dados categóricos

def tranforma(df):

    labelencoder_X=LabelEncoder()

    for col in df.columns:

        if df[col].dtype == object:

            df[col] = labelencoder_X.fit_transform(df[col].astype(str))

    return df       
# Aplicando transformação nos dados categóricos

df_tratado = tranforma(df_treino)

dft_tratado = tranforma(df_teste)
# Imputando recursos multivariados para os valores missings



X_tratado = df_tratado.iloc[:,1:]

imp = IterativeImputer(max_iter=10, initial_strategy='median', random_state=0)

imp.fit(X_tratado)

X_tratado = imp.transform(X_tratado)
# Aplicando algoritmo de Regressão Logistica para seleção automática de recursos para o aprendizado

lsvc = logisticRegr = LogisticRegression(C=10, l1_ratio=0.25, max_iter=800, solver='saga',penalty="elasticnet",fit_intercept=True,multi_class='ovr').fit(X_tratado,  df_tratado["target"])

model = SelectFromModel(lsvc, prefit=True)

X_svc = model.transform(X_tratado)

feature_idx = model.get_support()

X_svc.shape
# Mantendo apenas as variáveis mais importantes para aplicar no modelo

columns = []

for i in range(feature_idx.shape[0]):

    if feature_idx[i] == True:

        columns.append(df_tratado.columns[i+1])
# Avaliação da importancia de cadas variável no conjunto de dados em relação a variável alvo



from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect

df_mutual_information = mutual_info_classif(X_svc, df_treino["target"])



plt.subplots(1, figsize=(26, 1))

sns.heatmap(df_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)

plt.yticks([], [])

plt.gca().set_xticklabels(columns, rotation=90, ha='right', fontsize=12)

plt.suptitle("Verificando a importancia de cada variável  (mutual_info_classif)", fontsize=18, y=1.2)

plt.gcf().subplots_adjust(wspace=0.2)

pass
# Gerando os dados de treino e de teste para o modelo

X_feature_train, X_feature_test, y_train, y_test = train_test_split(X_svc, df_treino["target"], test_size=0.2, random_state=42)
from xgboost import XGBClassifier

import scipy.stats as st

xgb = XGBClassifier(nthread=1,

                    silent=False, 

                    scale_pos_weight=0.761199,

                    scale_neg_weight=0.238801,

                    learning_rate=0.01,  

                    colsample_bytree = 0.4,

                    subsample = 0.33,

                    rate_drop=0.4,

                    objective='binary:logistic', 

                    eval_metric='logloss',

                    n_estimators=600, 

                    reg_alpha = 0.4,

                    max_depth=7, 

                    gamma=10)

eval_set = [(X_feature_train, y_train), (X_feature_test, y_test)]

xgb.fit(X_feature_train, y_train,early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

#generate predicted classes for test data

pred = xgb.predict(X_feature_test)

#generate predicted probabilites for test data

pred_prob = xgb.predict_proba(X_feature_test)
#from xgboost import XGBClassifier

#import scipy.stats as st

#from sklearn.model_selection import RandomizedSearchCV

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=1,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    n_estimators=1000, 

#                    reg_alpha = 0.4,

#                    max_depth=7, 

#                    gamma=10)



#one_to_left = st.beta(10, 1)  

#from_zero_positive = st.expon(0, 50)



#params = {  

#    "n_estimators": st.randint(900, 1000),

#    "max_depth": st.randint(3, 10),

#    "learning_rate": st.uniform(0.05, 0.4),

#    "colsample_bytree": one_to_left,

#    "subsample": one_to_left,

#    "objective": 'binary:logistic',

#    "gamma": st.uniform(0, 10),

#    'reg_alpha': from_zero_positive,

#    "min_child_weight": from_zero_positive,

#}

# 16/12/2019 as 23:40 log-loss 0.47062557673195327

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=1,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    eval_metric='logloss',

#                    n_estimators=1000, 

#                    reg_alpha = 0.4,

#                    max_depth=7, 

#                    gamma=10)

# 16/12/2019 as 23:48 log-loss 0.47062557673195327

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=1,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    eval_metric='logloss',

#                    n_estimators=1000, 

#                    reg_alpha = 0.4,

#                    max_depth=10, 

#                    gamma=10)

# 0.469607022417657

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=1,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    eval_metric='logloss',

#                    n_estimators=1000, 

#                    reg_alpha = 0.4,

#                    max_depth=12, 

#                    gamma=10)

# 0.4692065531642619 -> max_depth=15, 0.4691725650704019 -> max_depth=17

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=0.9,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    eval_metric='logloss',

#                    n_estimators=1000, 

#                    reg_alpha = 0.4,

#                    max_depth=10, 

#                    gamma=10)

#####################################################

#xgb = XGBClassifier(silent=False, 

#                    scale_pos_weight=1,

#                    learning_rate=0.01,  

#                    colsample_bytree = 0.4,

#                    subsample = 0.8,

#                    objective='binary:logistic', 

#                    eval_metric='logloss',

#                    n_estimators=1200, 

#                    reg_alpha = 0.4,

#                    max_depth=10, 

#                    gamma=10)

#xgb = XGBClassifier()

#gs = RandomizedSearchCV(xgb, params, n_jobs=1, verbose=1)  

#gs.fit(X_feature_train, y_train)  

#xgb = XGBClassifier()

#use logistic model to fit training data

#eval_set = [(X_feature_train, y_train), (X_feature_test, y_test)]

#xgb.fit(X_feature_train, y_train,early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

#generate predicted classes for test data

#pred = xgb.predict(X_feature_test)

#generate predicted probabilites for test data

#pred_prob = xgb.predict_proba(X_feature_test)
# Avaliação das métricas de desempenho

results = xgb.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()


#generate confusion matrix

cm_xgb = confusion_matrix(y_test, pred)

#put it into a dataframe

cm_xgb_df = pd.DataFrame(cm_xgb)



#plot CM

fig, ax = plt.subplots(figsize = (7,7))

sns.heatmap(pd.DataFrame(cm_xgb_df.T), annot=True, annot_kws={"size": 15}, cmap="Purples", vmin=0, vmax=500, fmt='.0f', linewidths=1, linecolor="white", cbar=False,

           xticklabels=["não ativado","ativado"], yticklabels=["não ativado","ativado"])

plt.ylabel("Predicted", fontsize=15)

plt.xlabel("Actual", fontsize=15)

ax.set_xticklabels(["não ativado","ativado"], fontsize=13)

ax.set_yticklabels(["não ativado","ativado"], fontsize=13)

plt.title("Confusion Matrix for Logistic Classifier (Threshold = 0.5) - Counts", fontsize=15)

plt.show()
#Generating a Confusion matrix of proportions for logistic model



#converting counts to proportions

cm_xgb = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]



cm_xgb_df = pd.DataFrame(cm_xgb)

fig, ax = plt.subplots(figsize = (7,7))

sns.heatmap(pd.DataFrame(cm_xgb_df.T), annot=True, annot_kws={"size": 15}, cmap="Purples", vmin=0, vmax=1, fmt='.3f', linewidths=1, linecolor="white", cbar=False,

           xticklabels=["não ativado","ativado"], yticklabels=["não ativado","ativado"])

plt.ylabel("Predicted", fontsize=15)

plt.xlabel("Actual", fontsize=15)

ax.set_xticklabels(["não ativado","ativado"], fontsize=13)

ax.set_yticklabels(["não ativado","ativado"], fontsize=13)

plt.title("Confusion Matrix for Logistic Classifier (Threshold = 0.5) - Proportions", fontsize=15)



plt.show()
#generating a report to extract the measure of interest using built-in sklearn function

print(classification_report(y_test,pred))
#Plotting the ROC curve



#Generating points to plot on ROC curve (logistic model)

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, pred_prob[:,1])





fig, ax = plt.subplots(figsize = (10,7))

#plotting the "guessing" model

plt.plot([0, 1], [0, 1], 'k--')

#plotting the logistic model

plt.plot(fpr_xgb, tpr_xgb)

plt.fill_between(fpr_xgb, tpr_xgb, alpha=0.2, color='b')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve: AUC={0:0.3f}'.format(roc_auc_score(y_test,pred_prob[:,1])))

plt.show()
#Plot PR curve



#Generating points to plot on recall precision curve

precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])

average_precision = average_precision_score(y_test, pred_prob[:,1])



#its a step function so plotting is different 

fig, ax = plt.subplots(figsize = (10,7))

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='orange', alpha=1,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='orange', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision={0:0.3f}'.format(average_precision))

plt.show()
print(log_loss(y_test, pred_prob))
print(accuracy_score(y_test,pred))
array_tratado = imp.transform(dft_tratado)
columns = dft_tratado.columns

indexs  = dft_tratado.index
dft_tratado = pd.DataFrame(array_tratado, columns=columns, index=indexs)
dft_tratado = dft_tratado.iloc[:, feature_idx]
dft_tratado.head()
dft_tratado.shape
pred_prob = xgb.predict_proba(dft_tratado.values)[:,1]
pred_prob
sub = pd.read_csv("../input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv", sep=',',

                    parse_dates = True, low_memory = False)
sub['PredictedProb'] = pred_prob
# Salvando 

sub.to_csv("sample_submission.csv", sep=',', index=False)