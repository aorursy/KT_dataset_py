# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importação dos pacotes

import pandas as ps

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import metrics

import lightgbm as lgb
treino = pd.read_csv("../input/kernelrandomensemble/dataset_treino.csv")

teste = pd.read_csv("../input/kernelrandomensemble/dataset_teste.csv")

submission = pd.read_csv("../input/kernelrandomensemble/sample_submission.csv")

transacoeshist = pd.read_csv("../input/kernelrandomensemble/transacoes_historicas.csv")

newtransacoeshist = pd.read_csv("../input/kernelrandomensemble/novas_transacoes_comerciantes.csv")
treino.head()
treino.info()
teste.head()
teste.info()
transacoeshist.head()
transacoeshist.info()
newtransacoeshist.head()
newtransacoeshist.info()
#Representando numericamente valores da variável "first_active_month"

treino['first_active_month']=pd.to_datetime(treino['first_active_month'])

teste['first_active_month']=pd.to_datetime(teste['first_active_month'])

treino["ano"] = treino["first_active_month"].dt.year

teste["ano"] = teste["first_active_month"].dt.year

treino["mes"] = treino["first_active_month"].dt.month

teste["mes"] = teste["first_active_month"].dt.month
treino.head(2)
teste.head(2)
treino.describe()
teste.describe()
#Distribuição da variável target

fig = plt.figure(figsize=(15,5))

sns.distplot(treino['target'])
fig = plt.figure(figsize=(15,5))

sns.boxplot(data = treino[['target']], orient = "h")
sns.countplot(x = "feature_1", data = treino, palette = "Greens_d")
sns.stripplot(x = "feature_1", y = "target", data = treino)
sns.countplot(x = "feature_2", data = treino, palette = "Greens_d")
sns.stripplot(x = "feature_2", y = "target", data = treino)
sns.countplot(x = "feature_3", data = treino, palette = "Greens_d")
sns.stripplot(x = "feature_3", y = "target", data = treino)
sns.countplot(x = "mes", data = treino, palette = "Greens_d")
sns.stripplot(x = "mes", y = "target", data = treino)
treino.isnull().sum()
teste.isnull().sum()
#Substituindo valores nulos dos dados de teste pelos valores mais frequentes

from statistics import mode

teste['first_active_month'].fillna(mode(teste['first_active_month']), inplace=True)

teste['ano'].fillna(mode(teste['ano']), inplace=True)

teste['mes'].fillna(mode(teste['mes']), inplace=True)
teste.isnull().sum()
newtransacoeshist.isnull().sum()
newtransacoeshist['category_2'].fillna(value=0.1,inplace=True)

newtransacoeshist['category_3'].fillna('B',inplace=True)

newtransacoeshist['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
transacoeshist.isnull().sum()
transacoeshist['category_2'].fillna(value=0.1,inplace=True)

transacoeshist['category_3'].fillna('B',inplace=True)

transacoeshist['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
hist = transacoeshist.groupby(["card_id"])

hist= hist["purchase_amount"].size().reset_index()

hist.columns = ["card_id", "hist_transactions"]

treino = pd.merge(treino,hist, on="card_id", how="left")

teste = pd.merge(teste,hist, on="card_id", how="left")



hist = transacoeshist.groupby(["card_id"])

hist = hist["purchase_amount"].agg(['sum','mean','max','min','std']).reset_index()

hist.columns = ['card_id','sum_hist_tran','mean_hist_tran','max_hist_tran','min_hist_tran','std_hist_tran']

treino = pd.merge(treino,hist,on='card_id',how='left')

teste = pd.merge(teste,hist,on='card_id',how='left')
merchant = newtransacoeshist.groupby(["card_id"])

merchant= merchant["purchase_amount"].size().reset_index()

merchant.columns = ["card_id", "merchant_transactions"]

treino = pd.merge(treino,merchant, on="card_id", how="left")

teste = pd.merge(teste,merchant, on="card_id", how="left")



merchant= newtransacoeshist.groupby(["card_id"])

merchant= merchant["purchase_amount"].agg(['sum','mean','max','min','std']).reset_index()

merchant.columns=['card_id','sum_merchant_tran','mean_merchant_tran','max_merchant_tran','min_merchant_tran','std_merchant_tran']

treino=pd.merge(treino,merchant,on='card_id',how='left')

teste=pd.merge(teste,merchant,on='card_id',how='left')
#treino

dumtreino_feature_1 = pd.get_dummies(treino['feature_1'],prefix = 'f1_')

dumtreino_feature_2 = pd.get_dummies(treino['feature_2'],prefix = 'f2_')

dumtreino_feature_3 = pd.get_dummies(treino['feature_3'],prefix = 'f3_')



#teste

dumteste_feature_1 = pd.get_dummies(teste['feature_1'], prefix = 'f1_')

dumteste_feature_2 = pd.get_dummies(teste['feature_2'], prefix = 'f2_')

dumteste_feature_3 = pd.get_dummies(teste['feature_3'], prefix = 'f3_')



#concatenando dados

treino = pd.concat([treino, dumtreino_feature_1, dumtreino_feature_2, dumtreino_feature_3], axis = 1, sort = False)

teste = pd.concat([teste, dumteste_feature_1, dumteste_feature_2, dumteste_feature_3], axis = 1, sort = False)
#visualizando o resultado dados de treino

treino.head()
#visualizando o resultado dados de teste

teste.head()
target_col = treino['target']

treino.drop('target',axis=1,inplace=True)
treino.columns
cols_to_use = ['ano', 'mes', 'hist_transactions', 'sum_hist_tran', 'mean_hist_tran',

       'max_hist_tran', 'min_hist_tran', 'std_hist_tran',

       'merchant_transactions', 'sum_merchant_tran', 'mean_merchant_tran',

       'max_merchant_tran', 'min_merchant_tran', 'std_merchant_tran', 'f1__1',

       'f1__2', 'f1__3', 'f1__4', 'f1__5', 'f2__1', 'f2__2', 'f2__3', 'f3__0',

       'f3__1']



def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.05,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_freq" : 5,

        "bagging_seed" : 2019,

        "verbosity" : -1

    }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], 

        early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result



train_X = treino[cols_to_use]

test_X = teste[cols_to_use]

train_y = target_col.values



pred_test = 0

kf = model_selection.KFold(n_splits=5, random_state=2019, shuffle=True)

for dev_index, val_index in kf.split(treino):

    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    pred_test += pred_test_tmp

pred_test /= 5.
fig, ax = plt.subplots(figsize=(15,12))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax,color='lightblue')

plt.title("Importância das Variáveis",color='black', fontsize = 18)

ax.grid(False)

plt.show()
submission['target'] = pred_test

submission.to_csv("submission1.csv", index=False)
submission.head()