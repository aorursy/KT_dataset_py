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
df= pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head(2)
df.info()
df[df['TotalCharges'].str.isspace()]
df['TotalCharges']=df['TotalCharges'].str.replace(' ','0').astype(float)
df2 = df.copy()
pd.get_dummies(df['InternetService'])
df= pd.get_dummies(df,columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',

                   'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',

                   'Contract','PaperlessBilling','PaymentMethod'])
df.shape
df.info()
from sklearn.model_selection import train_test_split 



train,test = train_test_split(df,test_size=0.2,random_state = 42)
train, valid = train_test_split(train, test_size=0.2,random_state=42)

train.shape,valid.shape,test.shape
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(n_estimators=200,random_state=42)
feats = [c for c in df.columns if c not in ['customerID','Churn']]
rf.fit(train[feats],train['Churn'])
preds_val=rf.predict(valid[feats])

preds_val
from sklearn.metrics import accuracy_score
accuracy_score(valid['Churn'],preds_val)
preds_test=rf.predict(test[feats])
accuracy_score(test['Churn'],preds_test)
df['Churn'].value_counts(normalize=True)
import seaborn as sn

confusion_matrix = pd.crosstab(test['Churn'], preds_test, rownames=['Actual'], colnames=['Predicted'])



sn.heatmap(confusion_matrix, annot=True)
#copiando df

df2.info()
df2['gender'].astype('category').cat.categories
df2['gender'].astype('category').cat.codes
df2['PaymentMethod'].astype('category').cat.categories.value_counts()
df2['PaymentMethod'].astype('category').cat.codes.value_counts()
# convertendo as colunas categoricas em conlunas numericas

for col in df2.columns:

    if df2[col].dtype == 'object':

        df2[col] = df2[col].astype('category').cat.codes
df2.info()
#separando o dataframe em train, valid e test

#primeiro split - train test

train,test =train_test_split(df2,test_size=0.2,random_state=42)

#segundo split train validatioon

train,valid = train_test_split(train, test_size = 0.2, random_state = 42)



train.shape, valid.shape, test.shape

#colunas a serem usadas para treino



feats = [c for c in df2.columns if c not in ['customerID','Churn'] ]
#instanciando o modelo

rf2 =RandomForestClassifier(n_estimators=200,random_state=42)



#treinando o modelo

rf2.fit(train[feats],train['Churn'])
#previsoes para os dados de validadcao

pred_val = rf2.predict(valid[feats])



#verificando acuracia

accuracy_score(valid['Churn'],pred_val)
#previsoes parar os dados de teste

preds_test = rf2.predict(test[feats])



#verificando a acuraria

accuracy_score(test['Churn'],preds_test)
rf2.feature_importances_,feats
#avaliando a importancia de cada coluna(cada variavel de entrada)

pd.Series(rf2.feature_importances_,index=feats).sort_values().plot.barh()
#importando a biblioteca para plotar o gráfico de Matriz de Confusao

import scikitplot as skplt
#matriz de confusao - dados validacao

skplt.metrics.plot_confusion_matrix(valid['Churn'],pred_val,normalize = True)
pd.Series(predval.feature_importances_,index=feats).sort_values().plot.barh()