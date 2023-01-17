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


import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn as sk
import seaborn as sns
import datetime
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#Reading the data
rdo_1 = pd.read_csv("/kaggle/input/crime-data-in-brazil/RDO_1.csv", low_memory = False)
rdo_2 = pd.read_csv("/kaggle/input/crime-data-in-brazil/RDO_2.csv", low_memory = False)
rdo_3 = pd.read_csv("/kaggle/input/crime-data-in-brazil/RDO_3.csv", low_memory = False)

#dropping extra column
rdo_1.drop('Unnamed: 30', axis = 1, inplace = True)
rdo_2.drop('Unnamed: 30', axis = 1, inplace = True)
rdo_3.drop('Unnamed: 30', axis = 1, inplace = True)

print (rdo_1.shape)
print (rdo_1.columns)
print (rdo_1.dtypes)

print (rdo_2.shape)
print (rdo_2.columns)
print (rdo_2.dtypes)

print (rdo_3.shape)
print (rdo_3.columns)
print (rdo_3.dtypes)

frames = [rdo_1, rdo_2, rdo_3]

data = pd.concat(frames, ignore_index=True)

print (data.shape)
print (data.columns)
print (data.dtypes)
data['IDADE_PESSOA'] = pd.to_numeric(data['IDADE_PESSOA'], errors='coerce')
data['NUMERO_LOGRADOURO'] = pd.to_numeric(data['NUMERO_LOGRADOURO'], errors='coerce')

data['LONGITUDE'] = pd.to_numeric(data['LONGITUDE'], errors='coerce')
data['LATITUDE'] = pd.to_numeric(data['LATITUDE'], errors='coerce')

data['DATA_OCORRENCIA_BO'] = pd.to_datetime(data['DATA_OCORRENCIA_BO'], errors='coerce')
data['HORA_OCORRENCIA_BO'] = pd.to_datetime(data['HORA_OCORRENCIA_BO'], format='%H:%M', errors='coerce').dt.time

data['CIDADE'] = pd.Categorical(data['CIDADE'])
data['NOME_SECCIONAL_CIRC'] = pd.Categorical(data['NOME_SECCIONAL_CIRC'])
data['NOME_DELEGACIA_CIRC'] = pd.Categorical(data['NOME_DELEGACIA_CIRC'])
data['DESCR_TIPO_BO'] = pd.Categorical(data['DESCR_TIPO_BO'])
data['RUBRICA'] = pd.Categorical(data['RUBRICA'])
data['DESCR_CONDUTA'] = pd.Categorical(data['DESCR_CONDUTA'])
data['DESDOBRAMENTO'] = pd.Categorical(data['DESDOBRAMENTO'])
data['DESCR_TIPOLOCAL'] = pd.Categorical(data['DESCR_TIPOLOCAL'])
data['DESCR_SUBTIPOLOCAL'] = pd.Categorical(data['DESCR_SUBTIPOLOCAL'])
data['LOGRADOURO'] = pd.Categorical(data['LOGRADOURO'])
data['DESCR_TIPO_PESSOA'] = pd.Categorical(data['DESCR_TIPO_PESSOA'])
data['SEXO_PESSOA'] = pd.Categorical(data['SEXO_PESSOA'])
data['COR_CUTIS'] = pd.Categorical(data['COR_CUTIS'])
print (data.dtypes)
print(data['FLAG_VITIMA_FATAL'].unique())
print(data['FLAG_STATUS'].unique())
print(data['SEXO_PESSOA'].unique())
print(data['COR_CUTIS'].unique())
bool = (data['FLAG_VITIMA_FATAL'] != 'N') & (data['FLAG_VITIMA_FATAL'] != 'S')
data['FLAG_VITIMA_FATAL'].loc[bool] = np.NaN
print(data['FLAG_VITIMA_FATAL'].value_counts())

bool = (data['SEXO_PESSOA'] != 'M') & (data['SEXO_PESSOA'] != 'F')
data['SEXO_PESSOA'].loc[bool] = np.NaN
print(data['SEXO_PESSOA'].value_counts())

#'Preta' 'Parda' 'Branca' 'Amarela' 'Outros' 'Vermelha'
data['COR_CUTIS'] =data['COR_CUTIS'].str.strip()
bool = (data['COR_CUTIS'] != 'Preta') & (data['COR_CUTIS'] != 'Parda') & (data['COR_CUTIS'] != 'Branca') & (data['COR_CUTIS'] != 'Amarela') & (data['COR_CUTIS'] != 'Outros') & (data['COR_CUTIS'] != 'Vermelha')
data['COR_CUTIS'].loc[bool] = np.NaN
print(data['COR_CUTIS'].value_counts())
plt.bar(data['ANO_BO'].unique(), data['ANO_BO'].value_counts(), align='center', alpha=0.5)
plt.ylabel('Quantity')
plt.title('Quantity of cases per year')

plt.show()
bool_logradouro = data['LOGRADOURO'].notnull()

data = data[bool_logradouro]

le = preprocessing.LabelEncoder()
data['CAT_LOGRADOURO'] = le.fit_transform(data['LOGRADOURO'])

print (data.shape)
print (data.columns)
#missing values
data['FLAG_VITIMA_FATAL'].value_counts()

bool_vitimas = data['FLAG_VITIMA_FATAL'].notnull()

vitimas = data[bool_vitimas]

print(data.shape)
print(vitimas.shape)

bool_values = vitimas['IDADE_PESSOA'].notnull()

scatter = vitimas[bool_values]

plt.scatter(scatter['CAT_LOGRADOURO'], scatter['IDADE_PESSOA'])
plt.xlabel('Logradouro')
plt.ylabel('Idade')
plt.show()
bool_values = vitimas['LATITUDE'].notnull()
scatter = vitimas[bool_values]
bool_values = scatter['LONGITUDE'].notnull()
scatter = scatter[bool_values]

plt.scatter(scatter['LATITUDE'], scatter['LONGITUDE'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
bool_values = data['COR_CUTIS'].notnull()

bar = data[bool_values]
bar = pd.DataFrame(bar['COR_CUTIS'].value_counts())
bar['COR'] = bar.index

plt.bar(bar['COR'], bar['COR_CUTIS'], align='center', alpha=0.5)
plt.ylabel('Quantity')
plt.title('Quantity of cases per skin color')

plt.show()
bool_values = data['COR_CUTIS'].notnull()
scatter = data[bool_values]

plt.scatter(scatter['CAT_LOGRADOURO'], scatter['COR_CUTIS'])
plt.xlabel('Logradouro')
plt.ylabel('Cor da Pele')
plt.show()
print(data['IDADE_PESSOA'].max())
print(data['IDADE_PESSOA'].min())

bool = (data['IDADE_PESSOA'] > 100) | (data['IDADE_PESSOA'] < 10)
data['IDADE_PESSOA'].loc[bool] = np.NaN
data['IDADE_PESSOA'].value_counts()
bool_values = data['IDADE_PESSOA'].notnull()

box = data[bool_values]

fig1, ax1 = plt.subplots()
ax1.set_title('Idade')
ax1.boxplot(box['IDADE_PESSOA'])
dd=pd.melt(box,id_vars=['COR_CUTIS'],value_vars=['IDADE_PESSOA'])
sns.boxplot(x='COR_CUTIS',y='value',data=dd)
bool_values = data['HORA_OCORRENCIA_BO'].notnull()

bar = data[bool_values]

list = pd.Series.tolist(bar['HORA_OCORRENCIA_BO'])

new_list = []
for item in list:
    new_list.append(item.hour)
    
plot = pd.DataFrame(new_list, columns = ['QTD'])

plot = pd.DataFrame(plot['QTD'].value_counts())

plot['HORA'] = plot.index

plt.bar(plot['HORA'], plot['QTD'], align='center', alpha=0.5)
plt.ylabel('Quantidade')
plt.title('Quantidade de ocorrências por hora')

plt.show()
#Quantidade por mês

bool_values = data['DATA_OCORRENCIA_BO'].notnull()

bar = data[bool_values]

list = pd.Series.tolist(bar['DATA_OCORRENCIA_BO'])

new_list = []
for item in list:
    new_list.append(item.month)
    
plot = pd.DataFrame(new_list, columns = ['QTD'])

plot = pd.DataFrame(plot['QTD'].value_counts())

plot['MES'] = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

plt.bar(plot['MES'], plot['QTD'], align='center', alpha=0.5)
plt.ylabel('Quantidade')
plt.title('Quantidade de ocorrências por mês')

plt.show()
#variação mensal por ano

bool_values = data['DATA_OCORRENCIA_BO'].notnull()

bar = data[bool_values]

bar = pd.DataFrame(bar['DATA_OCORRENCIA_BO'].value_counts())
bar['DATA'] = bar.index
bar['DATA'] = pd.to_datetime(bar['DATA'], errors='coerce')
bar['ATRASO'] = bar['DATA'].apply (lambda x: (x - x.now()).days)

bar.reset_index(drop=True, inplace=True)

plt.bar(bar['ATRASO'], bar['DATA_OCORRENCIA_BO'], align='center', alpha=0.5)
plt.ylabel('Quantidade')
plt.title('variação mensal por ano')

plt.show()
data.columns
#quantidade de vitimas fatais por cor de pele
#quantidade de vitimas fatais por hora
#quantidade de vitimas fatais por idade
#quantidade de vitimas fatais por sexo
#quantidade de vitimas fatais por municipio
bool_values = data['FLAG_VITIMA_FATAL'].notnull()

vitima = data[bool_values]

print(vitima.shape)
print(vitima['FLAG_VITIMA_FATAL'].value_counts())
#dropping useless columns
#['ID_DELEGACIA', 'NOME_DEPARTAMENTO', 'NOME_SECCIONAL', 'NOME_DELEGACIA',
#       'CIDADE', 'ANO_BO', 'NUM_BO', 'NOME_DEPARTAMENTO_CIRC',
#       'NOME_SECCIONAL_CIRC', 'NOME_DELEGACIA_CIRC', 'NOME_MUNICIPIO_CIRC',
#       'DESCR_TIPO_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO',
#       'DATAHORA_COMUNICACAO_BO', 'FLAG_STATUS', 'RUBRICA', 'DESCR_CONDUTA',
#       'DESDOBRAMENTO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 'LOGRADOURO',
#       'NUMERO_LOGRADOURO', 'LATITUDE', 'LONGITUDE', 'DESCR_TIPO_PESSOA',
#       'FLAG_VITIMA_FATAL', 'SEXO_PESSOA', 'IDADE_PESSOA', 'COR_CUTIS']
vitima.drop('ID_DELEGACIA', axis = 1, inplace = True)
vitima.drop('NOME_DEPARTAMENTO', axis = 1, inplace = True)
vitima.drop('NOME_SECCIONAL', axis = 1, inplace = True)
vitima.drop('NOME_DELEGACIA', axis = 1, inplace = True)
vitima.drop('NUM_BO', axis = 1, inplace = True)
vitima.drop('NOME_DEPARTAMENTO_CIRC', axis = 1, inplace = True)
vitima.drop('NOME_MUNICIPIO_CIRC', axis = 1, inplace = True)
vitima.drop('DATAHORA_COMUNICACAO_BO', axis = 1, inplace = True)
vitima.drop('NUMERO_LOGRADOURO', axis = 1, inplace = True)
vitima.drop('CAT_LOGRADOURO', axis = 1, inplace = True)
#one-hot encoding categorical variables

dmCidade = pd.get_dummies(vitima['CIDADE'])
dmNomeSceccional = pd.get_dummies(vitima['NOME_SECCIONAL_CIRC'])
dmNomeDelegacia = pd.get_dummies(vitima['NOME_DELEGACIA_CIRC'])
dmDescBO = pd.get_dummies(vitima['DESCR_TIPO_BO'])
dmRubrica = pd.get_dummies(vitima['RUBRICA'])
dmConduta = pd.get_dummies(vitima['DESCR_CONDUTA'])
dmDesdobramento = pd.get_dummies(vitima['DESDOBRAMENTO'])
dmTipoLocal = pd.get_dummies(vitima['DESCR_TIPOLOCAL'])
dmSubTipoLocal = pd.get_dummies(vitima['DESCR_SUBTIPOLOCAL'])
dmTipoPessoa = pd.get_dummies(vitima['DESCR_TIPO_PESSOA'])
dmSexoPessoa = pd.get_dummies(vitima['SEXO_PESSOA'])
dmCor = pd.get_dummies(vitima['COR_CUTIS'])
print(dmCidade.shape)
print(dmNomeSceccional.shape)
print(dmNomeDelegacia.shape)
print(dmDescBO.shape)
print(dmRubrica.shape)
print(dmConduta.shape)
print(dmDesdobramento.shape)
print(dmTipoLocal.shape)
print(dmSubTipoLocal.shape)
print(dmTipoPessoa.shape)
print(dmSexoPessoa.shape)
print(dmCor.shape)
vitima = pd.concat([vitima, dmCidade, dmNomeSceccional, dmDescBO, dmRubrica, dmConduta, dmDesdobramento, 
                    dmTipoLocal, dmTipoPessoa, dmSexoPessoa, dmCor], axis=1)
vitima.drop('CIDADE', axis = 1, inplace = True)
vitima.drop('NOME_SECCIONAL_CIRC', axis = 1, inplace = True)
vitima.drop('NOME_DELEGACIA_CIRC', axis = 1, inplace = True)
vitima.drop('DESCR_TIPO_BO', axis = 1, inplace = True)
vitima.drop('RUBRICA', axis = 1, inplace = True)
vitima.drop('DESCR_CONDUTA', axis = 1, inplace = True)
vitima.drop('DESDOBRAMENTO', axis = 1, inplace = True)
vitima.drop('DESCR_TIPOLOCAL', axis = 1, inplace = True)
vitima.drop('DESCR_SUBTIPOLOCAL', axis = 1, inplace = True)
vitima.drop('LOGRADOURO', axis = 1, inplace = True)
vitima.drop('DESCR_TIPO_PESSOA', axis = 1, inplace = True)
vitima.drop('SEXO_PESSOA', axis = 1, inplace = True)
vitima.drop('COR_CUTIS', axis = 1, inplace = True)
print(vitima.shape)
vitima.head(5)
enc = preprocessing.OrdinalEncoder()

vitima['FLAG_STATUS'] = enc.fit_transform(vitima['FLAG_STATUS'].values.reshape(-1, 1))
vitima['FLAG_VITIMA_FATAL'] = enc.fit_transform(vitima['FLAG_VITIMA_FATAL'].values.reshape(-1, 1))
#'ANO_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 'FLAG_STATUS', 'LATITUDE', 'LONGITUDE', 'FLAG_VITIMA_FATAL', 'IDADE_PESSOA'
print(vitima['ANO_BO'].isnull().value_counts())
print(vitima['DATA_OCORRENCIA_BO'].isnull().value_counts())
print(vitima['HORA_OCORRENCIA_BO'].isnull().value_counts())
print(vitima['FLAG_STATUS'].isnull().value_counts())
print(vitima['LATITUDE'].isnull().value_counts())
print(vitima['LONGITUDE'].isnull().value_counts())
print(vitima['FLAG_VITIMA_FATAL'].isnull().value_counts())
print(vitima['IDADE_PESSOA'].isnull().value_counts())
bool_values = vitima['HORA_OCORRENCIA_BO'].notnull()

vitima = vitima[bool_values]

bool_values = vitima['LATITUDE'].notnull()

vitima = vitima[bool_values]

bool_values = vitima['IDADE_PESSOA'].notnull()

vitima = vitima[bool_values]
vitima.reset_index(inplace=True, drop = True)

list = pd.Series.tolist(vitima['DATA_OCORRENCIA_BO'])

months = []
days = []
for item in list:
    months.append(item.month)
    days.append(item.day)

list = pd.Series.tolist(vitima['HORA_OCORRENCIA_BO'])

hours = []
for item in list:
    hours.append(item.hour)

d = {'MES': months, 'DIA': days, 'HORA': hours}
df = pd.DataFrame(d)

vitima = pd.concat([vitima,df], axis = 1, sort=False)

vitima.drop('DATA_OCORRENCIA_BO', axis = 1, inplace = True)
vitima.drop('HORA_OCORRENCIA_BO', axis = 1, inplace = True)
print(vitima.columns)
vitima['FLAG_VITIMA_FATAL'].value_counts()
y = vitima['FLAG_VITIMA_FATAL']
X = vitima.drop('FLAG_VITIMA_FATAL', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#Downsampling majority class

train = pd.concat([X_train, y_train], axis = 1)

vitima_majority = train[train['FLAG_VITIMA_FATAL']==0]
vitima_minority = train[train['FLAG_VITIMA_FATAL']==1]


vitima_majority_downsampled = resample(vitima_majority, replace=False, n_samples=3581, random_state=100)
 
vitima_downsampled = pd.concat([vitima_majority_downsampled, vitima_minority])
 
print(vitima_downsampled['FLAG_VITIMA_FATAL'].value_counts())

y_train = vitima_downsampled['FLAG_VITIMA_FATAL']
X_train = vitima_downsampled.drop('FLAG_VITIMA_FATAL', axis = 1)
#kNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=100)

knn_model.fit(X_train,y_train)

y_pred = knn_model.predict(X_test)
print("kNN Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))
#Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train,y_train)

y_pred = log_reg_model.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_model = LDA(n_components=1)

lda_model.fit(X_train, y_train)

y_pred = lda_model.predict(X_test)
print("LDA Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))
#Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train,y_train)

y_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))
#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators =100, random_state = 100)

rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))

#SVM
from sklearn import svm

svm_model = svm.SVC(kernel='linear')

svm_model.fit(X_train,y_train)

y_pred = svm_model.predict(X_test)
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))

#Neural Network
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(hidden_layer_sizes=(30,30,30), random_state = 100)

nn_model.fit(X_train,y_train)

y_pred = nn_model.predict(X_test)
print("Neural Network Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = [0,1])
print("     Predito")
print("Real", cm[0][0], cm[0][1])
print("     ", cm[1][0], cm[1][1])
#false negative rate (FNR)
print("False Negative Rate:",cm[1][0]/(cm[1][0]+cm[1][1]))