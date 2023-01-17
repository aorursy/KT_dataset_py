import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')
dfTreino = pd.read_csv('../input/train.csv')

dfTeste = pd.read_csv('../input/test.csv')
# Podemos usar a coluna "id" como o índice dos Datasets, sem nenhum prejuizo

dfTreino.set_index('id', inplace=True)

dfTeste.set_index('id', inplace=True)
headNumber = 5

print(f'Dataset de treino - Primeiras {headNumber} linhas')

display(dfTreino.head(headNumber))



print(f'Dataset de teste - Primeiras {headNumber} linhas')

display(dfTeste.head(headNumber))
print('Dataset de treino - Estatistica descritiva')

display(dfTreino.describe())



print('Dataset de teste - Estatistica descritiva')

display(dfTeste.describe())
print('Dataset de treino - Sumário das Features')

print(dfTreino.info())

print('---')

print('Dataset de treino - Sumário das Features')

print(dfTeste.info())
print(f'Dataset de treino tem {dfTreino.shape[0]} linhas por {dfTreino.shape[1]} colunas ({dfTreino.shape[0] * dfTreino.shape[1]} celulas)')

print(f'Dataset de teste tem {dfTeste.shape[0]} linhas por {dfTeste.shape[1]} colunas ({dfTeste.shape[0] * dfTeste.shape[1]} celulas)')
nonUsed, used = dfTreino.groupby('target').size()

print(f'Das {dfTreino.shape[0]} entradas no dataset, {nonUsed} foram de casos onde não foi acionado o seguro e {used} foram caso onde houve acionamento')

print(f'Temos assim {round((used/nonUsed) * 100,6)}% de ocorrencias em que o resultado (1 ou "houve acionamento") desejamos prever')
print(f'Antes - Treino tem {dfTreino.shape[0]} linhas por {dfTreino.shape[1]} colunas ({dfTreino.shape[0] * dfTreino.shape[1]} celulas)')

dfTreino.drop_duplicates()

print(f'Depois - Treino tem {dfTreino.shape[0]} linhas por {dfTreino.shape[1]} colunas ({dfTreino.shape[0] * dfTreino.shape[1]} celulas)')

print('---')

print(f'Antes - Teste tem {dfTeste.shape[0]} linhas por {dfTeste.shape[1]} colunas ({dfTeste.shape[0] * dfTeste.shape[1]} celulas)')

dfTeste.drop_duplicates()

print(f'Depois - Teste tem {dfTeste.shape[0]} linhas por {dfTeste.shape[1]} colunas ({dfTeste.shape[0] * dfTeste.shape[1]} celulas)')
def generateMetadata(dfInput):

    data = []

    for f in dfInput.columns:

        # definindo o uso (entre rótulo, id e atributos)

        if f == 'target':

            role = 'target' # rótulo

        elif f == 'id':

            role = 'id'

        else:

            role = 'input' # atributos



        # definindo o tipo do dado

        if 'bin' in f or f == 'target':

            level = 'binary'

        elif 'cat' in f or f == 'id':

            level = 'nominal'

        elif dfInput[f].dtype == np.float64:

            level = 'interval'

        elif dfInput[f].dtype == np.int64:

            level = 'ordinal'



        # mantem keep como verdadeiro pra tudo, exceto id

        keep = True

        if f == 'id':

            keep = False



        # cria o tipo de dado

        dtype = dfInput[f].dtype



        # cria dicionário de metadados

        f_dict = {

            'varname': f,

            'role': role,

            'level': level,

            'keep': keep,

            'dtype': dtype

        }

        data.append(f_dict)



    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

    meta.set_index('varname', inplace=True)

    

    return meta
meta_train = generateMetadata(dfTreino)

meta_test = generateMetadata(dfTeste)
display(meta_train)

display(meta_test)
print('Metadados categoricos da base de treino')

print(meta_train[(meta_train.level == 'nominal') & (meta_train.keep)].index)

print('---')

print('Metadados categoricos da base de teste')

print(meta_test[(meta_test.level == 'nominal') & (meta_test.keep)].index)
print('Tipos e quantidade de features do dataset de treino')

display(pd.DataFrame({'count' : meta_train.groupby(['role', 'level'])['role'].size()}).reset_index())



print('Tipos e quantidade de features do dataset de teste')

display(pd.DataFrame({'count' : meta_test.groupby(['role', 'level'])['role'].size()}).reset_index())
def getMissingAttributes(dfInput):

    atributos_missing = []

    return_missing = []



    for f in dfInput.columns:

        missings = dfInput[dfInput[f] == -1][f].count()

        if missings > 0:

            atributos_missing.append(f)

            missings_perc = missings/dfInput.shape[0]

            

            return_missing.append([f, missings, missings_perc])



            print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))

            



    print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))

    

    return pd.DataFrame(return_missing).rename(index=str, columns={0: "column_name", 1: "column_nulls", 2: "column_percentage"})
missing_Train = getMissingAttributes(dfTreino)

display(missing_Train)
missing_Test = getMissingAttributes(dfTeste)

display(missing_Test)
# limiar de remoção - 42.5% de nulos

remove_threshold = 0.425
columns_to_remove = np.array(missing_Train.column_name[(missing_Train.column_percentage >= remove_threshold)])
# removendo as colunas que tem muitos valores faltantes

dfTreino = dfTreino.drop(columns_to_remove, axis=1)

dfTeste = dfTeste.drop(columns_to_remove, axis=1)



# atualiza os metadados para ter como referência

meta_train.loc[(columns_to_remove),'keep'] = False  

meta_test.loc[(columns_to_remove),'keep'] = False



# remove do frame de colunas com falta de dados as colunas que foram dropadas

missing_Train.drop(missing_Train[(np.isin(missing_Train.column_name, columns_to_remove))].index)
# Usa ou moda ou média para preencher os valores "vazios" que nosso dataset contem, baseado nos metadados do mesmo

def fillNullNumbers(dfInput, dfMetadata, dfMissing, missing_default, label):



    from sklearn.impute import SimpleImputer



    media_imp = SimpleImputer(missing_values=missing_default, strategy='mean')

    moda_imp = SimpleImputer(missing_values=missing_default, strategy='most_frequent')



    for index,row in dfMissing.iterrows():

        columnName = row['column_name']

        columnType = dfMetadata.level[(dfMetadata.index == columnName)][0]



        if (columnType == 'interval'):

            imputerToUse = media_imp

            imputerString = 'media_imp'

        elif (columnType == 'ordinal'):

            imputerToUse = moda_imp

            imputerString = 'moda_imp'

        else:

            imputerToUse = None

            imputerString = None



        if (imputerToUse != None):

            dfInput[columnName] = imputerToUse.fit_transform(dfInput[[columnName]]).ravel()

            print(f"{label} - Preenchida coluna {columnName}, cujo tipo é {columnType}, usando o Imputer {imputerString}")



    return dfInput
dfTreino = fillNullNumbers(dfTreino, meta_train, missing_Train, -1, 'Treino')

print('---')

dfTeste = fillNullNumbers(dfTeste, meta_test, missing_Train, -1, 'Teste')
def performOneHotEncoding(dfTrain, dfTest, meta_generic):

    v = meta_generic[(meta_generic.level == 'nominal') & (meta_generic.keep)].index



    for f in v:

        dist_values = dfTrain[f].value_counts().shape[0]

        print('Atributo {} tem {} valores distintos'.format(f, dist_values))

        

    print('Antes do one-hot encoding tinha-se {} atributos'.format(dfTrain.shape[1]))

    dfTrain = pd.get_dummies(dfTrain, columns=v, drop_first=True)

    print('Depois do one-hot encoding tem-se {} atributos'.format(dfTrain.shape[1]))



    dfTest = pd.get_dummies(dfTest, columns=v, drop_first=True)

    missing_cols = set( dfTrain.columns ) - set( dfTest.columns )

    for c in missing_cols:

        dfTest[c] = 0



    dfTrain, dfTest = dfTrain.align(dfTest, axis=1)

    

    return dfTrain, dfTest
dfTreino, dfTeste = performOneHotEncoding(dfTreino, dfTeste, meta_train)
from sklearn.preprocessing import MinMaxScaler



min_max_scaler = MinMaxScaler()



dfTreino[dfTreino.columns] = min_max_scaler.fit_transform(dfTreino[dfTreino.columns])

dfTeste[dfTeste.columns] = min_max_scaler.fit_transform(dfTeste[dfTeste.columns])
print(dfTreino.shape)

print(dfTeste.shape)
from sklearn.model_selection import train_test_split



X = dfTreino.drop(['id', 'target'], axis=1)

y = dfTreino['target']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



model = LogisticRegression(solver='lbfgs')



model.fit(X_train, y_train)

y_pred_class = model.predict(X_val)

y_pred_proba = model.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Regressão Logistica / Baseline')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



clf = DecisionTreeClassifier(max_depth=80,min_samples_leaf=1)

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_val)

y_pred_proba = clf.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Decision Tree with 80 layers and at least 1 leaves')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



clf = RandomForestClassifier(n_estimators=1, max_depth=80, min_samples_leaf=1, random_state=0)

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_val)

y_pred_proba = clf.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Random Forest with 1 estimator, 80 layers and at least 1 leaf per layer')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))
'''

from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



svm = LinearSVC()

clf = CalibratedClassifierCV(svm, cv=ShuffleSplit())

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_val)

y_pred_proba = clf.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Regressão Logistica / Baseline')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))

'''
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



clf = SGDClassifier(loss='modified_huber', shuffle=True, random_state=0, max_iter=1000, tol=1e-3)

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_val)

y_pred_proba = clf.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Regressão Logistica / Baseline')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))
# Rodar depois, fora do Kaggle - Provavel que vá demorar 6h+

'''

from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score



svm = SVC(gamma=2, C=1)

clf = CalibratedClassifierCV(svm, cv=ShuffleSplit())

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_val)

y_pred_proba = clf.predict_proba(X_val)



recall = recall_score(y_val, y_pred_class)

accuracy = accuracy_score(y_val, y_pred_class)

logloss = log_loss(y_val, y_pred_proba)

precision =  precision_score(y_val, y_pred_class)

f1 = f1_score(y_val, y_pred_class)



print(f'Regressão Logistica / Baseline')

print('---')

print(f'Acurácia: {round(accuracy, 6)}%')

print(f'Recall: {round(recall, 6)}%')

print(f'Precisão: {round(precision, 6)}%')

print(f'Log Loss: {round(logloss, 6)}')

print(f'F1 Score: {round(f1, 6)}')



print('---')

print('Matriz de Confusão')

display(pd.DataFrame(confusion_matrix(y_val, y_pred_class)))

'''