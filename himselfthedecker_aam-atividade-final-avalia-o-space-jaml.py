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
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
dfBase = pd.read_csv('../input/kobe-bryant-shot-selection/data.csv')

dfBase.dataframeName = 'kobe-bryant-shot-selection.csv'
print('Dataset Base - Sumário das Features')

print(dfBase.info())
headNumber = 5

print(f'Dataset Base - Primeiras {headNumber} linhas')

display(dfBase.head(headNumber))
print('Dataset Base - Estatistica descritiva')

display(dfBase.describe())
print(f'Dataset de treino tem {dfBase.shape[0]} linhas por {dfBase.shape[1]} colunas ({dfBase.shape[0] * dfBase.shape[1]} celulas)')
nonUsed, used = dfBase.groupby('shot_made_flag').size()

print(f'Das {dfBase.shape[0]} entradas no dataset, {nonUsed} foram de lances não convertidos e {used} foram de lances convertidos.')

print(f'Temos assim {round((used/dfBase.shape[0]) * 100,6)}% de lances que foram convertidos em pontos.')

print('---')



data = [go.Bar(

            x=['Lances Convertidos', 'Lances Ñ Convertidos'],

            y=[used, nonUsed],

            marker=dict(

                color=['rgba(38,222,47,0.8)','rgba(222,45,38,0.8)'])

    )]



py.iplot(data)
plotPerColumnDistribution(dfBase, len(dfBase), 5)
plotCorrelationMatrix(dfBase, 15)
plotScatterMatrix(dfBase, 25, 10)
## Ao trabalharmos com coordenadas polares, podemos definir com mais facilidade um ponto "zero", além de facilitar a visualização dos dados e remover features redundantes



dfPreprocess = dfBase.copy()



dfPreprocess['dist'] = np.sqrt(dfPreprocess['loc_x']**2 + dfPreprocess['loc_y']**2)



loc_x_zero = dfPreprocess['loc_x'] == 0

dfPreprocess['angle'] = np.array([0]*len(dfPreprocess))

dfPreprocess['angle'][~loc_x_zero] = np.arctan(dfPreprocess['loc_y'][~loc_x_zero] / dfPreprocess['loc_x'][~loc_x_zero])

dfPreprocess['angle'][loc_x_zero] = np.pi / 2 
## Tempo restante até o termino do período, em segundos.

dfPreprocess['remaining_time'] = dfPreprocess['minutes_remaining'] * 60 + dfPreprocess['seconds_remaining']
## Tempo decorrido de partida, gerado a partir do "period" e do "remaining_time"

dfPreprocess['match_elapsed_time'] = (dfPreprocess['period'] * 720) + (720 -  dfPreprocess['remaining_time'])
dfPreprocess.head(5)
dfPreprocess = dfPreprocess.drop(axis=1, columns=[

    ## Removido em detrimento a conversão para coordenadas polares

    'shot_zone_range', 

    'shot_zone_area', 

    'shot_distance',

    'lat', 

    'lon',

    'loc_x',

    'loc_y',

    'shot_zone_basic', 

    'shot_type', 

    

    # Não apresentam variação no arquivo - Ambos tem o mesmo valor como constante

    'team_name',  

    'team_id', 

    

    'matchup', ## Duplicidade desnormalizada de "Opponent"

    

    ## Identificador incremental sem valor para analise

    'game_event_id',

    'game_id',

    

    ## Não trataremos o dataset como série temporal

    'season', 

    'game_date',

    

    ## Eliminação por conversão - Deram origem a duas novas features

    'seconds_remaining',

    'minutes_remaining',

    'period',

    

])



## Promovido a Index por simples facilidade durante geração do arquivo de saída

dfPreprocess.set_index('shot_id', inplace=True)
## Reordenando as colunas por um simples fator de comodidade

dfPreprocess = dfPreprocess[['dist','angle', 'action_type', 'combined_shot_type', 'playoffs', 'match_elapsed_time', 'remaining_time', 'opponent', 'shot_made_flag']]
dfPreprocess
dfPreprocess.columns = [

    'dist',

    'angle',

    'action_type_cat', 

    'combined_shot_type_cat', 

    'playoffs_cat', 

    'match_elapsed_time', 

    'remaining_time', 

    'opponent_cat', 

    'target'

]
dfPreprocess
print(f'Antes - Preprocess tem {dfPreprocess.shape[0]} linhas por {dfPreprocess.shape[1]} colunas ({dfPreprocess.shape[0] * dfPreprocess.shape[1]} celulas)')

dfPreprocess.drop_duplicates()

print(f'Depois - Preprocess tem {dfPreprocess.shape[0]} linhas por {dfPreprocess.shape[1]} colunas ({dfPreprocess.shape[0] * dfPreprocess.shape[1]} celulas)')
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

        if f == 'target':

            level = 'binary'

        elif 'cat' in f or f == 'id':

            level = 'nominal'

        elif dfInput[f].dtype == float or dfInput[f].dtype == np.float64:

            level = 'interval'

        elif dfInput[f].dtype == int or dfInput[f].dtype == np.int64:

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
meta_preprocess = generateMetadata(dfPreprocess)
display(meta_preprocess)
## Simples demonstração para conferencia eventual

print('Metadados categoricos da base pré processada')

print(meta_preprocess[(meta_preprocess.level == 'nominal') & (meta_preprocess.keep)].index)
## Simples demonstração para conferencia eventual

print('Tipos e quantidade de features do dataset')

display(pd.DataFrame({'count' : meta_preprocess.groupby(['role', 'level'])['role'].size()}).reset_index())
def getMissingAttributes(dfInput):

    atributos_missing = []

    return_missing = []



    for f in dfInput.columns:

        missings = dfInput[f].isna().sum()

        if missings > 0:

            atributos_missing.append(f)

            missings_perc = missings/dfInput.shape[0]

            

            return_missing.append([f, missings, missings_perc])



            print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))

            



    print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))

    

    return pd.DataFrame(return_missing).rename(index=str, columns={0: "column_name", 1: "column_nulls", 2: "column_percentage"})
## Ao gerar nossa matriz de atributos faltantes, claramente vamos ignorar a coluna de alvo, que sabemos conter 5000 registros NaN (que desejamos prever ao termino da atividade)

missing = getMissingAttributes(dfPreprocess[meta_preprocess[(meta_preprocess.role != 'target')].index])

display(missing)
## limiar de remoção - 42.5% de nulos (Assumido como "horizonte de evento" em alguma aula passada e mantido desde então)

remove_threshold = 0.425
if (len(missing) > 0):

    columns_to_remove = np.array(missing.column_name[(missing.column_percentage >= remove_threshold)])

else:

    columns_to_remove = None
if (columns_to_remove != None):

    # removendo as colunas que tem muitos valores faltantes

    dfTreino = dfPreprocess.drop(columns_to_remove, axis=1)



    # atualiza os metadados para ter como referência

    meta_preprocess.loc[(columns_to_remove),'keep'] = False  



    # remove do frame de colunas com falta de dados as colunas que foram dropadas

    missing.drop(missing[(np.isin(missing.column_name, columns_to_remove))].index)
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
dfPreprocess = fillNullNumbers(dfPreprocess, meta_preprocess, missing, -1, 'Pré Processado')
def performOneHotEncoding(dfInput, meta_generic, dist_limit):

    v = meta_generic[(meta_generic.level == 'nominal') & (meta_generic.keep)].index

    display(v)

    for f in v:

        dist_values = dfInput[f].value_counts().shape[0]

        print('Atributo {} tem {} valores distintos'.format(f, dist_values))

        if (dist_values > dist_limit):

            print('Atributo {} tem mais de {} valores distintos e por isso será ignorado'.format(f, dist_limit))

            dfInput.drop([f], axis=1)

            v = v.drop([f])

        

    print('Antes do one-hot encoding tinha-se {} atributos'.format(dfInput.shape[1]))

    dfInput = pd.get_dummies(dfInput, columns=v, drop_first=True)

    print('Depois do one-hot encoding tem-se {} atributos'.format(dfInput.shape[1]))

    

    return dfInput
dfPreprocess = performOneHotEncoding(dfPreprocess, meta_preprocess, 200)
dfPreprocess.head(5)
from sklearn.preprocessing import MinMaxScaler



min_max_scaler = MinMaxScaler()



dfPreprocess[dfPreprocess.columns] = min_max_scaler.fit_transform(dfPreprocess[dfPreprocess.columns])
dfPreprocess.head(5)
# Models

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



# Feature Selection

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit, KFold, train_test_split, StratifiedKFold



# Auxiliary Scores

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score
def showDistribution(val_classes, targetName):

    nonUsed, used = pd.DataFrame(val_classes).groupby(targetName).size()

    print('---')

    print(f'Das {pd.DataFrame(val_classes).shape[0]} entradas no dataset, {nonUsed} foram de lances não convertidos e {used} foram de lances convertidos.')

    print(f'Temos assim {round((used/len(val_classes)) * 100,6)}% de lances que foram convertidos em pontos.')

    print('---')
def logisticRegression(X_Train, y_Train, X_Val, y_Val):



    model = LogisticRegression(solver='lbfgs')



    model.fit(X_Train, y_Train)



    y_pred_class = model.predict(X_Val)

    y_pred_proba = model.predict_proba(X_Val)



    recall = recall_score(y_Val, y_pred_class)

    accuracy = accuracy_score(y_Val, y_pred_class)

    logloss = log_loss(y_Val, y_pred_proba)

    precision =  precision_score(y_Val, y_pred_class)

    f1 = f1_score(y_Val, y_pred_class)



    print(f'Baseline - Regressão Logistica')

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return model, 'Baseline - Regressão Logistica'
def xGBClassifier(X_Train, y_Train, X_Val, y_Val, modelName, modelParams):



    if (modelParams == None):

        clf = XGBClassifier()

    else:

        clf = XGBClassifier(**modelParams)  

        modelName = modelName + ' - Parameters: ' + str(modelParams)

    

    clf.fit(X_Train, y_Train)



    y_pred_class = clf.predict(X_Val)

    y_pred_proba = clf.predict_proba(X_Val)



    recall = recall_score(y_Val, y_pred_class)

    accuracy = accuracy_score(y_Val, y_pred_class)

    logloss = log_loss(y_Val, y_pred_proba)

    precision =  precision_score(y_Val, y_pred_class)

    f1 = f1_score(y_Val, y_pred_class)



    print(modelName)

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return clf, modelName
def xGB_KFold(X, y, kfoldAmount, modelName, modelParams):



    if (modelParams == None):

        clf = XGBClassifier()

    else:

        clf = XGBClassifier(**modelParams)  

        modelName = modelName + ' - Parameters: ' + str(modelParams)

        

    clf_score = []

    iterator = 1

    

    for train_index, test_index in KFold(shuffle=True, n_splits=kfoldAmount, random_state=42).split(X):

        

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        

        print(f'Processando Fold {iterator}/{kfoldAmount}')

        

        clf.fit(X_train, y_train)

        y_pred_class = clf.predict(X_test)

        y_pred_proba = clf.predict_proba(X_test)

        

        

        recall = recall_score(y_test, y_pred_class)

        accuracy = accuracy_score(y_test, y_pred_class)

        logloss = log_loss(y_test, y_pred_proba)

        precision =  precision_score(y_test, y_pred_class)

        f1 = f1_score(y_test, y_pred_class)

        

        print(f'Fold {iterator}/{kfoldAmount} - Resultados')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'Log Loss: {round(logloss, 6)}')

        print(f'F1 Score: {round(f1, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_test, y_pred_class)))

        print('---')

        

        clf_score.append(logloss)

        

        iterator += 1

        

    print('Score Médio = ', round(np.array(clf_score).mean(), 6))

    return clf, modelName
def decisionTreeClassifier(X_Train, y_Train, X_Val, y_Val):



    clf = DecisionTreeClassifier()



    clf.fit(X_Train, y_Train)



    y_pred_class = clf.predict(X_Val)

    y_pred_proba = clf.predict_proba(X_Val)



    recall = recall_score(y_Val, y_pred_class)

    accuracy = accuracy_score(y_Val, y_pred_class)

    logloss = log_loss(y_Val, y_pred_proba)

    precision =  precision_score(y_Val, y_pred_class)

    f1 = f1_score(y_Val, y_pred_class)



    print(f'Decision Tree - Default Parameters')

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return clf, f'Decision Tree - Default Parameters'
def gridSearchKNN(X_Train, y_Train, X_Val, y_Val, k_range):

    clf=KNeighborsClassifier()

    param_grid=dict(n_neighbors=k_range)

    scores = ['neg_log_loss']

    for sc in scores:

        grid=GridSearchCV(clf,param_grid,cv=2,scoring=sc,n_jobs=-1)

        print("K-Nearest Neighbors - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)

        y_pred_proba = grid.predict_proba(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        logloss = log_loss(y_Val, y_pred_proba)

        precision =  precision_score(y_Val, y_pred_class)

        f1 = f1_score(y_Val, y_pred_class)



        print(f'KNN with recall-maxing hyperparameters - {grid.best_params_}')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'Log Loss: {round(logloss, 6)}')

        print(f'F1 Score: {round(f1, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

        print('---')

        

        return grid, f'KNN with recall-maxing hyperparameters - {grid.best_params_}'
def gridSearchSVC(X_Train, y_Train, X_Val, y_Val):

    svc=SVC()

    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 100, 1000]},

                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['neg_log_loss']

    for sc in scores:

        grid=GridSearchCV(svc,param_grid,cv=4,scoring=sc,n_jobs=-1)

        

        print("Support Vector Classifier - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        precision =  precision_score(y_Val, y_pred_class)

        f1 = f1_score(y_Val, y_pred_class)



        print(f'SVC with recall-maxing hyperparameters - {grid.best_params_}')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'F1 Score: {round(f1, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

        print('---')

        

        return grid, f'SVC with recall-maxing hyperparameters - {grid.best_params_}'
def gridSearchXGB(X_Train, y_Train, X_Val, y_Val, score):

    xgb=XGBClassifier(random_state = 0)

    ## Parametros para hiperparametrização tirados de um artigo do Medium, sugeridos como mais significativos e rápidos para classificação via XGB

    param_grid = [{'subsample': [0.3, 0.6, 0.9], 'colsample_bytree': [0.3, 0.6, 0.9], 'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75],'max_depth': [3, 7, 11, 15], 'gamma': [3, 6, 9]}]

    scores = [score]

    for sc in scores:

        grid=GridSearchCV(xgb,param_grid,cv=2,scoring=sc,n_jobs=-1)

        

        print("XGBoost - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)

        y_pred_proba = grid.predict_proba(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        logloss = log_loss(y_Val, y_pred_proba)

        precision =  precision_score(y_Val, y_pred_class)

        f1 = f1_score(y_Val, y_pred_class)



        print(f'XGBoost with {sc}-maxing hyperparameters - {grid.best_params_}')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'Log Loss: {round(logloss, 6)}')

        print(f'F1 Score: {round(f1, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

        print('---')

        

        return grid, f'XGBoost with {sc}-maxing hyperparameters - {grid.best_params_}'
def predictTestDataset(X_Test, y_Test, clfModel, clfName):

    y_pred_class = clfModel.predict(X_Test)

    y_pred_proba = clfModel.predict_proba(X_Test)



    recall = recall_score(y_Test, y_pred_class)

    accuracy = accuracy_score(y_Test, y_pred_class)

    logloss = log_loss(y_Test, y_pred_proba)

    precision =  precision_score(y_Test, y_pred_class)

    f1 = f1_score(y_Test, y_pred_class)



    print(clfName)

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Test, y_pred_class)))

    print('---')
def predictContestDataset(X_Test, clfModel, clfName): 

    

    print(clfName) 

    

    print('---') 

    

    y_pred_class = clfModel.predict(X_Test) 

    y_pred_proba = clfModel.predict_proba(X_Test) 

    

    pd_prediction = pd.DataFrame(y_pred_class) 

    pd_prediction.columns = ['target']

    

    showDistribution(pd_prediction, 'target') 

    

    return y_pred_class, y_pred_proba
print(dfPreprocess.shape)
def performSubSampling(sample_size_target, sample_size_non_target, dfInput, targetValue):

    target_indices = dfInput[dfInput.target == targetValue].index

    target_values = dfInput.loc[np.random.choice(activated_indices, sample_size, replace=False)]



    non_target_indices = dfInput[dfInput.target != targetValue].index

    non_target_values = dfInput.loc[np.random.choice(inactive_indices, sample_size_non_target, replace=False)]



    subsampled = pd.concat([target_values, non_target_values])



    subsampled.sort_index(inplace=True)

    

    return subsampled
## Como nosso objetivo final é submeter uma previsão dos valores nulos para a competição, vamos remover eles do dataset de onde tiraremos nossos dados de treino e validação

dfPredict = dfPreprocess[dfPreprocess['target'].isnull()]

dfPreprocess = dfPreprocess.dropna()
## Separamos os dados restantes em X (features) e y (alvo), para que possamos separar em treino e validação no futuro

X = dfPreprocess.drop(['target'], axis=1)

y = dfPreprocess['target']

y.columns = ['target']
## Nossa coluna "target" do dataset da competição é composta inteiramente de nulos e, sendo assim, pode ser descartada sem problemas

X_predict = dfPredict.drop(['target'], axis=1)
print(X.shape)

print(y.shape)
print(X_predict.shape)
## Como era de se esperar (e foi mostrado antes), nosso dataset é razoavelmente balanceado no que tange a distribuição das classes

showDistribution(y, 'target')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
## Graças a estratificação, mantemos a mesma proporção entre classes na massa de treino, com 70% dos dados originais para treino, 30% para validação do modelo e 5k fixo para submissão

showDistribution(y_train, 'target')
# Baseline - Regressão Logistica

## Como sempre, Regressão Logistica é nossa metrica mais básica a ser batida

logRegModel, logRegName = logisticRegression(X_train, y_train, X_val, y_val)
# Modelos baseados em arvore

## Na minha limitada experiencia, algoritmos baseados em arvore que usam gradiente (como o XGBoost) apresentam resultados bom para classificação

## No que tange aos hiperparametros, o "Preset" é o resultado da hiperparametrização do exercicio anterior enquanto que o "Grid Search Outputted" veio do algoritmo de grid search implementado nesse notebook, mas que foi comentado após a execução em virtude da demora

xgbPureModel, xgbPureName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Base', None)

xgbPresetModel, xgbPresetName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Preset', {'n_estimator':400, 'learning_rate' : 0.5,'random_state' : 0,'max_depth':70,'objective':"binary:logistic",'subsample':.8,'min_child_weig':6,'colsample_bytr':.8,'scale_pos_weight':1.6, 'gamma':10, 'reg_alph':8, 'reg_lambda':1})

xgbHyperParametrizedModel, xgbHyperParametrizedName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Grid Search Outputted',{'colsample_bytree': 0.6, 'gamma': 9, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.6, 'random_state': 42})



## Usamos também Decision Tree para comparar os resultados

decTreeModel, decTreeName = decisionTreeClassifier(X_train, y_train, X_val, y_val)
# Otimizações via GridSearch - DEMORAM PARA EXECUTAR

## Como dito acima, os casos onde usamos hiperparametrização vieram desses algoritmos; Infelizmente executar eles toda vez é inviavel devido a demora

# xgbGSModel, xgbGSName = gridSearchXGB(X_train, y_train, X_val, y_val,'neg_log_loss')

# knnModel, knnName = gridSearchKNN(X_train, y_train, X_val, y_val, list(range(1,20)))
# KFolding - Usando separação em treino e validação internamente

## No caso do K-Fold, usamos a massa de dados toda para podermos quebrar nos K segmentos utilizados pelo algoritmo e testar uns contra os outros

showDistribution(y, 'target')

xgbGSModel, xgbGSName = xGB_KFold(X, y, 10, 'XGBoost - KFolded',

                                  {'colsample_bytree': 0.6, 

                                   'gamma': 9, 

                                   'learning_rate': 0.01, 

                                   'max_depth': 7, 

                                   'n_estimators': 500, 

                                   'subsample': 0.6, 

                                   'random_state': 42

                                  })
## Considerando o resultado dos modelos acima, acabei por escolher o modelo saido do K-Fold, visto que ele entrega uma acurácia satisfatória e o melhor Logloss entre os testados

contest_prediction, contest_prediction_probability = predictContestDataset(X_predict, xgbGSModel, xgbGSName)
## Finalmente, vamos gerar o arquivo de submissão da competição e torcer por resultados bons!

sample    = pd.read_csv('../input/kobe-bryant-shot-selection/sample_submission.csv', low_memory=False)

sample.shot_made_flag = contest_prediction_probability

sample.shot_made_flag = 1 - sample.shot_made_flag

sample.to_csv("submission.csv", float_format='%.6f', index=False)