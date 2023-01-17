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



from numba import jit
@jit

def eval_gini(y_true, y_prob):

    """

    Original author CPMP : https://www.kaggle.com/cpmpml

    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
dfTotal = pd.read_csv('../input/avaliacao.csv')
headNumber = 5

print(f'Dataset de treino - Primeiras {headNumber} linhas')

display(dfTotal.head(headNumber))
print('Dataset - Estatistica descritiva')

display(dfTotal.describe())
print('Dataset - Sumário das Features')

print(dfTotal.info())
print(f'Dataset tem {dfTotal.shape[0]} linhas por {dfTotal.shape[1]} colunas ({dfTotal.shape[0] * dfTotal.shape[1]} celulas)')
dfTotal = dfTotal.drop(columns=['game_event_id',  

'game_id',

'team_id',  

'team_name',

'period',

'playoffs',   

'shot_id'])
dfComplete = dfTotal.dropna(subset=['shot_made_flag'])

dfComplete.minutes_remaining = dfComplete.minutes_remaining*60

dfComplete.seconds_remaining = pd.to_numeric(dfComplete.seconds_remaining)

dfComplete
dfComplete['total_seconds_remaining'] = dfComplete.seconds_remaining + dfComplete.minutes_remaining

dfComplete
dfComplete.drop(columns=['minutes_remaining', 'seconds_remaining'],inplace=True)
nonScore, score = dfComplete.groupby('shot_made_flag').size()

print(f'Das {dfComplete.shape[0]} entradas no dataset, {nonScore} erraram,  e {score} foram caso onde houve cesta')

print(f'Temos assim {round((score/nonScore) * 100,6)}% de ocorrencias em que o resultado desejamos prever')
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

missing_Train = getMissingAttributes(dfComplete)

display(missing_Train)
def generateMetadata(dfInput):

    data = []

    level = ''

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
# Models

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



# Feature Selection

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit



# Auxiliary Scores

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score
def showDistribution(val_classes):

    nonUsed, used = pd.DataFrame(val_classes).groupby('shot_made_flag').size()

    print('---')

    print(f'Das {pd.DataFrame(val_classes).shape[0]} entradas no dataset, {nonUsed} foram cestas')

    print(f'Temos assim {round((used/len(val_classes)) * 100,6)}% de cestas que desejamos prever')

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

    gini = eval_gini(y_Val, y_pred_class)



    print(f'Baseline - Regressão Logistica')

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')

    print(f'Gini: {round(gini, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return model, 'Baseline - Regressão Logistica'
# n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 70, random_state = 0 - Recall: 0.057621%



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

    gini = eval_gini(y_Val, y_pred_class)

    precision =  precision_score(y_Val, y_pred_class)

    f1 = f1_score(y_Val, y_pred_class)



    print(modelName)

    print('---')

    print(f'Acurácia: {round(accuracy, 6)}%')

    print(f'Recall: {round(recall, 6)}%')

    print(f'Precisão: {round(precision, 6)}%')

    print(f'Log Loss: {round(logloss, 6)}')

    print(f'F1 Score: {round(f1, 6)}')

    print(f'Gini: {round(gini, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return clf, modelName
def decisionTreeClassifier(X_Train, y_Train, X_Val, y_Val):



    clf = DecisionTreeClassifier()



    clf.fit(X_Train, y_Train)



    y_pred_class = clf.predict(X_Val)

    y_pred_proba = clf.predict_proba(X_Val)



    recall = recall_score(y_Val, y_pred_class)

    accuracy = accuracy_score(y_Val, y_pred_class)

    gini = eval_gini(y_Val, y_pred_class)

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

    print(f'Gini: {round(gini, 6)}')



    print('---')

    print('Matriz de Confusão')

    display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

    print('---')

    

    return clf, f'Decision Tree - Default Parameters'
def gridSearchKNN(X_Train, y_Train, X_Val, y_Val, k_range):

    clf=KNeighborsClassifier()

    param_grid=dict(n_neighbors=k_range)

    scores = ['f1']

    for sc in scores:

        grid=GridSearchCV(clf,param_grid,cv=4,scoring=sc,n_jobs=-1)

        print("K-Nearest Neighbors - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)

        y_pred_proba = grid.predict_proba(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        gini = eval_gini(y_Val, y_pred_class)

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

        print(f'Gini: {round(gini, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

        print('---')

        

        return grid, f'KNN with recall-maxing hyperparameters - {grid.best_params_}'
def gridSearchSVC(X_Train, y_Train, X_Val, y_Val):

    svc=SVC()

    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 100, 1000]},

                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['f1']

    for sc in scores:

        grid=GridSearchCV(svc,param_grid,cv=4,scoring=sc,n_jobs=-1)

        

        print("Support Vector Classifier - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        gini = eval_gini(y_Val, y_pred_class)

        precision =  precision_score(y_Val, y_pred_class)

        f1 = f1_score(y_Val, y_pred_class)



        print(f'SVC with recall-maxing hyperparameters - {grid.best_params_}')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'F1 Score: {round(f1, 6)}')

        print(f'Gini: {round(gini, 6)}')



        print('---')

        print('Matriz de Confusão')

        display(pd.DataFrame(confusion_matrix(y_Val, y_pred_class)))

        print('---')

        

        return grid, f'SVC with recall-maxing hyperparameters - {grid.best_params_}'
def gridSearchXGB(X_Train, y_Train, X_Val, y_Val, score):

    xgb=XGBClassifier(random_state = 0)

    param_grid = [{'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.1, 0.25, 0.5, 0.75],'max_depth': [25, 50, 75, 100], 'gamma': [0, 3, 6, 9]}]

    scores = [score]

    for sc in scores:

        grid=GridSearchCV(xgb,param_grid,cv=2,scoring=sc,n_jobs=-1)

        

        print("XGBoost - Tuning hyper-parameters for %s" % sc)

        

        grid.fit(X_Train,y_Train)

        print(grid.best_params_)

        print(np.round(grid.best_score_,3))

        

        y_pred_class = grid.predict(X_Val)



        recall = recall_score(y_Val, y_pred_class)

        accuracy = accuracy_score(y_Val, y_pred_class)

        gini = eval_gini(y_Val, y_pred_class)

        precision =  precision_score(y_Val, y_pred_class)

        f1 = f1_score(y_Val, y_pred_class)



        print(f'XGBoost with {sc}-maxing hyperparameters - {grid.best_params_}')

        print('---')

        print(f'Acurácia: {round(accuracy, 6)}%')

        print(f'Recall: {round(recall, 6)}%')

        print(f'Precisão: {round(precision, 6)}%')

        print(f'F1 Score: {round(f1, 6)}')

        print(f'Gini: {round(gini, 6)}')



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

    gini = eval_gini(y_Test, y_pred_class)

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

    print(f'Gini: {round(gini, 6)}')



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

    pd_prediction.columns = ['dfComplete']

    showDistribution(pd_prediction)



    return y_pred_class, y_pred_proba
def performOneHotEncoding(dfTrain, dfTest, meta_generic, dist_limit):

    v = meta_generic[(meta_generic.level == 'nominal') & (meta_generic.keep)].index

    display(v)

    for f in v:

        dist_values = dfTrain[f].value_counts().shape[0]

        print('Atributo {} tem {} valores distintos'.format(f, dist_values))

        if (dist_values > dist_limit):

            print('Atributo {} tem mais de {} valores distintos e por isso será ignorado'.format(f, dist_limit))

            dfTrain.drop([f], axis=1)

            v = v.drop([f])

        

    print('Antes do one-hot encoding tinha-se {} atributos'.format(dfTrain.shape[1]))

    dfTrain = pd.get_dummies(dfTrain, columns=v, drop_first=True)

    print('Depois do one-hot encoding tem-se {} atributos'.format(dfTrain.shape[1]))



    dfTest = pd.get_dummies(dfTest, columns=v, drop_first=True)

    missing_cols = set( dfTrain.columns ) - set( dfTest.columns )

    for c in missing_cols:

        dfTest[c] = 0



    dfTrain, dfTest = dfTrain.align(dfTest, axis=1)

    

    return dfTrain, dfTest
dfReplaced = dfComplete

dfReplaced.action_type = pd.factorize(dfComplete.action_type)[0] + 1

dfReplaced.combined_shot_type = pd.factorize(dfComplete.combined_shot_type)[0] + 1

dfReplaced.shot_type = pd.factorize(dfComplete.shot_type)[0] + 1

dfReplaced.shot_zone_basic = pd.factorize(dfComplete.shot_zone_basic)[0] + 1

dfReplaced.shot_zone_area = pd.factorize(dfComplete.shot_zone_area)[0] + 1

dfReplaced.shot_zone_range = pd.factorize(dfComplete.shot_zone_range)[0] + 1

dfReplaced.opponent = pd.factorize(dfComplete.opponent)[0] + 1

dfReplaced.matchup = pd.factorize(dfComplete.matchup)[0] + 1

dfReplaced = dfReplaced.drop(columns=['season','game_date'])

dfReplaced
X = dfReplaced.drop(['shot_made_flag'], axis=1)

y = dfReplaced['shot_made_flag']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
meta_train = generateMetadata(X_train)

meta_test = generateMetadata(X_val)

display(meta_train)

display(meta_test)
print('Tipos e quantidade de features do dataset de treino')

display(pd.DataFrame({'count' : meta_train.groupby(['role', 'level'])['role'].size()}).reset_index())



print('Tipos e quantidade de features do dataset de teste')

display(pd.DataFrame({'count' : meta_test.groupby(['role', 'level'])['role'].size()}).reset_index())
X_train_one, X_test_one = performOneHotEncoding(X_train, y_train, meta_train, 200)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    

showDistribution(y_train)

logRegModel, logRegName = logisticRegression(X_train, y_train, X_val, y_val)

xgbPureModel, xgbPureName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Base',None)

#xgbPresetModel, xgbPresetName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Preset', {'n_estimator':400, 'learning_rate' : 0.5,'random_state' : 0,'max_depth':70,'objective':"binary:logistic",'subsample':.8,'min_child_weig':6,'colsample_bytr':.8,'scale_pos_weight':1.6, 'gamma':10, 'reg_alph':8, 'reg_lambda':1})

xgbHyperParametrizedModel, xgbHyperParametrizedName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Hyperparametrized',{'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 16, 'n_estimators': 100, 'subsample': 1.0})

#decTreeModel, decTreeName = decisionTreeClassifier(X_train, y_train, X_val, y_val)

#knnModel, knnName = gridSearchKNN(X_train, y_train, X_val, y_val, list(range(1,20)))

#svcModel, svcName = gridSearchSVC(X_train, y_train, X_val, y_val)

    

showDistribution(y)

predictTestDataset(X, y, logRegModel, logRegName)

predictTestDataset(X, y, xgbPureModel, xgbPureName)

#predictTestDataset(X_supersampled, y_supersampled, xgbPresetModel, xgbPresetName)

predictTestDataset(X, y, xgbHyperParametrizedModel, xgbHyperParametrizedName)

#predictTestDataset(X_supersampled, y_supersampled, decTreeModel, decTreeName)

#predictTestDataset(X_supersampled, y_supersampled, knnModel, knnName)

#predictTestDataset(X_supersampled, y_supersampled, svcModel, svcName)
showDistribution(y_train)

xgbGridSearchModel, xgbGridSearchName = gridSearchXGB(X_train, y_train, X_val, y_val, 'f1')
showDistribution(y)

predictTestDataset(X, y, xgbGridSearchModel, xgbGridSearchName)
X
X = X.drop(columns=['lat','lon','matchup','opponent'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    

showDistribution(y_train)

logRegModel, logRegName = logisticRegression(X_train, y_train, X_val, y_val)

xgbPureModel, xgbPureName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Base',None)

#xgbPresetModel, xgbPresetName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Preset', {'n_estimator':400, 'learning_rate' : 0.5,'random_state' : 0,'max_depth':70,'objective':"binary:logistic",'subsample':.8,'min_child_weig':6,'colsample_bytr':.8,'scale_pos_weight':1.6, 'gamma':10, 'reg_alph':8, 'reg_lambda':1})

xgbHyperParametrizedModel, xgbHyperParametrizedName = xGBClassifier(X_train, y_train, X_val, y_val, 'XGBoost - Hyperparametrized',{'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 16, 'n_estimators': 100, 'subsample': 1.0})

#decTreeModel, decTreeName = decisionTreeClassifier(X_train, y_train, X_val, y_val)

#knnModel, knnName = gridSearchKNN(X_train, y_train, X_val, y_val, list(range(1,20)))

#svcModel, svcName = gridSearchSVC(X_train, y_train, X_val, y_val)

    

showDistribution(y)

predictTestDataset(X, y, logRegModel, logRegName)

predictTestDataset(X, y, xgbPureModel, xgbPureName)

#predictTestDataset(X_supersampled, y_supersampled, xgbPresetModel, xgbPresetName)

predictTestDataset(X, y, xgbHyperParametrizedModel, xgbHyperParametrizedName)

#predictTestDataset(X_supersampled, y_supersampled, decTreeModel, decTreeName)

#predictTestDataset(X_supersampled, y_supersampled, knnModel, knnName)

#predictTestDataset(X_supersampled, y_supersampled, svcModel, svcName)