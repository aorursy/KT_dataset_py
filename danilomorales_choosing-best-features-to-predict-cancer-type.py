#Importing fundamental libraries for data science

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Reading CSV file with Pandas Library

dados = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
dados.head()
dados = dados.drop(['id','Unnamed: 32'],axis=1)
dados.head()
num_colunas = dados.shape[1]
sns.countplot(x='diagnosis',data=dados)

plt.xlabel('Diagnóstico')

plt.ylabel('Count')

plt.title('Kind of diagnostic')
corr = dados.corr()

f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
colunas = dados.columns

print("Number of columns = {}".format(len(colunas)))
fig,ax=plt.subplots(6,5,figsize=(12,15))

sns.boxplot(y=dados[colunas[1]],x=dados['diagnosis'],ax=ax[0][0])

sns.boxplot(y=dados[colunas[2]],x=dados['diagnosis'],ax=ax[0][1])

sns.boxplot(y=dados[colunas[3]],x=dados['diagnosis'],ax=ax[0][2])

sns.boxplot(y=dados[colunas[4]],x=dados['diagnosis'],ax=ax[0][3])

sns.boxplot(y=dados[colunas[5]],x=dados['diagnosis'],ax=ax[0][4])



sns.boxplot(y=dados[colunas[6]],x=dados['diagnosis'],ax=ax[1][0])

sns.boxplot(y=dados[colunas[7]],x=dados['diagnosis'],ax=ax[1][1])

sns.boxplot(y=dados[colunas[8]],x=dados['diagnosis'],ax=ax[1][2])

sns.boxplot(y=dados[colunas[9]],x=dados['diagnosis'],ax=ax[1][3])

sns.boxplot(y=dados[colunas[10]],x=dados['diagnosis'],ax=ax[1][4])



sns.boxplot(y=dados[colunas[11]],x=dados['diagnosis'],ax=ax[2][0])

sns.boxplot(y=dados[colunas[12]],x=dados['diagnosis'],ax=ax[2][1])

sns.boxplot(y=dados[colunas[13]],x=dados['diagnosis'],ax=ax[2][2])

sns.boxplot(y=dados[colunas[14]],x=dados['diagnosis'],ax=ax[2][3])

sns.boxplot(y=dados[colunas[15]],x=dados['diagnosis'],ax=ax[2][4])



sns.boxplot(y=dados[colunas[16]],x=dados['diagnosis'],ax=ax[3][0])

sns.boxplot(y=dados[colunas[17]],x=dados['diagnosis'],ax=ax[3][1])

sns.boxplot(y=dados[colunas[18]],x=dados['diagnosis'],ax=ax[3][2])

sns.boxplot(y=dados[colunas[19]],x=dados['diagnosis'],ax=ax[3][3])

sns.boxplot(y=dados[colunas[20]],x=dados['diagnosis'],ax=ax[3][4])



sns.boxplot(y=dados[colunas[21]],x=dados['diagnosis'],ax=ax[4][0])

sns.boxplot(y=dados[colunas[22]],x=dados['diagnosis'],ax=ax[4][1])

sns.boxplot(y=dados[colunas[23]],x=dados['diagnosis'],ax=ax[4][2])

sns.boxplot(y=dados[colunas[24]],x=dados['diagnosis'],ax=ax[4][3])

sns.boxplot(y=dados[colunas[25]],x=dados['diagnosis'],ax=ax[4][4])



sns.boxplot(y=dados[colunas[26]],x=dados['diagnosis'],ax=ax[5][0])

sns.boxplot(y=dados[colunas[27]],x=dados['diagnosis'],ax=ax[5][1])

sns.boxplot(y=dados[colunas[28]],x=dados['diagnosis'],ax=ax[5][2])

sns.boxplot(y=dados[colunas[29]],x=dados['diagnosis'],ax=ax[5][3])

sns.boxplot(y=dados[colunas[30]],x=dados['diagnosis'],ax=ax[5][4])



plt.tight_layout()
colunas_normalizar = colunas.drop('diagnosis')

dados_norm = dados.copy()
from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold

from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder
enconder = LabelEncoder()

dados_norm['diagnosis'] = enconder.fit_transform(dados_norm['diagnosis'])
dados_norm.head()
scaler = RobustScaler()

for col in colunas_normalizar:

    dados_norm[col] = scaler.fit_transform(dados_norm[col].values.reshape(-1,1))
dados.head()
X = dados_norm.drop(['diagnosis'],axis=1)

Y = dados_norm['diagnosis']
strat_kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for indice_treino, indice_teste in strat_kfold.split(X, Y):

    #print("Treino:", indice_treino, "Teste:", indice_teste)

    X_treino, X_teste = X.iloc[indice_treino], X.iloc[indice_teste]

    Y_treino, Y_teste = Y.iloc[indice_treino], Y.iloc[indice_teste]
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,classification_report,recall_score
from sklearn.model_selection import GridSearchCV
nome_modelo = []

resultados = []
accuracy = []

precision =[]

recall = []

f1 = []
from sklearn.linear_model import LogisticRegression
print("Logistic Regression")

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000], 

                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

grid_log_reg = GridSearchCV(LogisticRegression(max_iter=2000), log_reg_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grid_log_reg.fit(X_treino, Y_treino)

logreg = grid_log_reg.best_estimator_

log_reg_score = cross_val_score(logreg, X_treino, Y_treino, cv=10,scoring='recall')

print("Best Estimator")

print(logreg)

print('Score Regressao Logistica Validacao Cruzada: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
nome_modelo.append("Logistic Regression")

resultados.append(log_reg_score)
logreg.fit(X_treino,Y_treino)

Y_pred_logreg = logreg.predict(X_teste)

cm_logreg = confusion_matrix(Y_teste,Y_pred_logreg)

acc_score_logreg = accuracy_score(Y_teste,Y_pred_logreg)

f1_score_logreg = f1_score(Y_teste,Y_pred_logreg)

precisao_logreg = average_precision_score(Y_teste,Y_pred_logreg)

recall_logreg = recall_score(Y_teste,Y_pred_logreg)

print('Acuracia Regressão Logistica ',round(acc_score_logreg*100,2).astype(str)+'%')

print('Precião média Regressão Logistica ',round(precisao_logreg*100,2).astype(str)+'%')

print('F1 Regressão Logistica ',round(f1_score_logreg*100,2).astype(str)+'%')

print('Recall Regressão Logistica ',round(recall_logreg*100,2).astype(str)+'%')
accuracy.append(acc_score_logreg)

precision.append(precisao_logreg)

recall.append(recall_logreg)

f1.append(f1_score_logreg)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_logreg, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Regressão Logistica \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
from sklearn.neighbors import KNeighborsClassifier
print("KNN")

knears_params = {"n_neighbors": list(range(5,40,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

                'leaf_size' : list(range(3,40,1))}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grid_knears.fit(X_treino, Y_treino)

knn = grid_knears.best_estimator_

knears_score = cross_val_score(knn, X_treino, Y_treino, cv=10,scoring='recall')

print("Best Estimator")

print(knn)

print('Score KNN Validacao Cruzada: ', round(knears_score.mean() * 100, 2).astype(str) + '%')
nome_modelo.append("KNN")

resultados.append(knears_score)
knn.fit(X_treino,Y_treino)

Y_pred_knn = knn.predict(X_teste)

cm_knn = confusion_matrix(Y_teste,Y_pred_knn)

acc_score_knn = accuracy_score(Y_teste,Y_pred_knn)

f1_score_knn = f1_score(Y_teste,Y_pred_knn)

precisao_knn = average_precision_score(Y_teste,Y_pred_knn)

recall_knn = recall_score(Y_teste,Y_pred_knn)

print('Acuracia KNN ',round(acc_score_knn*100,2).astype(str)+'%')

print('Precião média KNN ',round(precisao_knn*100,2).astype(str)+'%')

print('F1 KNN ',round(f1_score_knn*100,2).astype(str)+'%')

print('Recall KNN ',round(recall_knn*100,2).astype(str)+'%')
accuracy.append(acc_score_knn)

precision.append(precisao_knn)

recall.append(recall_knn)

f1.append(f1_score_knn)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_knn, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("KNN \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
print("Ada Boost Classifier")

ada_params = {'n_estimators' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80], 'learning_rate' : [0.001,0.01,0.1,1.0], 'algorithm' : ['SAMME','SAMME.R']}

grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grid_ada.fit(X_treino, Y_treino)

ada = grid_ada.best_estimator_

print("Best Estimator")

print(ada)

ada_score = cross_val_score(ada, X_treino, Y_treino, cv=10,scoring='recall')

print('Score AdaBoost Validacao Cruzada: ', round(ada_score.mean() * 100, 2).astype(str) + '%')
nome_modelo.append("AdaBoost")

resultados.append(ada_score)
ada.fit(X_treino,Y_treino)

Y_pred_ada = ada.predict(X_teste)

cm_ada = confusion_matrix(Y_teste,Y_pred_ada)

acc_score_ada = accuracy_score(Y_teste,Y_pred_ada)

f1_score_ada = f1_score(Y_teste,Y_pred_ada)

precisao_ada = average_precision_score(Y_teste,Y_pred_ada)

recall_ada = recall_score(Y_teste,Y_pred_ada)

print('Acuracia ADA Boost ',round(acc_score_ada*100,2).astype(str)+'%')

print('Precião média Ada Boost ',round(precisao_ada*100,2).astype(str)+'%')

print('F1 Ada Boost ',round(f1_score_ada*100,2).astype(str)+'%')

print('Recall Ada Boost ',round(recall_ada*100,2).astype(str)+'%')
accuracy.append(acc_score_ada)

precision.append(precisao_ada)

recall.append(recall_ada)

f1.append(f1_score_ada)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_ada, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Ada Boost \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
print("Random Forest Classifier")

forest_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,20,1)), 

              "min_samples_leaf": list(range(3,20,1)), 'max_features' : ['auto','sqrt','log2']}

forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

forest.fit(X_treino, Y_treino)

random_forest = forest.best_estimator_

print("Best Estimator")

print(random_forest)

forest_score = cross_val_score(random_forest, X_treino, Y_treino, cv=10,scoring='recall')

print('Score RFC Validacao Cruzada: ', round(forest_score.mean() * 100, 2).astype(str) + '%')
nome_modelo.append("RFC")

resultados.append(forest_score)
random_forest.fit(X_treino,Y_treino)

Y_pred_rf = random_forest.predict(X_teste)

cm_rf = confusion_matrix(Y_teste,Y_pred_rf)

acc_score_rf = accuracy_score(Y_teste,Y_pred_rf)

f1_score_rf = f1_score(Y_teste,Y_pred_rf)

precisao_rf = average_precision_score(Y_teste,Y_pred_rf)

recall_rf = recall_score(Y_teste,Y_pred_rf)

print('Acuracia Random Forest ',round(acc_score_rf*100,2).astype(str)+'%')

print('Precião média Random Forest ',round(precisao_rf*100,2).astype(str)+'%')

print('F1 Random Forest ',round(f1_score_rf*100,2).astype(str)+'%')

print('Recall Random Forest ',round(recall_rf*100,2).astype(str)+'%')
accuracy.append(acc_score_rf)

precision.append(precisao_rf)

recall.append(recall_rf)

f1.append(f1_score_rf)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_rf, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Random Forest \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
print("Gradient Boost Classifier")

grad_params = {'n_estimators' : [30,35,40,45,50,55,60,65,70], 'learning_rate' : [0.001,0.01,0.1,1.0], 'loss' : ['deviance','exponential'],

              'max_depth' : [3,4,5,6,7], 'max_features' : ['auto','sqrt','log2'], 'min_samples_leaf' : [2,3,4,5,6]}

grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grad.fit(X_treino, Y_treino)

grad_boost = grad.best_estimator_

print("Best Estimator")

print(grad_boost)

grad_score = cross_val_score(grad_boost, X_treino, Y_treino, cv=10,scoring='recall')

print('Score GradBoost Validacao Cruzada: ', round(grad_score.mean() * 100, 2).astype(str) + '%')
nome_modelo.append("GradBoost")

resultados.append(grad_score)
grad_boost.fit(X_treino,Y_treino)

Y_pred_gb = grad_boost.predict(X_teste)

cm_gb = confusion_matrix(Y_teste,Y_pred_gb)

acc_score_gb = accuracy_score(Y_teste,Y_pred_gb)

f1_score_gb = f1_score(Y_teste,Y_pred_gb)

precisao_gb = average_precision_score(Y_teste,Y_pred_gb)

recall_gb = recall_score(Y_teste,Y_pred_gb)

print('Acuracia Gradient Boosting ',round(acc_score_gb*100,2).astype(str)+'%')

print('Precião média Gradient Boosting  ',round(precisao_gb*100,2).astype(str)+'%')

print('F1 Gradient Boosting  ',round(f1_score_gb*100,2).astype(str)+'%')

print('Recall Gradient Boosting  ',round(recall_gb*100,2).astype(str)+'%')
accuracy.append(acc_score_gb)

precision.append(precisao_gb)

recall.append(recall_gb)

f1.append(f1_score_gb)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_gb, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Gradient Boosting  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
fig,ax=plt.subplots(figsize=(10,7))

plt.boxplot(resultados)

ax.set_xticklabels(nome_modelo)

plt.tight_layout()
from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Dropout

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.callbacks import ReduceLROnPlateau,EarlyStopping
n_inputs = X_treino.shape[1]
modelo = Sequential()

modelo.add(Dense(32, input_shape=(n_inputs, ), activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(Dropout(0.5))

modelo.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(Dropout(0.5))

modelo.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, mode='auto', min_delta=0.0001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
callbacks_list = [reduce_lr,es]
modelo.compile(Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['top_k_categorical_accuracy'])

modelo.fit(X_treino, Y_treino, batch_size=20, epochs=200, verbose=2, validation_data=(X_teste,Y_teste),callbacks=callbacks_list)
Y_pred_keras = modelo.predict_classes(X_teste, batch_size=50, verbose=0)
cm_keras = confusion_matrix(Y_teste,Y_pred_keras)

acc_score_keras = accuracy_score(Y_teste,Y_pred_keras)

f1_score_keras = f1_score(Y_teste,Y_pred_keras)

precisao_keras = average_precision_score(Y_teste,Y_pred_keras)

recall_keras = recall_score(Y_teste,Y_pred_keras)

print('Acuracia Keras ',round(acc_score_keras*100,2).astype(str)+'%')

print('Precião média Keras  ',round(precisao_keras*100,2).astype(str)+'%')

print('F1 Gradient Boosting  ',round(f1_score_keras*100,2).astype(str)+'%')

print('Recall Keras  ',round(recall_keras*100,2).astype(str)+'%')
nome_modelo.append("Keras")

accuracy.append(acc_score_keras)

precision.append(precisao_keras)

recall.append(recall_keras)

f1.append(f1_score_keras)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_keras, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Keras  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
from sklearn.feature_selection import SelectKBest,chi2
dados['diagnosis'] = enconder.fit_transform(dados['diagnosis'])
X = dados.drop(['diagnosis'],axis=1)

Y = dados['diagnosis']
def model_params(model):

    if(model == 'logistic_regression'):

        modelo = LogisticRegression(max_iter=2000)

        modelo_params = {"penalty": ['l1', 'l2'], 'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000], 

                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

    

    elif(model == 'KNN'):

        modelo = KNeighborsClassifier()

        modelo_params = {"n_neighbors": list(range(5,40,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

                'leaf_size' : list(range(2,40,1))}

    

    elif(model == 'AdaBoost'):

        modelo = AdaBoostClassifier()

        modelo_params = {'n_estimators' : list(range(5,81)), 'learning_rate' : [0.001,0.01,0.1,1.0], 'algorithm' : ['SAMME','SAMME.R']}

        

    elif(model == 'RFC'):

        modelo = RandomForestClassifier()

        modelo_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,20,1)), 

              "min_samples_leaf": list(range(3,20,1)), 'max_features' : ['auto','sqrt','log2']}

        

    elif(model == 'GradBoost'):

        modelo = GradientBoostingClassifier()

        modelo_params = {'n_estimators' : [30,35,40,45,50,55,60,65,70], 'learning_rate' : [0.001,0.01,0.1,1.0], 'loss' : ['deviance','exponential'],

              'max_depth' : [3,4,5,6,7], 'max_features' : ['auto','sqrt','log2'], 'min_samples_leaf' : [2,3,4,5,6]}

    

    return modelo,modelo_params
def find_best_features(modelo,X,Y,n):

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=42)

    best_features = SelectKBest(chi2, k=n).fit(X_treino, Y_treino)

    X_treino = best_features.transform(X_treino)

    X_teste = best_features.transform(X_teste)

    acc,precision,recall,f1 = best_model(modelo,X_treino,Y_treino,X_teste,Y_teste)

    

    return acc, precision, recall, f1
def best_model(model,X_treino_best,Y_treino_best,X_teste,Y_teste):

    

    modelo, parametros = model_params(model)

    grid_log_reg = GridSearchCV(modelo, parametros,n_jobs=8,cv=10,scoring=['recall','f1'],refit='f1')

    grid_log_reg.fit(X_treino_best, Y_treino_best)

    logreg = grid_log_reg.best_estimator_

    logreg.fit(X_treino_best,Y_treino_best)

    Y_pred_Kbest = logreg.predict(X_teste)

    acc_kest = accuracy_score(Y_teste,Y_pred_Kbest)

    f1_kbest = f1_score(Y_teste,Y_pred_Kbest)

    precisao_kbest = average_precision_score(Y_teste,Y_pred_Kbest)

    recall_kbest = recall_score(Y_teste,Y_pred_Kbest)

    

    return acc_kest,precisao_kbest,recall_kbest,f1_kbest
def find_bestn(modelo,X,Y,number):



    acc_findbest = []

    rec_findbest = []

    prec_findbest = []

    f1s_findbest= []

    n_idex = []



    for n in range(5,number):

        acuraciax,precisaox,recallx,f1x = find_best_features(modelo,X,Y,n)

        acc_findbest.append(acuraciax)

        rec_findbest.append(recallx)

        prec_findbest.append(precisaox)

        f1s_findbest.append(f1x)

        n_idex.append(n)

        print("N = ",n,"Acc = ",acuraciax, "Prec = ",precisaox, "Rec = ",recallx, "F1 = ",f1x)

    

    dic_kbest = {"N" : n_idex, "Acuracia" : acc_findbest, "Recall" : rec_findbest, "Precision" : prec_findbest, "F1" : f1s_findbest}



    dataframe_kbest = pd.DataFrame(dic_kbest)

    

    dataframe_kbest = dataframe_kbest.sort_values(by=['Acuracia','Recall','F1','Precision'],ascending=False).reset_index()

    

    best_n = int(dataframe_kbest.iloc[0]['N'])

    

    return best_n
modelos = ['logistic_regression']
dic_bestn = {}



for models in modelos:

    bestn = find_bestn(models,X,Y,num_colunas-1)

    dic_bestn[models] = bestn

    print("Modelo = ",models," ","N = ",bestn)
X_treino_best, X_teste_best, Y_treino_best, Y_teste_best = train_test_split(X, Y, test_size=0.3, random_state=42)
best_n = dic_bestn['logistic_regression']
modelo_kbest = SelectKBest(chi2, k=int(best_n)).fit(X_treino_best, Y_treino_best)

X_treino_best = modelo_kbest.transform(X_treino_best)#.values

X_teste_best = modelo_kbest.transform(X_teste_best)#.values

Y_treino_best = Y_treino_best.values

Y_teste_best = Y_teste_best.values
acc_kbest = []

precison_kbest =[]

recall_kbest = []

f1_kbest = []
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1e3,1e4], 

                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

grid_log_reg = GridSearchCV(LogisticRegression(max_iter=2000), log_reg_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='f1')

grid_log_reg.fit(X_treino_best, Y_treino_best)

logreg = grid_log_reg.best_estimator_

logreg.fit(X_treino_best,Y_treino_best)

Y_pred_best = logreg.predict(X_teste_best)

cm_best = confusion_matrix(Y_teste_best,Y_pred_best)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_best, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Regressão logistica  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
acc_score_logreg_best = accuracy_score(Y_teste_best,Y_pred_best)

f1_score_logreg_best = f1_score(Y_teste_best,Y_pred_best)

precisao_logreg_best = average_precision_score(Y_teste_best,Y_pred_best)

recall_logreg_best = recall_score(Y_teste_best,Y_pred_best)

print('Acuracia Regressão Logistica ',round(acc_score_logreg_best*100,2).astype(str)+'%')

print('Precião média Regressão Logistica ',round(precisao_logreg_best*100,2).astype(str)+'%')

print('F1 Regressão Logistica ',round(f1_score_logreg_best*100,2).astype(str)+'%')

print('Recall Regressão Logistica ',round(recall_logreg_best*100,2).astype(str)+'%')
acc_kbest.append(acc_score_logreg_best)

precison_kbest.append(precisao_logreg_best)

recall_kbest.append(recall_logreg_best)

f1_kbest.append(f1_score_logreg_best)
knears_params = {"n_neighbors": list(range(5,40,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

                'leaf_size' : list(range(2,40,1))}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grid_knears.fit(X_treino_best, Y_treino_best)

knn = grid_knears.best_estimator_

knn.fit(X_treino_best,Y_treino_best)

Y_pred_best_knn = knn.predict(X_teste_best)

cm_best_knn = confusion_matrix(Y_teste_best,Y_pred_best_knn)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_best_knn, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("KNN  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
acc_score_knn = accuracy_score(Y_teste_best,Y_pred_best_knn)

f1_score_knn = f1_score(Y_teste_best,Y_pred_best_knn)

precisao_knn = average_precision_score(Y_teste_best,Y_pred_best_knn)

recall_knn = recall_score(Y_teste_best,Y_pred_best_knn)

print('Acuracia KNN ',round(acc_score_knn*100,2).astype(str)+'%')

print('Precião média KNN ',round(precisao_knn*100,2).astype(str)+'%')

print('F1 KNN ',round(f1_score_knn*100,2).astype(str)+'%')

print('Recall KNN ',round(recall_knn*100,2).astype(str)+'%')
acc_kbest.append(acc_score_knn)

precison_kbest.append(precisao_knn)

recall_kbest.append(recall_knn)

f1_kbest.append(f1_score_knn)
ada_params = {'n_estimators' : list(range(5,81)), 'learning_rate' : [0.001,0.01,0.1,1.0], 'algorithm' : ['SAMME','SAMME.R']}

grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='f1')

grid_ada.fit(X_treino_best, Y_treino_best)

ada = grid_ada.best_estimator_

ada.fit(X_treino_best,Y_treino_best)

Y_pred_best_ada = ada.predict(X_teste_best)

cm_best_ada = confusion_matrix(Y_teste_best,Y_pred_best_ada)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_best_ada, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Ada Boost  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
acc_score_ada_best = accuracy_score(Y_teste_best,Y_pred_best_ada)

f1_score_ada_best = f1_score(Y_teste_best,Y_pred_best_ada)

precisao_ada_best = average_precision_score(Y_teste_best,Y_pred_best_ada)

recall_ada_best = recall_score(Y_teste_best,Y_pred_best_ada)

print('Acuracia Ada Boost ',round(acc_score_ada_best*100,2).astype(str)+'%')

print('Precião média Ada Boost ',round(precisao_ada_best*100,2).astype(str)+'%')

print('F1 Ada Boost ',round(f1_score_ada_best*100,2).astype(str)+'%')

print('Recall Ada Boost ',round(recall_ada_best*100,2).astype(str)+'%')
acc_kbest.append(acc_score_ada_best)

precison_kbest.append(precisao_ada_best)

recall_kbest.append(recall_ada_best)

f1_kbest.append(f1_score_ada_best)
forest_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,20,1)), 

              "min_samples_leaf": list(range(3,20,1)), 'max_features' : ['auto','sqrt','log2']}

forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

forest.fit(X_treino_best, Y_treino_best)

random_forest = forest.best_estimator_

random_forest.fit(X_treino_best,Y_treino_best)

Y_pred_best_rf = random_forest.predict(X_teste_best)

cm_best_rf = confusion_matrix(Y_teste_best,Y_pred_best_rf)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_best_rf, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Random Forest  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
acc_score_rf_best = accuracy_score(Y_teste_best,Y_pred_best_rf)

f1_score_rf_best = f1_score(Y_teste_best,Y_pred_best_rf)

precisao_rf_best = average_precision_score(Y_teste_best,Y_pred_best_rf)

recall_rf_best = recall_score(Y_teste_best,Y_pred_best_rf)

print('Acuracia Random Forest ',round(acc_score_rf_best*100,2).astype(str)+'%')

print('Precião média Random Forest ',round(precisao_rf_best*100,2).astype(str)+'%')

print('F1 Random Forest ',round(f1_score_rf_best*100,2).astype(str)+'%')

print('Recall Random Forest ',round(recall_rf_best*100,2).astype(str)+'%')
acc_kbest.append(acc_score_rf_best)

precison_kbest.append(precisao_rf_best)

recall_kbest.append(recall_rf_best)

f1_kbest.append(f1_score_rf_best)
grad_params = {'n_estimators' : [30,35,40,45,50,55,60,65,70], 'learning_rate' : [0.001,0.01,0.1,1.0], 'loss' : ['deviance','exponential'],

              'max_depth' : [3,4,5,6,7], 'max_features' : ['auto','sqrt','log2'], 'min_samples_leaf' : [2,3,4,5,6]}

grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=8,cv=10,scoring=['recall','f1'],refit='recall')

grad.fit(X_treino_best, Y_treino_best)

grad_boost = grad.best_estimator_

grad_boost.fit(X_treino_best,Y_treino_best)

Y_pred_best_grad = grad_boost.predict(X_teste_best)

cm_best_grad = confusion_matrix(Y_teste_best,Y_pred_best_grad)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_best_grad, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Gradient Boosting  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
acc_score_grad_best = accuracy_score(Y_teste_best,Y_pred_best_grad)

f1_score_grad_best = f1_score(Y_teste_best,Y_pred_best_grad)

precisao_grad_best = average_precision_score(Y_teste_best,Y_pred_best_grad)

recall_grad_best = recall_score(Y_teste_best,Y_pred_best_grad)

print('Acuracia Random Forest ',round(acc_score_grad_best*100,2).astype(str)+'%')

print('Precião média Random Forest ',round(precisao_grad_best*100,2).astype(str)+'%')

print('F1 Random Forest ',round(f1_score_grad_best*100,2).astype(str)+'%')

print('Recall Random Forest ',round(recall_grad_best*100,2).astype(str)+'%')
acc_kbest.append(acc_score_grad_best)

precison_kbest.append(precisao_grad_best)

recall_kbest.append(recall_grad_best)

f1_kbest.append(f1_score_grad_best)
n_inputs = X_treino_best.shape[1]
modelo2 = Sequential()

modelo2.add(Dense(32, input_shape=(n_inputs, ), activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo2.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo2.add(Dropout(0.5))

modelo2.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo2.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo2.add(Dropout(0.5))

modelo2.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
modelo2.compile(Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['top_k_categorical_accuracy'])

modelo2.fit(X_treino_best, Y_treino_best, batch_size=20, epochs=200, verbose=2, validation_data=(X_teste_best,Y_teste_best),callbacks=callbacks_list)
Y_pred_keras = modelo2.predict_classes(X_teste_best, batch_size=50, verbose=0)
cm_keras = confusion_matrix(Y_teste_best,Y_pred_keras)

acc_score_keras = accuracy_score(Y_teste_best,Y_pred_keras)

f1_score_keras = f1_score(Y_teste_best,Y_pred_keras)

precisao_keras = average_precision_score(Y_teste_best,Y_pred_keras)

recall_keras = recall_score(Y_teste_best,Y_pred_keras)

print('Acuracia Keras ',round(acc_score_keras*100,2).astype(str)+'%')

print('Precião média Keras  ',round(precisao_keras*100,2).astype(str)+'%')

print('F1 Gradient Boosting  ',round(f1_score_keras*100,2).astype(str)+'%')

print('Recall Keras  ',round(recall_keras*100,2).astype(str)+'%')
acc_kbest.append(acc_score_keras)

precison_kbest.append(precisao_keras)

recall_kbest.append(recall_keras)

f1_kbest.append(f1_score_keras)
dic_metrics = {'Model' : nome_modelo, 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1' : f1, 

              'Accuracy_Kbest' : acc_kbest, 'Precision_Kbest' : precison_kbest, 'Recall_Kbest' : recall_kbest,

              'F1_Kbest' : f1_kbest}



dataframe = pd.DataFrame(dic_metrics)
dataframe