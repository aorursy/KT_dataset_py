import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,recall_score,roc_auc_score

from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder,MinMaxScaler



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest,chi2



from keras.models import Sequential

from keras.layers import Activation,BatchNormalization

from keras.layers.core import Dense,Dropout

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.callbacks import ReduceLROnPlateau,EarlyStopping
dados = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv',sep=';')
dados.head()
dados = dados.drop('id',axis=1)
dados.head()
dados.duplicated().sum()
dados = dados.drop_duplicates()
dados['age'] = dados['age']/365
dados.head()
dados.info()
dados.isna().sum()
dados['bmi'] = dados["weight"]/(dados["height"]/100)**2
#number of columns

num_colunas = dados.shape[1]
corr = dados.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
dados = dados[(dados["ap_hi"]<=250) & (dados["ap_lo"]<=200)]

dados = dados[(dados["ap_hi"] >= 0) & (dados["ap_lo"] >= 0)]
corr = dados.corr()

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.countplot(x='cardio',data=dados)
fig,ax = plt.subplots(2,3,figsize=(13,5))

sns.countplot(x='gender',data=dados,ax=ax[0][0])

sns.countplot(x='cholesterol',data=dados,ax=ax[0][1])

sns.countplot(x='smoke',data=dados,ax=ax[0][2])

sns.countplot(x='gluc',data=dados,ax=ax[1][0])

sns.countplot(x='alco',data=dados,ax=ax[1][1])

sns.countplot(x='active',data=dados,ax=ax[1][2])

plt.tight_layout()
print("Maximum age = ",dados['age'].max())

print("Minimum age = ",dados['age'].min())
print("Maximum height = ",dados['height'].max())

print("Minimum height = ",dados['height'].min())
print("Maximum ap_high = ",dados['ap_hi'].max())

print("Minimum ap_high = ",dados['ap_hi'].min())
print("Maximum ap_low = ",dados['ap_lo'].max())

print("Minimum ap_low = ",dados['ap_lo'].min())
dados_norm = dados.copy()
colunas_normalizar = ['ap_hi','ap_lo','age','height','weight']



tipo_scaler = 'MinMax'

if(tipo_scaler=='Standard'):

    scaler = StandardScaler((0,1))

elif(tipo_scaler=='Robust'):

    scaler = RobustScaler()

elif(tipo_scaler=='MinMax'):

    scaler = MinMaxScaler(feature_range=(0, 1))



for col in colunas_normalizar:

    dados_norm[col] = scaler.fit_transform(dados_norm[col].values.reshape(-1,1))
dados_norm.head()
fig,ax = plt.subplots(1,2,figsize=(13,5))

sns.boxplot(y=dados_norm['ap_hi'],x=dados_norm['cardio'],ax=ax[0])

sns.boxplot(y=dados_norm['ap_lo'],x=dados_norm['cardio'],ax=ax[1])

plt.tight_layout()
fig,ax = plt.subplots(1,3,figsize=(13,5))

sns.boxplot(y=dados_norm['age'],x=dados_norm['cardio'],ax=ax[0])

sns.boxplot(y=dados_norm['height'],x=dados_norm['cardio'],ax=ax[1])

sns.boxplot(y=dados_norm['weight'],x=dados_norm['cardio'],ax=ax[2])

plt.tight_layout()
def remover_outlier(dados,coluna_input,coluna_output,tipo):

    dados_tmp = dados[dados[coluna_output]==tipo]

    q25, q75 = np.percentile(dados_tmp[coluna_input], 25), np.percentile(dados_tmp[coluna_input], 75)

    iqr = q75 - q25

    cut_off = iqr * 1.5

    x_inferior, x_superior = q25 - cut_off, q75 + cut_off

    outliers = [x for x in dados_tmp[coluna_input] if x < x_inferior or x > x_superior]

    dados_novo = dados.drop(dados[(dados[coluna_input] > x_superior) | (dados[coluna_input] < x_inferior)].index)

    return dados_novo
dados_norm = remover_outlier(dados_norm,'ap_hi','cardio',1)
fig,ax = plt.subplots(1,2,figsize=(13,5))

sns.boxplot(y=dados_norm['ap_hi'],x=dados_norm['cardio'],ax=ax[0])

sns.boxplot(y=dados_norm['ap_lo'],x=dados_norm['cardio'],ax=ax[1])

plt.tight_layout()
dados_norm = remover_outlier(dados_norm,'ap_lo','cardio',0)
fig,ax = plt.subplots(1,2,figsize=(13,5))

sns.boxplot(y=dados_norm['ap_hi'],x=dados_norm['cardio'],ax=ax[0])

sns.boxplot(y=dados_norm['ap_lo'],x=dados_norm['cardio'],ax=ax[1])

plt.tight_layout()
dados_norm = remover_outlier(dados_norm,'age','cardio',0)

dados_norm = remover_outlier(dados_norm,'height','cardio',0)

dados_norm = remover_outlier(dados_norm,'weight','cardio',0)
fig,ax = plt.subplots(1,3,figsize=(13,5))

sns.boxplot(y=dados_norm['age'],x=dados_norm['cardio'],ax=ax[0])

sns.boxplot(y=dados_norm['height'],x=dados_norm['cardio'],ax=ax[1])

sns.boxplot(y=dados_norm['weight'],x=dados_norm['cardio'],ax=ax[2])

plt.tight_layout()
sns.countplot(x='cardio',data=dados_norm)

plt.tight_layout()
X = dados_norm.drop('cardio',axis=1).values

Y = dados_norm['cardio'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=42)
#List to compute metrics

accuracy = []

precision =[]

recall = []

f1 = []

roc = []
#print("Logistic Regression")

#log_reg_params = {"penalty": ['l1', 'l2','elasticnet'], 'C': [1, 10], 

#                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

#grid_log_reg = GridSearchCV(LogisticRegression(max_iter=10000), log_reg_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')

#grid_log_reg.fit(X_train, y_train)

#logreg = grid_log_reg.best_estimator_

#print(logreg)
#Parameters have been choosing based on GridSearchCV

logreg = LogisticRegression(C=1,max_iter=10000,penalty='l1',solver='liblinear')

logreg.fit(X_train,y_train)
log_reg_score = cross_val_score(logreg, X_train, y_train, cv=10,scoring='roc_auc_ovo')

log_reg_score_teste = cross_val_score(logreg, X_test, y_test, cv=10,scoring='roc_auc_ovo')

print('Score Regressao Logistica Treino: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

print('Score Regressao Logistica Teste: ', round(log_reg_score_teste.mean() * 100, 2).astype(str) + '%')
Y_pred_logreg = logreg.predict(X_test)
cm_logreg = confusion_matrix(y_test,Y_pred_logreg)
acc_score_logreg = accuracy_score(y_test,Y_pred_logreg)

f1_score_logreg = f1_score(y_test,Y_pred_logreg)

precisao_logreg = average_precision_score(y_test,Y_pred_logreg)

recall_logreg = recall_score(y_test,Y_pred_logreg)

roc_logreg = roc_auc_score(y_test,Y_pred_logreg,multi_class='ovo')

print('Acuracia Regressão Logistica ',round(acc_score_logreg*100,2).astype(str)+'%')

print('Precião média Regressão Logistica ',round(precisao_logreg*100,2).astype(str)+'%')

print('F1 Regressão Logistica ',round(f1_score_logreg*100,2).astype(str)+'%')

print('Recall Regressão Logistica ',round(recall_logreg*100,2).astype(str)+'%')

print('ROC Regressão Logistica ',round(roc_logreg*100,2).astype(str)+'%')
accuracy.append(acc_score_logreg)

precision.append(precisao_logreg)

recall.append(recall_logreg)

f1.append(f1_score_logreg)

roc.append(roc_logreg)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_logreg, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Regressão Logistica \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
#print("KNN")

#knears_params = {"n_neighbors": list(range(20,30,1)),'leaf_size' : list(range(5,11,1)), 'weights': ['uniform', 'distance']}

#grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')

#grid_knears.fit(X_train, y_train)

#knn = grid_knears.best_estimator_

#print("Best Estimator")

#print(knn)
#Parameters have been choosing based on GridSearchCV

knn = KNeighborsClassifier(weights='uniform',n_neighbors=27,leaf_size=6)

knn.fit(X_train,y_train)
knears_score = cross_val_score(knn, X_train, y_train, cv=10,scoring='roc_auc_ovo')

knears_score_teste = cross_val_score(knn, X_test, y_test, cv=10,scoring='roc_auc_ovo')

print('Score KNN Treino: ', round(knears_score.mean() * 100, 2).astype(str) + '%')

print('Score KNN Teste: ', round(knears_score_teste.mean() * 100, 2).astype(str) + '%')
Y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test,Y_pred_knn)
acc_score_knn = accuracy_score(y_test,Y_pred_knn)

f1_score_knn = f1_score(y_test,Y_pred_knn)

precisao_knn = average_precision_score(y_test,Y_pred_knn)

recall_knn = recall_score(y_test,Y_pred_knn)

roc_knn = roc_auc_score(y_test,Y_pred_knn,multi_class='ovo')

print('Acuracia KNN ',round(acc_score_knn*100,2).astype(str)+'%')

print('Precião média KNN ',round(precisao_knn*100,2).astype(str)+'%')

print('F1 KNN ',round(f1_score_knn*100,2).astype(str)+'%')

print('Recall KNN ',round(recall_knn*100,2).astype(str)+'%')

print('ROC KNN ',round(roc_knn*100,2).astype(str)+'%')
accuracy.append(acc_score_knn)

precision.append(precisao_knn)

recall.append(recall_knn)

f1.append(f1_score_knn)

roc.append(roc_knn)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_knn, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("KNN \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
#print("Ada Boost Classifier")

#ada_params = {'n_estimators' : list(range(100,200))}

#grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')

#grid_ada.fit(X_train, y_train)

#ada = grid_ada.best_estimator_

#print("Best Estimator")

#print(ada)
#Parameters have been choosing based on GridSearchCV

ada = AdaBoostClassifier(n_estimators=102)

ada.fit(X_train,y_train)
ada_score = cross_val_score(ada, X_train, y_train, cv=10,scoring='roc_auc_ovo')

ada_score_teste = cross_val_score(ada, X_test, y_test, cv=10,scoring='roc_auc_ovo')

print('Score AdaBoost Treino: ', round(ada_score.mean() * 100, 2).astype(str) + '%')

print('Score AdaBoost Teste: ', round(ada_score_teste.mean() * 100, 2).astype(str) + '%')
Y_pred_ada = ada.predict(X_test)
cm_ada = confusion_matrix(y_test,Y_pred_ada)
acc_score_ada = accuracy_score(y_test,Y_pred_ada)

f1_score_ada = f1_score(y_test,Y_pred_ada)

precisao_ada = average_precision_score(y_test,Y_pred_ada)

recall_ada = recall_score(y_test,Y_pred_ada)

roc_ada = roc_auc_score(y_test,Y_pred_ada,multi_class='ovo')

print('Acuracia ADA Boost ',round(acc_score_ada*100,2).astype(str)+'%')

print('Precião média Ada Boost ',round(precisao_ada*100,2).astype(str)+'%')

print('F1 Ada Boost ',round(f1_score_ada*100,2).astype(str)+'%')

print('Recall Ada Boost ',round(recall_ada*100,2).astype(str)+'%')

print('ROC Ada Boost ',round(roc_ada*100,2).astype(str)+'%')
accuracy.append(acc_score_ada)

precision.append(precisao_ada)

recall.append(recall_ada)

f1.append(f1_score_ada)

roc.append(roc_ada)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_ada, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Ada Boost \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
#print("Random Forest Classifier")

#forest_params = {"max_depth": list(range(10,50,1)),"n_estimators" : [350,400,450]}

#forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')

#forest.fit(X_train, y_train)

#random_forest = forest.best_estimator_

#print("Best Estimator")

#print(random_forest)
#Parameters have been choosing based on GridSearchCV

random_forest = RandomForestClassifier(max_depth=10,n_estimators=350)

random_forest.fit(X_train,y_train)
forest_score = cross_val_score(random_forest, X_train, y_train, cv=10,scoring='roc_auc_ovo')

forest_score_teste = cross_val_score(random_forest, X_test, y_test, cv=10,scoring='roc_auc_ovo')

print('Score RFC Treino: ', round(forest_score.mean() * 100, 2).astype(str) + '%')

print('Score RFC Teste: ', round(forest_score_teste.mean() * 100, 2).astype(str) + '%')
Y_pred_rf = random_forest.predict(X_test)
cm_rf = confusion_matrix(y_test,Y_pred_rf)
acc_score_rf = accuracy_score(y_test,Y_pred_rf)

f1_score_rf = f1_score(y_test,Y_pred_rf)

precisao_rf = average_precision_score(y_test,Y_pred_rf)

recall_rf = recall_score(y_test,Y_pred_rf)

roc_rf = roc_auc_score(y_test,Y_pred_rf,multi_class='ovo')

print('Acuracia Random Forest ',round(acc_score_rf*100,2).astype(str)+'%')

print('Precião média Random Forest ',round(precisao_rf*100,2).astype(str)+'%')

print('F1 Random Forest ',round(f1_score_rf*100,2).astype(str)+'%')

print('Recall Random Forest ',round(recall_rf*100,2).astype(str)+'%')

print('ROC Random Forest ',round(roc_rf*100,2).astype(str)+'%')
accuracy.append(acc_score_rf)

precision.append(precisao_rf)

recall.append(recall_rf)

f1.append(f1_score_rf)

roc.append(roc_rf)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_rf, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Random Forest \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
#print("Gradient Boost Classifier")

#grad_params = {'n_estimators' : [50,55,60,65,70,75,80,85,90],'max_depth' : list(range(3,11,1))}

#grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')

#grad.fit(X_train, y_train)

#grad_boost = grad.best_estimator_

#print("Best Estimator")

#print(grad_boost)
#Parameters have been choosing based on GridSearchCV

grad_boost = GradientBoostingClassifier(n_estimators=65,max_depth=4)

grad_boost.fit(X_train, y_train)
grad_score = cross_val_score(grad_boost, X_train, y_train, cv=10,scoring='roc_auc_ovo')

grad_score_teste = cross_val_score(grad_boost, X_test, y_test, cv=10,scoring='roc_auc_ovo')

print('Score GradBoost Treino: ', round(grad_score.mean() * 100, 2).astype(str) + '%')

print('Score GradBoost Teste: ', round(grad_score_teste.mean() * 100, 2).astype(str) + '%')
Y_pred_gb = grad_boost.predict(X_test)
cm_gb = confusion_matrix(y_test,Y_pred_gb)
acc_score_gb = accuracy_score(y_test,Y_pred_gb)

f1_score_gb = f1_score(y_test,Y_pred_gb)

precisao_gb = average_precision_score(y_test,Y_pred_gb)

recall_gb = recall_score(y_test,Y_pred_gb)

roc_gb = roc_auc_score(y_test,Y_pred_gb,multi_class='ovo')

print('Acuracia Gradient Boosting ',round(acc_score_gb*100,2).astype(str)+'%')

print('Precião média Gradient Boosting  ',round(precisao_gb*100,2).astype(str)+'%')

print('F1 Gradient Boosting  ',round(f1_score_gb*100,2).astype(str)+'%')

print('Recall Gradient Boosting  ',round(recall_gb*100,2).astype(str)+'%')

print('ROC Gradient Boosting ',round(roc_gb*100,2).astype(str)+'%')
accuracy.append(acc_score_gb)

precision.append(precisao_gb)

recall.append(recall_gb)

f1.append(f1_score_gb)

roc.append(roc_gb)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_gb, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Gradient Boosting  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
resultados = [log_reg_score,knears_score,ada_score,forest_score,grad_score]

resultados_teste = [log_reg_score_teste,knears_score_teste,ada_score_teste,forest_score_teste,grad_score_teste]

nome_modelo = ["Logistic Regression","KNN","AdaBoost","RFC","GradBoost"]
fig,ax=plt.subplots(figsize=(10,5))

ax.boxplot(resultados)

ax.set_xticklabels(nome_modelo)

plt.tight_layout()
fig,ax=plt.subplots(figsize=(10,5))

ax.boxplot(resultados_teste)

ax.set_xticklabels(nome_modelo)

plt.tight_layout()
n_inputs = X_train.shape[1]
modelo = Sequential()

modelo.add(Dense(128, input_shape=(n_inputs, ), activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dropout(0.5))

modelo.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dropout(0.5))

modelo.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))

modelo.add(BatchNormalization())

modelo.add(Dropout(0.5))

modelo.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, mode='auto', min_delta=0.0001)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

callbacks_list = [reduce_lr,es]

bsize = 2000
modelo.compile(Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train, batch_size=bsize, epochs=200, verbose=2, validation_data=(X_test,y_test),callbacks=callbacks_list)
Y_pred_keras = modelo.predict_classes(X_test, batch_size=bsize, verbose=0)
cm_keras = confusion_matrix(y_test,Y_pred_keras)

acc_score_keras = accuracy_score(y_test,Y_pred_keras)

f1_score_keras = f1_score(y_test,Y_pred_keras)

precisao_keras = average_precision_score(y_test,Y_pred_keras)

recall_keras = recall_score(y_test,Y_pred_keras)

roc_keras = roc_auc_score(y_test,Y_pred_keras,multi_class='ovo')

print('Acuracia Keras ',round(acc_score_keras*100,2).astype(str)+'%')

print('Precião média Keras  ',round(precisao_keras*100,2).astype(str)+'%')

print('F1 Keras  ',round(f1_score_keras*100,2).astype(str)+'%')

print('Recall Keras  ',round(recall_keras*100,2).astype(str)+'%')

print('ROC Keras ',round(roc_keras*100,2).astype(str)+'%')
accuracy.append(acc_score_keras)

precision.append(precisao_keras)

recall.append(recall_keras)

f1.append(f1_score_keras)

roc.append(roc_keras)
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(cm_keras, ax=ax, annot=True, cmap=plt.cm.copper)

ax.set_title("Keras  \n Matriz de Confusão", fontsize=14)

ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)

ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)
nome_modelo = ["Logistic Regression","KNN","AdaBoost","RFC","GradBoost","Keras"]

dic_metrics = {'Model' : nome_modelo, 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1' : f1, 'ROC' : roc}

dataframe = pd.DataFrame(dic_metrics)
dataframe_sorted =  dataframe.sort_values(by=['ROC','Accuracy','Recall','F1','Precision'],ascending=False).reset_index().drop('index',axis=1)
dataframe_sorted