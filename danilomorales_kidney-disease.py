import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dados = pd.read_csv('../input/ckdisease/kidney_disease.csv')
dados.head()
dados = dados.drop('id',axis=1)
dados.head()
dados.info()
dados.isna().sum()
dados = dados.dropna()
dados.isna().sum()
dados.head()
dados = dados.reset_index().drop('index',axis=1)
sns.countplot(dados['classification'])
fig, ax = plt.subplots(2,5,figsize=(12,7))
sns.countplot(dados['ane'],ax=ax[0][0],hue=dados['classification'])
sns.countplot(dados['pe'],ax=ax[0][1],hue=dados['classification'])
sns.countplot(dados['appet'],ax=ax[0][2],hue=dados['classification'])
sns.countplot(dados['cad'],ax=ax[0][3],hue=dados['classification'])
sns.countplot(dados['dm'],ax=ax[0][4],hue=dados['classification'])
sns.countplot(dados['htn'],ax=ax[1][0],hue=dados['classification'])
sns.countplot(dados['rbc'],ax=ax[1][1],hue=dados['classification'])
sns.countplot(dados['pc'],ax=ax[1][2],hue=dados['classification'])
sns.countplot(dados['pcc'],ax=ax[1][3],hue=dados['classification'])
sns.countplot(dados['ba'],ax=ax[1][4],hue=dados['classification'])
plt.tight_layout()
dados['pcv'] = dados['pcv'].astype(int)
dados['wc'] = dados['wc'].astype(int)
dados['rc'] = dados['rc'].astype(float)
fig, ax = plt.subplots(4,3,figsize=(12,7))
ax[0][0].hist(dados['bgr'])
ax[0][0].set_title('bgr')
ax[0][1].hist(dados['bu'])
ax[0][1].set_title('bu')
ax[0][2].hist(dados['sc'])
ax[0][2].set_title('sc')

ax[1][0].hist(dados['sod'])
ax[1][0].set_title('sod')
ax[1][1].hist(dados['pot'])
ax[1][1].set_title('pot')
ax[1][2].hist(dados['hemo'])
ax[1][2].set_title('hemo')

ax[2][0].hist(dados['pcv'])
ax[2][0].set_title('pcv')
ax[2][1].hist(dados['wc'])
ax[2][1].set_title('wc')
ax[2][2].hist(dados['rc'])
ax[2][2].set_title('rc')

ax[3][0].hist(dados['age'])
ax[3][0].set_title('age')
ax[3][1].hist(dados['sg'])
ax[3][1].set_title('sg')
ax[3][2].hist(dados['bp'])
ax[3][2].set_title('bp')
plt.tight_layout()
colunas_normalizar = ['bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','age','bp']
from sklearn.preprocessing import MinMaxScaler
for col in colunas_normalizar:
    scaler = MinMaxScaler(feature_range=(0,1))
    dados[col] = scaler.fit_transform(dados[col].values.reshape(-1,1))
dados.head()
colunas_onehot = ['rbc','pc','ba','pcc','pe','appet','cad','dm','htn','ane']
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
for col in colunas_onehot:
    enc = OneHotEncoder()
    dados[col] = enc.fit_transform(dados[col].values.reshape(-1,1)).toarray()
enc = LabelEncoder()
dados['classification'] = enc.fit_transform(dados['classification'])
dados.head()
corr = dados.corr()
fig,ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,ax=ax)
plt.tight_layout()
from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,recall_score,roc_auc_score
colunas_X = dados.columns.drop('classification')
X = dados.drop('classification',axis=1).values
Y = dados['classification'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=42,stratify=Y)
accuracy = []
precision =[]
recall = []
f1 = []
roc = []
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
print("Logistic Regression")
log_reg_params = {"penalty": ['None','l1', 'l2','elasticnet'], 'C': [1, 10, 100], 
                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_log_reg = GridSearchCV(LogisticRegression(max_iter=10000), log_reg_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
grid_log_reg.fit(X_train, y_train)
logreg = grid_log_reg.best_estimator_
log_reg_score = cross_val_score(logreg, X_train, y_train, cv=10,scoring='roc_auc_ovo')
log_reg_score_teste = cross_val_score(logreg, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print("Best Estimator")
print(logreg)
print('Score Regressao Logistica Train: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
print('Score Regressao Logistica Test: ', round(log_reg_score_teste.mean() * 100, 2).astype(str) + '%')
logreg.fit(X_train,y_train)
Y_pred_logreg = logreg.predict(X_test)
cm_logreg = confusion_matrix(y_test,Y_pred_logreg)
acc_score_logreg = accuracy_score(y_test,Y_pred_logreg)
f1_score_logreg = f1_score(y_test,Y_pred_logreg)
precisao_logreg = average_precision_score(y_test,Y_pred_logreg)
recall_logreg = recall_score(y_test,Y_pred_logreg)
roc_logreg = roc_auc_score(y_test,Y_pred_logreg,multi_class='ovo')
print('Accuracy Logistic Regression ',round(acc_score_logreg*100,2).astype(str)+'%')
print('Precision Logistic Regression ',round(precisao_logreg*100,2).astype(str)+'%')
print('F1 Logistic Regression ',round(f1_score_logreg*100,2).astype(str)+'%')
print('Recall Logistic Regression ',round(recall_logreg*100,2).astype(str)+'%')
print('ROC Logistic Regression ',round(roc_logreg*100,2).astype(str)+'%')
accuracy.append(acc_score_logreg)
precision.append(precisao_logreg)
recall.append(recall_logreg)
f1.append(f1_score_logreg)
roc.append(roc_logreg)
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_logreg, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
importance_logreg = logreg.coef_[0]
feature_series_logreg = pd.Series(data=importance_logreg,index=colunas_X)
feature_series_logreg.plot.bar()
plt.title('Feature Importance Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
print("KNN")
knears_params = {"n_neighbors": list(range(5,30,1)),'leaf_size' : list(range(3,11,1)), 'weights': ['uniform', 'distance']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
grid_knears.fit(X_train, y_train)
knn = grid_knears.best_estimator_
knears_score = cross_val_score(knn, X_train, y_train, cv=10,scoring='roc_auc_ovo')
knears_score_teste = cross_val_score(knn, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print("Best Estimator")
print(knn)
print('Score KNN Train: ', round(knears_score.mean() * 100, 2).astype(str) + '%')
print('Score KNN Test: ', round(knears_score_teste.mean() * 100, 2).astype(str) + '%')
knn.fit(X_train,y_train)
Y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test,Y_pred_knn)
acc_score_knn = accuracy_score(y_test,Y_pred_knn)
f1_score_knn = f1_score(y_test,Y_pred_knn)
precisao_knn = average_precision_score(y_test,Y_pred_knn)
recall_knn = recall_score(y_test,Y_pred_knn)
roc_knn = roc_auc_score(y_test,Y_pred_knn,multi_class='ovo')
print('Accuracy KNN ',round(acc_score_knn*100,2).astype(str)+'%')
print('Precision KNN ',round(precisao_knn*100,2).astype(str)+'%')
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
ax.set_title("KNN \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
print("Ada Boost Classifier")
ada_params = {'n_estimators' : list(range(5,200))}
grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
grid_ada.fit(X_train, y_train)
ada = grid_ada.best_estimator_
print("Best Estimator")
print(ada)
ada_score = cross_val_score(ada, X_train, y_train, cv=10,scoring='roc_auc_ovo')
ada_score_teste = cross_val_score(ada, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score AdaBoost Train: ', round(ada_score.mean() * 100, 2).astype(str) + '%')
print('Score AdaBoost Test: ', round(ada_score_teste.mean() * 100, 2).astype(str) + '%')
ada.fit(X_train,y_train)
Y_pred_ada = ada.predict(X_test)
cm_ada = confusion_matrix(y_test,Y_pred_ada)
acc_score_ada = accuracy_score(y_test,Y_pred_ada)
f1_score_ada = f1_score(y_test,Y_pred_ada)
precisao_ada = average_precision_score(y_test,Y_pred_ada)
recall_ada = recall_score(y_test,Y_pred_ada)
roc_ada = roc_auc_score(y_test,Y_pred_ada,multi_class='ovo')
print('Accuracy ADA Boost ',round(acc_score_ada*100,2).astype(str)+'%')
print('Precision Ada Boost ',round(precisao_ada*100,2).astype(str)+'%')
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
ax.set_title("Ada Boost \n Confusion matrix", fontsize=14)
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
importance_ada = ada.feature_importances_
feature_series_ada = pd.Series(data=importance_ada,index=colunas_X)
feature_series_ada.plot.bar()
plt.title('Feature Importance Ada Boost')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
print("Random Forest Classifier")
forest_params = {"max_depth": list(range(5,10,1)),"n_estimators" : list(range(5,10,1))}
forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
forest.fit(X_train, y_train)
random_forest = forest.best_estimator_
print("Best Estimator")
print(random_forest)
forest_score = cross_val_score(random_forest, X_train, y_train, cv=10,scoring='roc_auc_ovo')
forest_score_teste = cross_val_score(random_forest, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score RFC Train: ', round(forest_score.mean() * 100, 2).astype(str) + '%')
print('Score RFC Test: ', round(forest_score_teste.mean() * 100, 2).astype(str) + '%')
random_forest.fit(X_train,y_train)
Y_pred_rf = random_forest.predict(X_test)
cm_rf = confusion_matrix(y_test,Y_pred_rf)
acc_score_rf = accuracy_score(y_test,Y_pred_rf)
f1_score_rf = f1_score(y_test,Y_pred_rf)
precisao_rf = average_precision_score(y_test,Y_pred_rf)
recall_rf = recall_score(y_test,Y_pred_rf)
roc_rf = roc_auc_score(y_test,Y_pred_rf,multi_class='ovo')
print('Accuracy Random Forest ',round(acc_score_rf*100,2).astype(str)+'%')
print('Precision Random Forest ',round(precisao_rf*100,2).astype(str)+'%')
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
ax.set_title("Random Forest \n Confusion matrix", fontsize=14)
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
importance_rfc = random_forest.feature_importances_
feature_series_rfc = pd.Series(data=importance_rfc,index=colunas_X)
feature_series_rfc.plot.bar()
plt.title('Feature Importance Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
print("Gradient Boost Classifier")
grad_params = {'n_estimators' : list(range(4,21,1)),'max_depth' : list(range(5,21,1))}
grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
grad.fit(X_train, y_train)
grad_boost = grad.best_estimator_
print("Best Estimator")
print(grad_boost)
grad_score = cross_val_score(grad_boost, X_train, y_train, cv=10,scoring='roc_auc_ovo')
grad_score_teste = cross_val_score(grad_boost, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score GradBoost Train: ', round(grad_score.mean() * 100, 2).astype(str) + '%')
print('Score GradBoost Test: ', round(grad_score_teste.mean() * 100, 2).astype(str) + '%')
grad_boost.fit(X_train, y_train)
Y_pred_gb = grad_boost.predict(X_test)
cm_gb = confusion_matrix(y_test,Y_pred_gb)
acc_score_gb = accuracy_score(y_test,Y_pred_gb)
f1_score_gb = f1_score(y_test,Y_pred_gb)
precisao_gb = average_precision_score(y_test,Y_pred_gb)
recall_gb = recall_score(y_test,Y_pred_gb)
roc_gb = roc_auc_score(y_test,Y_pred_gb,multi_class='ovo')
print('Accuracy Gradient Boosting ',round(acc_score_gb*100,2).astype(str)+'%')
print('Precision Gradient Boosting  ',round(precisao_gb*100,2).astype(str)+'%')
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
ax.set_title("Gradient Boosting  \n Confusion matrix", fontsize=14)
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
importance_gradboost = grad_boost.feature_importances_
feature_series_gradboost = pd.Series(data=importance_gradboost,index=colunas_X)
feature_series_gradboost.plot.bar()
plt.title('Feature Importance Gradient Boosting')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
n_inputs = X_train.shape[1]
from keras.models import Sequential
from keras.layers import Activation,BatchNormalization
from keras.layers.core import Dense,Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
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
bsize = 50
modelo.compile(Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X_train, y_train, batch_size=bsize, epochs=200, verbose=2, validation_data=(X_test,y_test),callbacks=callbacks_list)
Y_pred_keras = modelo.predict_classes(X_test, batch_size=bsize, verbose=0)
cm_keras = confusion_matrix(y_test,Y_pred_keras)
acc_score_keras = accuracy_score(y_test,Y_pred_keras)
f1_score_keras = f1_score(y_test,Y_pred_keras)
precisao_keras = average_precision_score(y_test,Y_pred_keras)
recall_keras = recall_score(y_test,Y_pred_keras)
roc_keras = roc_auc_score(y_test,Y_pred_keras,multi_class='ovo')
print('Accuracy Keras ',round(acc_score_keras*100,2).astype(str)+'%')
print('Precision Keras  ',round(precisao_keras*100,2).astype(str)+'%')
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
ax.set_xticklabels(['ckd', 'notckd'], fontsize=14, rotation=0)
ax.set_yticklabels(['ckd', 'notckd'], fontsize=14, rotation=360)
nome_modelo = ["Logistic Regression","KNN","AdaBoost","RFC","GradBoost","Keras"]
dic_metrics = {'Model' : nome_modelo, 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1' : f1, 'ROC' : roc}
dataframe = pd.DataFrame(dic_metrics)
dataframe_sorted =  dataframe.sort_values(by=['ROC','Accuracy','Recall','F1','Precision'],ascending=False).reset_index().drop('index',axis=1)
dataframe_sorted