# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix,roc_curve,auc,roc_auc_score

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Churn_Modelling_Uni.csv')
data.drop(['RowNumber','CustomerId','Surname'],axis=1, inplace=True)
data.head()
fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(40,30))
tipo_plot = ['hist','bar','bar','hist','bar','hist','bar','bar','bar','hist']
for i in range(data.shape[1]-1):
    if tipo_plot[i] == 'hist':
        pd.DataFrame([data.loc[data.Exited==0,data.columns[i]],\
                      data.loc[data.Exited==1,data.columns[i]]])\
        .T.plot(kind=tipo_plot[i],ax=axes[i//4,i%4],title = data.columns[i],legend = False,fontsize=20,rot=30,alpha=0.5)
    elif tipo_plot[i] == 'bar':
        pd.DataFrame([data.loc[data.Exited==0,data.columns[i]].value_counts(),\
                      data.loc[data.Exited==1,data.columns[i]].value_counts()])\
        .T.plot(kind=tipo_plot[i],ax=axes[i//4,i%4],title = data.columns[i],legend = False,fontsize=20,rot=0)
    axes[i//4,i%4].title.set_size(20)
sns.heatmap(data.loc[:,~data.columns.isin(['Geography','Gender','HasCrCard','IsActiveMember','Exited'])].corr(),\
           vmin = -1.0, vmax=1.0, annot=True)
dummies = pd.get_dummies(data[['Geography','Gender']])
data = data.join(dummies).copy()
data.drop(['Geography','Gender','Gender_Female'],axis=1,inplace= True)
data.head()
y = data.Exited.copy()
X = data.drop('Exited',axis=1).copy()
parameters = {'max_depth':range(3,20)}
scoring = ['accuracy','balanced_accuracy','roc_auc','f1','precision','recall']

for i in range(X.shape[1]-2):
    
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring= scoring, refit='roc_auc', return_train_score =True)
    clf.fit(X=X, y=y)
    
    
    idx = clf.cv_results_['params'].index(clf.best_params_)
    resultado =  clf.best_params_
    resultado.update({'{}'.format(s) : clf.cv_results_['mean_test_{}'.format(s)][idx] for s in scoring})
    
    if i==0:
        df_resultado = pd.Series(resultado,name = 'todos_incluidos').to_frame()
        modelos = {'todos_incluidos':clf}
        col_variables = {'todos_incluidos':X.columns}
    else:
        df_resultado = df_resultado.join(pd.Series(resultado,name = 'sin_{}'.format(drop_col))).copy()
        modelos.update({'sin_{}'.format(drop_col):clf })
        col_variables.update({'sin_{}'.format(drop_col):X.columns })
    
    drop_col = X.columns[np.argmin(clf.best_estimator_.feature_importances_)]
    X.drop(drop_col,axis=1,inplace=True)
df_resultado
pd.Series(modelos['todos_incluidos'].best_estimator_.feature_importances_*100,\
          index=col_variables['todos_incluidos']).sort_values().plot.barh(figsize=(15,10))
ax = pd.DataFrame({'Prueba':        modelos['sin_EstimatedSalary'].cv_results_['mean_test_roc_auc'],\
                   'Entrenamiento': modelos['sin_EstimatedSalary'].cv_results_['mean_train_roc_auc']},\
                  index=[i['max_depth'] for i in modelos['sin_EstimatedSalary'].cv_results_['params']]).plot(figsize=(10,10))
ax.set_xlabel('Max_Depth')
ax.set_ylabel('ROC_AUC')
dot_data = tree.export_graphviz(modelos['sin_EstimatedSalary'].best_estimator_, out_file=None, 
                     feature_names=col_variables['sin_EstimatedSalary'],  
                     class_names=np.array(['No Exited', 'Exited']),  
                     filled=True, rounded=True,  
                     special_characters=True)  
graphviz.Source(dot_data)  
cm = confusion_matrix(y,modelos['sin_EstimatedSalary'].best_estimator_.predict(data[col_variables['sin_EstimatedSalary']]))
ax = sns.heatmap(cm,annot=True, fmt='.0f',cbar=False)
ax.set(xlabel='Condición Predecida', ylabel='Condición Actual')
plt.show()
print('Sensibilidad: {:.1f}%\nEspecificidad: {:.1f}%'.format(cm[1,1]/cm[1].sum()*100,cm[0,0]/cm[0].sum()*100))
from sklearn import metrics
probs = modelos['sin_EstimatedSalary'].best_estimator_.predict_proba(data[col_variables['sin_EstimatedSalary']])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
y = data.Exited.copy()
X = data.drop('Exited',axis=1).copy()
parameters = {'C':[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30]}
scoring = ['accuracy','balanced_accuracy','roc_auc','f1','precision','recall']

for i in range(X.shape[1]-4):
    
    clf = GridSearchCV(LogisticRegression(solver  ='liblinear' ), parameters, cv=5, scoring= scoring, refit='roc_auc', return_train_score =True)
    clf.fit(X=X, y=y)
        
    idx = clf.cv_results_['params'].index(clf.best_params_)
    resultado =  clf.best_params_
    resultado.update({'{}'.format(s) : clf.cv_results_['mean_test_{}'.format(s)][idx] for s in scoring})
    
    if i==0:
        df_resultado = pd.Series(resultado,name = 'todos_incluidos').to_frame()
        modelos = {'todos_incluidos':clf}
        col_variables = {'todos_incluidos':X.columns}
    else:
        df_resultado = df_resultado.join(pd.Series(resultado,name = 'sin_{}'.format(drop_col))).copy()
        modelos.update({'sin_{}'.format(drop_col):clf })
        col_variables.update({'sin_{}'.format(drop_col):X.columns })
    
    drop_col = X.columns[np.argmin(abs(clf.best_estimator_.coef_ ))]
    X.drop(drop_col,axis=1,inplace=True)
    
df_resultado
pd.Series(modelos['todos_incluidos'].best_estimator_.coef_.ravel()/max(abs(modelos['todos_incluidos'].best_estimator_.coef_.ravel()))*100,\
          index=col_variables['todos_incluidos']).sort_values().plot.barh(figsize=(15,10))
ax = pd.DataFrame({'Prueba':        modelos['sin_HasCrCard'].cv_results_['mean_test_roc_auc'],\
                   'Entrenamiento': modelos['sin_HasCrCard'].cv_results_['mean_train_roc_auc']},\
                  index=[i['C'] for i in modelos['sin_HasCrCard'].cv_results_['params']]).plot(figsize=(10,10))
ax.set_xlabel('C')
ax.set_ylabel('ROC_AUC')
ax.set_xscale('log')
ax.set_xlim((0.1,30))
ax.set_ylim((0.76,0.765))
