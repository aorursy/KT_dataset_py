import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

%matplotlib inline
plt.rcParams.update(params)
da=pd.read_excel(r"C:\Users\aferr\Documents\MAE_5905_ime_usp_2020\Exercícios_mae_5905\Lista 4\EX3.xlsx")
da.head()
daX=da[["X1","X2"]]
daY=da[["Y"]]
plt.scatter(da["X1"],da["X2"])
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(daX, daY)
from sklearn import preprocessing
le = preprocessing.LabelBinarizer()
y = le.fit_transform(daY)
y = np.ravel(daY)
from sklearn.svm import SVC
svm = SVC()
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10],'gamma':[0.001, 0.01, 0.1, 1]}
grid_search_svm = GridSearchCV(svm,cv=3, param_grid=param_grid,scoring='recall')
grid_search_svm.fit(daX, y)
# altera-se aqui o scoring para 'recall', dentro do conceito da perda de oportunidade
grid_search_svm.best_estimator_
# mostra-se o conjunto de resultados, inclusive o kernel que resultou no melhor "rbf" = radius base function
# estes são os parâmetros que devem ser considerados, não são mais necessários os 30 originais
# para evolução, use doravante esse conjunto e não mais o original
# o grid search já salva esses parâmetros.
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
svm_pred = grid_search_svm.predict(daX)
acuracia_svm = accuracy_score(y_pred=svm_pred, y_true=y)
precisao_svm = precision_score(y_pred=svm_pred, y_true=y)
recall_svm = recall_score(y_pred=svm_pred, y_true=y)
svm_cm = confusion_matrix(y,svm_pred)
print(acuracia_svm,precisao_svm,recall_svm)
print(svm_cm)
grid_search_svm = GridSearchCV(svm,cv=3, param_grid=param_grid,scoring='accuracy')
grid_search_svm.fit(daX, y)
grid_search_svm.best_estimator_
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
svm_pred = grid_search_svm.predict(daX)

acuracia_svm = accuracy_score(y_pred=svm_pred, y_true=y)
precisao_svm = precision_score(y_pred=svm_pred, y_true=y)
recall_svm = recall_score(y_pred=svm_pred, y_true=y)
svm_cm = confusion_matrix(y,svm_pred)
print(acuracia_svm,precisao_svm,recall_svm)
print(svm_cm)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
svc = SVC(gamma="auto")
cv_result = cross_val_score(svc,daX,daY, cv=2, scoring="accuracy")
print("Acurácia com cross validation:", cv_result.mean()*100)
x1=np.random.uniform(-0.5,0.5,500)
x2=np.random.uniform(-0.5,0.5,500)
y=abs((x1**2 - x2**2))
y
daXY = dat[['x1','x2','y']]
ax1 = daXY.plot(kind='scatter', x='x1', y='y', color='r', label = "x1 versus y") 
ax2 = daXY.plot(kind='scatter', x='x2', y='y', color='b', label = "x2 versus y",ax=ax1) 
print(ax1 == ax2)
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
dat = pd.DataFrame({'x1': x1, 'x2': x2, "y": y}, columns=['x1',"x2",'y'])
dat.head
dat.describe()
# Tornar Y - Categórico (0 ou 1)
def num (data):
    nota = 2
    if data <= 0.0073911:
        nota = 0
    elif data > 0.0073911:
        nota = 1
    return nota
dat['y1'] = dat['y'].apply(num)
daX=dat[["x1","x2"]]
daY=dat[["y1"]]
from sklearn import preprocessing
le = preprocessing.LabelBinarizer()
y = le.fit_transform(daY)
y = np.ravel(daY)
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(daX,y,test_size = 0.25) 
x_train.head()
y_train.head()
# esperado_nao_ter_resultado
from sklearn import svm
# criação do classificador
clf = svm.SVC(kernel='linear') # Linear Kernel
# treinamento
clf.fit(x_train, y_train)
# predição
y_pred = clf.predict(x_test)
# Avaliação
from sklearn import metrics
# Acurácia
print("Acurácia:",metrics.accuracy_score(y_test, y_pred))
# Avaliação Complementar: Precisão e Recall
print("Precisao:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
from sklearn.svm import SVC
svm = SVC()
from sklearn.model_selection import GridSearchCV
# C = parâmetro de penalização. C_pequeno = margem pequena no hiperplano. C_grande = margem grande no hiperplano.
# Gamma factor = ajuste aos pontos. Gamma_grande = overfitting.
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10],'gamma':[0.001, 0.01, 0.1, 1]}
grid_search_svm = GridSearchCV(svm,cv=3, param_grid=param_grid,scoring='accuracy')
grid_search_svm.fit(x_train,y_train)
grid_search_svm.best_estimator_
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
svm_pred = grid_search_svm.predict(daX)

acuracia_svm = accuracy_score(y_pred=svm_pred, y_true=y)
precisao_svm = precision_score(y_pred=svm_pred, y_true=y)
recall_svm = recall_score(y_pred=svm_pred, y_true=y)
svm_cm = confusion_matrix(y,svm_pred)
print(acuracia_svm,precisao_svm,recall_svm)
print(svm_cm)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
svc = SVC(gamma="auto")
cv_result = cross_val_score(svc,x_train, y_train, cv=10, scoring="accuracy")
print("Acurácia com cross validation:", cv_result.mean()*100)
