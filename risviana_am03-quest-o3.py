# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import seaborn as sn
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import category_encoders as ce
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importando dataset
data = pd.read_csv("/kaggle/input/iris/Iris.csv",usecols=['SepalLengthCm','SepalWidthCm',
                                                          'PetalLengthCm','PetalWidthCm','Species'])
#Motando conjunto de teste e treino
def montar_conjunto_teste_treino():
    
    frame=[data.iloc[0:75],data.iloc[75:]]
    test_train_set=pd.concat(frame)
    #criando coluna classificação
    test_train_set['conjunto']=None
    test_train_set.iloc[0:75,5:]='conjunto de treino'
    test_train_set.iloc[75:,5:]='conjunto de teste'
    return test_train_set

#motando a mastriz de distância euclidiana entre o conjunto de teste e treino

def montar_matriz_de_distancia(test_train_set):

    distance_matrix= spatial.distance.cdist(test_train_set.iloc[75:,:4],
                                            test_train_set.iloc[:75,:4], metric='euclidean')

    #coluna representa o indice do conjunto de treino e linha o indece do conjunto de teste
    return pd.DataFrame(distance_matrix,index=test_train_set.iloc[75:].index)

#selecionando 1_nn vizinhos mais próximo
def selecionar_k_vizinhos_proximos(quant_vizinho,distance_matrix):
    trans_matrix=distance_matrix.T
    lista=[]
    for k in distance_matrix.index:
        ex_kNN=trans_matrix[k].nsmallest(n=quant_vizinho, keep='first')
        serie={'menor_peso':ex_kNN.values[0], 'index_menor_peso':ex_kNN.index[0]}
        lista.append(serie)
    return  pd.DataFrame(lista,index=[distance_matrix.index])

#classificar espécie do conjunto de teste utilizando 1_nn sem peso  
def classificar_especie_knn_sem_peso(lista_k_nn,test_train_set):
    
    #criando colunas
    lista_k_nn_copy=lista_k_nn.copy()
    lista_k_nn['Species']=''
    lista_k_nn['classificação']=''
    lista_classif=[]
    lista_index=lista_k_nn_copy['index_menor_peso']
    for i in range(len(lista_k_nn_copy)):
        
        lista_k_nn.iloc[i:i+1,3:]=[test_train_set.iloc[lista_index.iloc[i]]['Species']]
        lista_k_nn.iloc[i:i+1,2:3]=[test_train_set.iloc[lista_k_nn.index[i]]['Species']]

    return lista_k_nn
test_train_set=montar_conjunto_teste_treino()
#test_train_set




distance_matrix=montar_matriz_de_distancia(test_train_set)
lista_k_nn=selecionar_k_vizinhos_proximos(1,distance_matrix)
tabela_classif=classificar_especie_knn_sem_peso(lista_k_nn,test_train_set)
#tabela_classif

def plot_confusion_matrix(cm, classes,normalize,title,cmap):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.tight_layout()


#cnf_matrix=confusion_matrix(tabela_classif['Species'].values,tabela_classif['classificação'].values)
#classes=class_names[[1,2]]
# Plot non-normalized confusion matrix
#pyplot.figure()
#plot_confusion_matrix(cnf_matrix, classes,False,'Confusion matrix, without normalization',pyplot.cm.Blues)
#pyplot.show()

#recall para classe Iris-setosa
VP_setosa=len(tabela_classif[(tabela_classif['classificação']=='Iris-setosa') &
                             (tabela_classif['Species']=='Iris-setosa')])
FN_setosa=len(tabela_classif[(tabela_classif['classificação']!='Iris-setosa') &
                             (tabela_classif['Species']=='Iris-setosa')])
recall_setosa=0.0
#recall para classe Iris-versicolor
VP_versicolor=len(tabela_classif[(tabela_classif['classificação']=='Iris-versicolor') &
                             (tabela_classif['Species']=='Iris-versicolor')])
FN_versicolor=len(tabela_classif[(tabela_classif['classificação']!='Iris-versicolor') &
                             (tabela_classif['Species']=='Iris-versicolor')])
recall_versicolor=VP_versicolor/(VP_versicolor+FN_versicolor)
#recall para classe Iris-virginica
VP_virginica=len(tabela_classif[(tabela_classif['classificação']=='Iris-Iris-virginica') &
                             (tabela_classif['Species']=='Iris-Iris-virginica')])
FN_virginica=len(tabela_classif[(tabela_classif['classificação']!='Iris-virginica') &
                             (tabela_classif['Species']=='Iris-virginica')])
recall_virginica=VP_virginica/(VP_virginica+FN_virginica)

#visualização
recall_por_classe = { 'recall_setosa': recall_setosa, 'recall_virginica': recall_virginica,'recall_versicolor':recall_versicolor}

#print(recall_por_classe)

#média ponderada do recall
media_recall=(recall_setosa+recall_virginica+recall_versicolor)/3
#media_recall
#precisão Iris-setosa
FP_setosa=len(tabela_classif[(tabela_classif['classificação']=='Iris-setosa') &
                             (tabela_classif['Species']!='Iris-setosa')])
precisao_setosa=0.0
#precisão Iris-versicolor
FP_versicolor=len(tabela_classif[(tabela_classif['classificação']=='Iris-versicolor') &
                             (tabela_classif['Species']!='Iris-versicolor')])
precisao_versicolor=VP_versicolor/(VP_versicolor+FP_versicolor)
#precisão Iris-virginica
FP_virginica=len(tabela_classif[(tabela_classif['classificação']=='Iris-virginica') &
                             (tabela_classif['Species']!='Iris-virginica')])
precisao_virginica=0.0
#visualização
precisao_por_classe = { 'precisão_setosa': precisao_setosa, 'precisão_virginica': precisao_virginica,
                     'precisão_versicolor':precisao_versicolor}

#print(precisao_por_classe)
#média ponderada da precisão
media_precisao=(precisao_setosa+precisao_virginica+precisao_versicolor)/3
#media_precisao
#Média-F setosa
#F_setosa= 2 * precisao_setosa *recall_setosa / (precisao_setosa + recall_setosa)
F_setosa=0
#Média-F versicolor
F_versicolor= 2 * precisao_versicolor *recall_versicolor / (precisao_versicolor + recall_versicolor)
#Média-F virginica
#F_virginica= 2 * precisao_virginica *recall_virginica / (precisao_virginica + recall_virginica)
F_virginica=0
#visualização
mediaF_por_classe = { 'mediaF_setosa': F_setosa, 'mediaF_virginica': F_versicolor,
                     'mediaF_versicolor':F_virginica}

#print(mediaF_por_classe)
#média ponderada da média F
mediaF_ponderada=(F_setosa+F_versicolor+F_virginica)/3
#media_precisao
#Taxa de FP Iris-setosa
taxa_fp_setosa=0
#Taxa de FP Iris-versicolor
VN_versicolor=len(tabela_classif[(tabela_classif['classificação']=='Iris-versicolor') &
                             (tabela_classif['Species']!='Iris-versicolor')])
taxa_fp_versicolor=FP_versicolor/(FP_versicolor+VN_versicolor)

#Taxa de FP Iris-virginica
VN_virginica=len(tabela_classif[(tabela_classif['classificação']=='Iris-virginica') &
                             (tabela_classif['Species']!='Iris-virginica')])
#taxa_fp_virginica=FP_virginica/(FP_virginica+VN_virginica)
taxa_fp_virginica=0
#visualização
taxa_fp_por_classe = { 'taxa_fp_setosa': taxa_fp_setosa, 'taxa_fp_virginica': taxa_fp_versicolor,
                     'taxa_fp_versicolor':taxa_fp_virginica}

#print(taxa_fp_por_classe)

#média ponderada da Taxa de FP
media_Taxa_FP=(taxa_fp_setosa+taxa_fp_versicolor+taxa_fp_virginica)/3
#media_precisao
x=data.iloc[:,:4]
y=data['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=0.5)

neigh=KNeighborsClassifier(n_neighbors=1,metric='euclidean')
clf=neigh.fit(x_train, y_train)
y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)*100
#Pré-processamento
y_pred=pd.DataFrame(y_pred)
encoder = ce.OrdinalEncoder(cols=[0])
y_pred = encoder.fit_transform(y_pred)

y_test=pd.DataFrame(y_test)
encoder = ce.OrdinalEncoder(cols=['Species'])
y_test = encoder.fit_transform(y_test)

fpr, tpr, thresholds=metrics.roc_curve(y_test,y_pred,2)

roc_auc=auc(fpr,tpr)

pyplot.figure()
lw = 2
pyplot.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.plot(0.2, 0.98, 'o')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic example')
pyplot.legend(loc="lower right")

pyplot.show()


