import numpy as np

import matplotlib.pyplot as plt

import sklearn as skl

import pandas as pd
adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape # nome_da_base.shape , shape == forma
adult.head() # nome_da_base.head() , head() == cabeçalho {Pequena demonstração da base}
nadult = adult.dropna() #Elimina-se observações com "missing data"
nadult.shape
adult.shape
Observacoes_com_missing_data = 32561 - 30162

print(Observacoes_com_missing_data) #numero absoluto de missing data

print((Observacoes_com_missing_data/32561)*100) #percentual de missing data
#importantdo a base de teste

testAdult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nTestAdult = testAdult.dropna()

print(testAdult.shape)

print(nTestAdult.shape)
from sklearn.linear_model import LogisticRegression #Chamando a Regressao Logistica na biblioteca sklearn

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
#Função Regressão Logistica com cross validation

def LR_CV(do_predict):

#Area de ajsustes de parametros da funcao:_______________________________________________________________

    pcv = 10 #Numero de divisoes da base de dados para cross validation

    

#A funcao:_______________________________________________________________________________________________

    from sklearn.linear_model import LogisticRegression #Chamando a Regressao Logistica na biblioteca sklearn

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import accuracy_score

    from sklearn import preprocessing

    #Instanciando e organizando a base de dados

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,0:14]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,0:14]

    YtestAdult = numTestAdult.Target

    #Realizando a regressão Logistica e corss validation

    LR = LogisticRegression(random_state=0)

    scores_cv = cross_val_score(LR, Xadult, Yadult, cv=pcv, scoring='accuracy')

    

    #Observacao das features relevantes por meio dos pesos calculados, "betas"

    LR.fit(Xadult,Yadult)

    coefs = pd.Series(LR.coef_[0], index=Xadult.columns)

    coefs.sort_values(ascending = False)

    

    i=0

    average = 0

    while i<pcv :

        average = average + scores_cv[i]

        i = i + 1

    average = average/pcv

    

    if(do_predict == 1):

        YtestPred = LR.predict(XtestAdult)

        print(YtestPred)

    return average #Retorna a media dos resultados do cross validation



#COMO USAR: do_predict eh um booleano[0,1] que indica se

#uma previsao deve ser feita sobre a base de testes ou nao.
#Função para mostrar os pesos da regressao logistica, mostra "betas"

def betas():

    pcv = 10;

    from sklearn.linear_model import LogisticRegression #Chamando a Regressao Logistica na biblioteca sklearn

    

    #Instanciando e organizando a base de dados

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,0:14]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,0:14]

    YtestAdult = numTestAdult.Target

    #Realizando a regressão Logistica e corss validation

    LR = LogisticRegression(random_state=0)

    scores_cv = cross_val_score(LR, Xadult, Yadult, cv=pcv, scoring='accuracy')

    

    #Observacao das features relevantes por meio dos pesos calculados, "betas"

    LR.fit(Xadult,Yadult)

    coefs = pd.Series(LR.coef_[0], index=Xadult.columns)

    return coefs.sort_values(ascending = False)
#Função para mostrar os pesos da regressao logistica, mostra "betas"

#com selecao de atributos

def betas_select(Atributos):

    pcv = 10;

    from sklearn.linear_model import LogisticRegression #Chamando a Regressao Logistica na biblioteca sklearn

    

    #Instanciando e organizando a base de dados

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,Atributos]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,Atributos]

    YtestAdult = numTestAdult.Target

    

    #Realizando a regressão Logistica e corss validation

    LR = LogisticRegression(random_state=0)

    scores_cv = cross_val_score(LR, Xadult, Yadult, cv=pcv, scoring='accuracy')

    

    #Observacao das features relevantes por meio dos pesos calculados, "betas"

    LR.fit(Xadult,Yadult)

    coefs = pd.Series(LR.coef_[0], index=Xadult.columns)

    return coefs.sort_values(ascending = False)

#Função Regressão Logistica com cross validation e selecao de atributos via 

# vetor numerico 

def LR_cv_select(do_predict,Atributos):

    pcv = 10;

    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import accuracy_score

    from sklearn import preprocessing

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,Atributos]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,Atributos]

    YtestAdult = numTestAdult.Target

    #Realizando a regressão Logistica e corss validation

    LR = LogisticRegression(random_state=0)

    scores_cv = cross_val_score(LR, Xadult, Yadult, cv=pcv, scoring='accuracy')

    

    

    i=0

    average = 0

    while i<pcv :

        average = average + scores_cv[i]

        i = i + 1

    average = average/pcv

    

    if(do_predict == 1):

        LR.fit(Xadult,Yadult)

        YtestPred = LR.predict(XtestAdult)

        accuracy_pred = accuracy_score(YtestAdult,YtestPred)

        print(accuracy_pred)

    return average #Retorna a media dos resultados do cross validation

    
LR_CV(1)
betas()
betas_select([4,9,11,10,0,12])
LR_cv_select(1,[4,9,11,10,0,12]) #Com parametros selecionados via RL
LR_cv_select(0,[0,3,6,7,10,11]) #com parametros selecionados no data prep do EP1
#Funcao KNN com cross validation

def knn_cv_select(KN,Atributos):

    pcv = 10;

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import accuracy_score

    from sklearn import preprocessing

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,Atributos]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,Atributos]

    YtestAdult = numTestAdult.Target

    knn = KNeighborsClassifier(n_neighbors=KN)

    scores_cv = cross_val_score(knn, Xadult, Yadult, cv=pcv)

    

    i=0

    average = 0

    while i<pcv :

        average = average + scores_cv[i]

        i = i + 1

    average = average/pcv

    

    knn.fit(Xadult,Yadult)

    YtestPred = knn.predict(XtestAdult)

    accuracy_pred = accuracy_score(YtestAdult,YtestPred)

    print(accuracy_pred)

        

    return average
knn_cv_select(16,[0,3,6,7,10,11]) #com parametros selecionados no EP1 pra KNN
knn_cv_select(16,[4,9,11,10,0,12]) #com parametros selcionados via RegLogist
LR_cv_select(0,[4,9,0]) #somente os mais importantes
LR_cv_select(0,[4,9,11,10,0,12,2]) #adicionando o melhor dos piores (fnlwgt)
LR_cv_select(0,[2,4,6,8,10,12]) #Alto grau de aleatoriedade em relacao ao problema

LR_cv_select(1,[4,9,11,10,0,12])
#Funcao Random Forest com cross validation

def RF_cv_select_pred(do_predict,rs,Atributos):

    pcv = 10;

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import accuracy_score

    from sklearn import preprocessing

    #Data base

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,Atributos]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,Atributos]

    YtestAdult = numTestAdult.Target

    #Processing Data

    RF = RandomForestClassifier(random_state=rs)

    scores_cv = cross_val_score(RF, Xadult, Yadult, cv=pcv)

    

    i=0

    average = 0

    while i<pcv :

        average = average + scores_cv[i]

        i = i + 1

    average = average/pcv

    

    if(do_predict == 1):

        RF.fit(Xadult,Yadult)

        YtestPred = RF.predict(XtestAdult)

        accuracy_pred = accuracy_score(YtestAdult,YtestPred)

        print("Teste")

        print(accuracy_pred)

    print("Cross_Validation")

    return average #Retorna a media dos resultados do cross validation
RF_cv_select_pred(1,1,[1,2,3,4,5,6,7,8,9,10,11,12,13]) #1 arvore
RF_cv_select_pred(1,2,[1,2,3,4,5,6,7,8,9,10,11,12,13]) #2 arvores
RF_cv_select_pred(1,10,[1,2,3,4,5,6,7,8,9,10,11,12,13]) #10 arvores
RF_cv_select_pred(1,20,[1,2,3,4,5,6,7,8,9,10,11,12,13])
RF_cv_select_pred(1,50,[1,2,3,4,5,6,7,8,9,10,11,12,13])
RF_cv_select_pred(1,100,[1,2,3,4,5,6,7,8,9,10,11,12,13])
RF_cv_select_pred(1,500,[1,2,3,4,5,6,7,8,9,10,11,12,13])
i=0

while i<21 :

    print(i,RF_cv_select_pred(1,i,[1,2,3,4,5,6,7,8,9,10,11,12,13]))

    i = i + 1
arr = np.array([0.8347255179535971,0.8339962288266571,0.8329685422307043,0.8336314633233639,0.8331009922850943,0.8338305562832211,0.8330681545952977,0.8343938319905859,0.8330680006639337,0.8347252431042806,0.8311115913310386,0.835256054931208,0.8337641112686788,0.833731405588787,0.8332335962343975,0.8344600570761023,0.8336314633671048,0.8341292175241787,0.8328359932818827,0.8345267877986287])

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=1

arr_res = []

while i<31 :

    arr_res.insert(i, RF_cv_select_pred(1,i,[0,3,6,7,10,11]))

    i = i + 1

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
i=1

arr_res = []

while i<31 :

    rf = RF_cv_select_pred(1,i,[4,9,11,10,0,12])

    arr_res.insert(i, rf)

    print(i)

    i = i + 1

arr = np.array(arr_res)

Index_Max_Accuracy = np.where(arr == np.amax(arr))

print('List of Indices of maximum element :', Index_Max_Accuracy[0])

# Get the maximum element from a Numpy array

print('Max accuracy is :', np.amax(arr))
RF_cv_select_pred(1,19,[0,3,6,7,10,11])
#Funcao SVM - Polinomial gamma auto

def SVM_cv_select_pred_pol(do_predict,Atributos):

    pcv = 10;

    from sklearn.svm import SVC

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import accuracy_score

    from sklearn import preprocessing

    #Data base

    numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

    numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

    Xadult = numAdult.iloc[:,Atributos]

    Yadult = numAdult.Target

    XtestAdult = numTestAdult.iloc[:,Atributos]

    YtestAdult = numTestAdult.Target

    #Processing Data

    SVM = SVC(kernel='rbf')

    scores_cv = cross_val_score(SVM, Xadult, Yadult, cv=pcv)

    

    i=0

    average = 0

    while i<pcv :

        average = average + scores_cv[i]

        i = i + 1

    average = average/pcv

    

    if(do_predict == 1):

        SVM.fit(Xadult,Yadult)

        YtestPred = SVM.predict(XtestAdult)

        accuracy_pred = accuracy_score(YtestAdult,YtestPred)

        print("Teste")

        print(accuracy_pred)

    print("Cross_Validation")

    return average #Retorna a media dos resultados do cross validation
SVM_cv_select_pred_pol(1,[0])
SVM_cv_select_pred_pol(1,[0,3,6,7,10,11])
SVM_cv_select_pred_pol(1,[4,9,11,10,0,12])