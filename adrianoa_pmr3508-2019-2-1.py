#Tratamento de dados

%matplotlib inline

import pandas as pd

import numpy as np

import sklearn 

import os



#Métodos de classificação

#a. Berouli Naive-Bayes

from sklearn.naive_bayes import BernoulliNB

#b. Random Forest

from sklearn.ensemble import RandomForestClassifier

#c. Logist Regression

from sklearn.linear_model import LogisticRegression



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets, linear_model

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
#cwd = os.getcwd()

#os.listdir(cwd)
#base_file = (r"C:\Users\adria\Documents\POLI USP\7º Semestre\PMR3508\1_trabalho\train_data.csv")
#base_adult = pd.read_csv(base_file, 

                                         #names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", 

                                                        #"Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                                                        #"Hours per week", "Country", "Target"],      

                                        #sep=r'\s*,\s*',

                                        #engine='python',

                                        #na_values = "?")
#test_file = (r"C:\Users\adria\Documents\POLI USP\7º Semestre\PMR3508\1_trabalho\test_data.csv")
#test_adult = pd.read_csv(test_file, 

                                         #names=[ 'Id', "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", 

                                                        #"Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                                                        #"Hours per week", "Country", "Target"],      

                                        #sep=r'\s*,\s*',

                                        #engine='python',

                                        #na_values = "?")
base_adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

                         names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                                                        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", 

                                                        "Capital Loss","Hours per week", "Country", "Target"], 

                        sep = r'\s*,\s*',

                        engine = 'python',

                        na_values = "?")



test_adult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                         names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                                                        "Occupation","Relationship", "Race", "Sex", "Capital Gain", 

                                                        "Capital Loss", "Hours per week", "Country", "Target"], 

                        sep = r'\s*,\s*',

                        engine = 'python',

                        na_values = "?")
#Nesta célula, preencheu-se os espaços em branco das variáveis que tinham sua caegoria com frequência maior que 50%  com essa maior frequência.

base_adult["Workclass"] = base_adult["Workclass"].fillna('Private')

base_adult["Race"] = base_adult["Race"].fillna('White')

base_adult["Sex"] = base_adult["Sex"].fillna('Male')

base_adult["Capital Gain"] = base_adult["Capital Gain"].fillna('0')

base_adult["Capital Loss"] = base_adult["Capital Loss"].fillna('0')

base_adult["Country"] = base_adult["Country"].fillna('United-States')

base_adult["Target"] = base_adult["Target"].fillna('<=50K ') 
#Aqui, após o preenchimento de alguns dados, elimina-se todas as linhas que tenham dados faltantes

nadult = base_adult.dropna()
adult_treated = nadult.apply(preprocessing.LabelEncoder().fit_transform)
adult_treated = adult_treated.drop(adult_treated.index[0]) #Aqui elimina-se a linha "intrusa" observada acima
#Agora comecei a repetir o tratamento dado na base de aprendizado na base objeto

test_adult["Workclass"] = test_adult["Workclass"].fillna('Private')

test_adult["Race"] = test_adult["Race"].fillna('White')

test_adult["Sex"] = test_adult["Sex"].fillna('Male')

test_adult["Capital Gain"] = test_adult["Capital Gain"].fillna(0)

test_adult["Capital Loss"] = test_adult["Capital Loss"].fillna(0)

test_adult["Country"] = test_adult["Country"].fillna('United-States')

test_adult["Target"] = test_adult["Target"].fillna('<=50K ') 
natest = test_adult.dropna()
test_treated = nadult.apply(preprocessing.LabelEncoder().fit_transform)
test_treated = test_treated.drop(test_treated.index[0]) #Aqui elimina-se a linha "intrusa" observada acima
adult_parametro = adult_treated

aux = adult_parametro.astype(np.int)



corr_mat = aux.corr()

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm", annot = True)
Xadult = adult_treated[["Age", "Martial Status", "Relationship","Sex", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = adult_treated.Target



XtestAdult = test_treated[["Age", "Martial Status", "Relationship", "Sex", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = test_treated.Target
# a. Implementação Bernoulli Naive-Bayes

clf = BernoulliNB()

clf.fit(Xadult, Yadult )
scoresB = cross_val_score(clf, Xadult, Yadult, cv = 10)

scoresB
# b. Random Forest

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(Xadult, Yadult)
rfc_pred = rfc.predict(XtestAdult)

print('Accuracy: ',accuracy_score(YtestAdult, rfc_pred))
#c. Regrssão Logística

logmodel = LogisticRegression()

logmodel.fit(Xadult,Yadult)
log_pred = logmodel.predict(XtestAdult)

print('Accuracy: ',accuracy_score(YtestAdult, log_pred))