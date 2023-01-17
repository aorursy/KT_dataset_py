# Importa Bibliotecas

import numpy as np
import pandas as pd
import copy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Este vai converter a última coluna de categorical data para integer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
print(os.listdir("../input"))

# Faz a leitura do csv

dados = pd.read_csv('../input/sonar.all-data.csv', header = None)
   
#x = dados[:,0:60].astype(float)
#y = dados[:,60]
# Erro -> NumPy-style indexing doesn't work on a Pandas DataFrame; use loc or iloc
# https://www.kaggle.com/residentmario/indexing-selecting-assigning-reference
# https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c

tipo = dados[60]
valores = dados.iloc[:, :-1]

# Converter última coluna usando LabelEncoder()
# https://stackoverflow.com/questions/34915813/convert-text-columns-into-numbers-in-sklearn
# https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

le = LabelEncoder()
tipo_le = le.fit_transform(tipo)

# colocar um parâmetro a mais "n" e incluir o "n" na função aleatória

def vizinhanca(gamma, C, n):         
    
    gamma_copy = copy.copy(gamma)
    C_copy = copy.copy(C)
    
    # aumentar o raio
    gamma = gamma + (2 * np.random.rand() - 1) * gamma * n  
    C = (C + (2 * np.random.rand() - 1) * C) * n 
    
   
    while gamma <= 0 or C <= 0:
        print('Entrou no while gamma/C <= 0')
        gamma = gamma_copy
        C = C_copy
        gamma = (gamma + (2 * np.random.rand() - 1) * gamma) * n
        C = (C + (2 * np.random.rand() - 1) * C) * n
        
        
    print('C: {}'.format(C))    
    print('gamma: {}'.format(gamma))   
    return gamma, C
def f(y_true, y_pred):
    
    return np.linalg.norm(y_true - y_pred)

def VNS(gamma_inicial, C_inicial, max_iter, max_vizinh, max_fail):
    
    
    X_train, X_test, y_train, y_test = train_test_split(valores,
                                                       tipo_le,
                                                       test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    i = 0 
    # n não pode ser 0
    n = 1
    
    gamma = gamma_inicial
    C = C_inicial
    fail = 0
    
    for i in range(max_iter):
        print('Iniciou o for....')        
        i += 1                
        print('i: {}'.format(i))
        
        while n < max_vizinh:
            
            gamma_s1, C_s1 = vizinhanca(gamma, C,n)                        

            
            while fail < max_fail:                
                gamma_s2 = gamma_s1 + gamma_s1 * 0.3
                C_s2 = C_s1 + C_s1 * 0.7
                
                svm_s1 = SVC(gamma=gamma_s1, C=C_s1)
                svm_s2 = SVC(gamma=gamma_s2, C=C_s2)
                
                svm_s1 = svm_s1.fit(X_train, y_train)
                svm_s2 = svm_s2.fit(X_train, y_train)
                
                y_pred_s1 = svm_s1.predict(X_train)
                y_pred_s2 = svm_s2.predict(X_train)
                
                f_s1 = f(y_train, y_pred_s1)
                f_s2 = f(y_train, y_pred_s2)
                                
                if f_s2 <= f_s1:
                    gamma_s1 = gamma_s2
                    C_s1 = C_s2
                    
                else:
                    fail = fail + 1
                    print('fail: {}'.format(fail))
            
            svm_s = SVC(gamma=gamma, C=C)
            svm_s = svm_s.fit(X_train, y_train)
       
            y_pred_s = svm_s.predict(X_train)
            
            f_s = f(y_train, y_pred_s)
            
            if f_s2 <= f_s:
                gamma = gamma_s2
                C = C_s2
                n = 1
            else:
                n = n + 1
               
    print('SA terminado!\n')
    print('Melhor gamma: {}'.format(gamma_s1))
    print('Melhor C: {}'.format(C_s1))
    
    print('Treinando modelo final....')
    final_svm = SVC(gamma=gamma, C=C)
    final_svm = final_svm.fit(X_train, y_train)
    print('Avaliando acurácia no conjunto de teste...')
    accuracy = final_svm.score(X_test, y_test)
    print('Acurácia: {}'.format(accuracy))
VNS(0.1, 2, 10, 10, 5)