import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import os

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head()
number_of_fraud = len(data[data.Class == 1])

number_of_normal= len(data[data.Class == 0])



print ("Fraude:", number_of_fraud)

print ("Normal:",number_of_normal)
sns.countplot("Class",data=data)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50



ax1.hist(data.Time[data.Class == 1], bins = bins)

ax1.set_title('Fraude')



ax2.hist(data.Time[data.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Tempo')

plt.ylabel('Numero de transacoes')

plt.show()
print ("Fraude")

print (data.Amount[data.Class == 1].describe())

print ()

print ("Normal")

print (data.Amount[data.Class == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 10



ax1.hist(data.Amount[data.Class == 1], bins = bins)

ax1.set_title('Fraude')



ax2.hist(data.Amount[data.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Montante')

plt.ylabel('Numero de transacoes')

plt.show()
PCA_features = data.iloc[:,1:29].columns
plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(data[PCA_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(data[cn][data.Class == 1], bins=50)

    sns.distplot(data[cn][data.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histograma de recurso: ' + str(cn))

plt.show()
data = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

data.head()
data['Normalized_Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data.head()
data = data.drop(['Time','Amount'],axis=1)

data.head()
#índices of normal class

indices_of_normal = data[data.Class==0].index

#escolha aleatoriamente a mesma quantidade de amostras que a fraude e retorne seus índices

random_indices_of_normal = np.array(np.random.choice(indices_of_normal, number_of_fraud, replace=False))

#indices of fraud class

indices_of_fraud = np.array(data[data.Class == 1].index)

#indices of undersampled dataset

indices_of_undersampled = np.concatenate([random_indices_of_normal, indices_of_fraud])

#conjunto de dados com pouca amostra

data_of_undersampled = data.iloc[indices_of_undersampled,:]



print(len(data_of_undersampled))
#conjunto de dados inteiro

X = data.loc[:,data.columns!='Class']

y = data.loc[:,data.columns=='Class']



#treinar e testar o conjunto de dados dividido em todo o conjunto de dados, com proporção 70/30

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



print("Numero de transacoes treinar conjunto de dados: ", len(X_train))

print("Conjunto de dados de teste de transacoes numericas: ", len(X_test))

print("Numero total de transacoes: ", len(X_train)+len(X_test))
#conjunto de dados com pouca amostra

X_undersampled = data_of_undersampled.loc[:,data_of_undersampled.columns!='Class']

y_undersampled = data_of_undersampled.loc[:,data_of_undersampled.columns=='Class']



#conjunto de dados de trem e teste dividido a partir de um conjunto de dados com pouca amostra, com razão 70/30

X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(X_undersampled,y_undersampled,test_size = 0.3, random_state = 0)



print("Numero de transacoes treinar conjunto de dados: ", len(X_train_undersampled))

print("Conjunto de dados de teste de transacoes numericas: ", len(X_test_undersampled))

print("Numero total de transacoes: ", len(X_train_undersampled)+len(X_test_undersampled))
def train(model,X,y):

    

    # Recordar para o modelo

    clf = model

    

    # Diferentes parâmetros C para regularização

    C_param = [0.01,0.1,1,10,100]



    # Validação cruzada do K-Fold

    kf = KFold(n_splits=5)

    

    # Inicializacao

    scores     =[]

    best_score = 0

    best_C     = 0

    

    for C in C_param:

        

        clf.C = C



        score = []

        for train_index, test_index in kf.split(X): 



            # Use os dados de treinamento divididos para ajustar-se ao modelo.

            clf.fit(X.iloc[train_index,:].values,y.iloc[train_index,:].values.ravel())



            # Prever valores usando os dados de teste divididos

            y_pred = clf.predict(X.iloc[test_index,:].values)

            

            # Calcular a pontuação de rechamada e anexá-la a uma lista de pontuações de rechamada representando o parâmetro c_ atual

            rec = recall_score(y.iloc[test_index,:].values.ravel(),y_pred)

            

            # Anexar pontuação de recordar de cada iteração à pontuação

            score.append(rec)



        # Calcule a pontuação média real para todas as iterações e compare-a com a melhor pontuação.

        mean_score = np.mean(score)

        if mean_score > best_score:

            best_score = mean_score

            best_C     = C

        

        # Anexar a pontuação média de cada C às pontuações

        scores.append(np.mean(score))

        

    # Crie um quadro de dados para mostrar a pontuação média para cada parâmetro C    

    lr_results = pd.DataFrame({'Pontuacao':scores, 'C':C_param}) 

    print(lr_results)

    

    print("A melhor pontuacao de recordacao eh: ", best_score)

    print("O melhor parametro C eh: ", best_C)

    

    return best_score, best_C
def predict(model, X_train, y_train, X_test, y_test):

    # Recordar para o modelo

    clf = model

    #clf = Regressão logística (C = C, penalidade = 'l1')

    # Use todo o conjunto de dados de trem com pouca amostra para ajustar-se ao modelo.

    clf.fit(X_train.values,y_train.values.ravel())

    # Previsão no conjunto de dados de teste com pouca amostra



    y_pred = clf.predict(X_test.values)



    # Matriz de confusão

    CM = confusion_matrix(y_test.values, y_pred)

    # Obter verdadeiros positivos (tp), falsos positivos (fp), falsos negativos (fn)

    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()



    # Prediction report

    sns.heatmap(CM,cmap="coolwarm_r",annot=True,linewidths=0.5)

    plt.title("Matriz_de_confusao")

    plt.xlabel("Classe_prevista")

    plt.ylabel("Classe Real")

    plt.show()

    print("\n----------Relatorio de classificacao------------------------------------")

    print(classification_report(y_test.values, y_pred))
clf = LogisticRegression(penalty = 'l2', solver ='lbfgs')

best_score, best_C = train(clf, X_train_undersampled,y_train_undersampled)
clf = LogisticRegression(C=best_C, penalty = 'l2', solver ='lbfgs')

predict(clf, X_train_undersampled,y_train_undersampled,X_test_undersampled,y_test_undersampled)
predict(clf,X_train_undersampled,y_train_undersampled,X_test,y_test)
clf = LogisticRegression(penalty = 'l2',solver ='lbfgs')

best_score_whole, best_C_whole = train(clf,X_train,y_train)
clf = LogisticRegression(C=best_C_whole,penalty = 'l2',solver='lbfgs')

predict(clf,X_train,y_train,X_test,y_test)
clf = SVC(gamma='auto')

best_score, best_C = train(clf, X_train_undersampled,y_train_undersampled)
clf = SVC(C=best_C,gamma='auto')

predict(clf, X_train_undersampled,y_train_undersampled,X_test_undersampled,y_test_undersampled)
predict(clf,X_train_undersampled,y_train_undersampled,X_test,y_test)
predict(clf,X_train,y_train,X_test,y_test)