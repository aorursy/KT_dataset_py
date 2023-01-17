import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



df = pd.read_csv("../input/creditcard.csv",encoding="utf-8")



# Fill NA/NaN values with 0

df = df.fillna(0)

df.head(5)

print("Número de amostras da classe 0 (transação normal):")

print(sum(df.Class==0))



print("Número de amostras da classe 1 (transação fraudulenta):")

print(sum(df.Class==1))
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler



# Separando os dados em X, com os dados/features e y com os labels

X = df.iloc[:, df.columns != "Class"]

y = df.iloc[:, df.columns == "Class"]



# Separando os dados em test e train

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



# Undersampling data

rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)



print("Número de amostras na base de treino da classe 0 (transação normal):")

print(sum(y_resampled==0))



print("Número de amostras na base de treino da classe 1 (transação fraudulenta):")

print(sum(y_resampled==1))
from sklearn.metrics import confusion_matrix, classification_report

def evaluates_model(model, X, y, probability):

    '''

    Reveices a model, and prints report and confusion matrix

    for a given dataset and a defined probability.

    '''



    predicted = model.predict_proba(X)

    predicted = [0 if x[0]>probability else 1 for x in predicted]

    target_names = ['Normal', 'Fraude']

    print(classification_report(y, predicted, target_names=target_names))

    print("Confusion matrix:")

    print(confusion_matrix(y, predicted))

    return predicted



from sklearn.metrics import recall_score, precision_score, f1_score

import numpy as np

import matplotlib.pyplot as plt



def plotRecallPrecision(mdl, X_test, y_test, space):

    recall_axis = []

    prec_axis = []

    f1_axis = []

    x_axis = np.linspace(space[0],space[1], 20)

    for prob in x_axis:

        predicted = mdl.predict_proba(X_test)

        predicted = [0 if x[0]>prob else 1 for x in predicted]

        # Calculating the recall 

        recall_axis.append(np.trunc(recall_score(y_test, predicted, average='binary')*100))

        # Calculating the precision 

        prec_axis.append(np.trunc(precision_score(y_test, predicted, average='binary')*100))

        # Calculating the precision 

        f1_axis.append(np.trunc(f1_score(y_test, predicted, average='binary')*100))

    # Plot Grid search scores

    _, ax = plt.subplots(1,1)

    ax.set_title("Detecção de Fraudes", fontsize=20, fontweight='bold')

    ax.set_ylabel("%", fontsize=16)

    ax.set_xlabel("Probabilidade de corte", fontsize=16)

    ax.plot(x_axis, prec_axis, label='Precision')

    ax.plot(x_axis, recall_axis, label='Recall')

    ax.plot(x_axis, f1_axis, label='f1 score')

    ax.legend()

    ax.grid('on')
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



mdl = ExtraTreesClassifier(n_estimators=25, max_depth=10, bootstrap=True, n_jobs = 1,

                            max_leaf_nodes=40, min_samples_leaf = 2, 

                            min_samples_split=5, random_state=10)

mdl.fit(X_resampled, y_resampled.ravel())



plotRecallPrecision(mdl, X_test, y_test,[0.1,0.7])



print("_____________________________________________________________")

print("\nModelo: ExtraTreesClassifier   Probabilidade de corte: 0.3\n")

_ = evaluates_model(mdl, X_test, y_test, .3)
mdl = RandomForestClassifier(n_estimators=25, max_depth=10, bootstrap=True, n_jobs = 1,

                            max_leaf_nodes=40, min_samples_leaf = 2, 

                            min_samples_split=5, random_state=10)

mdl.fit(X_resampled, y_resampled.ravel())



plotRecallPrecision(mdl, X_test, y_test,[0.1,0.7])



print("_____________________________________________________________")

print("\nModelo: RandomForest   Probabilidade de corte: 0.4\n")

_ = evaluates_model(mdl, X_test, y_test, .4)
mdl = XGBClassifier()

mdl.fit(X_train, y_train)



#X_test = X_test[X_train.columns]

# Todo, solve problem with 



plotRecallPrecision(mdl, X_test, y_test, [0.4,0.98])
print("_____________________________________________________________")

print("\nModelo:  XGBClassifier   Probabilidade de corte: 0.8\n")

_ = evaluates_model(mdl, X_test, y_test, .8)





print("_____________________________________________________________")

print("\nModelo:  XGBClassifier   Probabilidade de corte: 0.95\n")

_ = evaluates_model(mdl, X_test, y_test, .95)