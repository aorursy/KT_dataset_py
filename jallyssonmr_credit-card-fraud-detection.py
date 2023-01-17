import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import LeaveOneOut

from sklearn import metrics

%matplotlib inline
def read(dataset_path):

    try:

        return pd.read_csv(dataset_path)

    except:

        print("FILE NOT FOUND: " + dataset_path)
# INFO: READ DATASET

# -- leitura dos dados originais

dataframe_credit_card_fraud = read("../input/creditcardfraud/creditcard.csv")
# INFO: DATA TYPES

# -- verificando se é preciso codificar algum atributo (nominal)

dataframe_credit_card_fraud.dtypes
# INFO: NUMBER OF ROWS

# -- saber o tamanho do dataset para analisar se tem valores faltantes nos atributos

len(dataframe_credit_card_fraud)
# INFO: CHECKING EMPTY FIELDS

# -- substituindo valores faltantes por np.nan para remover do dataframe 

# -- e saber se o número de registros é menor que o original

len(dataframe_credit_card_fraud.replace(r'^\s*$', np.nan, regex=True).dropna())
# INFO: SCALE OF VALUES

# -- verificando atributos em escalas diferentes (Time e Amount)

dataframe_credit_card_fraud.describe()
# INFO: UNDERSTANDING DATA

# -- verificando a distribuição das classes (TOTALMENTE desbalanceado)

dataframe_credit_card_fraud["Class"].value_counts().plot.bar()
# -- função para plotar gráficos

def plot_graphic(x, y, x_label, y_label):

    plt.plot(x, y, color='tab:blue')

    plt.grid()

    plt.xlabel(x_label)

    plt.ylabel(y_label)
# INFO: PLOT AMOUNT X TIME

# -- verificando os atributos de escala maiores para normalizar

column_x = "Time"

column_y = "Amount"

plot_graphic(dataframe_credit_card_fraud[column_x], dataframe_credit_card_fraud[column_y], column_x, column_y)
# INFO: PLOT TIME

# -- verificando a quantidade de transações ao longo do tempo (segundos)

# -- as partes com menor frequencia parecem ser a mudanca de dia (periodo da noite)

dataframe_credit_card_fraud["Time"].plot.hist(bins=30, edgecolor='black')
# INFO: PLOT CLASS X TIME

# -- verificando os segundos que aparecem classes positivas

column_x = "Time"

column_y = "Class"

plot_graphic(dataframe_credit_card_fraud[column_x], dataframe_credit_card_fraud[column_y], column_x, column_y)
# -- função para pegar apenas os registros com classe positiva

def get_class_positive(dataframe_credit_card_fraud):

    return dataframe_credit_card_fraud.loc[dataframe_credit_card_fraud["Class"] == 1]
# INFO: PLOT CLASS POSITIVE X TIME

# -- verificando os segundos que aparecem classes positivas

column_x = "Time"

column_y = "Class"

dataframe_class_positive = get_class_positive(dataframe_credit_card_fraud)

plot_graphic(dataframe_class_positive[column_x], dataframe_class_positive[column_y], column_x, column_y)
# INFO: PLOT TIME

# -- verificando a quantidade de transações ao longo do tempo (segundos) para as classes positivas

dataframe_class_positive["Time"].plot.hist(bins=30, edgecolor='black')
# -- função para ajustar o desbalanceamento

# -- seleciona aleatoriamente N registros da classe negativa onde N é igual ao número de regitros da classe positiva

# -- e junta esses registros com os da classe positiva

def adjusts_the_unbalance(lenght_dataframe_class_positive, dataframe_class_positive, dataframe_credit_card_fraud):

    dataframe_class_false = dataframe_credit_card_fraud.loc[dataframe_credit_card_fraud["Class"] == 0]

    return pd.concat([dataframe_class_false.sample(n=lenght_dataframe_class_positive), dataframe_class_positive])
# -- função para saber a quantidade de registros da classe positiva

def get_lenght_class_positive(dataframe_class_positive):

    return len(dataframe_class_positive)
# -- verificando a distribuição das classes com o ajuste do desbalanceamento

# -- salva o dataframe no formato csv como um novo dataset (controle)

lenght_class_positive = get_lenght_class_positive(dataframe_class_positive)

dataframe_balanced = adjusts_the_unbalance(lenght_class_positive, dataframe_class_positive, dataframe_credit_card_fraud)

dataframe_balanced["Class"].value_counts().plot.bar()
# -- função para normalizar os atributos de Time e Amount

# -- usa o metodo min-max

def normalize_dataframe(dataframe_balanced):

    cols_to_norm = ["Time", "Amount"]

    dataframe_balanced[cols_to_norm] = dataframe_balanced[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return dataframe_balanced
# -- faz a normalização de Time e Amount e salva o csv do dataframe como um novo dataset

normalize = normalize_dataframe(dataframe_balanced)

normalize.head()
# INFO: TRAINING MODEL (KNN AND LEAVE_ONE_OUT)

# -- visto que ficaram poucos registro (haviam poucos da classe positiva)

# -- assim foi utilizado o metodo leave-one-out para validar os modelos

# -- garantindo que cada registro aparece pelo menos uma vez no treino e teste

def training_with_leave_one_out(execution, k_range, data, target):

    scores_accuracy = {}

    scores_f1_score = {}

    confusion_matrix_values = {}  

    for k in k_range:

        loo = LeaveOneOut()

        targets_test = []

        targets_predict = [] 

        target_test_index = 0



        for train_index, test_index in loo.split(data):

            data_train, data_test = data.loc[train_index], data.loc[test_index]

            target_train, target_test = target.loc[train_index], target.loc[test_index]

            knn = KNeighborsClassifier(n_neighbors=k)

            knn.fit(data_train, target_train)

            target_predict = knn.predict(data_test)

            

            targets_test.append(target_test[target_test_index])

            targets_predict.append(target_predict[0])

            

            target_test_index += 1

        

        scores_accuracy[k] = metrics.accuracy_score(targets_test, targets_predict)

        scores_f1_score[k] = metrics.f1_score(targets_test, targets_predict)

        

        confusion_matrix_values[k] = targets_test, targets_predict

    return scores_accuracy, scores_f1_score, confusion_matrix_values
# -- definindo o range do parametro k do algoritmo knn

# -- preparando o dataset balanceado e normalizado

# -- executando o range de k 10 vezes pegando variações aleatorias ao fazer o balanceamento dos dados

# -- fazendo a media das metricas de avaliação (acurácia e f1-score) nas 10 execuções

k_range = range(1, 10, 2)



scores_accuracy = {}

scores_f1_score = {}



execution_number = 10



for execution in range(execution_number):

    dataframe_class_positive = get_class_positive(dataframe_credit_card_fraud)

    lenght_class_positive = get_lenght_class_positive(dataframe_class_positive)

    

    dataframe_balanced = adjusts_the_unbalance(lenght_class_positive, dataframe_class_positive, dataframe_credit_card_fraud)

    dataframe_normalize = normalize_dataframe(dataframe_balanced)

    

    dataframe_normalize = read("../input/datasets-balancead/creditcard_balanced_normalize_" + str(execution) + ".csv")

    

    data = dataframe_normalize.iloc[:, 2:29]

    target = dataframe_normalize["Class"]



    accuracy, f1_score, confusion_matrix_values = training_with_leave_one_out(execution, k_range, data, target)

    

    for k in k_range:

        if (k in scores_accuracy):

            scores_accuracy[k] = float(scores_accuracy[k]) + float(accuracy[k])

            scores_f1_score[k] = float(scores_f1_score[k]) + float(f1_score[k])

        else:

            scores_accuracy[k] = float(accuracy[k])

            scores_f1_score[k] = float(f1_score[k])

    

for k in k_range:

    scores_accuracy[k] = scores_accuracy[k]/execution_number

    scores_f1_score[k] = scores_f1_score[k]/execution_number
# INFO: SCORE GRAPHIC

# -- funçào para plotar o score de cada parâmetro k do algoritmo

def score_graphic(x, y, y_label):

    plt.plot(x, y)

    plt.xlabel("Value of k for KNN")

    plt.ylabel(y_label)

    plt.grid()
# INFO: SCORE GRAPHIC (ACCURACY)

score_graphic(k_range, scores_accuracy.values(), "Testing Accuracy")
# INFO: CONFUSION MATRIX

# -- necessário para analisar os dados desbalanceados

# -- extrair as metricas de avaliação

def plot_confusion_matrix(values_true, values_predict):

    cm = metrics.confusion_matrix(values_true, values_predict)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(tick_marks, ["False", "True"])

    plt.yticks(tick_marks, ["False", "True"])

    plt.ylabel('Value true')

    plt.xlabel('Value predict')
# INFO: CONFUSION MATRIX (BEST RESULT)

# -- função para plotar a matriz de confusão que teve melhor média de score

def best_confusion_matrix(scores, confusion_matrix_values):

    INDEX_VALUES_TRUE = 0

    INDEX_VALUES_PREDICT = 1

    

    key_max_score = max(scores, key=scores.get)

    

    values_true = confusion_matrix_values[key_max_score][INDEX_VALUES_TRUE]

    values_predict = confusion_matrix_values[key_max_score][INDEX_VALUES_PREDICT]

    

    plot_confusion_matrix(values_true, values_predict)

    print("------------------------------------")

    print(metrics.confusion_matrix(values_true, values_predict))

    print("------------------------------------")

    tn, fp, fn, tp = metrics.confusion_matrix(values_true, values_predict).ravel()

    

    print("TN: " + str(tn))

    print("FP: " + str(fp))

    print("FN: " + str(fn))

    print("TP: " + str(tp))

    print("------------------------------------")
# INFO: CONFUSION MATRIX (BEST RESULT - ACCURACY)

best_confusion_matrix(scores_accuracy, confusion_matrix_values)
# INFO: SCORE GRAPHIC (F1-SCORE)

score_graphic(k_range, scores_f1_score.values(), "Testing F1-Score")
# INFO: CONFUSION MATRIX (BEST RESULT - F1-SCORE)

best_confusion_matrix(scores_f1_score, confusion_matrix_values)