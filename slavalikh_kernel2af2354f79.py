# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#импортируем библиотеки необходимые для подготовки данных

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')

data.head()
data.info()
data['Tenure'] = data['Tenure'].fillna(-1)
data['Geography'].unique()
data.duplicated().sum()
data['CustomerId'].value_counts().sum()
#Обозначим новый датафрейм "data_ml" — данные для машинного обучения

#Удалим столбцы-идентификаторы, не представляющие ценностия для алгоритма 

#for_drop = ['RowNumber','CustomerId', 'Surname']

for_drop = ['RowNumber','CustomerId', 'Surname']

data_ml = data.drop(for_drop, axis=1)

data_ml.head()
data_ml.shape
data_ml = pd.get_dummies(data_ml, drop_first=True)

data_ml.head()
data_ml.shape
#разделим на признаки и целевой признак

features = data_ml.drop('Exited', axis=1)

target = data_ml['Exited']
features.head()
target.head()
#1. Выделим валидационную 60%

features_train, features_validtest, target_train, target_validtest = train_test_split(features,

                                                    target,

                                                    train_size=0.6,

                                                    random_state=12345, 

                                                    stratify=target)
print('Признаки обучающей выборки:',features_train.shape,  

      'Целевой признак обучающей выборки:', target_train.shape, 

      'Валидационная и тестовая вместе', features_validtest.shape, target_validtest.shape)
#2. Разделим оставшиеся 40% на 2 равные части (валидационная и тестовая)

features_valid, features_test, target_valid, target_test = train_test_split(features_validtest,

                                                    target_validtest,

                                                    train_size=0.5,

                                                    random_state=12345, 

                                                    stratify=target_validtest)
print(features_valid.shape, target_valid.shape, features_test.shape, target_test.shape)
#импортируем библиотеку для стандартноо масштабирования

from sklearn.preprocessing import StandardScaler
features_train.head()
#Для масштабирования зафиксируем численные признаки

numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#Создадим объект этой структуры и настроим его на обучающих данных:

scaler = StandardScaler()

scaler.fit(features_train[numeric])

#Масштабируем численные признаки обучающей выборки 

features_train[numeric] = scaler.transform(features_train[numeric])

features_train.head()
#Масштабируем численные признаки валидационной выборки 

features_valid[numeric] = scaler.transform(features_valid[numeric])

features_valid.head()
#Масштабируем численные признаки тестовой выборки 

features_test[numeric] = scaler.transform(features_test[numeric])

features_test.head()
#импорт библиотек трех моделей

#Деревое решений

from sklearn.tree import DecisionTreeClassifier



#Случайный лес 

from sklearn.ensemble import RandomForestClassifier



#Логистическая регрессия

from sklearn.linear_model import LogisticRegression
#Импортируем необходимые метики

#метика accuracy

from sklearn.metrics import accuracy_score



#матрица ошибок

from sklearn.metrics import confusion_matrix



#полнота

from sklearn.metrics import recall_score



#точность

from sklearn.metrics import precision_score



#F-1 мера

from sklearn.metrics import f1_score



#AUC-ROC

from sklearn.metrics import roc_auc_score



#ROC-кривая

from sklearn.metrics import roc_curve
def all_models_accuracy(features_train, target_train, features_valid, target_valid):

    model_DTC = DecisionTreeClassifier(random_state=123)

    DTC_score = model_DTC.fit(features_train, target_train).score(features_valid, target_valid)

    

    model_RFC = RandomForestClassifier(random_state=12345, n_estimators = 100)

    RFC_score = model_RFC.fit(features_train, target_train).score(features_valid, target_valid)

    

    model_LgR = LogisticRegression(solver = 'liblinear')

    LgR_score = model_LgR.fit(features_train, target_train).score(features_valid, target_valid)

    print("Точность:" "дерево решений", DTC_score, "случайный лес ", RFC_score, "логистческая регрессия", LgR_score)
all_models_accuracy(features_train, target_train, features_valid, target_valid)
target_train.value_counts(normalize = 1)
target_valid.value_counts(normalize = 1)
def all_models_share(features_train, target_train, features_valid, target_valid):

    model_DTC = DecisionTreeClassifier(random_state=123)

    model_DTC.fit(features_train, target_train)

    DTC_share = pd.Series(model_DTC.predict(features_valid)).value_counts(normalize = 1)

    

    

    

    model_RFC = RandomForestClassifier(random_state=12345, n_estimators = 100)

    model_RFC.fit(features_train, target_train)

    RFC_share = pd.Series(model_RFC.predict(features_valid)).value_counts(normalize = 1)

    

    model_LgR = LogisticRegression(solver = 'liblinear')

    model_LgR.fit(features_train, target_train)

    LgR_share = pd.Series(model_LgR.predict(features_valid)).value_counts(normalize = 1)

    



    

    print("Доли ответов:" "дерево решений", DTC_share, "случайный лес ", RFC_share, "логистческая регрессия", LgR_share , end='')
all_models_share(features_train, target_train, features_valid, target_valid)
#Создаем константную модель

target_predict_constant = pd.Series([0]*len(target_valid))

target_predict_constant.shape
accuracy_score_constant = accuracy_score(target_valid, target_predict_constant)

accuracy_score_constant
#матрица ошибок для дерево решений

model_DTC = DecisionTreeClassifier(random_state=123)

model_DTC.fit(features_train, target_train)

DTC_prediction = model_DTC.predict(features_valid)

confusion_matrix(target_valid, DTC_prediction)

def rec_prec_f1(target_valid, prediction):

    print("Полнота" , recall_score(target_valid, prediction))

    print("Точность", precision_score(target_valid, prediction))

    print("F1-мера", f1_score(target_valid, prediction))

    print("AUC-ROC", roc_auc_score(target_valid, prediction))

    
#полнота, точность и F1-мера для дерева решений

rec_prec_f1(target_valid, DTC_prediction)
#матрица ошибок для случайного леса

model_RFC = RandomForestClassifier(random_state=12345, n_estimators = 100)

model_RFC.fit(features_train, target_train)

RFC_prediction = model_RFC.predict(features_valid)

confusion_matrix(target_valid, RFC_prediction)

#полнота, точность и F1-мера для случайного леса

rec_prec_f1(target_valid, RFC_prediction)
#матрица ошибок для логистической регрессии

model_LgR = LogisticRegression(solver = 'liblinear')

model_LgR.fit(features_train, target_train)

LgR_prediction = model_LgR.predict(features_valid)

confusion_matrix(target_valid, LgR_prediction)



LgR_probabilities_one_valid = model_LgR.predict_proba(features_valid)[:, 1]



auc_roc_LgR = roc_auc_score(target_valid, LgR_probabilities_one_valid)



auc_roc_LgR
fpr, tpr, thresholds = roc_curve(target_valid, LgR_probabilities_one_valid) 



plt.figure()

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()
LgR_probabilities_one_valid = model_LgR.predict_proba(features_valid)[:, 1]



auc_roc_LgR = roc_auc_score(target_valid, LgR_probabilities_one_valid)



auc_roc_LgR
fpr, tpr, thresholds = roc_curve(target_valid, LgR_probabilities_one_valid) 



plt.figure()

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()
#попробуем обучать логистическую регресию сбалансировав классы

model_LgR = LogisticRegression(solver = 'liblinear', class_weight='balanced')

model_LgR.fit(features_train, target_train)

LgR_probabilities_one_valid_class_weight = model_LgR.predict_proba(features_valid)[:, 1]

print("Score", model_LgR.score(features_valid, target_valid))

print("AUC-ROC", roc_auc_score(target_valid, LgR_probabilities_one_valid_class_weight))



fpr, tpr, thresholds = roc_curve(target_valid, LgR_probabilities_one_valid_class_weight) 

plt.figure()

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()

from sklearn.utils import shuffle
#Ознакомимся с пероначальным распределением классов

target_train.value_counts(normalize = 1)
target_train.shape
target_train.plot(kind ='hist', bins=2, figsize=(1,1))
#создадим функцию для увеличения представленной класса в выборке 

def upsample(features, target, repeat, upsampled_сlass):

    features_zeros = features[target == 0]

    features_ones = features[target == 1]

    target_zeros = target[target == 0]

    target_ones = target[target == 1]

    

    if upsampled_сlass == 0:

        features_upsampled = pd.concat([features_zeros]* repeat + [features_ones] )

        target_upsampled = pd.concat([target_zeros]* repeat + [target_ones] )

        features_upsampled, target_upsampled = shuffle(

        features_upsampled, target_upsampled, random_state=12345)

        

    elif upsampled_сlass == 1:

        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)

        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

        features_upsampled, target_upsampled = shuffle(

        features_upsampled, target_upsampled, random_state=12345)

    else:

        features_upsampled = 0

        target_upsampled = 0  

        

        

       

    return features_upsampled, target_upsampled

    "Функция принимаем значение признаков (features[]), целевого признака (target[]), repeat(int / float), "

    " класс который будет увеличен (upsampled_сlass (0 or 1))"
#Протестируем функцию (верное значение)

features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 0)

print(target_train_upsampled.value_counts(normalize = 1))

print(target_train_upsampled.shape)
#Протестируем функцию (верное значение)

features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 3)

features_train_upsampled
#применим функцию upsample 

#увеличим количество положительных ответов в 4 раза

features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 1)

print(target_train_upsampled.value_counts(normalize = 1))

print(target_train_upsampled.shape)
target_train_upsampled.plot(kind ='hist', bins=2, figsize=(1,1))
#точность моделей на выборке с дисбалансом

all_models_accuracy(features_train, target_train, features_valid, target_valid)
#точность моделей на сбалансированной выборке

all_models_accuracy(features_train_upsampled, target_train_upsampled, features_valid, target_valid)
#Решающее дерево

model_DTC_upsampled = DecisionTreeClassifier(random_state=123)

model_DTC_upsampled.fit(features_train_upsampled, target_train_upsampled)

DTC_prediction_upsampled = model_DTC_upsampled.predict(features_valid)

rec_prec_f1(target_valid, DTC_prediction_upsampled)
#Случайный лес

model_RFC_upsampled = RandomForestClassifier(random_state=12345, n_estimators = 100)

model_RFC_upsampled.fit(features_train_upsampled, target_train_upsampled)

RFC_prediction_upsampled = model_RFC_upsampled.predict(features_valid)

rec_prec_f1(target_valid, RFC_prediction_upsampled)
#Логистическая регрессия

model_LgR_upsampled = LogisticRegression(solver = 'liblinear')

model_LgR_upsampled.fit(features_train_upsampled, target_train_upsampled)

LgR_prediction_upsampled = model_LgR_upsampled.predict(features_valid)

rec_prec_f1(target_valid, LgR_prediction_upsampled)
from itertools import product

import tqdm
def RandomForestQuality(features_train, target_train, features_valid, target_valid):

    

    #Параметры для перебора

    bootstrap = [True, False]

    class_weight = ['balanced', 'balanced_subsample', None]

    max_features = ['auto', 'sqrt', 'log2'] 

    max_depth = [] #диапазон изменения параметра мксимальной глубины каждого дерева

    for i in range(1, 20):

        max_depth.append(i)



    #Метод itertools.product для перебора нескольких параметров

    myproduct = product(bootstrap, class_weight, max_features, max_depth)

    

    #Строки, котоыре будут наполняться циклом при переборе параметров

    bootstrap_table = []

    class_weight_table = []

    features_table = []

    depth_table = []

    f1_table = []

    recall_table = []

    precision_table = []

    score_train_table = []

    score_valid_table = []

    

    #Цикл перебора всех параметров: bootstrap, class_weight, max_features, max_depth

    for p in tqdm.tqdm(myproduct,):

        #Обучение модели

        model_forest = RandomForestClassifier(

            bootstrap=p[0] , class_weight= p[1], max_features = p[2], max_depth = p[3], 

            n_estimators = 10, random_state=12345)

        model_forest.fit(features_train, target_train)

        prediction = model_forest.predict(features_valid) #предсказание целевого признака

        

        #расчет параметров

        f1 = f1_score(target_valid, prediction)

        recall = recall_score(target_valid, prediction)

        precision = precision_score(target_valid, prediction)

        score_train = model_forest.score(features_train, target_train)

        score_valid = model_forest.score(features_valid, target_valid)

        

        #внесение значиний параметров в строки

        bootstrap_table.append(p[0])

        class_weight_table.append(p[1])

        features_table.append(p[2])

        depth_table.append(p[3])



        #внесение значений метрик в строки

        f1_table.append(f1)

        recall_table.append(recall)

        precision_table.append(precision)

        score_train_table.append(score_train)

        score_valid_table.append(score_valid)

               

    

    #Обоъединение строк в датафрем

    quality_table = pd.DataFrame(data = (

        bootstrap_table, class_weight_table, features_table, depth_table, 

        f1_table, recall_table, precision_table, score_train_table, score_valid_table)).T

    quality_table.columns = (

        'bootstrap', 'class_weight', 'max_features', 'max_depth', 'f1', 'recall', 'precision', 'score_train', 'score_valid')

    return quality_table



    "4 параметра: features_train, target_train — признаки и целевой признак обучающей выборки"

    "features_valid, target_valid — признаки и целевой признак обучающей выборки"



%%time

quality_table = RandomForestQuality(features_train_upsampled, target_train_upsampled, features_valid, target_valid)
quality_table.query('score_valid>=score_train').sort_values('f1', ascending = False).head()
model_RFC_final = RandomForestClassifier(

    bootstrap = True, class_weight = 'balanced', max_depth= 7,  n_estimators = 100, random_state=12345)

model_RFC_final.fit(features_train_upsampled, target_train_upsampled)
model_RFC_final_prediction = model_RFC_final.predict(features_valid)

rec_prec_f1(target_valid, model_RFC_final_prediction)
#Создаем константную модель

target_predict_constant = pd.Series([0]*len(target_valid))

target_predict_constant.value_counts()
#Сравним показатель точности (accuracy_score) константной модели и финальной

print('accuracy_score константой модели:', accuracy_score(target_valid, target_predict_constant))

print('accuracy_score финальной модели:', accuracy_score(target_valid, model_RFC_final_prediction))

#Дополнительно сравним AUC-ROC — единственный параметр подающийся сравнению, потому что константная подель содержит только негативные ответы

print('AUC-ROC константой модели:', roc_auc_score(target_valid, target_predict_constant))

print('AUC-ROC финальной модели:', roc_auc_score(target_valid, model_RFC_final_prediction))
model_RFC_final

model_RFC_final_prediction = model_RFC_final.predict(features_test)

rec_prec_f1(target_test, model_RFC_final_prediction)
final_model_probabilities_one = model_LgR.predict_proba(features_test)[:, 1]



fpr, tpr, thresholds = roc_curve(target_test, final_model_probabilities_one) 



plt.figure()

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()
results = pd.DataFrame(model_RFC_final_prediction)

results.head(5)
results.to_csv('/kaggle/working/Churn_Modelling_results.csv')