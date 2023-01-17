import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score, train_test_split

train_file_path = '../input/train.csv'

valid_file_path = '../input/valid.csv'

test_file_path = '../input/test.csv'



#leitura dos arquivos

train_file_reader = pd.read_csv(train_file_path)

valid_file_reader = pd.read_csv(valid_file_path)

test_file_reader = pd.read_csv(test_file_path)
#crianção do y

y_column = 'default payment next month'

y = train_file_reader[y_column]



#tratamento para trabalhar apenas com valores positivos

train_file_reader[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']] +=2

valid_file_reader[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']] +=2

test_file_reader[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']] +=2



#selecionando colunas do dataset para o X

feature_columns = ['LIMIT_BAL','SEX','EDUCATION', 'MARRIAGE',

                   'AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',

                   'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',

                   'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3',

                   'PAY_AMT4','PAY_AMT5','PAY_AMT6']

X = train_file_reader[feature_columns]



#split dos Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y)



#variaveis para uso na predição

valid_features = valid_file_reader[feature_columns]

test_features = test_file_reader[feature_columns]
print(y.value_counts())
kf = KFold(n_splits=5)

model = RandomForestClassifier(n_jobs=-1, n_estimators=100, 

                        max_features=6, min_samples_leaf=10, random_state=0, class_weight='balanced')



model.fit(X_train, y_train)
from sklearn.metrics import roc_curve

# computação da texa de true positive e de false positive

y_scores = model.predict_proba(X_test)

y_scores = y_scores[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_scores)



# Plot do gráfico

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()



# Taxa Numerica do Gráfico



auc = roc_auc_score(y_test, y_scores)



print(f'ROC_AUC: {auc}')
val_predictions_valid = model.predict(valid_features)



#exporta resultados da predição com o dados de validação para um arquivo CSV

csv_valid_predict = pd.DataFrame({"ID": range(21001,25501), "Default": val_predictions_valid}) 

csv_valid_predict
val_predictions_test = model.predict(test_features)



#exporta resultados da predição com o dados de teste para um arquivo CSV

csv_test_predict = pd.DataFrame({"ID": range(25501,30001), "Default": val_predictions_test})

csv_test_predict

result = pd.concat([csv_valid_predict, csv_test_predict])

result.to_csv("result.csv", index=False)



result