import numpy as np # Álgebra linear

import pandas as pd # Processamento de dados e I/O de CSVs

import seaborn as sns

import missingno as msno #Verificar valores faltantes

import matplotlib.pyplot as plt #Plotar gráficos

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier #Gerar modelo de classificação

from sklearn.model_selection import cross_validate #Calcular pontuação







%matplotlib inline

import os

print(os.listdir("../input")) # Apresentar os arquivos que estão no diretório "input"
train_data_path = '../input/train.csv'

valid_data_path = '../input/valid.csv'

test_data_path = '../input/test.csv'



train_data = pd.read_csv(train_data_path)

valid_data = pd.read_csv(valid_data_path)

test_data = pd.read_csv(test_data_path)

total_data = pd.concat([train_data, valid_data, test_data], sort=False)
print(total_data.count())

msno.matrix(total_data,figsize=(12,5))
def plotCorrelationMatrix(data, method_='spearman'):

    

    dataCorrelations = data.corr(method=method_)



    fig, ax = plt.subplots(figsize=(15,15))

    sns.heatmap(dataCorrelations, vmax=1.0, center=0, fmt='.2f',

                    square=True, linewidths=.1, annot=True, cbar_kws={"shrink": .90})

    plt.show();
plotCorrelationMatrix(total_data)
total_data = total_data.drop(columns=["AGE","BILL_AMT6"])



total_data
plotCorrelationMatrix(total_data)
print("Total_data: ",total_data.shape)

print("Train_data: ",train_data.shape)

print("Valid_data: ",valid_data.shape)

print("Test_data: ", test_data.shape)
train_data = total_data.iloc[:21000]

valid_test_data = total_data.iloc[21000:]



train_data.drop(train_data.index[train_data["BILL_AMT1"] == 0], inplace = True)

train_data.drop(train_data.index[train_data["PAY_0"] < -1], inplace = True)

train_data.drop(train_data.index[train_data["PAY_2"] < -1], inplace = True)

train_data.drop(train_data.index[train_data["PAY_3"] < -1], inplace = True)

train_data.drop(train_data.index[train_data["PAY_4"] < -1], inplace = True)

train_data.drop(train_data.index[train_data["PAY_5"] < -1], inplace = True)

train_data.drop(train_data.index[train_data["PAY_6"] < -1], inplace = True)



print("Train_data: ",train_data.shape)

print("Valid_Test_data: ",valid_test_data.shape)
trainX = train_data.drop(columns=["ID","default payment next month"])

trainY = train_data["default payment next month"]
rfModel = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 1)

rfModel.fit(trainX, trainY)
cross_validate(rfModel, trainX, trainY, scoring='roc_auc', cv=5, return_train_score=True)
validTestX = valid_test_data.drop(columns=["ID", "default payment next month"])

predicted = rfModel.predict(validTestX)
data = {'ID': valid_test_data.ID, 'Default': predicted}



output = pd.DataFrame(data)



output.to_csv("outputFinal7.csv", encoding="utf8", index=False)