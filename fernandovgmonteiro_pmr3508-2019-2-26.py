#Importando bibliotecas necessárias

import pandas as pd

import sklearn



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# Encontrando o endereço dos arquivos com as bases

import os

cwd = "/kaggle/input/adult-pmr3508/"

testDataPath = cwd + "/test_data.csv"

trainDataPath = cwd + "/train_data.csv"



# Lendo os arquivos csv com as bases de teste e treino e limpando as bases

headers = [

            "Age", "Workclass", "fnlwgt", "Education",

            "Education-Num", "Martial Status",

            "Occupation", "Relationship", "Race", "Sex",

            "Capital Gain", "Capital Loss",

            "Hours per week", "Country", "Target"

        ]



testData = pd.read_csv(

    testDataPath,

    names=headers[:-1],

    sep=r'\s*,\s*',

    engine='python',

    na_values="?",

    skiprows=1

).dropna()



trainData = pd.read_csv(

    trainDataPath,

    names=headers,

    sep=r'\s*,\s*',

    engine='python',

    na_values="?",

    skiprows=1

).dropna()
# Checando se os arquivos foram corretamente importados

print("TestData Shape: ", testData.shape)

print("TrainData Shape: ", trainData.shape)
# Convertendo categorias em novas colunas

def categoryToNumeric(trainData, columnName):

    classValues = set(trainData[columnName])

    print("Set of Features:", len(classValues))



    for value in classValues:

        newColumnName = columnName + '_'+ value

        trainData[newColumnName] = [0]*trainData.shape[0]





    for index in trainData.index:

        for value in classValues:

            if trainData[columnName][index] == value:

                trainData[columnName + '_' + value][index] = 1

                

    return trainData



nonNumericFeatures = [

            "Workclass", "Education", "Martial Status",

            "Occupation", "Relationship", "Race", "Sex", "Country"

        ]



for feature in nonNumericFeatures:

    print("Feature:", feature)

    trainData = categoryToNumeric(trainData, feature)

    testData = categoryToNumeric(testData, feature)



trainData.head()
# Observando quais features não estão presentes em ambos os sets

print(set(trainData.columns) - set(testData.columns))
# Dados que já eram numéricos

numericFeatures = [

    "Age",

    "Education-Num",

    "Capital Gain",

    "Capital Loss",

    "Hours per week"

]



# Dados que já eram numéricos juntos aos dados que foram convertidos para atributos binários

# São consideradas apenas as features que estão em ambos os conjuntos de dados

allFeatures = ['Age',

 'Education-Num',

 'Capital Gain',

 'Capital Loss',

 'Hours per week',

 'Workclass_Local-gov',

 'Workclass_Private',

 'Workclass_State-gov',

 'Workclass_Without-pay',

 'Workclass_Self-emp-inc',

 'Workclass_Federal-gov',

 'Workclass_Self-emp-not-inc',

 'Education_Assoc-voc',

 'Education_HS-grad',

 'Education_Some-college',

 'Education_1st-4th',

 'Education_5th-6th',

 'Education_7th-8th',

 'Education_Bachelors',

 'Education_Prof-school',

 'Education_11th',

 'Education_Preschool',

 'Education_Doctorate',

 'Education_Assoc-acdm',

 'Education_12th',

 'Education_9th',

 'Education_Masters',

 'Education_10th',

 'Martial Status_Widowed',

 'Martial Status_Divorced',

 'Martial Status_Married-AF-spouse',

 'Martial Status_Separated',

 'Martial Status_Married-spouse-absent',

 'Martial Status_Never-married',

 'Martial Status_Married-civ-spouse',

 'Occupation_Transport-moving',

 'Occupation_Exec-managerial',

 'Occupation_Machine-op-inspct',

 'Occupation_Other-service',

 'Occupation_Priv-house-serv',

 'Occupation_Farming-fishing',

 'Occupation_Handlers-cleaners',

 'Occupation_Sales',

 'Occupation_Adm-clerical',

 'Occupation_Tech-support',

 'Occupation_Armed-Forces',

 'Occupation_Craft-repair',

 'Occupation_Prof-specialty',

 'Occupation_Protective-serv',

 'Relationship_Husband',

 'Relationship_Not-in-family',

 'Relationship_Wife',

 'Relationship_Other-relative',

 'Relationship_Unmarried',

 'Relationship_Own-child',

 'Race_Asian-Pac-Islander',

 'Race_White',

 'Race_Amer-Indian-Eskimo',

 'Race_Other',

 'Race_Black',

 'Sex_Female',

 'Sex_Male',

 'Country_India',

 'Country_El-Salvador',

 'Country_Mexico',

 'Country_Dominican-Republic',

 'Country_Cambodia',

 'Country_France',

 'Country_Vietnam',

 'Country_Hungary',

 'Country_Germany',

 'Country_South',

 'Country_England',

 'Country_Cuba',

 'Country_Iran',

 'Country_Trinadad&Tobago',

 'Country_Thailand',

 'Country_Japan',

 'Country_Haiti',

 'Country_Greece',

 'Country_Yugoslavia',

 'Country_Scotland',

 'Country_Canada',

 'Country_Taiwan',

 'Country_Ireland',

 'Country_Italy',

 'Country_Peru',

 'Country_Philippines',

 'Country_China',

 'Country_Portugal',

 'Country_Ecuador',

 'Country_Nicaragua',

 'Country_Hong',

 'Country_Poland',

 'Country_Guatemala',

 'Country_Jamaica',

 'Country_Honduras',

 'Country_United-States',

 'Country_Outlying-US(Guam-USVI-etc)',

 'Country_Puerto-Rico',

 'Country_Laos',

 'Country_Columbia']



# Separando dados da base para construção do modelo de treino

trainX = trainData[allFeatures]

trainY = trainData.Target
# Criando o classificador KNN e gerando modelo

knn = KNeighborsClassifier(n_neighbors=3)



# CossValidation para o KNN

KNN_cross_val = cross_val_score(knn, trainX, trainY, cv=10)

print("Cross Validation Results: ", KNN_cross_val)

print("Cross Validation Average: ", sum(KNN_cross_val)/len(KNN_cross_val))
from sklearn import svm



# Criando modelo de SVM

SVM = svm.SVC(gamma='auto')



# CrossValidation para o SVM

SVM_cross_val = cross_val_score(SVM, trainX, trainY, cv=5)

print("Cross Validation Results: ", SVM_cross_val)

print("Cross Validation Average: ", sum(SVM_cross_val)/len(SVM_cross_val))
# Importando

from sklearn.tree import DecisionTreeClassifier



# Modelo de árvore de decisão

tree = DecisionTreeClassifier()



# CrossValidation para Árvore de Decisão

TREE_cross_val = cross_val_score(tree, trainX, trainY, cv=10)

print("Cross Validation Results: ", TREE_cross_val)

print("Cross Validation Average: ", sum(TREE_cross_val)/len(TREE_cross_val))
# Separando dados do arquivo de teste para fazer a predição

testX = testData[allFeatures]



# Predizendo o resultado

SVM.fit(trainX, trainY)

testY_pred = SVM.predict(testX)

testY_pred
# Exportando predições para um CSV

predictionList = pd.DataFrame(testData.index)

predictionList["income"] = testY_pred

predictionList.to_csv("prediction.csv", index=False)