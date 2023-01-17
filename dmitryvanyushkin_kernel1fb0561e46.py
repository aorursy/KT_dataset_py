infoColumnsAmount = 78

labelsAmountGlobal = 14

maxBalancingAmount = 100000

fileDirectories = ['/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 

                   '/kaggle/input/minorfolders/MachineLearningCVEGood/Monday-WorkingHours.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Morning.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Wednesday-workingHours.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',

                  '/kaggle/input/minorfolders/MachineLearningCVEGood/Tuesday-WorkingHours.pcap_ISCX.csv']
import gc

import datetime

import tensorflow as tf

import pandas as pd

import numpy as np

from tensorflow import keras

from sklearn.metrics import f1_score

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Activation

import keras.backend as K

import csv

import seaborn as sns

import sklearn.metrics

from keras import losses

import matplotlib.pyplot as pltMath

import scikitplot as plt

import sklearn.preprocessing as preproc

import sklearn as sk

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as mat

from sklearn.model_selection import KFold

#import tensorflow.compat.v1 as tf

tf.version

#tf.disable_v2_behavior()

tree = 0#SoftDecisionTree(max_depth=6,n_features=n_features,n_classes=n_classes,max_leafs=None)

    # optimizer

optimizer = 0#tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(tree.loss)
def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def readDataSets():



    dataSetFriday = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', low_memory = False)

    dataSetMonday = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Monday-WorkingHours.pcap_ISCX.csv', low_memory = False)

    dataSetThursday = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Morning.pcap_ISCX.csv', low_memory = False)

    dataSetWednesday = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Wednesday-workingHours.pcap_ISCX.csv', low_memory = False)

    dataSetThursdayMorning = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory = False)#dataSetThursdayAfterNoon = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', low_memory = False)

    dataSetFridayMorning = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', low_memory = False)

    dataSetTuesdayWorking = pd.read_csv(filepath_or_buffer = '/kaggle/input/minorfolders/MachineLearningCVEGood/Tuesday-WorkingHours.pcap_ISCX.csv', low_memory = False)

    

    dataSetRawLoad = pd.concat([dataSetFriday, dataSetMonday, dataSetThursday, dataSetWednesday, dataSetThursdayMorning, dataSetFridayMorning, dataSetTuesdayWorking])



    dataSetModified = pd.get_dummies(data = dataSetRawLoad,columns = [' Label'],dtype = float, drop_first = False) #one hot encoding в деле

    

    allAttacksType = dataSetModified.columns[infoColumnsAmount:]

    allAttacksTypeCorrected = []

    for attackType in allAttacksType:

        allAttacksTypeCorrected.append(attackType[7:])

    labelsAmountGlobal = len(allAttacksType)

    print(datetime.datetime.now(), "File read finished ^_^")

        

    return (dataSetModified.to_numpy(dtype = float), allAttacksTypeCorrected)

def prepareDataSet(dataSet):

    

    dataSet = np.nan_to_num(dataSet)

    np.random.shuffle(dataSet)



    #Скейлим данные для упрощения работы Вычитаем среднее и делим на стандартное отклонение

    #scaler = preproc.StandardScaler()

    #scaler.fit(dataSet[:,:infoColumnsAmount])

    dataSetOverall = np.concatenate((preproc.normalize(dataSet[:,:infoColumnsAmount], norm='l2'), dataSet[:,infoColumnsAmount:]), axis = 1)#np.concatenate((scaler.transform(dataSet[:,:infoColumnsAmount]), dataSet[:,infoColumnsAmount:]), axis = 1 )

    print(datetime.datetime.now(), "Scaling finished ^_^")

    # Распределяем все данные

    overallLen = len(dataSetOverall)

    dataAmountToTrain = int(0.8 * overallLen)

    dataAmountToPredict = int(0.2 * overallLen)

    

    dataSetTrain = dataSetOverall[:dataAmountToTrain,:]

    dataSetPredict = dataSetOverall[(dataAmountToTrain + 1):,:]

    

    print(datetime.datetime.now(), "DataSet prepared ^_^")

    

    return (dataSetTrain, dataSetPredict)
def synthesizeShortBalancedDataSet(dataSetInput, position):

    print(datetime.datetime.now(), "Balancing started ^_^")

    columnsAmount = len(dataSetInput[0])

    # Получаем все лейблы, которые у нас имеются

    labelArrays = dataSetInput[:, infoColumnsAmount:]

    #Считаем количество различных лейблов, среди имеющихся

    labelAmounts = len(labelArrays[0]) 

    rareLabels = []

    baseLabels = []

    allLabels = [[]]

    for i in range(0, labelAmounts - 1):

        allLabels.append([])

    #группируем элементы датасета по типам атаки (массив массивов с одинаковым типом атак)

    for i in range(0, len(labelArrays)):

        for j in range(0, len(labelArrays[i])):

            if labelArrays[i][j] == 1:

                allLabels[j].append(dataSetInput[i])

    # теперь создаем массив с количество типов каждых из атак

    lengthArray = []

    for i in allLabels:

        lengthArray.append(len(i))

    result = [np.zeros(columnsAmount)]

    elementsAmount = max(lengthArray)

    if elementsAmount > maxBalancingAmount:

        elementsAmount = maxBalancingAmount

    #print(elementsAmount)

    #Далее с помощью магического цикла дополняем типы атак с дефицитным числом лейблов их копиями

    for (i,j) in zip(allLabels, lengthArray):

        cuttedArr = [np.zeros(columnsAmount)]

        #если элементов меньше, чем нам кажется адекватным, то добавляем копии

        if (j <  elementsAmount):

            #количество копий

            if len(i) != 0:

                arrAmounts = int( elementsAmount / len(i))

                cuttedArr = np.repeat(i, arrAmounts, axis = 0)

                #for t in range(0, arrAmounts):

                #    cuttedArr = np.concatenate((cuttedArr, i))

        else:

            #если же их и так много, то просто обрезаем до заданного

            cuttedArr = i[1:]

        result = np.concatenate((result, cuttedArr))

    # обрезаю нулевой элемент (он был нужен для проблем с типизацией)

    result = result[1:,:]

    

    print(datetime.datetime.now(), "Balancing finished ^_^")

    return result
def createModel(labelsTypeAmount):

    print(labelsTypeAmount)

    model = Sequential([

        Dense(labelsTypeAmount, input_shape=(infoColumnsAmount,)),

        Dense(labelsTypeAmount),

        Dense(labelsTypeAmount),

        Dense(labelsTypeAmount, activation = 'softmax')])

    

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print(datetime.datetime.now(), "Model created ^_^")

    return model
 # Initialize the variables (i.e. assign their default value)

#init = tf.global_variables_initializer()

#sess = tf.Session()

#sess.run(init)



n_features = infoColumnsAmount

n_classes = labelsAmountGlobal

batch_size = 25 #60

    

def createDecisionTree():

    global tree

    tree = SoftDecisionTree(max_depth=6,n_features=n_features,n_classes=n_classes,max_leafs=None)

    tree.build_tree()

    # optimizer

    global optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(tree.loss)

    
def train(model, dataSet, labelsTypeAmount):

    

    inputMatrix = dataSet[:,:infoColumnsAmount]

    outputMatrix = dataSet[:,infoColumnsAmount:]



    epochsAmount = 1

    splitsAmount = 10

    acc = []

    val_acc = []

    loss = []

    val_loss = []

    

    # Вот здесь мы практикуем K-Fold (перетасовали и разрелали массивы)

    for train_index, test_index in KFold(n_splits = splitsAmount, shuffle = False).split(inputMatrix):

        X_train, X_test = inputMatrix[train_index], inputMatrix[test_index]

        y_train, y_test = outputMatrix[train_index], outputMatrix[test_index]

        res = model.fit(np.concatenate((X_train, X_test), axis = 0), np.concatenate((y_train, y_test), axis = 0) , batch_size=1000, validation_split = 0.2, shuffle = True, verbose = 1) #, class_weight = class_weights)

        gc.collect()

        for i in range(len(res.history['accuracy'])):   

            acc.append(res.history['accuracy'][i])

            val_acc.append(res.history['val_accuracy'][i])

            loss.append(res.history['loss'][i])

            val_loss.append(res.history['val_loss'][i])

            

    return acc, loss, val_acc, val_loss
def trainLSTM(model, dataSet, labelsTypeAmount):

    

    inputMatrix = dataSet[:,:infoColumnsAmount]

    outputMatrix = dataSet[:,infoColumnsAmount:]

    

    outputMatrixSimplified = np.empty(len(inputMatrix[:,0]), dtype=int)

    

    #convert output matrix to single form

    i = 0

    for element in outputMatrix:

        j = 0

        for k in range(len(element)):

            if element[j] == 1:

                break

            j += 1

        outputMatrixSimplified[i] = j

        i += 1

        

    print(np.amin(outputMatrixSimplified))

    print(np.amax(outputMatrixSimplified))

    acc = []

    val_acc = []

    loss = []

    val_loss = []

    

    # Вот здесь мы практикуем K-Fold (перетасовали и разрелали массивы)

    model.fit(inputMatrix, outputMatrixSimplified, batch_size=1000, validation_split = 0.2, shuffle = True, verbose = 1)
def plotBarChart(value1,validationValue1,description):

    pltMath.figure(figsize=(20,5))

    pltMath.bar([1,2],[value1[len(value1) - 1],validationValue1[len(validationValue1) - 1]])

    pltMath.ylabel(description, fontsize=16)

    pltMath.title("Histogram {}".format(description))

    pltMath.show()



    pltMath.plot(value1)

    pltMath.plot(validationValue1)

    pltMath.title(description)

    pltMath.ylabel('accuracy')

    pltMath.xlabel('epoch')

    pltMath.legend(['train', 'val'], loc='upper left')

    pltMath.show()

def decodeAttack(resultArray, attackTypeArray) -> str:

    positionOfPredict = 0

    isFound = False

    for result in resultArray:

        if result == 1:

            isFound = True

            break

        positionOfPredict += 1

    if isFound:

        return attackTypeArray[positionOfPredict]

    else:

        return "Unrecognized"
def convertToStringForm(predictions, attackTypeArray):

    elementsAmount = len(predictions[:,0])

    labelArr = np.empty(elementsAmount, dtype='string')

    i = 0

    for prediction in predictions:

        labelArr[i] = decodeAttack(prediction, attackTypeArray)

    return labelArr

        
def roundPredictions(predictions):

    i = 0

    j = 0

    for prediction in predictions:

        for labelPoss in prediction:

            predictions[i][j] = int(round(predictions[i][j]))

            j += 1

        j = 0

        i += 1

        

    return predictions
def makePrediction(model, dataSetPredict):

    print(datetime.datetime.now(),"Prediction Start ^_^")

    predictionsNew = model.predict(dataSetPredict[:,:infoColumnsAmount])

    roundedPrediction = roundPredictions(predictionsNew)

    

    

    print(datetime.datetime.now(),"Prediction Finish ^_^")

    return roundedPrediction

    
def countAccuracy(predicted, expected):

    overallAmount = len(predicted)

    correctAmount = 0

    i = 0

    for i in range(0, len(predicted) - 1):

        if predicted[i].all() == expected[i].all():

            correctAmount += 1

    

    return (correctAmount/overallAmount)
def buildConfusionMatrix(predictions, expected):

    plt.metrics.plot_confusion_matrix(

        expected.argmax(axis=1), 

        predictions.argmax(axis=1))
def createLSTMModel():

    inputTeam1 = tf.keras.Input(shape=(infoColumnsAmount,))

    

    team1Branch = tf.keras.layers.Embedding(labelsAmountGlobal, 64)(inputTeam1)



    team1Branch = tf.keras.Model(inputs=inputTeam1, outputs=team1Branch)

    

    

    combined = team1Branch.output



    result = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(combined)

    result = tf.keras.layers.Dense(labelsAmountGlobal, activation="softmax")(result)

    

    model = tf.keras.Model(inputs=[team1Branch.input], outputs=result)

    

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print(datetime.datetime.now(), "Model LSTM created ^_^")

    

    return model
print(datetime.datetime.now(),"Start ^_^")

dataSet, correctedLabels = readDataSets()

gc.collect()

dataSetTrain, dataSetPredict = prepareDataSet(dataSet)

gc.collect()



print(correctedLabels)



trainedModel = createModel(len(correctedLabels)) #createLSTMModel()#

trainedModelLSTM = createLSTMModel()



splittedSets = np.array_split(dataSetTrain, 10)



#createDecisionTree()



# Kaggle падает при слишком большом датасете. Фикс - деление на чанки для обучения и балансировки

for chunk in splittedSets:



    chunkBalanced = synthesizeShortBalancedDataSet(chunk, infoColumnsAmount)

    gc.collect()

    #train(trainedModel, chunkBalanced, len(correctedLabels))

    acc,loss,val_acc,val_loss = train(trainedModel, chunkBalanced, len(correctedLabels))

    plotBarChart(acc,val_acc,'accuracy comparing')

    plotBarChart(loss,val_loss,'loss comparing')

    #plotting

    #history = train(trainedModel, chunkBalanced, len(correctedLabels))

    #plotBarChart(0.8,2,'Models accuracy comparing', 'Validation models accuracy comparing', history)

    #plotVaryingGraph(history) 

    #gc.collect()

    #trainLSTM(trainedModelLSTM, chunkBalanced, len(correctedLabels))

    #gc.collect()



# base model prediction

    

preidctionResult = makePrediction(trainedModel, dataSetPredict)



expectedResult = dataSetPredict[:,infoColumnsAmount:]



accuracy = countAccuracy(preidctionResult, expectedResult)



buildConfusionMatrix(preidctionResult, expectedResult)



print(accuracy)



gc.collect()



# lstm prediction



#preidctionResult = makePrediction(trainedModelLSTM, dataSetPredict)



#expectedResult = dataSetPredict[:,infoColumnsAmount:]



#print(expectedResult)



#accuracy = countAccuracy(preidctionResult, expectedResult)



#buildConfusionMatrix(preidctionResult, expectedResult)



#print(accuracy)



#gc.collect()