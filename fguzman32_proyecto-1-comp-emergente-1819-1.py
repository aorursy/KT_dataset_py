# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

from operator import add

import matplotlib.pyplot as plt

from numpy import array

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import regularizers, initializers, optimizers

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from keras import regularizers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print(os.listdir("../input"))
trainingData = pd.read_csv('../input/titanic/train.csv')

testData = pd.read_csv('../input/titanic/test.csv')



print("Train shape: ",trainingData.shape)

print("Test shape: ",testData.shape)

trainingData[0:5]
list2 = trainingData['Cabin']

list3= []



for l in list2:

    if(pd.isnull(l)):

        list3.append(False)

    else:

        list3.append(True)

#print(list3)



trainingData2 = trainingData.copy(deep=True)

trainingData2['Cabin'] = list3

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.barplot(x='Cabin', y='Survived', data=trainingData2)



list3 = trainingData['Cabin']

list33 = testData['Cabin']

#mySet = set(list3)

#print(mySet)



#agrupar segun letra con que empiezan a ver que data arroja



list4 = []

for l in list3:

    if(pd.isnull(l)):

        list4.append('n')

    else:

        list4.append(l[0])



list44 = []

for l in list33:

    if(pd.isnull(l)):

        list44.append('n')

    else:

        list44.append(l[0])

    

trainingData3 = trainingData.copy(deep=True)

trainingData3['Cabin'] = list4 



trainingData['Cabin'] = list4

testData['Cabin'] = list44







sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.barplot(x='Cabin', y='Survived', data=trainingData3)

list44 = trainingData['Cabin']

list444 = testData['Cabin']

values = array(list44)

values444 = array(list444)

#print(values)



label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)

#print(integer_encoded)



label_encoder444 = LabelEncoder()

integer_encoded444 = label_encoder444.fit_transform(values444)



onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)





onehot_encoder444 = OneHotEncoder(sparse=False)

integer_encoded444 = integer_encoded444.reshape(len(integer_encoded444),1)

onehot_encoded444 = onehot_encoder444.fit_transform(integer_encoded444)





listA = []

for f in range(onehot_encoded.shape[0]):

    listA.append(onehot_encoded[f][0])

#print(listA)



listA2= []

for f in range(onehot_encoded444.shape[0]):

    listA2.append(onehot_encoded444[f][0])



listB= []

for f in range(onehot_encoded.shape[0]):

    listB.append(onehot_encoded[f][1])

    

listB2 = []

for f in range(onehot_encoded444.shape[0]):

    listB2.append(onehot_encoded444[f][1])

    

listC = []

for f in range(onehot_encoded.shape[0]):

    listC.append(onehot_encoded[f][2])

    

listC2 =[]

for f in range(onehot_encoded444.shape[0]):

    listC2.append(onehot_encoded444[f][2])

    

listD = []

for f in range(onehot_encoded.shape[0]):

    listD.append(onehot_encoded[f][3])

    

listD2 = []

for f in range(onehot_encoded444.shape[0]):

    listD2.append(onehot_encoded444[f][3])

    

listE=[]

for f in range(onehot_encoded.shape[0]):

    listE.append(onehot_encoded[f][4])

    

listE2=[]

for f in range(onehot_encoded444.shape[0]):

    listE2.append(onehot_encoded444[f][4])

    

listF=[]

for f in range(onehot_encoded.shape[0]):

    listF.append(onehot_encoded[f][5])

    

listF2 = []

for f in range(onehot_encoded444.shape[0]):

    listF2.append(onehot_encoded444[f][5])

    

listG= []

for f in range(onehot_encoded.shape[0]):

    listG.append(onehot_encoded[f][6])

    

listG2=[]

for f in range(onehot_encoded444.shape[0]):

    listG2.append(onehot_encoded444[f][6])



listT = []

for f in range(onehot_encoded.shape[0]):

    listT.append(onehot_encoded[f][7])

    

listT2 = []

for f in range(onehot_encoded444.shape[0]):

    listT2.append(0)

    

    

#T no hay para test set

#listT2 =[]

#for f in range(onehot_encoded444.shape[0]):

    #listT2.append(onehot_encoded444[f][7])

    

listN = []

for f in range(onehot_encoded.shape[0]):

    listN.append(onehot_encoded[f][8])



listN2 =[]

for f in range(onehot_encoded444.shape[0]):

    listN2.append(onehot_encoded444[f][7])

    

trainingData['CabinA'] = listA

trainingData['CabinB'] = listB

trainingData['CabinC'] = listC

trainingData['CabinD'] = listD

trainingData['CabinE'] = listE

trainingData['CabinF'] = listF

trainingData['CabinN'] = listN

trainingData['CabinT'] = listT



testData['CabinA'] = listA2

testData['CabinB'] = listB2

testData['CabinC'] = listC2

testData['CabinD'] = listD2

testData['CabinE'] = listE2

testData['CabinF'] = listF2

testData['CabinN'] = listN2

testData['CabinT'] = listT2









trainingData = trainingData.drop(['Cabin'], axis = 1)

testData = testData.drop(['Cabin'], axis=1)



#hay 9 categorias de letras que por las que empieza una cabina

#print(onehot_encoded.shape)

trainingData
list4 = trainingData['Name']

list42 = testData['Name']



#chequear probabilidades de salvarse una persona segun su titulo (Mr, Mrs, Miss, etc)





#lo que va despues de la coma y espacio hasta punto.

list5 = []

for l in list4:

    #l.index(', ')

    #print(l[l.index(', ')+2])

    list5.append(l[l.index(', ')+2:])

    #de ese substring quiero ver donde hay '. '

    #l[l.index(', ')+2:].index('. ')



#print(list5)

    

list52 = []

for l in list42:

    #print(l[l.index(', ')+2:])

    list52.append(l[l.index(', ')+2:])

    

    

list6 = []

for l in list5:

    #print(l[:l.index('. ')])

    list6.append(l[:l.index('. ')])

    

list62 = []

for l in list52:

    list62.append(l[:l.index('. ')])

    

    

trainingData4 = trainingData.copy(deep=True)

trainingData4['Name'] = list6

trainingData ['Name'] = list6 



testData['Name'] = list62





sns.set(rc={'figure.figsize':(25,8.27)})

sns.barplot(x='Name', y='Survived', data=trainingData4)

list20 = trainingData['Name']



trainingData['Name'] = trainingData['Name'].replace(['Lady', 'Capt', 'Col','Don'

, 'Dr', 'Major', 'Rev','Jonkheer', 'Dona'], 'Rare')

trainingData['Name'] = trainingData['Name'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

trainingData['Name'] = trainingData['Name'].replace('Mlle', 'Miss')

trainingData['Name'] = trainingData['Name'].replace('Ms', 'Miss')

trainingData['Name'] = trainingData['Name'].replace('Mme', 'Mrs')



list22 = testData['Name']

testData['Name'] = testData['Name'].replace(['Lady', 'Capt', 'Col','Don'

, 'Dr', 'Major', 'Rev','Jonkheer', 'Dona'], 'Rare')

testData['Name'] = testData['Name'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

testData['Name'] = testData['Name'].replace('Mlle', 'Miss')

testData['Name'] = testData['Name'].replace('Ms', 'Miss')

testData['Name'] = testData['Name'].replace('Mme', 'Mrs')

list21 = trainingData['Name']

mySet = set(list21)



print('Numero de titulos diferentes en la data (entrenamiento): ',len(mySet))



sns.barplot(x='Name', y='Survived', data=trainingData)

list32 = trainingData['Name']

values32 = array(list32)

#print(values32)

mySet = set(list32)

#print(mySet)



list3232 = testData['Name']

values3232 = array(list3232)



label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values32)



onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)





label_encoder3232 = LabelEncoder()

integer_encoded3232 = label_encoder3232.fit_transform(values3232)



onehot_encoder3232 = OneHotEncoder(sparse=False)

integer_encoded3232 = integer_encoded3232.reshape(len(integer_encoded3232),1)

onehot_encoded3232 = onehot_encoder3232.fit_transform(integer_encoded3232)



#print('shape 1: ',onehot_encoded.shape)

#print('shape 2: ',onehot_encoded3232.shape)



#print('tituloss en data de training: ',mySet)

#mySet2 = set(list3232)

#print('titulos en data de test: ',mySet2)





listMaster = []

for f in range(onehot_encoded.shape[0]):

    listMaster.append(onehot_encoded[f][0])

#print(listMaster)



listMaster2 = []

for f in range(onehot_encoded3232.shape[0]):

    listMaster2.append(onehot_encoded3232[f][0])



listMiss = []

for f in range(onehot_encoded.shape[0]):

    listMiss.append(onehot_encoded[f][1])

#print(listMiss)



listMiss2 = []

for f in range(onehot_encoded3232.shape[0]):

    listMiss2.append(onehot_encoded3232[f][1])



listMr = []

for f in range(onehot_encoded.shape[0]):

    listMr.append(onehot_encoded[f][2])

#print(listMr)



listMr2 = []

for f in range(onehot_encoded3232.shape[0]):

    listMr2.append(onehot_encoded3232[f][2])



listMrs = []

for f in range(onehot_encoded.shape[0]):

    listMrs.append(onehot_encoded[f][3])

#print(listMrs)



listMrs2 = []

for f in range(onehot_encoded3232.shape[0]):

    listMrs2.append(onehot_encoded3232[f][3])



listRare = []

for f in range(onehot_encoded.shape[0]):

    listRare.append(onehot_encoded[f][4])

#print(listRare)



listRare2 = []

for f in range(onehot_encoded3232.shape[0]):

    listRare2.append(onehot_encoded3232[f][4])



    

#en data entrenamiento no hay ni royal ni countess

listRoyal = []

for f in range(onehot_encoded.shape[0]):

    listRoyal.append(onehot_encoded[f][5])



listRoyal2 = []

for f in range(onehot_encoded3232.shape[0]):

    listRoyal2.append(0)

    

    

listTheCountess = []

for f in range(onehot_encoded.shape[0]):

    listTheCountess.append(onehot_encoded[f][6])



listTheCountess2 = []

for f in range(onehot_encoded3232.shape[0]):

    listTheCountess2.append(0)

    

    

trainingData['NameMaster'] = listMaster

trainingData['NameMiss'] = listMiss

trainingData['NameMr'] = listMr

trainingData['NameMrs'] = listMrs

trainingData['NameRare'] = listRare

trainingData['NameRoyal'] = listRoyal

trainingData['NameTheCountess'] = listTheCountess



testData['NameMaster'] = listMaster2

testData['NameMiss'] = listMiss2

testData['NameMr'] = listMr2

testData['NameMrs'] = listMrs2

testData['NameRare'] = listRare2

testData['NameRoyal'] = listRoyal2

testData['NameTheCountess'] = listTheCountess2



trainingData = trainingData.drop('Name', axis=1) 

testData = testData.drop('Name', axis=1)

submission = testData[['PassengerId']].copy()

trainingData = trainingData.drop(['Ticket'], axis = 1)

testData = testData.drop(['Ticket'], axis = 1)

trainingData = trainingData.drop(['PassengerId'], axis=1)

testData = testData.drop(['PassengerId'], axis=1)

trainingData
list7 = trainingData['Pclass']

list72 = testData['Pclass']

#se tiene que hacer onehotencoding de esta columna

values = array(list7)

values72 = array(list72)



label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)



label_encoder72 = LabelEncoder()

integer_encoded72 = label_encoder72.fit_transform(values72)



onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



onehot_encoder72 = LabelEncoder()

integer_encoded72 = integer_encoded72.reshape(len(integer_encoded72),1)

onehot_encoded72 = onehot_encoder.fit_transform(integer_encoded72)





listClass1=[]

for f in range(onehot_encoded.shape[0]):

    listClass1.append(onehot_encoded[f][0])

#print(listClass1)



list2Class1 = []

for f in range(onehot_encoded72.shape[0]):

    list2Class1.append(onehot_encoded72[f][0])



listClass2 = []

for f in range(onehot_encoded.shape[0]):

    listClass2.append(onehot_encoded[f][1])

#print(listClass2)



list2Class2 = []

for f in range(onehot_encoded72.shape[0]):

    list2Class2.append(onehot_encoded72[f][1])



listClass3 = []

for f in range(onehot_encoded.shape[0]):

    listClass3.append(onehot_encoded[f][2])

#print(listClass3)



list2Class3 = []

for f in range(onehot_encoded72.shape[0]):

    list2Class3.append(onehot_encoded72[f][2])







trainingData['Pclass1'] = listClass1

trainingData['Pclass2'] = listClass2

trainingData['Pclass3'] = listClass3



testData['Pclass1'] = list2Class1

testData['Pclass2'] = list2Class2

testData['Pclass3'] = list2Class3



sns.barplot(x='Pclass', y='Survived', data=trainingData)

trainingData = trainingData.drop('Pclass', axis=1)

testData = testData.drop('Pclass', axis=1)

trainingData
list8 = trainingData['Sex']

values = array(list8)



label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)



onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



list56 = testData['Sex']

values56 = array(list56)



label_encoder56 = LabelEncoder()

integer_encoded56 = label_encoder56.fit_transform(values56)



onehot_encoder56 = OneHotEncoder(sparse=False)

integer_encoded56 = integer_encoded56.reshape(len(integer_encoded56),1)

onehot_encoded56 = onehot_encoder56.fit_transform(integer_encoded56)





listFemale=[]

for f in range(onehot_encoded.shape[0]):

    listFemale.append(onehot_encoded[f][0])



list2Female = []

for f in range(onehot_encoded56.shape[0]):

    list2Female.append(onehot_encoded56[f][0])

    

listMale=[]

for f in range(onehot_encoded.shape[0]):

    listMale.append(onehot_encoded[f][1])



list2Male = []    

for f in range(onehot_encoded56.shape[0]):

    list2Male.append(onehot_encoded56[f][1])

    

trainingData['SexFemale'] = listFemale

trainingData['SexMale'] = listMale

    

testData['SexFemale'] = list2Female

testData['SexMale'] = list2Male





sns.barplot(x='Sex', y='Survived', data=trainingData)

trainingData = trainingData.drop('Sex', axis=1)

testData = testData.drop('Sex', axis=1)

trainingData
list9 = trainingData['Embarked']

mySet = set(list9)





nullCounter=0

for l in list9:

    if(pd.isnull(l)):

        nullCounter=nullCounter+1        

#print("the number of nulls: ",nullCounter)





list99 = testData['Embarked']

mySet99 = set(list99)



nullCounter99 = 0

for l in list99:

    if(pd.isnull(l)):

        nullCounter99=nullCounter99+1

        

#print('the number of nulls in the testdata: ',nullCounter99)

#0 null



values4 = array(list9)

#print(values4)





values99 = array(list99)





#hay 2 null, los pongo en S

for f in range(len(values4)):

    if pd.isnull(values4[f]):

        values4[f]='S'

        



label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values4)

#print(integer_encoded)



onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#print(onehot_encoded)





label_encoder99 = LabelEncoder()

integer_encoded99 = label_encoder99.fit_transform(values99)



onehot_encoder99 = OneHotEncoder(sparse=False)

integer_encoded99 = integer_encoded99.reshape(len(integer_encoded99),1)

onehot_encoded99 = onehot_encoder99.fit_transform(integer_encoded99)





#print('trainingData shape: ',onehot_encoded.shape)

#print('testData shape: ',onehot_encoded99.shape)





listC = []

for f in range(onehot_encoded.shape[0]):

    listC.append(onehot_encoded[f][0])

#print(listC)



listC2 = []

for f in range(onehot_encoded99.shape[0]):

    listC2.append(onehot_encoded99[f][0])



listQ=[]

for f in range(onehot_encoded.shape[0]):

    listQ.append(onehot_encoded[f][1])

    

listQ2 = []

for f in range(onehot_encoded99.shape[0]):

    listQ2.append(onehot_encoded99[f][1])

    

listS=[]

for f in range(onehot_encoded.shape[0]):

    listS.append(onehot_encoded[f][2])



listS2 = []

for f in range(onehot_encoded99.shape[0]):

    listS2.append(onehot_encoded99[f][2])

    

trainingData['EmbarkedC'] = listC

trainingData['EmbarkedQ'] = listQ

trainingData['EmbarkedS'] = listS

    

    

testData['EmbarkedC'] = listC2

testData['EmbarkedQ'] = listQ2

testData['EmbarkedS'] = listS2

    

sns.barplot(x='Embarked', y='Survived', data=trainingData)

trainingData = trainingData.drop('Embarked', axis=1)    

testData = testData.drop('Embarked', axis=1)

trainingData
list10 = trainingData['SibSp']

list11 = trainingData['Parch']



list12 = list(map(add, list10, list11))



list100 = testData['SibSp']

list110 = testData['Parch']



list120 = list(map(add,list100,list110))



trainingData['FamilyMembers'] = list12 



testData['FamilyMembers'] = list120



trainingData = trainingData.drop('SibSp', axis=1)    

trainingData = trainingData.drop('Parch', axis=1)

testData = testData.drop('SibSp', axis=1)

testData = testData.drop('Parch', axis=1)

list15 = trainingData['Age']



nullles = 0

for l in list15:

    if(pd.isnull(l)):

        nullles=nullles+1

        

print(nullles)



list150 = testData['Age']

nulles150 = 0

for l in list150:

    if(pd.isnull(l)):

        nulles150= nulles150+1



print('in test data',nulles150)

trainingData['Age'].fillna(trainingData['Age'].median(skipna=True), inplace=True)

testData['Age'].fillna(testData['Age'].median(skipna=True),inplace=True)

trainingData
trainingData_X = trainingData.drop('Survived', axis=1)

trainingData_Y = trainingData['Survived']

sc = StandardScaler()



toNormalize = trainingData_X[['Age', 'Fare', 'FamilyMembers']]



toNormalize2 = sc.fit_transform(toNormalize)





toNormalize32 = testData[['Age','Fare','FamilyMembers']]



toNormalize322 = sc.fit_transform(toNormalize32)



#print(toNormalize2.shape)



listAge = []

for f in range(toNormalize2.shape[0]):

    listAge.append(toNormalize2[f][0])

    

list2Age = []

for f in range(toNormalize322.shape[0]):

    list2Age.append(toNormalize322[f][0])

    

listFare = []

for f in range(toNormalize2.shape[0]):

    listFare.append(toNormalize2[f][1])



list2Fare = []

for f in range(toNormalize322.shape[0]):

    list2Fare.append(toNormalize2[f][1])

    

listFamilyMembers = []

for f in range(toNormalize2.shape[0]):

    listFamilyMembers.append(toNormalize2[f][2])



list2FamilyMembers = []

for f in range(toNormalize322.shape[0]):

    list2FamilyMembers.append(toNormalize322[f][2])

    

    

trainingData_X['Age'] = listAge

trainingData_X['Fare'] = listFare

trainingData_X['FamilyMembers'] = listFamilyMembers

 

    

testData['Age'] = list2Age

testData['Fare'] = list2Fare

testData['FamilyMembers'] = list2FamilyMembers

testData.shape
trainingData_X.shape
def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

def precision(model, registrar=False):

    y_pred = model.predict(train_dfX)

    train_auc = roc_auc_score(train_dfY, y_pred)

    y_pred = model.predict(val_dfX)

    val_auc = roc_auc_score(val_dfY, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])

def func_model(arquitectura): 

    np.random.seed(32)

    random_seed = 32

    first =True

    inp = Input(shape=(trainingData_X.shape[1],))

    for capa in arquitectura:        

        if first:

            x=Dense(capa, activation="relu", kernel_initializer=initializers.RandomNormal(seed= random_seed), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(inp)            

            first = False

        else:

            x=Dense(capa, activation="relu", kernel_initializer=initializers.RandomNormal(seed= random_seed), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)  

    x=Dense(1, activation="sigmoid", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros')(x)  

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002), metrics=['binary_accuracy'])

    return model

train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(trainingData_X,trainingData_Y , test_size=0.1, stratify=trainingData_Y)
print('Entrenamiento: ', train_dfX.shape)

print('Validacion: ', val_dfX.shape)

arq1 = [1024, 1024, 512, 256]

model1 = func_model(arq1)

print(model1.summary())

train_history = model1.fit(train_dfX, train_dfY, batch_size=32, epochs= 30, validation_data = (val_dfX, val_dfY))

graf_model(train_history)

precision(model1, False)
testData.shape
y_test =model1.predict(testData)

submission['Survived'] = np.rint(y_test).astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)