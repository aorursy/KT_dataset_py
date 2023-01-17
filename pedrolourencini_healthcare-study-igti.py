import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder #data preprocessing

from sklearn.model_selection import train_test_split #dataset break in training and test 

from sklearn.metrics import confusion_matrix, accuracy_score 

from sklearn.naive_bayes import GaussianNB #Naive Bayes classifier

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC #Support Vector Machine
nomeArquivo = '../input/healthcareigti/HealthCare.csv'

dataset = pd.read_csv(nomeArquivo, sep = ',')
dataset.head()
dataset.shape
dataset.info()
dataset.fillna(dataset.mean(), inplace = True)

dataset.head()
dataset_to_array = np.array(dataset)
target = dataset_to_array[:, 57]

target = target.astype('int')

target
dataset_sensor = np.column_stack((

    dataset_to_array[:,11], 

    dataset_to_array[:,33],

    dataset_to_array[:,34],

    dataset_to_array[:,35],

    dataset_to_array[:,36],

    dataset_to_array[:,38]

))
dataset_medico = np.column_stack((dataset_to_array[:, 4],

    dataset_to_array[:, 6],

    dataset_to_array[:, 9],

    dataset_to_array[:, 39],

    dataset.age,

    dataset.sex,

    dataset.hypertension

))
dataset_paciente = np.concatenate((dataset_medico, dataset_sensor), axis = 1)

dataset_paciente 
dataset_paciente.shape
x_train, x_test, y_train, y_test = train_test_split(dataset_paciente, target, random_state = 223)
modelSVM = SVC(kernel='linear')
modelSVM.fit(x_train, y_train)
previsao = modelSVM.predict(x_test)
accuracia = accuracy_score(y_test, previsao)

print("Acuracia utilizando o SVM : ", accuracia, "\nEm porcentagem : ", round(accuracia * 100), "%\n")
cm = confusion_matrix(y_test, previsao)

df_cm = pd.DataFrame(cm, index = [i for i in "01234"], columns = [i for i in "01234"])

plt.figure(figsize = (10, 7))

sns.heatmap(df_cm, annot = True)
dataset_to_array = np.array(dataset)

label = dataset_to_array[:, 57]

label = label.astype('int')

label[label > 0] = 1 #if 1 pacient has a disease
label
x_train, x_test, y_train, y_test = train_test_split(dataset_paciente, label, random_state = 223)
modelSVM = SVC(kernel= 'linear')
modelSVM.fit(x_train, y_train)
previsao = modelSVM.predict(x_test)
accuracia = accuracy_score(y_test, previsao)

print("Acuracia utilizando o SVM : ", accuracia, "\nEm porcentagem : ", round(accuracia*100), "%\n")
cm = confusion_matrix(y_test, previsao)

df_cm = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])

plt.figure(figsize= (10, 7))

sns.heatmap(df_cm, annot = True)