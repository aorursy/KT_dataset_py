import numpy as np
import pandas as pds
import collections
from sklearn import preprocessing
from pandas import concat
import matplotlib.pyplot as plt
import itertools

%matplotlib inline
#Load dos dados
originalData = pds.read_csv('../input/KaggleV2-May-2016.csv')
print(originalData.shape)
originalData.head()
#Drop dos IDS
data = originalData.copy()
data = data.drop('PatientId',1)
data = data.drop('AppointmentID',1)

#Renomeando a coluna No-show
data.rename(columns={'No-show':'Noshow'}, inplace=True)

#Transformacao dos dados
data.Noshow = data.Noshow.map({'Yes': 1, 'No': 0})
data.Gender = data.Gender.map({'F': 1, 'M': 0})
data.Age = pds.to_numeric(data.Age)
data.Neighbourhood = data.Neighbourhood.astype("category").cat.codes

#Construcao de novas colunas
data['SchMonth'] = data.ScheduledDay.str.slice(5,7)
data['SchDay'] = data.ScheduledDay.str.slice(8,10)

data.ScheduledDay = pds.to_datetime(data.ScheduledDay.str.slice(0,10))
data.AppointmentDay = pds.to_datetime(data.AppointmentDay.str.slice(0,10))
data['WaitingDays'] = abs((data.ScheduledDay - data.AppointmentDay).dt.days)

#Drop de colunas
data = data.drop('ScheduledDay',1)
data = data.drop('AppointmentDay',1)

#Normalizacao
data.SchMonth = preprocessing.scale(list(data.SchMonth.astype(float)))
data.SchDay = preprocessing.scale(list(data.SchDay.astype(float)))
data.Age = preprocessing.scale(list(data.Age.astype(float)))
data.Neighbourhood = preprocessing.scale(list(data.Neighbourhood.astype(float)))

#Remocao de dados invalidos
data = data[data.Age > 0]

data.head()
data = data.reindex(columns=sorted(data.columns))
data = data.reindex(columns=(['Noshow'] + list([a for a in data.columns if a != 'Noshow'])))
data.head()
corr = data.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

#Separação do dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = data.drop('Noshow',1)
y = data.Noshow

#Separação de conjunto de testes
X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#Separação de conjunto de validação
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.2, random_state=0)  
#Imbalancement treatment com over sampling
from imblearn.over_sampling import SMOTE

balancer = SMOTE(kind='regular')
x_resampled, y_resampled = balancer.fit_sample(X_train, y_train)

print('Normal Data: ', collections.Counter(y_train))
print('Resampled: ', collections.Counter(y_resampled))
#Criacao dos modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  

algsSize = 3

#Algoritimos que serao treinados
algs = []
algs.append(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', random_state=0))
algs.append(LogisticRegression(random_state=0, penalty='l2', C=1, fit_intercept=True, solver='liblinear'))
algs.append(DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='best')  )

#Mesmo algoritimos que serao treinados com o dataset balanceado
algsSampled = []
algsSampled.append(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', random_state=0))
algsSampled.append(LogisticRegression(random_state=0, penalty='l2', C=1, fit_intercept=True, solver='liblinear'))
algsSampled.append(DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='best')  )

for x in range(0, algsSize):
    print('Fitting: ', type(algs[x]).__name__)
    algs[x].fit(X_train, y_train) 
    algsSampled[x].fit(x_resampled, y_resampled) 
#Definição de dataframe para exibição de resultados
results = pds.DataFrame(columns=['Name', 'Type', 'Resampled', 'ACC'])
#Função para display de resultados
def appendResult(alg, dataType, resampled, X, y):
    algName = type(alg).__name__
    predicted = alg.predict(X)
    accuracy = accuracy_score(y, predicted)
    results.loc[len(results)]=[algName, dataType, resampled, accuracy]
    print('Confusion Matrix - ', algName, ' RESAMPLED = ', resampled)
    plot_confusion_matrix(cm=confusion_matrix(y, predicted), target_names=['Show', 'NoShow'])
#Acuracia do modelo em VALIDACAO
for x in range(0, algsSize):
    appendResult(algs[x], 'Validation', 'No', X_val, y_val)
    appendResult(algsSampled[x], 'Validation', 'Yes', X_val, y_val)
#Acuracia do modelo em TESTE
for x in range(0, algsSize):
    appendResult(algs[x], 'Test', 'No', X_test, y_test)
    appendResult(algsSampled[x], 'Test', 'Yes', X_test, y_test)
#Resultados
results