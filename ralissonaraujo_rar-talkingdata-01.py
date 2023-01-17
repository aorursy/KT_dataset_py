# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#ocultando mensagens de warning
import warnings
warnings.filterwarnings("ignore")
#importanto os principais pacotes
import pandas as pd
import numpy as np

import gc
import os


#Nomeando as colunas do dataset
#Definindo a menor "máscara" para a caputura dos dados, desta forma o dataset será menor que o padrão
#de importação da função read_csv do Pandas

dtypes = {
    
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    
}
# Usaremos 2/3 do dataset train.csv para a analise e criação do modelo de machine learning.

lines = 184903891
divisor = 10
amostra = int(lines / divisor)

skiplines = np.random.choice(np.arange(1, lines), size=lines-1-(amostra), replace=False)
skiplines= np.sort(skiplines)

#import joblib
#with joblib.parallel_backend('dask'):
%time train = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', skiprows=skiplines, dtype=dtypes, parse_dates=['click_time', 'attributed_time'])

# Limpeza de memória, etapa necessária para a continuação do processo se você tiver uma máquina com menos
#de 12gb de mmória Ram
del skiplines
gc.collect()
train.info()
train = train.drop(columns=[ 'ip', 'attributed_time'], inplace=False)
target_count = train['is_attributed'].value_counts()
print('Click 0:', target_count[0])
print('Click 1:', target_count[1])
print('Proporção:', (round(target_count[1] / target_count[0], 6)*100), '%')
target_count.plot(kind='bar', title='Distribuição de Clicks',color = ['#1F77B4', '#FF7F0E']);
# contagem das classes 0 e 1 da variavel target
clicks_0, clicks_1 = train['is_attributed'].value_counts()

# Divisão de Classes
df_class_0 = train[train['is_attributed'] == 0]
df_class_1 = train[train['is_attributed'] == 1]
df_class_0_under = df_class_0.sample(clicks_1)

train2 = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(train2['is_attributed'].value_counts())
train2['is_attributed'].value_counts().plot(kind='bar', title='Distribuição de clicks',color = ['#1F77B4', '#FF7F0E']);
#limpeza de memória

del clicks_0
del clicks_1
del df_class_0
del df_class_1
del df_class_0_under
del train

gc.collect()
train2.info()
train2["hour"] = train2["click_time"].dt.hour
train2.head()
# Vamos deletar a coluna de data e hora 
train2 = train2.drop(columns=[ 'click_time'], inplace=False)
train2['hour'] = train2['hour'].astype('uint16')
train2.info()
correl = train2.corr()
print(correl)
import seaborn as sns
# Pairplot
sns.pairplot(train2)
train2.columns
# Vamos separar as colunas em Features e Target
features = train2[
    [
        
    'app', 'os', 'channel', 'hour'
    ]
    
]
target = train2['is_attributed'] 
#Divisão do dataset em treino e teste

from sklearn.model_selection import train_test_split

X = features
y = target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=133)

# Importando os módulos necessários

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# Carregando features e Target
X = features
Y = target

# Definindo os valores para o número de folds
num_folds = 10
seed = 7

# Preparando uma lista de modelos que serão analisados.


modelos = []

#modelos.append(('LDA', LinearDiscriminantAnalysis()))
#modelos.append(('NB', GaussianNB()))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('RanF', RandomForestClassifier(bootstrap=False, max_depth=30, 
                                               max_features='log2', min_samples_split=10, 
                                               min_weight_fraction_leaf=0, n_estimators=50, warm_start=True)))

#modelos.append(('SGD', SGDClassifier()))
modelos.append(('ExtraTree', ExtraTreesClassifier()))
modelos.append(('GBoost', GradientBoostingClassifier()))
#modelos.append(('AdaBoost', AdaBoostClassifier()))
modelos.append(('DesTree', DecisionTreeClassifier()))
#modelos.append(('MLP', MLPClassifier(hidden_layer_sizes=(50,50,50), activation='logistic', solver='adam', alpha=0.0001)))
#modelos.append(('SVM', SVC()))

# Avaliando cada modelo em um loop
resultados = []
nomes = []



for nome, modelo in modelos:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(modelo, X, Y, cv = kfold, scoring = 'accuracy')
    resultados.append(cv_results)
    nomes.append(nome)
    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())
    print(msg)


# Boxplot para comparar os algoritmos
fig = plt.figure()
fig.suptitle('Comparação de Algoritmos de Classificação')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nomes)
plt.show()
# Avaliação do modelo usando o 

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Separando as features e target
X = features
Y = target

# Divide os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

# Criando o modelo
Model = RandomForestClassifier(bootstrap=False, max_depth=30, max_features='log2', min_samples_split=10,
                               min_weight_fraction_leaf=0, n_estimators=50, warm_start=True)


# Definindo os valores para o número de folds
num_folds = 10
seed = 7
    
# Separando os dados em folds
kfold = KFold(num_folds, True, random_state = seed)    
    

# Treinando o modelo
Model.fit(X_train, Y_train)


# Previsão do modelo

Predict = Model.predict(X_test)


# Acurácia final e ROC
resultadoAC = cross_val_score(Model, X, Y, cv = kfold, scoring = 'accuracy')
resultadoROC = cross_val_score(Model, X, Y, cv = kfold, scoring = 'roc_auc')

print("ROC foi de: %.3f" % (resultadoROC.mean() * 100))
print("A Acurácia foi de: %.3f%%" % (resultadoAC.mean() * 100.0))


report = classification_report(Y_test, Predict)

# Imprimindo o relatório
print(report)
