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
import pandas as pd
directory = '../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
# Lendo a base de dados.

data = pd.read_csv(directory)

data.shape
data.head()
# A variável TotalCharges deveria ser um float, porém tem algumas linhas com caracteres bizarros.

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

data.isnull().sum()
# As 11 linhas vazias em TotalCharges são resultado dos caracteres bizarros. Como são só 11 linhas, vamos deletá-las.

data.dropna(inplace = True)
# Tipos das variáveis.

tipos_das_variaveis = data.dtypes

tipos_das_variaveis
# Substituindo as strings das variáveis categóricas por números.

data['gender'] = data['gender'].map(lambda x: 1 if x == 'Male' else 0)

data['Partner'] = data['Partner'].map(lambda x: 1 if x == 'Yes' else 0)

data['Dependents'] = data['Dependents'].map(lambda x: 1 if x == 'Yes' else 0)

data['PhoneService'] = data['PhoneService'].map(lambda x: 1 if x == 'Yes' else 0)

data['MultipleLines'] = data['MultipleLines'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['InternetService'] = data['InternetService'].map(lambda x: 1 if x  == 'Fiber optic' else (0 if x == 'No' else 2))

data['OnlineSecurity'] = data['OnlineSecurity'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['OnlineBackup'] = data['OnlineBackup'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['DeviceProtection'] = data['DeviceProtection'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['TechSupport'] = data['TechSupport'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['StreamingTV'] = data['StreamingTV'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['StreamingMovies'] = data['StreamingMovies'].map(lambda x: 1 if x  == 'Yes' else (0 if x == 'No' else 2))

data['Contract'] = data['Contract'].map(lambda x: 1 if x  == 'Month-to-month' else (0 if x == 'One year' else 2))

data['PaperlessBilling'] = data['PaperlessBilling'].map(lambda x: 1 if x == 'Yes' else 0)

data['PaymentMethod'] = data['PaymentMethod'].map(lambda x: 0 if x == 'Electronic check' else(1 if x == 'Mailed check' else(2 if x == 'Bank transfer (automatic)' else 3)))
# Verificando balanceamento da variável resposta.

data['Churn'].value_counts()
# Divisão em treino/teste

from sklearn.model_selection import train_test_split



data['Churn'] = data['Churn'].map(lambda x: 1 if x == 'Yes' else 0)

y = data['Churn']

data = data.drop(['customerID', 'Churn'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=11)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Normalizando variáveis numéricas.

from sklearn.preprocessing import MinMaxScaler



columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()

X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
# Treinando uma Random Forest para classificação através de GridSearch

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



forest = RandomForestClassifier(random_state=11)



n_estimators = [100, 300, 500, 800, 1200]

max_depth = [5, 8, 15, 25, 30]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]



hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  

              min_samples_split = min_samples_split, 

              min_samples_leaf = min_samples_leaf)



gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 

                     n_jobs = -1)

bestF = gridF.fit(X_train, y_train)
final_table = pd.DataFrame(bestF.cv_results_)

final_table
# Verificando a matriz de confusão.

from sklearn.metrics import confusion_matrix



predicted = bestF.best_estimator_.predict(X_test)

mc = confusion_matrix(y_test, predicted)

mc
# Algumas métricas.

def acc(tn, fp,fn, tp):

    return ((tn+tp)/(tn+fp+fn+tp))



def precision(tp, fp):

    return (tp/(tp+fp))



def recall(tp, fn):

    return (tp/(tp+fn))



print('acc: {}'.format(acc(mc[0, 0], mc[0, 1], mc[1, 0], mc[1, 1])))

print('precision: {}'.format(precision(mc[1, 1], mc[0, 1])))

print('recall: {}'.format(recall(mc[1, 1], mc[1, 0])))
# Feature importance.

import matplotlib.pyplot as plt



importances = bestF.best_estimator_.feature_importances_

std = np.std([tree.feature_importances_ for tree in bestF.best_estimator_.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Printando o ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plotando o gráfico de importância

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

        color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), indices)

plt.xlim([-1, X_train.shape[1]])

plt.show()