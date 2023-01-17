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
data = pd.read_csv("/kaggle/input/fetal-health-classification/fetal_health.csv")

data.head().transpose()
data.info()
data.fetal_health.unique()
from sklearn.model_selection import train_test_split

y = data.fetal_health

X = data.drop(['fetal_health'], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3,

                                                      random_state=12)
lista = (X.dtypes != 'object')

numerical_cols = list(lista[lista].index)



print("Variáveis Numéricos:")

print(numerical_cols)
from sklearn.preprocessing import StandardScaler



std_X_train = X_train.copy()

std_X_valid = X_valid.copy()



std_numerical_cols = numerical_cols.copy()

del std_numerical_cols[1]



for column in std_numerical_cols:

    transformer = StandardScaler()

    

    values = np.array(std_X_train[column]).reshape(-1,1)

    std_X_train[column] = transformer.fit_transform(values)

    

    values = np.array(std_X_valid[column]).reshape(-1,1)

    std_X_valid[column] = transformer.transform(values)
std_X_train.head().transpose()
#Model

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix  

from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier

import xgboost 



#Viz

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

%matplotlib inline
modelo = XGBClassifier()

modelo.fit(X_train, y_train)
def matrix_confusao(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred) 



    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,

                xticklabels=['1', '2', '3'],

                yticklabels=['1', '2', '3']

               )

    plt.xlabel('Valores Reais')

    plt.ylabel('Valores Previstos')
def train_model(classifier, X_test, y_test):    

    

    # predict the labels on validation dataset

    predictions = classifier.predict(X_test)

    

    report = classification_report(y_test, predictions)

    print("\033[1m" + "{:>50}".format("%s" % (report)))

        

    mat = matrix_confusao(y_test, predictions)

        

    return predictions
predicoes = train_model(modelo, X_valid, y_valid)
from sklearn.metrics import roc_auc_score, f1_score

roc_auc_score_cbc = roc_auc_score(y_valid, modelo.predict_proba(X_valid), multi_class='ovr')

print('Area under the ROC Curve',roc_auc_score_cbc)
print("F1 Score: ", f1_score(y_valid, modelo.predict(X_valid), average=None))