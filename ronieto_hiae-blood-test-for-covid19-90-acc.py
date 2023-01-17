# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import itertools

%matplotlib inline
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx', index_col=0)

warnings='ignore'
df.reset_index(drop=False, inplace=True)
df.tail()
df.isnull().sum()
# o número de células com valores NaN é muito grande. Exposição gráfica para melhor visualização

plt.figure(figsize=(14,9))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df.notnull().sum()
cols=df.columns
relevant = df[cols[0:20]]
relevant['Neutrophils'] = df['Neutrophils']
relevant.info()
col = relevant.columns
def falta_numero(x):

      return sum(x.isnull())
relevant.apply(falta_numero, axis=0)
relevant.head()
def normaliza (x):

    y = x + 5

    return y
relevant[col[6:21]] = relevant[col[6:21]].apply(normaliza)
relevant[col[6:21]] = relevant[col[6:21]].fillna(9)
plt.figure(figsize=(14,9))

sns.heatmap(relevant.isnull(), cbar=False, yticklabels=False)
## Eliminamos todos os valores nulos ou negativos

relevant.notnull().sum()
covid = relevant
covid.reset_index(drop=True, inplace=True)
covid.tail()
covid = covid.drop(['Patient ID'], axis=1, inplace=True)

covid = relevant
covid = covid.drop(['Patient addmited to regular ward (1=yes, 0=no)'], axis=1, inplace=True)

covid = relevant
covid = covid.drop(['Patient addmited to semi-intensive unit (1=yes, 0=no)'], axis=1, inplace=True)

covid = relevant
covid = covid.drop(['Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1, inplace=True)

covid = relevant
covid.head()
# Verificando os valores da coluna target

covid['SARS-Cov-2 exam result'].value_counts()
# convertendo celulas categóricas em numericas

covid['SARS-Cov-2 exam result'] = pd.get_dummies(covid['SARS-Cov-2 exam result'])
covid['SARS-Cov-2 exam result'].value_counts()
# verificação final

covid.tail()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(covid.drop('SARS-Cov-2 exam result',axis=1), covid['SARS-Cov-2 exam result'], test_size=0.25)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
reg_cm=confusion_matrix(y_test,predictions)

print(reg_cm)
def matriz_deconfusao(cm, target_names,

                          title='Matriz de Confusão',

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

matriz_deconfusao(cm = reg_cm, normalize  = False,

                      target_names = ['Positivo - COVID-19', 'Não Infectado'],

                      title        = "Matriz de Confusão")