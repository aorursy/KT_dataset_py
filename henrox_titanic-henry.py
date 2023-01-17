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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
train = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')
test = pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv')
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators = 100, n_jobs = 1, random_state=0)
#fazendo o onehotencoder manual para sexo
def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0 
#adicionando uma coluna desse sexo 0,1 na ultima coluna
variaveis = ['sexo_binario','Age']
train['sexo_binario'] = train['Sex'].map(transformar_sexo)
train.head(2)
X = train[variaveis]
y = train['Survived']
X = X.fillna(-1)
modelo.fit(X,y)
test['sexo_binario'] = test['Sex'].map(transformar_sexo)
X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)
X_prev.head()
y_pred = modelo.predict(X_prev)
y_pred
sub = pd.Series(y_pred, index=test['PassengerId'], name = 'Survived')
sub.to_csv('primeiro_modelo.csv', header = True)
!head -n10 primeiro_modelo.csv
