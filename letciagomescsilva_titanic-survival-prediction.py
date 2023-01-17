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
# Algebra Linear:
import numpy as np 

# Processamento/manipulação dos dados:
import pandas as pd 

# Visualização dos dados:
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algoritimos Machine Learning:
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
train_df.info()
# Identificando quais features possuem valores faltantes:

Null_values = train_df.isnull().sum().sort_values(ascending=False)
Null_values
# Porcentagem de dados da feature que possui valores faltanteS:

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_1.sort_values(ascending=False)
# Transformando os valores anteriores em porcentagem: 

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
percent_2.sort_values(ascending=False)
# Total de valores faltantes x Porcentagem:

missing_data = pd.concat([Null_values, percent_2], axis=1, keys=['Total', '%'])
missing_data
# Matriz de correlação para entendermos quais features mais influenciam a sobrevivência:
train_df.corr()
# Matriz de calor:

heat_map = sns.heatmap(train_df.corr(),annot = True,fmt = '.2g',cmap='Blues')
plt.title('Correlação entre variáveis do dataset Notas Pism')
plt.show()

train_df = train_df.drop(['PassengerId'], axis=1)
