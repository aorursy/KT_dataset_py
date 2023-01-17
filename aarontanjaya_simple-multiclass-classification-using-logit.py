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





data=pd.read_csv('/kaggle/input/uci-wine-dataset/wine.data',names=['Cultivars','Alcohol','Malic acid','Ash','Alcalinity of ash',

'Magnesium','Total Phenols','Flavanoids','Nonflavanoid phenols',

'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])

print(data)

input=['Alcohol','Malic acid','Ash','Alcalinity of ash',

'Magnesium','Total Phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

x=data[input]

y=data['Cultivars']
print(data.describe())

print(data.dtypes)

print(data.corr())

for dat in input:

    plt.hist(data[dat])

    plt.title(dat)

    plt.show()
from  sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

print(xtrain.shape)

print(ytrain.shape)

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

ex=ss.fit_transform(xtrain)

extest=ss.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

paramlogi={'multi_class':['ovr','multinomial'],'solver':['newton-cg','saga','lbfgs'],'C':[1.0,0.8,0.7,0.6],'max_iter':[150]}

logi=GridSearchCV(LogisticRegression(), param_grid=paramlogi,scoring=['accuracy'],refit='accuracy')

logi.fit(ex,ytrain)
from sklearn.metrics import classification_report

def show(grid,xex,yey):

    for a,b in zip(grid.cv_results_['params'],grid.cv_results_['mean_test_accuracy']):#,grid.cv_results_['mean_test_precision'],grid.cv_results_['mean_test_recall']):

        print('parameters',a,'test accuracy',b)#,'precision',c,'recall',d)

    print('best param:',grid.best_estimator_,'score',grid.best_score_)

    print(classification_report(yey,grid.predict(xex)))

show(logi,extest,ytest)
logis=LogisticRegression(C=0.7,max_iter=150,multi_class='ovr',solver='newton-cg')

logis.fit(ex,ytrain)

print('score:', logis.score(extest,ytest))