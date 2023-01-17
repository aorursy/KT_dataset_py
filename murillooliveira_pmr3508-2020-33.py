# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt



import sklearn



import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score



from sklearn import preprocessing



from IPython.display import HTML



import base64



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Criação da Base de Treino



#importacao do arquivo csv como a base de treino train_base



train_base = pd.read_csv("../input/adult-pmr3508/train_data.csv",

names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", 

"Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Income"],

sep=r'\s*,\s*',

engine='python',

na_values="?")



# Exploração da base 



#visualizar headers da base

train_base.head()

#mostrar tamanho da Base em (linhas, colunas)



train_base.shape 
#Tratamento da Base



#Criação de uma nova base de treino tratada -> new_train_base



#Remover a primeira linha da Base

train_base.drop(train_base.index[0],inplace=True) 



#Criação de nova base, removendo linhas c/ falta de infos

new_train_base = train_base.dropna()



#preview da nova base

new_train_base
#Exploracao da nova Base



#Uso do matplotlib para criar histogramas,

#contendo algumas distribuições,

#conforme os parâmetros dos indivíduos da Base



new_train_base["Race"].value_counts().plot(kind="bar")
new_train_base["Sex"].value_counts().plot(kind="bar")
new_train_base["Relationship"].value_counts().plot(kind="bar")
new_train_base["Income"].value_counts().plot(kind="bar")
#Criação da Base de Teste



test_base = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "ID","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



#Visualizacao dos headers da base de teste



test_base.head()
#Tratamento da Base



#Criação de uma nova base de teste tratada -> new_test_base



#Remover a primeira linha da Base

test_base.drop(test_base.index[0],inplace=True) 



new_test_base = test_base



#Criação de nova base, removendo linhas c/ falta de infos

new_test_base.fillna(method ='ffill', inplace = True) 



new_test_base
train_base_x = new_train_base[["Age", "Marital Status", "Workclass", "fnlwgt", "Education", "Education-Num",

                               "Relationship", "Hours per week", 

                               "Country"]].apply(preprocessing.LabelEncoder().fit_transform)



train_base_y = new_train_base.Income 



train_base_y.head()
new_test_base
train_base_y.shape
test_base_x = new_test_base[["Age", "Marital Status", "Workclass", "fnlwgt", "Education", "Education-Num",

                             "Relationship","Hours per week", 

                             "Country"]].apply(preprocessing.LabelEncoder().fit_transform) 

 

test_base_y = new_test_base.Income 
#Implementacao do classificador KNN



KNN_class = KNeighborsClassifier(n_neighbors=15)



KNN_class.fit(train_base_x,train_base_y)
cross = 30



cross_score = cross_val_score(KNN_class, train_base_x,train_base_y, cv=cross)



cross_score

prediction_y = KNN_class.predict(test_base_x)



prediction_y
count = 0



for cs in cross_score:

    count = count + cs

    

accuracy = count/cross



accuracy
prediction_y



prediction_y.shape
# Exportacao dos outputs



output = pd.DataFrame()



output[0] = new_test_base.ID



output[1] = prediction_y



output.columns = ['Id', 'Income']



output.to_csv('output.csv', index=False)



output
