# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#data = pd.concat(map(pd.read_csv, ['../input/crime-data-in-brazil/BO_2007_1.csv', '../input/crime-data-in-brazil/BO_2007_2.csv','../input/crime-data-in-brazil/BO_2008_1.csv','../input/crime-data-in-brazil/BO_2008_1.csv','../input/crime-data-in-brazil/BO_2008_2.csv','../input/crime-data-in-brazil/BO_2009_1.csv','../input/crime-data-in-brazil/BO_2009_2.csv','../input/crime-data-in-brazil/BO_2010_1.csv','../input/crime-data-in-brazil/BO_2010_2.csv','../input/crime-data-in-brazil/BO_2011_1.csv','../input/crime-data-in-brazil/BO_2011_2.csv','../input/crime-data-in-brazil/BO_2012_1.csv','../input/crime-data-in-brazil/BO_2012_2.csv']))

data = pd.concat(map(pd.read_csv, ['../input/crime-data-in-brazil/BO_2007_1.csv']))
data.head()
data.info()
plt.figure(figsize=(12,6))

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
del data['DESDOBRAMENTO']

del data['LATITUDE']

del data['LONGITUDE']

del data['IDADE_PESSOA']

del data['DESCR_PROFISSAO']

del data['DESCR_GRAU_INSTRUCAO']

del data['CONDUTA']

del data['COR']
plt.figure(figsize=(12,6))

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data = data.dropna()

x = data.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,15,16,18,19,20]].values 

x[0]
y = data.iloc[:,14].values

y
from sklearn.preprocessing import LabelEncoder

train = x

train[0]
label_encoder = LabelEncoder();

train[:,2] = label_encoder.fit_transform(train[:,2])

train[:,3] = label_encoder.fit_transform(train[:,3])

train[:,4] = label_encoder.fit_transform(train[:,4])

train[:,5] = label_encoder.fit_transform(train[:,5])

train[:,6] = label_encoder.fit_transform(train[:,6])

train[:,7] = label_encoder.fit_transform(train[:,7])

train[:,8] = label_encoder.fit_transform(train[:,8])

train[:,9] = label_encoder.fit_transform(train[:,9])

train[:,10] = label_encoder.fit_transform(train[:,10])

train[:,11] = label_encoder.fit_transform(train[:,11])

train[:,12] = label_encoder.fit_transform(train[:,12])

train[:,13] = label_encoder.fit_transform(train[:,13])

train[:,14] = label_encoder.fit_transform(train[:,14])

train[:,15] = label_encoder.fit_transform(train[:,15])

train[:,16] = label_encoder.fit_transform(train[:,16])

#train[:,17] = label_encoder.fit_transform(train[:,17])

#train[:,18] = label_encoder.fit_transform(train[:,18])

#train[:,19] = label_encoder.fit_transform(train[:,19])

#train[:,20] = label_encoder.fit_transform(train[:,20])









train[0]
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()

train = scaler_x.fit_transform(train)

train
from sklearn.model_selection import train_test_split

x_treinamento,x_teste,y_treinamento,y_teste = train_test_split(train,y,test_size = 0.3)
from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression(max_iter=10000)

classificador.fit(x_treinamento,y_treinamento)
previsoes = classificador.predict(x_teste)
previsoes
y_teste
from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(y_teste,previsoes) 

taxa_acerto
sns.pairplot(data,hue='RUBRICA')
x_treinamento,x_teste,y_treinamento,y_teste = train_test_split(train,y,test_size = 0.3,random_sate=101)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(x_treinamento,y_treinamento)
pred = dtree.predict(x_teste)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_teste,pred))
len(train[1])
