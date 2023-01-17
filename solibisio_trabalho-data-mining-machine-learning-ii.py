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
# Análise exploratória dos dados

# Estatística do negócio

# correlação

# distribuição

# interpretação desses dados

# tratamento converter, dividir, criar colunas, etc

# grid search

# hiperparametros
df = pd.read_csv('../input/hmeq-data/hmeq.csv')



df.info() 
df.shape
df.head()
df.sample(30)
df.describe().T
print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())

print ("\nPorcentagem Missing: \n", df.isna().mean().round(4) * 100)
df['REASON'].astype('category').cat.categories
df['JOB'].astype('category').cat.categories
df['REASON'].astype('category').cat.codes



# -1 é nulo
df['JOB'].astype('category').cat.codes



# -1 é nulo
# manter apenas rows com pelo menos 8 campos não nulos 

df.dropna(thresh=8, inplace=True)
# preencher no campo REASON DebtCon como default para nulo

df.REASON.fillna('DebtCon', inplace=True)



# preencher no campo JOB Other como default para nulo

df.JOB.fillna('Other', inplace=True)

df.JOB.fillna('Other', inplace=True)



print ("INFO     : " ,df.info())

print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())

print ("\nPorcentagem Missing: \n", df.isna().mean().round(4) * 100)
# Convertendo as colunas categórias em colunas numéricas

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes


print ("INFO     : " ,df.info())

print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())

print ("\nTotal Missing: \n", df.isna().sum())

print ("\nPorcentagem Missing: \n", df.isna().mean().round(4) * 100)
df["DEROG"].value_counts()

df["DEROG"].value_counts()

df["DELINQ"].value_counts()
df["MORTDUE"].value_counts()

df["VALUE"].value_counts()
df["YOJ"].value_counts()
df["CLAGE"].value_counts()
df["NINQ"].value_counts()
df["CLNO"].value_counts()
df["DEROG"].fillna(value=0,inplace=True)

df["DELINQ"].fillna(value=0,inplace=True)
print ("INFO     : " ,df.info())

print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())

print ("\nTotal Missing: \n", df.isna().sum())

print ("\nPorcentagem Missing: \n", df.isna().mean().round(4) * 100)
# criar uma base com dados limpos (removendo todos os missing)

df2 = df.dropna()



# criar uma base com dados utilizando ffill e bfill

df3 = df.fillna(method='ffill')

df3 = df3.fillna(method='bfill')

import seaborn as sn

import matplotlib.pyplot as plt



plt.figure(figsize=(18,18))





corrMatrix = df[df['DEBTINC'].notnull()].corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
import seaborn as sns

from scipy.stats import norm

from pylab import rcParams

rcParams['figure.figsize'] = 30, 15



dic = {"LOAN":df["LOAN"],"BAD":df["BAD"],"MORTDUE":df["MORTDUE"],"VALUE":df["VALUE"],"YOJ":df["YOJ"]}

rcParams['figure.figsize'] = 5, 5



df_pair = pd.DataFrame(dic)

sns.pairplot(df_pair,vars=['LOAN', 'MORTDUE',"VALUE","YOJ"],hue="BAD")
sns.countplot(x=df2['BAD'], data=df2).set_title("Distribuição para df2")
ax = sns.countplot(x=df3['BAD'], data=df3).set_title("Distribuição para df3")
df2['BAD'].value_counts()
df3['BAD'].value_counts()
fcat = ['REASON','JOB']



for col in fcat:

    plt.figure()

    sns.countplot(x=df[col], data=df)

    plt.show()


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split



feats  = [c for c in df.columns if c not in ['BAD']]





# df2

df2_x_treino, df2_x_valid, df2_y_treino, df2_y_valid = train_test_split(df2[feats],

                                                                              df2['BAD'], 

                                                                              train_size=0.7, test_size=0.3,

                                                                random_state=123)





# df3

df3_x_treino, df3_x_valid, df3_y_treino, df3_y_valid = train_test_split(df3[feats],

                                                                              df3['BAD'], 

                                                                              train_size=0.7, test_size=0.3,

                                                                random_state=123)





smote = SMOTE()



# resample df2

df2_rx, df2_ry = smote.fit_sample(df2[feats], df2['BAD'])



df2_rx_treino, df2_rx_valid, df2_ry_treino, df2_ry_valid = train_test_split(df2_rx,

                                                                              df2_ry, 

                                                                              train_size=0.7, test_size=0.3,

                                                                random_state=123)





# resample df3

df3_rx, df3_ry = smote.fit_sample(df3[feats], df3['BAD'])



df3_rx_treino, df3_rx_valid, df3_ry_treino, df3_ry_valid = train_test_split(df3_rx,

                                                                              df3_ry, 

                                                                              train_size=0.7, test_size=0.3,

                                                                random_state=123)

df2_rx.shape, df2_ry.shape, df2_rx_treino.shape, df2_rx_valid.shape, df2_ry_treino.shape, df2_ry_valid.shape
df3_rx.shape, df3_ry.shape, df3_rx_treino.shape, df3_rx_valid.shape, df3_ry_treino.shape, df3_ry_valid.shape
df2_x.shape, df2_y.shape, df2_x_treino.shape, df2_x_valid.shape, df2_y_treino.shape, df2_y_valid.shape
df3_x.shape, df3_y.shape, df3_x_treino.shape, df3_x_valid.shape, df3_y_treino.shape, df3_y_valid.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold



from sklearn.metrics import confusion_matrix



from xgboost import XGBClassifier





models = []

models.append(('RandomForest', RandomForestClassifier(random_state=123)))

models.append(('GBM', GradientBoostingClassifier(random_state=123)))

models.append(('XGB', XGBClassifier(random_state=123)))





X_train = df2_x_treino

X_valid =df2_x_valid

y_train = df2_y_treino

y_valid = df2_y_valid



score = []

for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v)

    f_dict = {

        'model': 'base dropada - ' + name ,

        'accuracy': accuracy_valid,

    }

    score.append(f_dict)

    





X_train = df3_x_treino

X_valid =df3_x_valid

y_train = df3_y_treino

y_valid = df3_y_valid



for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v)

    f_dict = {

        'model': 'base fill  - ' + name ,

        'accuracy': accuracy_valid,

    }

    

    score.append(f_dict)

    


X_train = df2_rx_treino

X_valid =df2_rx_valid

y_train = df2_ry_treino

y_valid = df2_ry_valid



for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v)

    f_dict = {

        'model': 'base dropada resample - ' + name,

        'accuracy': accuracy_valid,

    }

    score.append(f_dict)

    





X_train = df3_rx_treino

X_valid =df3_rx_valid

y_train = df3_ry_treino

y_valid = df3_ry_valid



for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v)

    f_dict = {

        'model': 'base fill resample - ' + name ,

        'accuracy': accuracy_valid,

    }

    

    score.append(f_dict)

    

score = pd.DataFrame(score, columns = ['model', 'accuracy'])
print(score)
# From https://www.kaggle.com/ajay1735/my-credit-scoring-model

# função alterada e utilizada de maneira simplificada para exemplo abaixo

import itertools

def plot_confusion_matrix(cm,title='Matrix de confusão',classes=[0,1], cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('Valor real')

    plt.xlabel('Valor predito')
X_train = df2_rx_treino

X_valid =df2_rx_valid

y_train = df2_ry_treino

y_valid = df2_ry_valid



param_grid = {}

my_model = GridSearchCV(RandomForestClassifier(random_state=123),param_grid)

my_model.fit(X_train, y_train)

predictions_t = my_model.predict(X_train) 

predictions_v = my_model.predict(X_valid)

accuracy_train = accuracy_score(y_train, predictions_t) 

accuracy_valid = accuracy_score(y_valid, predictions_v)
# informações do melhor modelo, random forest com resample na base com nulos dropados

print(my_model.best_estimator_)

pd.Series(my_model.best_estimator_.feature_importances_, index=feats).sort_values().plot.barh()
cm = confusion_matrix(y_valid, predictions_v)

np.set_printoptions(precision=2)

print('Matrix de confusão')

print(cm)

plt.figure()

plot_confusion_matrix(cm, 'base dropada resample - RandomForest')
X_train = df2_x_treino

X_valid =df2_x_valid

y_train = df2_y_treino

y_valid = df2_y_valid



param_grid = {}

my_model = GridSearchCV(RandomForestClassifier(random_state=123),param_grid)

my_model.fit(X_train, y_train)

predictions_t = my_model.predict(X_train) 

predictions_v = my_model.predict(X_valid)

accuracy_train = accuracy_score(y_train, predictions_t) 

accuracy_valid = accuracy_score(y_valid, predictions_v)
cm = confusion_matrix(y_valid, predictions_v)

np.set_printoptions(precision=2)

print('Matrix de confusão')

print(cm)

plt.figure()

plot_confusion_matrix(cm, 'base dropada - RandomForest')
# informações do melhor modelo, random forest com resample na base com nulos dropados

print(my_model.best_estimator_)

pd.Series(my_model.best_estimator_.feature_importances_, index=feats).sort_values().plot.barh()