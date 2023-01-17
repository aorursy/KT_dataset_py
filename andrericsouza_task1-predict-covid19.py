#import

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#pandas

pd.set_option("display.max_rows",500)

pd.set_option("display.max_columns",500)
# data

data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
# data size

data.shape
# view data

data.head()
# understand data

data.describe(include="all").T
#tamanho da base

print(data.shape)
# check null/nan

data_null=data.isna().sum()

(data_null/data.shape[0]*100).round(2)
#collect columns with good data

good_columns = data_null[data_null==0].reset_index()

good_columns
#create a new dataframe with good columns

data1=data[good_columns['index']]

data1.head()
# check correlation

data1.corr()
# check correlation with graphcs...

ax = sns.heatmap(data1.corr().round(2), annot=True)

plt.show()
# grouping data by result to check balance

data.groupby("SARS-Cov-2 exam result").count()
data[['Patient ID','SARS-Cov-2 exam result']].groupby("SARS-Cov-2 exam result").count()
#list datatype of columns

data.dtypes
#create a list of columns with datatype == 'object'

x1=[]

for c in data.columns:

    x=data[c].dtype

    if x == 'object' and c != 'Patient ID':

        x1.append(c)

print(x1)
#check unique values on list

for a in x1:

    print('analyzing column: ', a)

    print(data[a].unique())

    print()
#replace some datas

data=data.replace(['positive','negative','not_detected','detected','not_done','absent','Não Realizado','present','normal'],

                  [1,0,0,1,np.nan,0,np.nan,1,0])

data['Urine - Leukocytes'].replace('<1000', '999', inplace=True)

data['Urine - pH'] = data['Urine - pH'].astype("float64")

data['Urine - Leukocytes'] = data['Urine - Leukocytes'].astype("float64")

data['Urine - Urobilinogen'] = data['Urine - Urobilinogen'].astype("float64")
#replace nan by 0

data = data.fillna(0)
# Making dummies variable from categorical



#create dataframe with dummies

data_dummies=pd.get_dummies(data[data.dtypes[(data.dtypes == "object")].drop("Patient ID").index])

data_dummies.head()
#create dataframe without dtypes object

data2=pd.concat([data["Patient ID"], data[data.dtypes[(data.dtypes != "object")].index]], axis=1)
#concate dummies with not-dummies and target

data=pd.concat([data_dummies,data2], axis=1)
#check if have columns empty

data.describe(include="all").T.round().sort_values('max', ascending=True)
#remove columns without data



#create a list of columns

list_empyt=data[data.sum()[(data.sum() == 0)].index].columns

print(list_empyt)
#create a dataframe without empty data

data=data.drop(list_empyt,axis=1)

data.shape
#import

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import  classification_report, confusion_matrix
# remove some columns

data1 = data.drop([

    "Patient ID",

    'Patient addmited to regular ward (1=yes, 0=no)',

    'Patient addmited to semi-intensive unit (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)'

], axis=1)
# get target

target = data['SARS-Cov-2 exam result']
#get explanatory variable

expl = data1.drop(columns='SARS-Cov-2 exam result')
#split into training and testing

X_treino, X_teste, Y_treino, Y_teste = train_test_split(expl, target, test_size=0.3, random_state=30)
#fit model

model_lr = LogisticRegression()

model_lr.fit(X_treino, Y_treino)
#check score training data

print(round(model_lr.score(X_treino, Y_treino)*100,2), '%')
#check score test data

print(round(model_lr.score(X_teste, Y_teste)*100,2),'%')
#check confusion matrix



confusion_matrix(Y_teste, model_lr.predict(X_teste))