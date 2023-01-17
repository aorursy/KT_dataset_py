import pandas as pd

import numpy as np

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv"

                   , index_col = ['Id']

                   , sep=r'\s*,\s*'

                   , engine='python'

                   , na_values="?")



testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv"

                   , index_col = ['Id']

                   , sep=r'\s*,\s*'

                   , engine='python'

                   , na_values="?")
adult = adult.rename(columns={'age':'Age','workclass':'Workclass','education':'Education','education.num':'Education-Num'

                            , 'marital.status':'Marital Status', 'occupation':'Occupation', 'relationship':'Relationship'

                            , 'race':'Race', 'sex': 'Sex', 'capital.gain': 'Capital Gain', 'capital.loss': 'Capital Loss'

                            , 'hours.per.week': 'Hours per Week', 'native.country':'Native Country', 'income': 'Income'})



testAdult = testAdult.rename(columns={'age':'Age','workclass':'Workclass','education':'Education','education.num':'Education-Num'

                            , 'marital.status':'Marital Status', 'occupation':'Occupation', 'relationship':'Relationship'

                            , 'race':'Race', 'sex': 'Sex', 'capital.gain': 'Capital Gain', 'capital.loss': 'Capital Loss'

                            , 'hours.per.week': 'Hours per Week', 'native.country':'Native Country', 'income': 'Income'})
adult.head()
adult.describe()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, dpi=110, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Age']

    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Idade por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=110, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'fnlwgt']

    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de fnlwgt por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Education-Num']

    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Education-Num por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Capital Gain']

    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Capital Gain por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Capital Gain']

    ax.hist(x, alpha=0.5, bins=100, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Capital Gain por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Capital Loss']

    ax.hist(x, alpha=0.5, bins=100, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Capital Loss por Income', y=1.05, size=16)
fig, axes = plt.subplots(1, 2, dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue']

for i, (ax, Income) in enumerate(zip(axes.flatten(), adult.Income.unique())):

    x = adult.loc[adult.Income==Income, 'Hours per Week']

    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(Income), color=colors[i])

    ax.set_title(Income)

plt.suptitle('Distribuição de Hours per Week por Income', y=1.05, size=16)
import seaborn as sns

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

adult.loc[:,'Income'] = le.fit_transform(adult['Income'])



plt.figure(figsize=(10,10))

sns.heatmap(adult.corr(method="pearson"), square = True, annot = True, vmin=-1, vmax=1, cmap ='Blues')

plt.show()
adult.loc[adult.Income==0,"Workclass"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Workclass por Income')

adult.loc[adult.Income==1,"Workclass"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Marital Status"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Marital Status por Income')

adult.loc[adult.Income==1,"Marital Status"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Occupation"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Occupation por Income')

adult.loc[adult.Income==1,"Occupation"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Relationship"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Relationship por Income')

adult.loc[adult.Income==1,"Relationship"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Race"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Race por Income')

adult.loc[adult.Income==1,"Race"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Sex"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Sex por Income')

adult.loc[adult.Income==1,"Sex"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.loc[adult.Income==0,"Native Country"].value_counts().plot(kind="bar", color= 'Red', label= "Income <=50K")

plt.title('Distribuição de Native Country por Income')

adult.loc[adult.Income==1,"Native Country"].value_counts().plot(kind="bar", color= 'Blue', label= "Income >50K")

plt.legend(loc='upper right')
adult.drop_duplicates(keep='first', inplace=True)

adult = adult.drop(["fnlwgt", "Native Country", "Education"], axis=1)
yTrain = adult.pop("Income")

xTrain = adult
xTrain
from sklearn.pipeline import  Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
catPipeline = Pipeline(steps = [('imputador', SimpleImputer(strategy="most_frequent"))

                               ,('onehot', OneHotEncoder(drop="if_binary"))])



numPipeline = Pipeline(steps = [('scaler', StandardScaler())])



numColumns = list(xTrain.select_dtypes(include = [np.number]).columns.values)

catColumns = list(xTrain.select_dtypes(exclude = [np.number]).columns.values)



preprocessador = ColumnTransformer(transformers = [('numerico', numPipeline, numColumns)

                                                  ,('categorico', catPipeline, catColumns)])
xTrain = preprocessador.fit_transform(xTrain)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
acuracia_result={}

for k in range(15,30):

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k), xTrain, yTrain, cv = 5, scoring="accuracy").mean()

    acuracia_result[k-14] = acuracia
acuracia_result
KNN = KNeighborsClassifier(n_neighbors=26)

KNN.fit(xTrain, yTrain)



xTest = testAdult.drop(['fnlwgt', 'Native Country', 'Education'], axis=1)

xTest = preprocessador.transform(xTest)

predict = KNN.predict(xTest)



depara = {0: "<=50K", 1: ">50K"}

predictFinal = np.array([depara[i] for i in predict], dtype=object)
predictFinal
submissao = pd.DataFrame()

submissao[0] = testAdult.index

submissao[1] = predictFinal

submissao.columns = ['Id', 'income']



submissao.to_csv('submission.csv', index = False)