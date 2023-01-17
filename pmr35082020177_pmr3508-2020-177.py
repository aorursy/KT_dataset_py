import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('seaborn')
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
print('Formato do Dataset:', adult.shape)
adult.head()
adult.describe()
print("Workclass:")

print(adult.Workclass.describe())



print("\nOccupation:")

print(adult.Occupation.describe())



print("\nCountry:")

print(adult.Country.describe())
x = adult.Workclass.describe().top

adult.Workclass =  adult.Workclass.fillna(x)



x = adult.Country.describe().top

adult.Country =  adult.Country.fillna(x)



x = adult.Occupation.describe().top

adult.Occupation =  adult.Occupation.fillna(x)
adult.describe()
x = adult_test.Workclass.describe().top

adult_test.Workclass =  adult_test.Workclass.fillna(x)



x = adult_test.Country.describe().top

adult_test.Country =  adult_test.Country.fillna(x)



x = adult_test.Occupation.describe().top

adult_test.Occupation =  adult_test.Occupation.fillna(x)
adult.head()
adult.describe()
from sklearn import preprocessing

LabelEncoder = preprocessing.LabelEncoder()



correlationMap = adult.apply(preprocessing.LabelEncoder().fit_transform).corr()

correlationMap.head()
plt.figure(figsize=(10,7))

sns.heatmap(correlationMap, vmin=-0.3, vmax=0.3, cmap = 'mako')
def LabeledBarPlot(df, x, label):

# plota um gráfico de barra da coluna "column" enfatizando o rótulo "label"



    print('Graph of %s and %s' %(x,label))

    

    index = df[x].unique()

    columns = df[label].sort_values().unique()

    data_to_plot = pd.DataFrame({'index': index})

    

    for column in columns:

        temp = []

        for unique in index:

            filtered_data = df[df[x] == unique]

            filtered_data = filtered_data[filtered_data[label] == column]

            

            temp.append(filtered_data.shape[0])

        data_to_plot = pd.concat([data_to_plot, pd.DataFrame({column: temp})], axis = 1)

        

    data_to_plot = data_to_plot.set_index('index', drop = True)

    

    ax = data_to_plot.plot.bar(rot=0, figsize = (14,7), alpha = 0.9, cmap = 'viridis')
LabeledBarPlot(adult.drop('Id', axis=0), 'Sex', 'Target')
LabeledBarPlot(adult.drop('Id', axis=0), 'Sex', 'Education-Num')
adult['Race'].describe()
LabeledBarPlot(adult.drop('Id', axis=0), 'Race', 'Target')
adult['Country'].describe()
numericalColumns = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']



sns.set()

sns.pairplot(adult.drop('Id'), vars = numericalColumns, hue = 'Target', palette = 'mako', height = 2.5, diag_kws={'bw':'1.0'})
numericalFeatures = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']

adult = adult.drop(columns = ['fnlwgt'])

adult_test = adult_test.drop(columns = ['fnlwgt'])



categoricalFeatures = ['Workclass', 'Martial Status', 'Occupation', 'Relationship', 'Race', 'Sex']

adult = adult.drop(columns = ['Education', 'Country'])

adult_test = adult_test.drop(columns = ['Education', 'Country'])



print("Atributos numéricos: ",numericalFeatures)

print("Atributos categóricos: ", categoricalFeatures)
from sklearn import preprocessing



scaler = preprocessing.StandardScaler()
# Vamos criar uma cópia para conseguir reverter essa exclusão posteriormente



adultCopy = adult.drop('Id', axis=0)



adult_testCopy = adult_test.drop('Id', axis=0)
adultCopy[numericalFeatures] = scaler.fit_transform(adultCopy[numericalFeatures])



adult_testCopy[numericalFeatures] = scaler.fit_transform(adult_testCopy[numericalFeatures])
LabelEncoder = preprocessing.LabelEncoder()



# training set

adultCopy[categoricalFeatures] = adult[categoricalFeatures].apply(LabelEncoder.fit_transform)



# testing set

adult_testCopy[categoricalFeatures] = adult_test[categoricalFeatures].apply(LabelEncoder.fit_transform)
adultCopy.head()
adult_testCopy.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



# Training set

Xadult = adultCopy[['Age','Workclass','Education-Num','Martial Status','Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss','Hours per week']]

Yadult = adultCopy.Target



# Testing set

XtestAdult = adult_testCopy[['Age','Workclass','Education-Num','Martial Status','Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss','Hours per week']]



accuracy = 0

bestK = 10

print('\nBuscando o melhor K...\n')



for k in range(10,30):

    

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, Xadult, Yadult, cv=5)

    accuracy_ = scores.mean()

    

    if accuracy_ > accuracy:

        bestK = k

        accuracy = accuracy_

        

print('O hiperparâmetro K que minimiza a taxa de erro é: %d, com acurácia de %0.2f %%' %(bestK, accuracy*100))
knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(Xadult,Yadult)
prediction = knn.predict(XtestAdult)



print('O nosso K-NN preveu os seguintes rótulos: ',prediction)
submission = pd.DataFrame(prediction)
submission[1] = prediction

submission[0] = adult_testCopy.index

submission.columns = ['Id','Income']
submission.head()
submission.to_csv('submission.csv',index = False)