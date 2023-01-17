import pandas as pd

import sklearn 

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from tabulate import tabulate

from matplotlib import pyplot
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = "?")
adult.rename(columns={'fnlwgt':'final_weight',

                      'education.num':'education_num',

                      'marital.status':'marital_status',

                      'capital.gain':'capital_gain',

                      'capital.loss':'capital_loss',

                     'hours.per.week':'hours_per_week',

                     'native.country':'native_country'},

            inplace = True)
adult.shape
adult.head()
sns.pairplot(adult,hue="income",corner=True,diag_kind="hist",palette='winter')
sns.distributions._has_statsmodels = False
fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (25,20),constrained_layout=True)

plt.tight_layout(pad = 1, w_pad = 1, h_pad = 2)

fig.subplots_adjust(wspace = 0.2,hspace = 1)



#fig.delaxes(ax=axes[3,1])



workclass = adult.groupby(['workclass', 'income']).size().unstack()

workclass['sum'] = adult.groupby('workclass').size()

workclass = workclass.sort_values('sum', ascending = False)[['<=50K', '>50K']]

workclass.plot(kind = 'bar', stacked = True, ax = axes[0, 0],color=['grey','gold'])



marital_status = adult.groupby(['marital_status', 'income']).size().unstack()

marital_status['sum'] = adult.groupby('marital_status').size()

marital_status = marital_status.sort_values('sum', ascending = False)[['<=50K', '>50K']]

marital_status.plot(kind = 'bar', stacked = True, ax = axes[0, 1],color=['grey','gold'])



occupation = adult.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = adult.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True, ax = axes[1, 0],color=['grey','gold'])



relationship = adult.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = adult.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True, ax = axes[1, 1],color=['grey','gold'])



race = adult.groupby(['race', 'income']).size().unstack()

race['sum'] = adult.groupby('race').size()

race = race.sort_values('sum', ascending = False)[['<=50K', '>50K']]

race.plot(kind = 'bar', stacked = True, ax = axes[2, 0],color=['grey','gold'])



sex = adult.groupby(['sex', 'income']).size().unstack()

sex['sum'] = adult.groupby('sex').size()

sex = sex.sort_values('sum', ascending = False)[['<=50K', '>50K']]

sex.plot(kind = 'bar', stacked = True, ax = axes[2, 1],color=['grey','gold'])



education = adult.groupby(['education', 'income']).size().unstack()

education['sum'] = adult.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True, ax = axes[3, 0],color=['grey','gold'])



native_country = adult.groupby(['native_country', 'income']).size().unstack()

native_country['sum'] = adult.groupby('native_country').size()

native_country = native_country.sort_values('sum', ascending = False)[['<=50K', '>50K']]

native_country.plot(kind = 'bar', stacked = True, ax = axes[3, 1],color=['grey','gold'])
def missing_data(data,columns):

    aux = data

    columns = columns

    for column in columns:

        most_frequent = data[column].value_counts().index[0]

        aux[column] = aux[column].fillna(most_frequent)

    return aux

    

def marital_status(value):

    if value == "Married-civ-spouse":

        return 1

    else:

        return 0



def race(value):

    if value == "White":

        return 1

    else:

        return 0

    

def native_country(value):

    if value == "United-States":

        return 1

    else:

        return 0

    

def income(value):

    if value == ">50K":

        return 1

    else:

        return 0
def encoder(data,columns,show):



    values = []

    encoder = {}



    for feature in columns:

        encoder[feature] = preprocessing.LabelEncoder()

        data[feature] = encoder[feature].fit_transform(data[feature])

        values.append([feature,encoder[feature].classes_])



    for i in range(len(values)):

        header = [values[i][0],'Code']

        value = []

        for j in range(len(values[i][1])):

            value.append([values[i][1][j],j])

        if show == 0:

            print(tabulate(value,headers=header,tablefmt='psql'))
def prepare_data(data,tipo):

    

    aux = pd.read_csv(data, na_values = "?")

    aux = aux.drop(columns=["Id","education"])

    aux.rename(columns={'fnlwgt':'final_weight','education.num':'education_num','marital.status':'marital_status',

                        'capital.gain':'capital_gain','capital.loss':'capital_loss','hours.per.week':'hours_per_week',

                        'native.country':'native_country'}, inplace = True)

                  

    column_missing = []

    for i in range(aux.shape[1]):

        if aux.isnull().sum()[i] != 0:  

            column_missing.append(aux.isnull().sum().index[i])

            

    missing_data(aux,column_missing)

    

    categoric_columns = ['workclass','occupation','relationship','sex']

    aux['race'] = aux['race'].apply(race)

    aux['marital_status'] = aux['marital_status'].apply(marital_status)    

    aux['native_country'] = aux['native_country'].apply(native_country)

    

    if tipo == "Training":    

        encoder(aux,categoric_columns,0)

        aux['income_num'] = aux['income'].apply(income)

        

    elif tipo == "Testing":

        encoder(aux,categoric_columns,1)

        

    return aux
adult_training = prepare_data("/kaggle/input/adult-pmr3508/train_data.csv","Training")
adult_training.shape
adult_training.isnull().sum()
adult_training.head()
columns = ['age', 'workclass', 'final_weight', 'education_num',

           'marital_status', 'occupation', 'relationship', 'race', 

           'sex','capital_gain', 'capital_loss', 'hours_per_week',

           'native_country']
corr = adult_training.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(10, 8))

ax = sns.heatmap(adult_training.loc[:,[*columns,'income_num']].corr().round(2),mask=mask ,vmin = -.5, vmax =.5, cmap=sns.diverging_palette(20, 220, n=200),linewidths=.5, annot = True)
X_adult_train = adult_training[["age","education_num","capital_gain","capital_loss","hours_per_week"]]

Y_adult_train = adult_training.income
max_score = 0



for i in range(10,41):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn,X_adult_train,Y_adult_train,cv=10)

    

    if scores.mean() > max_score:

        max_score = scores.mean()

        n_neighbors = i

        best_knn = KNeighborsClassifier(n_neighbors = i)

    

print("Melhor KNN = " + str(n_neighbors))

print("Acurácia = " + str(max_score))
X_adult_train = adult_training[["age","education_num","marital_status","sex","capital_gain","capital_loss","hours_per_week"]]

Y_adult_train = adult_training.income
scores = cross_val_score(best_knn,X_adult_train,Y_adult_train,cv=10)

print("Acurária = " + str(scores.mean()))
max_score = 0



for i in range(10,41):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn,X_adult_train,Y_adult_train,cv=10)

    

    if scores.mean() > max_score:

        max_score = scores.mean()

        n_neighbors = i

        best_knn = KNeighborsClassifier(n_neighbors = i)

    

print("Melhor KNN = " + str(n_neighbors))

print("Score = " + str(max_score))
best_knn.fit(X_adult_train,Y_adult_train)
adult_testing = prepare_data("/kaggle/input/adult-pmr3508/test_data.csv","Testing")
adult_testing.head()
X_adult_test = adult_testing[["age","education_num","marital_status","sex","capital_gain","capital_loss","hours_per_week"]]
Y_adult_pred = best_knn.predict(X_adult_test)
output = pd.DataFrame({'income': Y_adult_pred})
output
output.to_csv('submission.csv',index=True,index_label='Id')