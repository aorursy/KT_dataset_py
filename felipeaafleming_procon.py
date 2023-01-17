import pandas as pd
import numpy as np
import io
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import os
from sklearn import preprocessing

# define example
# reading data
df12 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2012.csv', header = 0)
df13 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2013.csv', header = 0)
df14 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2014.csv', header = 0)
df15 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2015.csv', header = 0)
df16 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2016.csv', header = 0)

print("Read Data")

df = pd.concat([df12, df13, df14, df15, df16])

df['Interval'] = (pd.to_datetime(df['DataArquivamento']) - pd.to_datetime(df['DataAbertura']))
df.Interval = df.Interval.dt.days

# filtering columns
#df = df[['Interval','CodigoRegiao','UF','Tipo','Atendida','CodigoAssunto','CodigoProblema','SexoConsumidor','FaixaEtariaConsumidor']]
#RadicalCNPJ
df = df[['AnoCalendario','Interval','CodigoRegiao','UF','Tipo','Atendida','CodigoAssunto','CodigoProblema','SexoConsumidor','FaixaEtariaConsumidor']]
#RadicalCNPJ

# replacing S per 1 and N per 0
df['Atendida'] = pd.Series(np.where(df['Atendida'].values == 'S', 1, 0), df['Atendida'].index)
df['SexoConsumidor'] = pd.Series(np.where(df['SexoConsumidor'].values == 'M', 1, 0), df['SexoConsumidor'].index)

# replacing each state by a number
df['UF'] = df['UF'].apply(lambda x: cidades[x])

# cleaning up na and duplicates
df = df.dropna()

print("Get Dummies")

#df = pd.get_dummies(df, columns =['UF','FaixaEtariaConsumidor', 'CodigoProblema', 'Tipo','CodigoRegiao','RadicalCNPJ'])
df = pd.get_dummies(df, columns =['UF','FaixaEtariaConsumidor','AnoCalendario', 'CodigoRegiao','Tipo'])

#df = df.drop(columns=['UF','FaixaEtariaConsumidor','AnoCalendario', 'CodigoRegiao','Tipo'])

# Separate majority and minority classes
df_majority = df[df.Atendida==1]
df_minority = df[df.Atendida==0]

df = None

print(df_minority.shape)
print(df_majority.shape)

print("Downsample")

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=df_minority.shape[0],     # to match minority class
                                 random_state=123) # reproducible results
 
print(df_majority_downsampled.shape)

df = df_minority
df_minority = None

# Combine minority class with downsampled majority class
df = pd.concat([df, df_majority_downsampled])

df_majority_downsampled = None

X = df.loc[:, df.columns != 'Atendida']
Y = df.Atendida

df = None 
print(X.columns)
scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=8,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

#RandomForestClassifier()
clf.fit(X_train, y_train)

#print(clf.feature_importances_)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy of random forest classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20,
                         learning_rate=0.002)

bdt.fit(X_train, y_train)
y_pred = bdt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy of AdaBoost classifier on test set: {:.2f}'.format(bdt.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression
lrt = LogisticRegression()

lrt.fit(X_train, y_train)
y_pred = lrt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lrt.score(X_test, y_test)))
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras.layers import Flatten

# creating model
model = Sequential()
model.add(Dense(64, input_dim=51, activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25, noise_shape=None, seed=None))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.005), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1, batch_size=12)

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = model.evaluate(X_test, y_test, batch_size=10)
print(score)
import pandas as pd
import numpy as np
import io
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import random
from scipy.stats import mode
from sklearn.metrics import mean_absolute_error
import datetime
import matplotlib.pyplot as plt


# carrega os dados do dataset procon e manipula pra deixar nos conformes (categoricos -> numericos)
def loadData():
    # reading the data
    df12 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2012.csv', header = 0)
    df13 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2013.csv', header = 0)
    df14 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2014.csv', header = 0)
    df15 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2015.csv', header = 0)
    df16 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2016.csv', header = 0)

    df = pd.concat([df12, df13, df14, df15, df16])
    df = df.dropna()
    df['Atendida'] = pd.Series(np.where(df['Atendida'].values == 'S', 1, 0), df['Atendida'].index)
    df['SexoConsumidor'] = pd.Series(np.where(df['SexoConsumidor'].values == 'M', 1, 0), df['SexoConsumidor'].index)
    return df

# dado um dataset (procon) e um cnpj (radicalCNPJ) retornamos um novo dataset com as seguintes colunas:
# 'Year','Month','ReclamacoesEmAberto','Atendida', 'AbertasNoMes'
def reclamacoesAbertasNoPeriodo(df, cnpj): 
    columns = ['Year','Month','ReclamacoesEmAberto','Atendida', 'AbertasNoMes']
    ndf = pd.DataFrame(columns=columns)
    
    bycnpj = df.loc[(df['RadicalCNPJ'] == cnpj)]
    
    for year in range(2012, 2017):
        for month in range(1, 13):
            currents =  bycnpj[(pd.to_datetime(bycnpj['DataArquivamento']) >= pd.to_datetime(str(year)+'-'+str(month))) &
                                  (pd.to_datetime(bycnpj['DataAbertura']) <= pd.to_datetime(str(year)+'-'+str(month)))
                                 ]
            abertasNoMes = bycnpj[
                            (pd.to_datetime(bycnpj.DataAbertura).dt.year == year) &
                            (pd.to_datetime(bycnpj.DataAbertura).dt.month == month)
            ].shape[0]

            data = pd.to_datetime(str(year)+'-'+str(month))
            if currents.shape[0] > 0:
                emAberto = currents.shape[0]
                atendida = currents['Atendida'].sum()
                row = pd.DataFrame([[data.year, data.month, emAberto, atendida, abertasNoMes]], columns=columns)
                ndf = ndf.append(row)
            else:
                row = pd.DataFrame([[data.year, data.month, 0, 0, abertasNoMes]], columns=columns)
                ndf = ndf.append(row)
    return ndf

# retorna os n elementos com maior frequencia de uma lista
def moda(lista, n=3):
    l = lista
    t = ()
    for i in range(0,n):
        moda = mode(l)[0][0]
        t = t + (moda,)
        l = l[l != moda]
        if l.shape[0] == 0:
            return t
    return t

# cria os elementos de X pra um determinado ano segundo padrao adotado
def createXFor(year=2017):
    columns = ['Year','Month']
    ndf = pd.DataFrame(columns=columns)
    
    for month in range(1,13):
        row = pd.DataFrame([[year,month]], columns=columns)
        ndf = ndf.append(row)
    return ndf


# faz a previsao do numero de reclamacoes abertas mes a mes para um determinado cnpj e ano. se graph = true entao plotamos o grafico
def forecast(cnpj, year, title="", graph=True):
    ndf = reclamacoesAbertasNoPeriodo(df, cnpj)    
    X = ndf[['Year','Month']]
    y = ndf.AbertasNoMes
    
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)

    x_n = createXFor(year=year)
    y_n = regr.predict(x_n)
    if graph == True:
        plt.figure(figsize=(12,8))
        dates = []
        for index, row in X.iterrows():
            month = row.Month
            year = row.Year
            dates.append(datetime.datetime(year, month, 1))

        plt.plot(dates, y, label = '2012-2016')

        dates_n = []
        for index, row in x_n.iterrows():
            month = row.Month
            year = row.Year
            dates_n.append(datetime.datetime(year, month, 1))
        
        plt.plot(dates_n, y_n, label = str(year))
        plt.legend(loc='best')
        
        plt.show()
# leitura dos dados 
df = loadData()
cnpjs = moda(df.RadicalCNPJ) # vai ler 3 cnpjs que correspondem aos com maior ocorrencia no dataset

maes = []
# validating model
for cnpj in cnpjs:
    ndf = reclamacoesAbertasNoPeriodo(df, cnpj)
    
    X = ndf[['Year','Month']]
    Y = ndf.AbertasNoMes
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    
    #calcula o mae
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
print("MAES:")
print(maes)

# previsao para 2017 3 empresas
forecast(97422620.0, 2017, title="SAMSUNG ELETRONICA DA AMAZONIA LTDA")
forecast(61695227.0, 2017, title="ELETROPAULO METROPOLITANA ELETRICIDADE DE S PAULO")
forecast(40432544.0, 2017, title="CLARO S/A")