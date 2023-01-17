import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import time

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics
import os

os.listdir('../input')
train = pd.read_csv("../input/adult-pmr3508/train_data.csv")

train = train.dropna()

test = pd.read_csv("../input/adulttest-only/adult.test.csv",

        names=["age", "workclass", "fnlwgt", "education", "education.num", "marital.status",

        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country", "income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

test = test.dropna()
train.head()
train.dtypes
train.describe()
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):

    if x == None:

        column_interested = y

    else:

        column_interested = x

    series = dataframe[column_interested]

    print(series.describe())

    print('mode: ', series.mode())

    if verbose:

        print('='*80)

        print(series.value_counts())



    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)

    plt.show()
categorical_summarized(train, x = "sex", hue = "income", verbose = True)
categorical_summarized(train, x = "education.num", hue = "income", verbose = True)
categorical_summarized(train, x = "race", hue = "income", verbose = True)
le = preprocessing.LabelEncoder()

train["workclass"] = le.fit_transform(train["workclass"])

train["education"] = le.fit_transform(train["education"])

train["marital.status"] = le.fit_transform(train["marital.status"])

train["occupation"] = le.fit_transform(train["occupation"])

train["relationship"] = le.fit_transform(train["relationship"])

train["race"] = le.fit_transform(train["race"])

train["sex"] = le.fit_transform(train["sex"])

train["native.country"] = le.fit_transform(train["native.country"])

train["income"] = le.fit_transform(train["income"])
y_train = train["income"]

cols = [col for col in train.columns if col not in ["Id", "fnlwgt", "income"]]

x_train = train[cols]
test["workclass"] = le.fit_transform(test["workclass"])

test["education"] = le.fit_transform(test["education"])

test["marital.status"] = le.fit_transform(test["marital.status"])

test["occupation"] = le.fit_transform(test["occupation"])

test["relationship"] = le.fit_transform(test["relationship"])

test["race"] = le.fit_transform(test["race"])

test["sex"] = le.fit_transform(test["sex"])

test["native.country"] = le.fit_transform(test["native.country"])

test["income"] = le.fit_transform(test["income"])
y_test = test["income"]

cols2 = [col for col in test.columns if col not in ["fnlwgt", "income"]]

x_test = test[cols2]
x_train
sc = StandardScaler()

x_train = np.array(sc.fit_transform(x_train))

x_test = np.array(sc.transform(x_test))

y_train = np.array(y_train)

y_test = np.array(y_test)
pca = PCA(0.95)

x_train_pca = pca.fit_transform(x_train)

x_test_pca = pca.transform(x_test)

pca.n_components_
exp_var = pca.explained_variance_ratio_

exp_var
logReg = LogisticRegression(solver = 'lbfgs')

logReg.fit(x_train_pca, y_train)

logReg.score(x_test_pca, y_test)
logReg.fit(x_train, y_train)

logReg.score(x_test, y_test)
def PCA_RegLog():

    var = [0.85, 0.90, 0.95, 0.99]

    n, t, s = [], [], []

    for x in var:

        tempo_0 = time.time()

        pca = PCA(x)

        x_train_pca = pca.fit_transform(x_train)

        x_test_pca = pca.transform(x_test)

        logReg.fit(x_train_pca, y_train)

        n_comp = pca.n_components_

        score = logReg.score(x_test_pca, y_test)

        n.append(n_comp)

        t.append(round(time.time() - tempo_0, 2))

        s.append(round(score, 4))

    #var, n, s = pd.DataFrame(data=var, columns=["Variância"]), pd.DataFrame(data=n, columns=["Número de componentes"]), pd.DataFrame(data=s, columns=["Acurácia"])

    table = list(zip(var, n, t, s))

    tabela = pd.DataFrame(data=table, columns = ["Variância", "Número de componentes", "Tempo", "Acurácia"])

    return tabela



PCA_RegLog()
import time

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x_train, y_train)
# Get numerical feature importances

importances = list(rf.feature_importances_)# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(cols, importances)]# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
t_i = time.time()

rf.fit(x_train, y_train)

pred_rf = rf.predict(x_test)

erro_rf = abs(pred_rf - y_test)

mape = 100 * (erro_rf / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

print("Erro médio: ", round(np.mean(erro_rf), 2))

print("Tempo de Execução: ", round(time.time() - t_i, 2), "s")
x_train2, x_test2 = x_train[:,:7], x_test[:,:7]

t_i2 = time.time()

rf.fit(x_train2, y_train)

pred_rf2 = rf.predict(x_test2)

erro_rf2 = abs(pred_rf2 - y_test)

print("Erro médio: ", round(np.mean(erro_rf2), 2))

print("Tempo de Execução: ", round(time.time() - t_i2, 2), "s")
lm = LinearRegression()



X = x_train

y = y_train

t0 = time.time()

model = lm.fit(X, y)

    

    

scores = cross_val_score(model, X, y, cv=6)

print("Acurácias de validação cruzada:", scores)

    

predictions = cross_val_predict(model, X, y, cv=6)

accuracy = metrics.r2_score(y, predictions)

print("Acurácia de predição cruzada:", accuracy)



print("Tempo de execução: ", round(time.time() - t0, 2), "s")