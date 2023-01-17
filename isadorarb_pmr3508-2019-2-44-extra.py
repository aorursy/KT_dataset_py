import pandas as pd

import numpy as np

import seaborn as sns
#aqui lemos os dados e retiramos todas as features que não vão ser usadas na previsão por razões variadas e retiramos as linhas em que há dados faltantes

teste = pd.read_csv("../input/adult-pmr3508/test_data.csv")

treino = pd.read_csv("../input/adult-pmr3508/train_data.csv")

nteste = teste.dropna()

ntreino = treino.dropna()

nteste = nteste.drop(["workclass", "fnlwgt", "education", "relationship", "marital.status", "occupation", "native.country"], axis=1)

ntreino = ntreino.drop(["workclass", "fnlwgt", "education", "relationship", "marital.status", "occupation", "native.country"], axis=1)
#aqui convertemos os valores de income, sex e race de strings para inteiros ou booleanos, para que possam

#ser analisadas numericamente

ntreino["income"] = ntreino["income"].map({"<=50K":0, ">50K":1})

ntreino["sex"] = ntreino["sex"].map({"Male":0, "Female":1})

ntreino["race"] = ntreino["race"].map({"White":0, "Asian-Pac-Islander":1, "Other":1, "Amer-Indian-Eskimo":2, "Black":3})

nteste["sex"] = nteste["sex"].map({"Male": 0, "Female":1})

nteste["race"] = nteste["race"].map({"White":0, "Asian-Pac-Islander":1, "Other":1, "Amer-Indian-Eskimo":2, "Black":3})
ntreino.head()
sns.heatmap(ntreino.corr(), annot=True, vmin=-1, vmax=1)
sns.pairplot(ntreino, hue='income')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



treino_features = ntreino[['age', 'education.num', 'race', 'sex', 'capital.gain', 'capital.loss','hours.per.week']]

treino_features = treino_features.values

treino_resultado = ntreino[['income']]

treino_resultado = treino_resultado.values



model = AdaBoostClassifier(RandomForestClassifier(n_estimators = 100),

                         algorithm="SAMME",

                         n_estimators=500)
model = model.fit(treino_features, treino_resultado)
print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(treino_resultado, model.predict(treino_features))))
nteste1 = nteste[['age', 'education.num', 'race', 'sex', 'capital.gain', 'capital.loss','hours.per.week']]

predicao = model.predict(nteste1.values)
df = pd.DataFrame(predicao)

df = df.replace(0,"<=50K")

df = df.replace(1,">50K")

df
arquivo = "predicao.csv"

predicao = pd.DataFrame(nteste, columns = ["income"])

predicao["income"] = df

predicao.to_csv(arquivo, index_label="Id")

predicao
from sklearn import svm



clf_svm = svm.SVC()

clf_svm.fit(treino_features, treino_resultado)
print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(treino_resultado, clf_svm.predict(treino_features))))
predicao_svm = clf_svm.predict(nteste1.values)
df_svm = pd.DataFrame(predicao_svm)

df_svm = df_svm.replace(0,"<=50K")

df_svm = df_svm.replace(1,">50K")

df_svm
arquivo = "predicao_svm.csv"

predicao_svm = pd.DataFrame(nteste, columns = ["income"])

predicao_svm["income"] = df_svm

predicao_svm.to_csv(arquivo, index_label="Id")

predicao_svm
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

naivebayes = gnb.fit(treino_features, treino_resultado)
print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(treino_resultado, naivebayes.predict(treino_features))))
predicao_gnb = naivebayes.predict(nteste1.values)
df_gnb = pd.DataFrame(predicao_gnb)

df_gnb = df_gnb.replace(0,"<=50K")

df_gnb = df_gnb.replace(1,">50K")

df_gnb
test = pd.read_csv("../input/atividade-3-pmr3508/test.csv")

train = pd.read_csv("../input/atividade-3-pmr3508/train.csv")

ntest = test.dropna()

ntrain = train.dropna()
train.head()
sns.heatmap(train.corr(), annot=True, vmin=-1, vmax=1)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split 



train_features = ntrain[['longitude', 'latitude', 'median_age', 'total_rooms', 'total_bedrooms', 'population','households','median_income']]

train_features = train_features.values

train_resultado = ntrain[['median_house_value']]

train_resultado = train_resultado.values



lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(train_features, train_resultado)
from sklearn.metrics import r2_score



pred_lasso = lasso_regressor.predict(train_features)

score = r2_score(train_resultado,pred_lasso)

print("Coeficiente de determinação:",score)