import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv")

treino = pd.read_csv("../input/adult-pmr3508/train_data.csv")

nteste = teste.dropna()

ntreino = treino.dropna()
ntreino["income"] = ntreino["income"].map({"<=50K": 0, ">50K":1})

ntreino["sex"] = ntreino["sex"].map({"Male": 0, "Female":1})
ntreino.head()
sns.heatmap(ntreino.corr(), annot=True, vmin=-1, vmax=1)
sns.lineplot('education.num', 'income', data=ntreino)
sns.lineplot('hours.per.week', 'income', data=ntreino)
sns.pairplot(ntreino, hue='income')
names = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

# Get column names first

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

x = ntreino.loc[:, names].values

x = scaler.fit_transform(x)

scaled_df = pd.DataFrame(x, columns=names)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(scaled_df)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

data_treino_principal = pd.concat([principalDf, ntreino[['income']]], axis = 1)

y = ntreino.income
y
sns.pairplot(data_treino_principal, hue='income')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
log_pred = logmodel.predict(X_test)
log_pred
print("Logistic Regression Metrics")

print(confusion_matrix(y_test, log_pred))

print(classification_report(y_test, log_pred))

print('Accuracy: ',accuracy_score(y_test, log_pred))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("Random Forest Metrics")

print(confusion_matrix(y_test, rfc_pred))

print(classification_report(y_test, rfc_pred))

print('Accuracy: ',accuracy_score(y_test, rfc_pred))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
print("Gaussian Naive Bayes Metrics")

print(confusion_matrix(y_test, gnb_pred))

print(classification_report(y_test, gnb_pred))

print('Accuracy: ',accuracy_score(y_test, gnb_pred))
ci = pd.read_csv("../input/dataci/train.csv")
ci
x_ci = ci.drop(columns=["Id","median_house_value"])

y_ci = ci["median_house_value"]
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x_ci,y_ci,test_size=0.25)
treino_si = ci.drop(columns="Id")

tab = treino_si.corr(method="spearman")

df = pd.DataFrame(tab)



def pinta(x):

    if abs(x) == 1:

        color = 'black'

    elif abs(x) >= abs(df.quantile(q=0.75).mean()):

        color = 'indianred' 

    elif abs(x) >= abs(df.mean().mean()):

        color = 'aqua'

    elif abs(x) <= abs(df.mean().mean())/6:

        color = 'white'

    elif abs(x) <= abs(df.mean().mean())/3:

        color = 'paleturquoise' 

    elif abs(x) <= abs(df.mean().mean()):

        color = 'dodgerblue'

    else:

        color = 'white'

            

    return 'background-color: %s' % color

    

print("Salmon : correlação maior que a média do 3º quartil das correlações ")

print("Azul mais vivo, aqua : correlação maior que a média das correlações ")

print("Azul mais escuro, dodgerblue : correlação pequena ")

print("Azul mais clara, paleturquoise : correlação menor que um terço da média")

print("Branco : correlação menor que um sexto da média ")

df.style.applymap(pinta)
fig, ax = plt.subplots()

ax.scatter(x = ci['median_income'], y = ci['median_house_value'] , s = 1)

plt.ylabel('median_house_value', fontsize=13)

plt.xlabel('median_income', fontsize=13)

plt.title("Relacionamento entre renda e valor do imóvel")

plt.show()



x_treino["median_age"].hist()

x_teste["median_age"].hist()

plt.ylabel('frequência', fontsize=13)

plt.xlabel('median_age', fontsize=13)

plt.title("Histograma de idades médias das regiões; laranja teste e azul treino")





plt.subplots()

plt.scatter(x = ci['median_age'], y = ci['median_house_value'], s=0.5)

plt.ylabel('median_house_value', fontsize=13)

plt.xlabel('median_age', fontsize=13)

plt.title("Relacionamento entre idade média e valor do imóvel")

plt.show()



plt.subplots()

plt.scatter(ci["longitude"],ci["latitude"], c= ci["median_age"] , cmap = "jet" , s = 5)

plt.title("Idade média por posição geográfica; quanto mais quente a cor, maior a idade")

plt.ylabel('latitude', fontsize=13)

plt.xlabel('longitude', fontsize=13)

plt.xlim(-130,-110)

plt.show()
from sklearn import linear_model

from sklearn.metrics import mean_squared_error





model = linear_model.Ridge(alpha = 300)

model.fit(x_treino, y_treino)
predictionTestSet = model.predict(x_teste)
from sklearn.metrics import mean_squared_error



errorTestSet = mean_squared_error(y_teste, predictionTestSet)
erro_normalizado = errorTestSet/10000000000
erro_normalizado
las = linear_model.Lasso(alpha=0.6)

las = las.fit(x_treino, y_treino)

pred = las.predict(x_treino)

erro_normalizado = mean_squared_error(y_treino, pred)/10000000000



print("Cross-value-score:  ",las.score(x_treino,y_treino, sample_weight=None))



print("Erro quadrático médio:  " ,erro_normalizado)
from catboost import CatBoostRegressor



tom = CatBoostRegressor(learning_rate=1, depth=4,num_trees= 20, loss_function='RMSE')

frajola = tom.fit(x_treino, y_treino)

garfield = tom.predict(x_treino)



gato_de_botas = mean_squared_error(y_treino, garfield)/10000000000





print("Erro quadrático médio:  " ,gato_de_botas)