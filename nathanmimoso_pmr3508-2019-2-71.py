import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)

import sklearn



#Importando as bibliotecas com as métricas

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Importando a base de treino

train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", engine='python', na_values="?")

train.set_index('Id', inplace=True)

train.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education_num", "Marital_status", "Occupation", "Relationship", "Race", "Sex", "Capital_gain", "Capital_loss", "Hours_per_week", "Native_country", "Income"]

train.head()
#Importando a base de testes

test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", engine='python', na_values="?")

test.set_index('Id', inplace=True)

test.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education_num", "Marital_status", "Occupation", "Relationship", "Race", "Sex", "Capital_gain", "Capital_loss", "Hours_per_week", "Native_country"]

test.head()
#Importando algumas bibliotecas para esta etapa

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
train = train.dropna()

Xtrain_0 = train[["Age", "Workclass", "fnlwgt", "Education", "Education_num", "Marital_status", "Occupation", "Relationship", "Race", "Sex", "Capital_gain", "Capital_loss", "Hours_per_week", "Native_country"]]

Xtrain_1 = Xtrain_0.apply(preprocessing.LabelEncoder().fit_transform)

scaler = StandardScaler()

Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain_1), columns=Xtrain_0.columns)

Ytrain = train[['Income']]



test = test.dropna()

Xtest_0 = test.apply(preprocessing.LabelEncoder().fit_transform)

Xtest = pd.DataFrame(scaler.transform(Xtest_0), columns=Xtest_0.columns)
train_num = train.apply(preprocessing.LabelEncoder().fit_transform)

f, ax = plt.subplots(figsize=(10, 6))

corr = train_num.corr()

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',

                 linewidths=.05)

f.subplots_adjust(top=0.93)

t = f.suptitle('Correlação entre Atributos na Adult', fontsize=14)
Xtrain = Xtrain[['Age', 'Education_num', 'Marital_status',

                 'Relationship', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week']]

Xtest = Xtest[['Age', 'Education_num', 'Marital_status',

                 'Relationship', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week']]
#Importando as bibliotecas necessárias

import sklearn.linear_model as linear_model



logistic = linear_model.LogisticRegression()



logistic_scores = cross_validate(logistic, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

logistic_scores
print("Acurácia média: ", logistic_scores['test_accuracy'].mean(), "\nF1-Score médio: ", logistic_scores['test_f1_macro'].mean(), "\nPrecisão média: ", logistic_scores['test_precision_macro'].mean(), "\nRecall médio: ", logistic_scores['test_recall_macro'].mean())
#importando as bibliotecas necessárias

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion="gini")



tree_scores = cross_validate(tree, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

tree_scores
print("Acurácia média: ", tree_scores['test_accuracy'].mean(), "\nF1-Score médio: ", tree_scores['test_f1_macro'].mean(), "\nPrecisão média: ", tree_scores['test_precision_macro'].mean(), "\nRecall médio: ", tree_scores['test_recall_macro'].mean())
results = []

for i in range(1,20):

    tree = DecisionTreeClassifier(criterion="gini",max_depth=i)

    tree_scores = cross_val_score(tree, Xtrain, Ytrain, cv=10)

    results.append(tree_scores.mean())

print(results)
results = []

for i in range(1,20):

    tree = DecisionTreeClassifier(criterion="entropy",max_depth=i)

    tree_scores = cross_val_score(tree, Xtrain, Ytrain, cv=10)

    results.append(tree_scores.mean())

print(results)
tree = DecisionTreeClassifier(criterion="gini", max_depth=9)



tree_scores = cross_validate(tree, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

tree_scores
print("Acurácia média: ", tree_scores['test_accuracy'].mean(), "\nF1-Score médio: ", tree_scores['test_f1_macro'].mean(), "\nPrecisão média: ", tree_scores['test_precision_macro'].mean(), "\nRecall médio: ", tree_scores['test_recall_macro'].mean())
#importando as bibliotecas necessárias

from sklearn.ensemble import AdaBoostClassifier



boosted_tree = AdaBoostClassifier(n_estimators=200)



boosted_scores = cross_validate(boosted_tree, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

boosted_scores
print("Acurácia média: ", boosted_scores['test_accuracy'].mean(), "\nF1-Score médio: ", boosted_scores['test_f1_macro'].mean(), "\nPrecisão média: ", boosted_scores['test_precision_macro'].mean(), "\nRecall médio: ", boosted_scores['test_recall_macro'].mean())
#importando as bibliotecas necessárias

from sklearn.ensemble import RandomForestClassifier



random = RandomForestClassifier(n_estimators=100)



random_scores = cross_validate(random, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

random_scores
print("Acurácia média: ", random_scores['test_accuracy'].mean(), "\nF1-Score médio: ", random_scores['test_f1_macro'].mean(), "\nPrecisão média: ", random_scores['test_precision_macro'].mean(), "\nRecall médio: ", random_scores['test_recall_macro'].mean())
#importando as bibliotecas necessárias

from sklearn.svm import SVC



svm = SVC()



svm_scores = cross_validate(svm, Xtrain, Ytrain, cv=10, scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'))

svm_scores
print("Acurácia média: ", svm_scores['test_accuracy'].mean(), "\nF1-Score médio: ", svm_scores['test_f1_macro'].mean(), "\nPrecisão média: ", svm_scores['test_precision_macro'].mean(), "\nRecall médio: ", svm_scores['test_recall_macro'].mean())
lr = pd.Series({'Classificador': 'Regressão logística', 'Acurácia': logistic_scores['test_accuracy'].mean(), 'F1-Score': logistic_scores['test_f1_macro'].mean(), 'Precisão': logistic_scores['test_precision_macro'].mean(), 'Recall': logistic_scores['test_recall_macro'].mean()})

t = pd.Series({'Classificador': 'Árvore de classificação', 'Acurácia': tree_scores['test_accuracy'].mean(), 'F1-Score': tree_scores['test_f1_macro'].mean(), 'Precisão': tree_scores['test_precision_macro'].mean(), 'Recall': tree_scores['test_recall_macro'].mean()})

bt = pd.Series({'Classificador': 'Boosted tree', 'Acurácia': boosted_scores['test_accuracy'].mean(), 'F1-Score': boosted_scores['test_f1_macro'].mean(), 'Precisão': boosted_scores['test_precision_macro'].mean(), 'Recall': boosted_scores['test_recall_macro'].mean()})

s = pd.Series({'Classificador': 'SVM', 'Acurácia': svm_scores['test_accuracy'].mean(), 'F1-Score': svm_scores['test_f1_macro'].mean(), 'Precisão': svm_scores['test_precision_macro'].mean(), 'Recall': svm_scores['test_recall_macro'].mean()}) 



relat = pd.DataFrame([lr, t, bt, s])

relat
#gerando arquivo para submeter na competição

boosted_tree.fit(Xtrain, Ytrain)



test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", engine='python')

test.set_index('Id', inplace=True)

test.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education_num", "Marital_status", "Occupation", "Relationship", "Race", "Sex", "Capital_gain", "Capital_loss", "Hours_per_week", "Native_country"]

Xtest_0 = test.apply(preprocessing.LabelEncoder().fit_transform)

Xtest = pd.DataFrame(scaler.transform(Xtest_0), columns=Xtest_0.columns)

Xtest = Xtest[['Age', 'Education_num', 'Marital_status',

                 'Relationship', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week']]



Ypred = boosted_tree.predict(Xtest)

Ypred
#gerando arquivo para submeter na competição

id_index = pd.DataFrame({'Id' : list(range(len(Ypred)))})

income = pd.DataFrame({'income' : Ypred})

result = id_index.join(income)

result.head()
#gerando arquivo para submeter na competição

result.to_csv("submission.csv", index = False)
df = pd.read_csv('/kaggle/input/californiahousing/train.csv', engine='python', na_values='?')

df.set_index('Id', inplace=True)

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
f, ax = plt.subplots(figsize=(10, 6))

corr = df.corr()

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',

                 linewidths=.05)

f.subplots_adjust(top=0.93)

t= f.suptitle('Correlação entre Atributos na CaliforniaHousing', fontsize=14)
ax = sns.pairplot(df, x_vars=['latitude','median_age','total_rooms','median_income'], 

                         y_vars='median_house_value', height=7, aspect=0.7, kind='reg')



ax.fig.suptitle('Correlação entre median_house_value e algumas variáveis', y=1.03)
sns.pairplot(df)
#Importando algumas bibliotecas para analisar dados espaciais

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon



#Importando o arquivo shapefile com o mapa da California

calif = gpd.read_file('/kaggle/input/shapefile-california/CA_Counties_TIGER2016.shp')



from pyproj import Proj, transform



#Transformando a latitude e a longitude dos pontos amostrais para o mesmo sistema de referência do mapa da CA

def transforma(x1, y1):

    inProj = Proj(init='epsg:4326')

    outProj = Proj(init='epsg:3857')

    x2,y2 = transform(inProj,outProj,x1,y1)

    return x2, y2



df2 = df.copy()

A = df2.apply(lambda x: transforma(x['longitude'], x['latitude']), axis=1)

#A.to_frame()



#tem jeito menos burro de fazer isso?

long = []

lat = []

for x in A:

    long.append(x[0])

    lat.append(x[1])

long = pd.DataFrame(long)

lat = pd.DataFrame(lat)

df2.longitude = long

df2.latitude = lat



geometry = [Point(xy) for xy in zip (df2['longitude'], df2['latitude'])]

crs = {'init': 'epsg:3857'}



#Transformando o DataFrame em um GeoDataFrame

geo_df = gpd.GeoDataFrame(df, crs = crs, geometry = geometry)



#Visualizando as casas sobre o mapa

fig, ax = plt.subplots(figsize = (12,12))

calif.plot(ax=ax, alpha=0.4, color='grey')

geo_df.plot(ax=ax)

plt.title("Houses in California")
sns.distplot(df.median_house_value)
sns.distplot(df.median_income)
sns.distplot(df.median_age)
sns.boxplot(df['median_house_value'], orient='v')
sns.boxplot(df['median_income'], orient='v')
jointgrid = sns.JointGrid(x='median_house_value', y='median_income', data=df)

jointgrid.plot_joint(sns.scatterplot)

jointgrid.plot_marginals(sns.distplot)
jointgrid = sns.JointGrid(x='median_house_value', y='median_age', data=df)

jointgrid.plot_joint(sns.scatterplot)

jointgrid.plot_marginals(sns.distplot)
import math

from sklearn.metrics import make_scorer

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import linear_model
Xtrain = df.iloc[:,:8]

Ytrain = df.iloc[:,8:9]
def RMSLE(Y, Ypred):

    n = len(Y)

    soma = 0

    Y = np.array(Y)

    for i in range(len(Y)):

        soma += ( math.log( abs(Ypred[i]) + 1 ) - math.log( Y[i] + 1 ) )**2

    return math.sqrt(soma / n)

scorer_rmsle = make_scorer(RMSLE)
linear = linear_model.LinearRegression()

linear.fit (Xtrain, Ytrain)

scores = cross_val_score (linear, Xtrain, Ytrain, cv = 10, scoring = scorer_rmsle)

total = 0

for j in scores:

    total += j

rmsle_linear = total/10



ridge = linear_model.Ridge (alpha=10000)

ridge.fit (Xtrain,Ytrain)

scores = cross_val_score (ridge,Xtrain,Ytrain, cv = 10,scoring = scorer_rmsle)

total = 0

for j in scores:

    total += j

rmsle_ridge = total/10



lasso = linear_model.Lasso(alpha=10000)

lasso.fit (Xtrain, Ytrain)

scores = cross_val_score (lasso,Xtrain,Ytrain, cv = 10,scoring = scorer_rmsle)

total = 0

for j in scores:

    total += j

rmsle_lasso = total/10



print('RMSLE para regressão linear: ', rmsle_linear, '\nRMSLE para regressão ridge: ', rmsle_ridge, '\nRMSLE para regressão lasso: ', rmsle_lasso)
def gerar_grafico(classificadores,X,f_o):

    if X is None:

        acc_total = {}

        for clf in classificadores:

            acc_clf = []

            for n in range(len(f_o)):

                Xn = df[f_o[:n+1]]

                clf.fit(Xn,Ytrain)

                scores = cross_val_score(clf, Xn, Ytrain, cv=10, scoring=scorer_rmsle)

                acc_clf.append(scores.mean())

            acc_total[clf] = acc_clf

    else:

        acc_total = {}

        for clf in  classificadores:

            acc_clf = []

            for n in range(len(f_o)):

                Xn = SelectKBest(chi2, k=n+1).fit_transform(abs(X),Y)

                clf.fit(Xn,Ytrain)

                scores = cross_val_score(clf, Xn, Ytrain, cv=10, scoring=scorer_rmsle)

                acc_clf.append(scores.mean())

            acc_total[clf] = acc_clf

    for clf in acc_total:

        plt.plot(np.arange(1,len(f_o)+1), acc_total[clf], 'o-', label=classificadores[clf])

    plt.ylabel('RMSLE esperada')

    plt.xlabel('Quantidade de features')

    plt.title('Uso das melhores features vs RMSLE esperada')

    plt.legend(loc='upper right')

    plt.grid(True)

    plt.show()

    return acc_total



features_ordenadas = ['median_income','latitude','total_rooms','median_age',

            'households','total_bedrooms','longitude','population']



classificadores_lineares = {lasso:'Lasso',

                            ridge:'Ridge',

                            linear:'Linear simples'}

acc_lineares = gerar_grafico(classificadores_lineares,None,features_ordenadas)
test = pd.read_csv('/kaggle/input/californiahousing/test.csv', engine='python', na_values='?')

test.set_index('Id', inplace=True)



Ypredict = ridge.predict(test)