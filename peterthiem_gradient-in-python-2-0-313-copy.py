#Python 2 !!!

#nie wszystko można odpalić w domyślnym 3.6.0 na kaggle

#najlepiej ściągnąć notebooka i zmienić kernel w jupyterze



#importuje biblioteki



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import  mean_squared_error, roc_auc_score,accuracy_score

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier



from pylab import rcParams



rcParams['figure.figsize'] = 10, 10

color = sns.color_palette()
#EKSPLORACJA DANYCH



#wgrywam dane



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



id_test = test_df.id



#sprawdzam ile rekordow i atrybutow

print('train_df shape:',train_df.shape)

print('test_df shape:',test_df.shape)
#jakie mam typy danych



dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

print('Variables data type:')

dtype_df.groupby("Column Type").aggregate('count').reset_index()
#POCZĄTKOWE DANE



#statystyka



train_df.describe().round(1)
#uzupelnianie



print(train_df.loc[train_df['build_year'] == 20052009].id)

print(train_df.loc[train_df['state'] == 33].id)

print('build_year:',train_df.ix[10090].build_year)

print('state:',train_df.ix[10090].state)



train_df.loc[train_df['id'] == 10092, 'build_year'] = 2007

train_df.loc[train_df['id'] == 10092, 'state'] = 3

train_df.loc[train_df['id'] == 10093, 'build_year'] = 2009
#BRAKUJĄCE DANE



#describe jeszcze raz po uzupelnieniu



train_df.describe().round(1)
#BRAKUJĄCE DANE



#sprawdzam gdzie są braki



train_na = (train_df.isnull().sum() / len(train_df)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

sns.barplot(y=train_na.index, x=train_na,color=color[0])

plt.xlabel('% missing')
#Transformacja zmiennych kategorycznych na ilościowe



for f in train_df.columns:

    if train_df[f].dtype=='object':

        lbl = LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        

for c in test_df.columns:

    if test_df[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(test_df[c].values)) 

        test_df[c] = lbl.transform(list(test_df[c].values))
#life_sq i kitch_sq sa powiazane z full_sq

#powierzchnia kuchnii i powierzchnia mieszkalna sa powiazane z powierzchnia calkowita

#dlatego uzupelniam z full_sq zamiast średniej



#pozostale uzupelniam ze sredniej



kitch_ratio = train_df['full_sq']/train_df['kitch_sq']

train_df['kitch_sq']=train_df['kitch_sq'].fillna(train_df['full_sq'] /kitch_ratio.median())

test_df['kitch_sq']=test_df['kitch_sq'].fillna(test_df['full_sq'] /kitch_ratio.median())



lifesq_ratio = train_df['full_sq']/train_df['life_sq']

train_df['life_sq']=train_df['life_sq'].fillna(train_df['full_sq'] /lifesq_ratio.median())

test_df['life_sq']=test_df['life_sq'].fillna(test_df['full_sq'] /lifesq_ratio.median())



train_df=train_df.fillna(train_df.median(),inplace=True)

test_df=test_df.fillna(test_df.median(),inplace=True)
#ZMIENNA DECYZYJNA



sns.distplot(train_df.price_doc.values, kde=None)

plt.xlabel('price')
#duzo lepiej sprawdza sie logarytm z price_doc aby uniknac dlugiego "ogona" z prawej



ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df.loc[train_df['price_doc'] >ulimit, 'price_doc'] = ulimit

train_df.loc[train_df['price_doc'] <llimit, 'price_doc'] = llimit



sns.distplot(np.log(train_df.price_doc.values),  bins=50,kde=None)

plt.xlabel('price')



train_df['price_doc_log'] = np.log1p(train_df['price_doc'])
#mamy 2 nienaturalne gorki na lewo



print(train_df['price_doc'].value_counts().head(10))



train_df['label_value'] = 0

train_df.loc[train_df['price_doc'] == 1000000, 'label_value'] = 1

train_df.loc[train_df['price_doc'] == 2000000, 'label_value'] = 2
# MODEL



# usuwanie kolumn z X i ustawienie y



data_X = train_df.drop(["id","timestamp","price_doc","price_doc_log",'label_value'],axis=1)

data_y = train_df['price_doc_log']
# cross walidacja RandomizedSearchCV

# zakomentowane poniewaz bardzo dlugie obliczenia - wyniki ponizej



#GBmodel = GradientBoostingRegressor()

#aram_dist = {"learning_rate": np.linspace(0.05, 0.15,5),

#              "max_depth": range(3, 5),

#              "min_samples_leaf": range(3, 5)}



#rand = RandomizedSearchCV(GBmodel, param_dist, cv=7,n_iter=10, random_state=5)

#rand.fit(data_X,data_y)

#rand.grid_scores_



#print(rand.best_score_)

#print(rand.best_params_)
#model gradient boost



GBmodel = GradientBoostingRegressor(min_samples_leaf= 4, learning_rate= 0.1, max_depth= 4)

GBmodel.fit(data_X,data_y)



sns.distplot(GBmodel.predict(data_X))
# DODATKOWA KLASYFIKACJA



#dodatkowa klasyfikacja na wykrycie wartosci szczytow na końcu sekcji "Eksploracja danych"



clfdata_X = train_df.drop(['id','timestamp','label_value','price_doc_log','price_doc'],axis=1)

clfdata_y = train_df['label_value']



clfX_train, clfX_test, clfY_train, clfY_test = train_test_split(clfdata_X, clfdata_y, test_size=0.30,random_state=21)



GBclf= GradientBoostingClassifier(max_depth=4,min_samples_leaf=2)
#model klasyfikacyjny



GBclf.fit(clfX_train,clfY_train)

GBclf.score(clfX_test,clfY_test)
# atrybuty dla regresji



importances = GBmodel.feature_importances_

importances_by_trees=[tree[0].feature_importances_ for tree in GBmodel.estimators_]

std = np.std(importances_by_trees,axis=0)

indices = np.argsort(importances)[::-1]





sns.barplot(importances[indices][:20],data_X.columns[indices[:20]].values)

plt.title("Waznosc atrybutow dla regresji")
#atrybuty dla klasyfikacji



clf_importances = GBclf.feature_importances_

clf_importances_by_trees=[tree[0].feature_importances_ for tree in GBclf.estimators_]

clf_std = np.std(clf_importances_by_trees,axis=0)

clf_indices = np.argsort(clf_importances)[::-1]





sns.barplot(clf_importances[clf_indices][:20],clfdata_X.columns[clf_indices[:20]].values)

plt.title("waznosc atrybutow dla klasyfikacji")
#predykcja



predict = GBmodel.predict(test_df.drop(["id", "timestamp"],axis=1))

lab = GBclf.predict(test_df.drop(['id','timestamp'],axis=1))

output = pd.DataFrame({'id': id_test, 'price_doc': np.expm1(predict)})

output['label'] = lab



output.loc[output['label'] == 1, 'price_doc'] = 1000000

output.loc[output['label'] == 2, 'price_doc'] = 2000000

output = output.drop(['label'],axis=1)
#do csv



output.to_csv('sub.csv', index=False)