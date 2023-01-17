#import pydotplus

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn

import numpy as np



from scipy import stats

from IPython.display import Image  

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz
df = pd.read_csv('../input/weatherAUS.csv')

df.info()
df['Location'].value_counts()
porcentagem_dados_nulo = pd.Series(index=df.columns)



for coluna in df:

    porcentagem_dados_nulo[coluna] = df[coluna].count()/df.shape[0]

    

porcentagem_dados_nulo_ordenado = porcentagem_dados_nulo.sort_values()



print(porcentagem_dados_nulo_ordenado)
porcentagem_dados_nulo_ordenado.plot.bar()
plot_sb = sbn.countplot(df.RainTomorrow, label='Total')

NotRain, Rain = df.RainTomorrow.value_counts()

print('Não Choveu: ',NotRain)

print('Choveu : ',Rain)
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
df.shape
df = df.dropna(how='any')
df.shape
z = np.abs(stats.zscore(df._get_numeric_data()))

print(z)

df = df[(z < 3).all(axis=1)]
df.shape
df['RainToday'] = df['RainToday'].replace({'No': 0,'Yes': 1})

df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0,'Yes': 1})
colunas_categoricas = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

for col in colunas_categoricas:

    print(np.unique(df[col]))



# transformar as colunas de categoria

df = pd.get_dummies(df, columns=colunas_categoricas)
df.head()
df.shape
scaler = preprocessing.MinMaxScaler()

scaler.fit(df)

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df.head()
y = df.RainTomorrow

x = df.drop(['RainTomorrow'], axis=1)
selector = SelectKBest(chi2, k=3)

selector.fit(x, y)

X_new = selector.transform(x)

print(x.columns[selector.get_support(indices=True)])
x = df[['Humidity3pm','Rainfall','RainToday']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=23)

clf_rf.fit(x_train,y_train)
y_pred = clf_rf.predict(x_test)

score = accuracy_score(y_test,y_pred)

print('Precisão: {:.3f}'.format(score))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots(figsize=(12,12))

sbn.heatmap(cm,xticklabels=['No Rain','Rain'],yticklabels=['No Rain','Rain'],annot=True,fmt="d")
score_cv = cross_val_score(clf_rf, x, y, cv=20, scoring="f1")

print(score_cv)

print(score_cv.mean())

print(score_cv.std())
treesDot = StringIO()

export_graphviz(clf_rf.estimators_[5], out_file=treesDot, filled=True, rounded=True, special_characters=True, class_names=["No Rain","Rain"], feature_names=x.columns.values)

#graph = pydotplus.graph_from_dot_data(treesDot.getvalue())  

#Image(graph.create_png())