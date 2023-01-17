# Bibliotecas
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import category_encoders as ce
# Datasets
df = pd.read_csv("../input/datasetstarefa2/train_data.csv")
dfEvaluation = pd.read_csv("../input/datasetstarefa2/test_features.csv")
# Head
df.head()
#Drop
df = df.drop('Id', axis =1)
df.head()
#Dividir em duas partes
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:,'word_freq_make':'capital_run_length_total'], df.ham, random_state=0, test_size=0.25)
train = pd.concat([X_train, Y_train], axis=1, sort=False)
test = pd.concat([X_test, Y_test], axis=1, sort=False)
graphCapital = train[['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'ham']].groupby("ham").mean()
graphCapital.head()
from matplotlib.pyplot import figure
hamFalse = graphCapital[:1].values.tolist()[0]
hamTrue = graphCapital[1:2].values.tolist()[0]

r1 = np.arange(len(hamFalse))
r2 = [x + 0.1 for x in r1]

fig = plt.gcf()
plt.bar(r1, hamFalse, color = "#ff0000", width=0.1, label= "ham = False")
plt.bar(r2, hamTrue, color = "#00ff00", width=0.1, label= "ham = True")
plt.xticks([r + 0.1 for r in range(len(hamFalse))],
           ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'])
plt.title("Valor médio das features por tipo de label")
plt.legend()
fig.set_size_inches(8,6)
plt.show()
graphWords = train[train.columns.difference(train.columns[train.columns.get_loc('char_freq_;') : train.columns.get_loc('capital_run_length_total')+1])]
graphWords = graphWords.groupby('ham').mean()
graphWords.head()
graphWordsHam = graphWords.iloc[1,:].sort_values(ascending=False)
graphWordsNotHam = graphWords.iloc[0,:].sort_values(ascending=False)
yPosNotHam = np.arange(graphWordsHam.size)
yPosHam = [x + 0.4 for x in yPosNotHam]
plt.barh(yPosNotHam, graphWordsNotHam.values.tolist(), color = "#ff0000", height=0.4, label= "ham = False")
plt.barh(yPosHam, graphWordsHam.values.tolist(), color = "#00ff00", height=0.4, label= "ham = True")
plt.yticks([r + 0.4 for r in range(graphWordsHam.size)],
           graphWordsHam.index.tolist())
plt.legend()
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.show()
graphChars = pd.concat([train.loc[:, 'char_freq_;':'char_freq_#'], train.ham], axis=1, sort=False)
graphChars = graphChars.groupby('ham').mean()
graphChars.head()
graphCharsHam = graphChars.iloc[1,:].sort_values(ascending=False)
graphCharsNotHam = graphChars.iloc[0,:].sort_values(ascending=False)
yPosNotHam = np.arange(graphCharsHam.size)
yPosHam = [x + 0.1 for x in yPosNotHam]
plt.barh(yPosNotHam, graphCharsNotHam.values.tolist(), color = "#ff0000", height=0.1, label= "ham = False")
plt.barh(yPosHam, graphCharsHam.values.tolist(), color = "#00ff00", height=0.1, label= "ham = True")
plt.yticks([r + 0.1 for r in range(graphCharsHam.size)],
           graphCharsHam.index.tolist())
plt.legend()
fig = plt.gcf()
fig.set_size_inches(5,5)
plt.show()
#Tirar colunas de features relacionadas a char
X_train = X_train.drop(X_train.loc[:,'char_freq_;':'char_freq_#'].columns, axis=1)
X_test = X_test.drop(X_test.loc[:,'char_freq_;':'char_freq_#'].columns, axis=1)
#Normalização das útlimas 3 features
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']] = scaler.fit_transform(X_train[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']])
X_test[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']] = scaler.fit_transform(X_test[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']])
Ordem = X_train.mean().sort_values().index
Ordem
X_train = X_train.reindex_axis(Ordem, axis=1)
X_test = X_test.reindex_axis(Ordem, axis=1)
X = X_train
Y = Y_train
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

list = []

i=5
knn = KNeighborsClassifier(n_neighbors=i)

for x in range(2, Ordem.size+1, 1):
    knn.fit(X.iloc[:, 0:x], Y)
    scores = cross_val_score(knn, X.iloc[:, 0:x], Y, cv=4)
    accuracy = sum(scores)/len(scores)
    list.append([x-1,accuracy])
    
resultados = pd.DataFrame(list,columns=["n° de features consideradas","accuracy"])
resultados.plot(x="n° de features consideradas",y="accuracy",style="")
resultados.accuracy.max()
resultados
i=5
knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(X.iloc[:, 0:49], Y)

Y_pred = knn.predict(X_test.iloc[:, 0:49])
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)
from sklearn.naive_bayes import MultinomialNB
list = []

NB = MultinomialNB()

for x in range(2, Ordem.size+1, 1):
    NB.fit(X.iloc[:, 0:x], Y)
    scores = cross_val_score(NB, X.iloc[:, 0:x], Y, cv=4)
    accuracy = sum(scores)/len(scores)
    list.append([x-1,accuracy])
    
resultados = pd.DataFrame(list,columns=["n° de features consideradas","accuracy"])
resultados.plot(x="n° de features consideradas",y="accuracy",style="")
resultados
resultados.accuracy.max()
NB.fit(X.iloc[:, 0:48], Y)
Y_pred_prob = NB.predict_proba(X_test.iloc[:, 0:48])[:, 1]
Y_pred_prob
from sklearn.metrics import roc_curve
fpr, tpr, threshholds = roc_curve(Y_test, Y_pred_prob)
plt.plot(fpr, tpr)
plt.title('Curva ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
