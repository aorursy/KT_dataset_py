import pandas as pd
import sklearn
import os
os.listdir('../input/')
adult = pd.read_csv('../input/adult-dataset/train_data.csv', names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult = pd.read_csv('../input/adult-dataset/test_data.csv', names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
nadult = adult.dropna()
nadult["Target"].value_counts()
X = pd.crosstab(nadult["Target"],nadult["Workclass"],margins=True)
X
def porcentagem(coluna):
    return coluna*100//float(coluna[2])
X = pd.crosstab(nadult["Target"], nadult["Occupation"],margins = True)


X.apply(porcentagem)
def aplicar_porcentagem(label):
    for i in range(len(scoreTipo)):
        if label == Tipo[i]:
            return scoreTipo[i]
    return label
Tipo = []
scoreTipo = []
X = pd.crosstab(nadult["Target"], nadult["Sex"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

X = pd.crosstab(nadult["Target"], nadult["Race"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    print(X.values[1][i])
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

Tipo
X = pd.crosstab(nadult["Target"], nadult["Workclass"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

X = pd.crosstab(nadult["Target"], nadult["Martial Status"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

X = pd.crosstab(nadult["Target"], nadult["Relationship"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

X = pd.crosstab(nadult["Target"], nadult["Occupation"],margins = True)
X = X.apply(porcentagem)
X.columns
for i in range(len(X.columns)-1):
    Tipo.append(X.columns[i])
    scoreTipo.append(X.values[1][i])

Tipo
scoreTipo
Tipo.append("Never-worked")
scoreTipo.append(0)
nadult["Sex"] = nadult["Sex"].apply(aplicar_porcentagem)
nadult["Workclass"] = nadult["Workclass"].apply(aplicar_porcentagem)
nadult["Martial Status"] = nadult["Martial Status"].apply(aplicar_porcentagem)
nadult["Relationship"] = nadult["Relationship"].apply(aplicar_porcentagem)
nadult["Occupation"] = nadult["Occupation"].apply(aplicar_porcentagem)
nadult["Race"] = nadult["Race"].apply(aplicar_porcentagem)
nadult.head
nadult  = pd.get_dummies(nadult, columns = [ "Country"])
Xadult = nadult[[ "Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country_United-States"]]
Yadult = nadult.Target
Xadult
testAdult  = pd.get_dummies(testAdult, columns = ["Country"])
XtestAdult = testAdult[[ "Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country_United-States"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=16, p=1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
XtestAdult["Sex"] = XtestAdult["Sex"].apply(aplicar_porcentagem)
XtestAdult["Workclass"] = XtestAdult["Workclass"].apply(aplicar_porcentagem)
XtestAdult["Martial Status"] = XtestAdult["Martial Status"].apply(aplicar_porcentagem)
XtestAdult["Relationship"] = XtestAdult["Relationship"].apply(aplicar_porcentagem)
XtestAdult["Occupation"] = XtestAdult["Occupation"].apply(aplicar_porcentagem)
XtestAdult["Race"] = XtestAdult["Race"].apply(aplicar_porcentagem)
XtestAdult.fillna(0)
XtestAdult["Workclass"].value_counts()
from sklearn.model_selection import GridSearchCV
#Usando GridSearch para achar os melhores parametros
modelo = KNeighborsClassifier(n_jobs=-1)
parametros = {'n_neighbors':[16, 22, 24, 26, 18],
          'leaf_size':[30],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1],
             'p':[1]}
#modelo1 = GridSearchCV(modelo, param_grid=parametros, n_jobs=1)
#modelo1.fit(Xadult,Yadult)
#modelo1.best_params_
#Teste para achar o melhor K
#for j in range(40):
 #   knn = KNeighborsClassifier(n_neighbors=j+1)
  #  scores = cross_val_score(knn, Xadult, Yadult, cv=10)
   # maxScore = 0
    #for i in range(len(scores)):
     #   if maxScore < scores[i]:
      #      maxScore = scores[i]
    #print(maxScore)
    #print(j+1)
   

testPred = knn.predict(XtestAdult.fillna(0))
w = pd.DataFrame(testPred)
w.to_csv("resposta27.csv")
