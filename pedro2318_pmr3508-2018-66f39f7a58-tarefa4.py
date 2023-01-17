import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

treino = pd.read_csv("../input/train_data.csv",
        names= None,
        engine='python',
        na_values = '?')

teste = pd.read_csv("../input/test_data.csv",
        names= None,
        engine='python')

treinoN = treino.dropna()
treinoN.info()
treino1 = treino[['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income']]
treino1.nunique()
treino2 = treinoN
treino2 = treino2.apply(preprocessing.LabelEncoder().fit_transform)
plt.matshow(treino2.corr())
treino3 = treino2.corr().income.sort_values(ascending=True)
treino3
treino4 = pd.get_dummies(treinoN[['relationship','marital.status','capital.loss', 'sex', 'hours.per.week', 'age', 'education.num', 'capital.gain', 'income']])
treino4 = treino4.corr().loc[:,'income_>50K'].sort_values(ascending=True)
treino4
treino5 = pd.get_dummies(treinoN)
treino5 = treino5.corr().loc[:,'income_>50K'].sort_values(ascending=True).where(lambda x : abs(x) > 0.15).dropna()
treino5
treino6 = treinoN[['occupation','income','race']]
treino6 = pd.get_dummies(treino6).drop(columns = 'income_<=50K')
treino6 = treino6.corr().loc[:,'income_>50K'].sort_values(ascending=True).where(lambda x : abs(x) > 0.088).dropna()
treino6
treino7 = pd.get_dummies(treinoN)
index = treino2.where(lambda x : abs(x) > 0.07).dropna().index[1:-1].append(treino6.index[:-1])
X_train = treino7[index]
Y_train = treino7.loc[:,'income_>50K']
teste1 = pd.get_dummies(teste)
teste1 = teste1.dropna()
X_test = teste1[index]
X_test.info()
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train,Y_train)
model.score(X_train,Y_train)
Y_test_1 = model.predict(X_test)
Y_test_1_copy = Y_test_1
Y_test_1_copy = Y_test_1_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_1_copy)):
    if Y_test_1_copy[output] == 0:
        Y_test_1_copy[output] = '<=50K'
    else:
        Y_test_1_copy[output] = '>50K'
    answer.append([output,Y_test_1_copy[output]])
result1 = np.vstack((teste["Id"],Y_test_1_copy)).T
result1
linear = linear_model.LinearRegression()
linear.fit(X_train,Y_train)
linear.score(X_train,Y_train)
Y_test_2 = linear.predict(X_test)
Y_test_2_copy = Y_test_2
Y_test_2_copy = Y_test_2_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_2_copy)):
    if Y_test_2_copy[output] == 0:
        Y_test_2_copy[output] = '<=50K'
    else:
        Y_test_2_copy[output] = '>50K'
    answer.append([output,Y_test_2_copy[output]])
result2 = np.vstack((teste["Id"],Y_test_2_copy)).T
result2
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
logmodel.score(X_train,Y_train)
Y_test_3= logmodel.predict(X_test)
Y_test_3_copy = Y_test_3
Y_test_3_copy = Y_test_3_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_3_copy)):
    if Y_test_3_copy[output] == 0:
        Y_test_3_copy[output] = '<=50K'
    else:
        Y_test_3_copy[output] = '>50K'
    answer.append([output,Y_test_3_copy[output]])
result3 = np.vstack((teste["Id"],Y_test_3_copy)).T
result3
treemodel = tree.DecisionTreeClassifier(criterion='gini') 
treemodel.fit(X_train,Y_train)
treemodel.score(X_train,Y_train)

Y_test_4= model.predict(X_test)
Y_test_4_copy = Y_test_4
Y_test_4_copy = Y_test_4_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_4_copy)):
    if Y_test_4_copy[output] == 0:
        Y_test_4_copy[output] = '<=50K'
    else:
        Y_test_4_copy[output] = '>50K'
    answer.append([output,Y_test_4_copy[output]])
result4 = np.vstack((teste["Id"],Y_test_4_copy)).T
result4
svmmodel = SVC(gamma='auto') 
svmmodel.fit(X_train,Y_train)
svmmodel.score(X_train,Y_train)
Y_test_5= svmmodel.predict(X_test)
Y_test_5_copy = Y_test_5
Y_test_5_copy = Y_test_5_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_5_copy)):
    if Y_test_5_copy[output] == 0:
        Y_test_5_copy[output] = '<=50K'
    else:
        Y_test_5_copy[output] = '>50K'
    answer.append([output,Y_test_5_copy[output]])
result5 = np.vstack((teste["Id"],Y_test_5_copy)).T
result5
NBmodel = GaussianNB() 
NBmodel.fit(X_train,Y_train)
NBmodel.score(X_train,Y_train)
Y_test_6= NBmodel.predict(X_test)
Y_test_6_copy = Y_test_6
Y_test_6_copy = Y_test_6_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_6_copy)):
    if Y_test_6_copy[output] == 0:
        Y_test_6_copy[output] = '<=50K'
    else:
        Y_test_6_copy[output] = '>50K'
    answer.append([output,Y_test_6_copy[output]])
result6 = np.vstack((teste["Id"],Y_test_6_copy)).T
result6
KMmodel = KMeans(n_clusters=2, random_state=0)
KMmodel.fit(X_train,Y_train)
KMmodel.score(X_train,Y_train)
Y_test_7= KMmodel.predict(X_test)
Y_test_7_copy = Y_test_7
Y_test_7_copy = Y_test_7_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_7_copy)):
    if Y_test_7_copy[output] == 0:
        Y_test_7_copy[output] = '<=50K'
    else:
        Y_test_7_copy[output] = '>50K'
    answer.append([output,Y_test_7_copy[output]])
result7 = np.vstack((teste["Id"],Y_test_7_copy)).T
result7
RFmodel= RandomForestClassifier()
RFmodel.fit(X_train,Y_train)
RFmodel.score(X_train,Y_train)
Y_test_8= RFmodel.predict(X_test)
Y_test_8_copy = Y_test_8
Y_test_8_copy = Y_test_8_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_8_copy)):
    if Y_test_8_copy[output] == 0:
        Y_test_8_copy[output] = '<=50K'
    else:
        Y_test_8_copy[output] = '>50K'
    answer.append([output,Y_test_8_copy[output]])
result8 = np.vstack((teste["Id"],Y_test_8_copy)).T
result8
GBMmodel= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
GBMmodel.fit(X_train,Y_train)
GBMmodel.score(X_train,Y_train)
Y_test_91= GBMmodel.predict(X_test)
Y_test_91_copy = Y_test_91
Y_test_91_copy = Y_test_91_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_91_copy)):
    if Y_test_91_copy[output] == 0:
        Y_test_91_copy[output] = '<=50K'
    else:
        Y_test_91_copy[output] = '>50K'
    answer.append([output,Y_test_91_copy[output]])
result91 = np.vstack((teste["Id"],Y_test_91_copy)).T
result91
ADAmodel= AdaBoostClassifier(n_estimators=100)
ADAmodel.fit(X_train,Y_train)
ADAmodel.score(X_train,Y_train)
Y_test_92= GBMmodel.predict(X_test)
Y_test_92_copy = Y_test_92
Y_test_92_copy = Y_test_92_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_92_copy)):
    if Y_test_92_copy[output] == 0:
        Y_test_92_copy[output] = '<=50K'
    else:
        Y_test_92_copy[output] = '>50K'
    answer.append([output,Y_test_92_copy[output]])
result92 = np.vstack((teste["Id"],Y_test_92_copy)).T
result92
NNmodel = MLPClassifier(solver='lbfgs', random_state=0)
NNmodel.fit(X_train,Y_train)
NNmodel.score(X_train,Y_train)
Y_test_10= NNmodel.predict(X_test)
Y_test_10_copy = Y_test_10
Y_test_10_copy = Y_test_10_copy.tolist()
answer = [["Id","income"]]
for output in range(len(Y_test_10_copy)):
    if Y_test_10_copy[output] == 0:
        Y_test_10_copy[output] = '<=50K'
    else:
        Y_test_10_copy[output] = '>50K'
    answer.append([output,Y_test_10_copy[output]])
result10 = np.vstack((teste["Id"],Y_test_10_copy)).T
result10
df1 = pd.DataFrame(result4)
df2 = pd.DataFrame(result7)
df3 = pd.DataFrame(result8)
df1.to_csv('result49887402.csv')
df2.to_csv('result79887402.csv')
df3.to_csv('result89887402.csv')