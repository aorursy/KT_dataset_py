import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
adults = pd.read_csv('../input/train_data.csv' ,
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adults = adults.drop(adults.index[0])
nadults = adults.dropna()
adults.shape
adults.head()
nadults.loc[nadults['Country']!='United-States','Country'] = 'non_usa'
import seaborn as sns
fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
plt.xticks(rotation=45)
sns.countplot(nadults['Workclass'],hue=nadults['Target'],ax=f)
sns.countplot(nadults['Relationship'],hue=nadults['Target'],ax=b)
sns.countplot(nadults['Martial Status'],hue=nadults['Target'],ax=c)
sns.countplot(nadults['Race'],hue=nadults['Target'],ax=d)
sns.countplot(nadults['Sex'],hue=nadults['Target'],ax=e)
sns.countplot(nadults['Country'],hue=nadults['Target'],ax=a)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in nadults.columns:
    nadults[i]=le.fit_transform(nadults[i])
TestAdults = pd.read_csv('../input/test_data.csv' ,
        names=[
        "id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

TestAdults = TestAdults.drop(TestAdults.index[0])

nTestAdults = TestAdults.fillna(0)

nTestAdults_backup = nTestAdults

nTestAdults = nTestAdults.drop('id', axis=1)
nTestAdults.loc[nTestAdults['Country']!='United-States','Country'] = 'non_usa'
nTestAdults.head()
for i in nTestAdults.columns:
    nTestAdults[i]=le.fit_transform(nTestAdults[i].astype(str))
nTestAdults.head()
nadults.head()
Xtrain = nadults[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
Ytrain = nadults.Target
XtestAdult = nTestAdults[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
adults.info()
knn = KNeighborsClassifier(n_neighbors=16)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
print('score:', scores.mean())
boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None),
                          n_estimators=100)        
boost.fit(Xtrain, Ytrain)
scores = cross_val_score(boost, Xtrain, Ytrain, cv=10)
boost_score = scores.mean()
print('accuracy =', boost_score)
lgr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(Xtrain, Ytrain)
scores = cross_val_score(lgr, Xtrain, Ytrain, cv=10)
scores.mean()
nnc = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(98,), random_state=1)
scores = cross_val_score(nnc, Xtrain, Ytrain, cv=10)
print('melhor n:', n, 'score:', scores.mean())
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(Xtrain, Ytrain, train_size=0.7)

from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=1000, learning_rate=0.02)
model.fit(X_train, y_train,eval_set=(X_validation, y_validation),plot=True)
YtestPred = model.predict(XtestAdult)
model.score(Xtrain,Ytrain)
YtestPred
Ypred = []
for i in YtestPred:
    if i == 0.:
        Ypred.append('<=50K')
    else:
        Ypred.append('>50K')
Ypred
arq = open ("submission.csv", "w")
arq.write("id,income\n")
for i, j in zip((nTestAdults_backup['id']), Ypred):
    arq.write(str(i)+ "," + str(j)+"\n")
arq.close()