import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train= pd.read_csv('../input/train_data.csv',na_values='?')
test=pd.read_csv('../input/test_data.csv',na_values='?')
train.head()
print(train.shape)
train.dropna(inplace=True)
print(train.shape)
train.drop(['Id','education','fnlwgt'],axis=1, inplace=True)
sns.pairplot(train,size=2.5)
plt.show()
List = ["workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
from sklearn import preprocessing
trainpreprocessing=train.apply(preprocessing.LabelEncoder().fit_transform)
train[List]=trainpreprocessing[List]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

ytrain=train.income
xtrain=train.drop(['income'],axis=1)
scoreant=0
indice=0
for i in range(1,40,3):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,xtrain,ytrain,cv=10)
    print(score.mean())
    if(score.mean()>scoreant):
        indice = i
        scoreant=score.mean()
print('Score KNN ',scoreant)
print('K = ',indice)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
LDA=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
LDA.fit(xtrain,ytrain)
LDA.predict(xtrain)
scoreLDA=cross_val_score(LDA,xtrain,ytrain,cv=10)
score2=scoreLDA.mean()
print('Score LDA = ', score2)
QDA=QuadraticDiscriminantAnalysis()
scoreQDA=cross_val_score(QDA,xtrain,ytrain,cv=10)
print('Score QDA ',scoreQDA.mean())
from sklearn.linear_model import LogisticRegression
Logistic=LogisticRegression(random_state=42,solver='newton-cg')
scoreLogistic=cross_val_score(Logistic,xtrain,ytrain,cv=10)
print('Score Logistic =',scoreLogistic.mean())
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
scoreBNB = cross_val_score(bnb, xtrain, ytrain, cv=10)
print('Score BernoulliNB',scoreBNB.mean())
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
scoreGNB = cross_val_score(gnb, xtrain, ytrain, cv=10)
print('Score GaussianNB',scoreGNB.mean())

from sklearn.svm import SVC
svm=SVC(gamma='auto')
scoreSVM=cross_val_score(svm,xtrain,ytrain,cv=4)
scoreSVM.mean()
from sklearn.svm import LinearSVC
linearSVM = LinearSVC(random_state=0, tol=1e-5)
scoreLinearSvm=cross_val_score(linearSVM,xtrain,ytrain,cv=5)
scoreLinearSvm.mean()
print('O melhor resultado foi do Support Vector Classification ',scoreSVM.mean())

from sklearn.ensemble import GradientBoostingClassifier
gbc= GradientBoostingClassifier(max_depth=2,learning_rate=1)
scoregbc=cross_val_score(gbc,xtrain,ytrain,cv=10)
print('Score Gradient Boosting ',scoregbc.mean())

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42)
scoreRF=cross_val_score(RF,xtrain,ytrain,cv=10)
print('Score Random Forest ',scoreRF.mean())
from sklearn.ensemble import ExtraTreesClassifier
ETC=ExtraTreesClassifier(n_estimators=150, max_depth=10,min_samples_split=5, random_state=42)
scoresETC = cross_val_score(ETC, xtrain, ytrain, cv=10)
print('Score Extra Trees ', scoresETC.mean())
from sklearn.ensemble import BaggingClassifier
Bag= BaggingClassifier(base_estimator=None,n_estimators=150,random_state=42,bootstrap=True, bootstrap_features=True,oob_score=True)
scoreBag=cross_val_score(Bag,xtrain,ytrain,cv=10)
print('Score Bagging ',scoreBag.mean())
from sklearn.ensemble import AdaBoostClassifier
Ada= AdaBoostClassifier(n_estimators=100,learning_rate=0.9)
scoreAda=cross_val_score(Ada,xtrain,ytrain,cv=10)
print('Score AdaBoost ',scoreAda.mean())
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=10,max_features='auto')
scoretree=cross_val_score(tree,xtrain,ytrain,cv=10)
print('Score Decision Tree ',scoretree.mean())

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
mlp=MLPClassifier(activation='logistic')
mlp.fit(xtrain,ytrain)
s=cross_val_score(mlp,xtrain,ytrain,cv=10)
print('Score MLP ',s.mean())
mlp=MLPClassifier(activation='relu')
mlp.fit(xtrain,ytrain)
s=cross_val_score(mlp,xtrain,ytrain,cv=10)
print('Score MLP ',s.mean())

from sklearn.ensemble import VotingClassifier
voting=VotingClassifier(estimators=[('tree',tree),('gbc',gbc),('bag',Bag)],voting='soft')
scorevoting=cross_val_score(voting,xtrain,ytrain,cv=10)
scorevoting.mean()
print(' O melhor classificador foi o Gradient Boosting com ',scoregbc.mean())
