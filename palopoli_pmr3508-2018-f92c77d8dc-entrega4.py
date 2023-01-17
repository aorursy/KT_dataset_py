%matplotlib inline
import sklearn
import numpy as np # linear algebra
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
adult = pd.read_csv('../input/train_data.csv',na_values='?')
adult.dropna(inplace=True)
adult.head()
# Distribuição do salário em relação à idade
adult['age_range'] = pd.cut(adult['age'],bins=[0,20,40,60,80,100])
sns.countplot(y='age_range',hue='income', data = adult)
# Distribuição do salário em relação ao tipo de serviço
sns.countplot(y='workclass',hue='income', data = adult)
# Distribuição do salário em relação à educação
sns.countplot(y='education',hue='income', data = adult)
# Distribuição do salário em relação ao tempo investido nos estudos
sns.countplot(y='education.num',hue='income', data = adult)
# Distribuição do salário em relação ao estado civil
sns.countplot(y='marital.status',hue='income', data = adult)
# Distribuição do salário em relação à ocupação
sns.countplot(y='occupation',hue='income', data = adult)
# Distribuição do salário em relação ao relacionamento
sns.countplot(y='relationship',hue='income', data = adult)
# Distribuição do salário em relação à raça
sns.countplot(y='race',hue='income', data = adult)
# Distribuição do salário em relação ao sexo
sns.countplot(y='sex',hue='income', data = adult)
# Distribuição do salário em relação ao ganho capital
adult['cg'] = 1
adult.loc[adult['capital.gain']==0,'cg']=0
adult.loc[adult['capital.gain']!=0,'cg']=1
sns.countplot(y='cg',hue='income', data = adult)
# Distribuição do salário em relação à perda capital
adult['cl'] = 1
adult.loc[adult['capital.loss']==0,'cl']=0
adult.loc[adult['capital.loss']!=0,'cl']=1
sns.countplot(y='cl',hue='income', data = adult)
# Distribuição dos salários menor que 50k em relação às horas trabalhadas na semana
adult[adult["income"]=="<=50K"]["hours.per.week"].describe()
# Distribuição dos salários maiores que 50k em relação às horas trabalhadas na semana
adult[adult["income"]==">50K"]["hours.per.week"].describe()
# Distribuição do salário em relação ao país de origem
adult["native.country"].value_counts().plot(kind="bar")
plt.ylim(top=750)
plt.xlim(left=0.5)
numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']
cor_mat = sns.heatmap(adult[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
from sklearn import preprocessing
num_adult = adult.apply(preprocessing.LabelEncoder().fit_transform)
num_adult.head()
num_adult.corr(method='pearson').income.sort_values(ascending=True)
label_train = num_adult.income
features_train = num_adult.drop(['Id','age_range','cg','cl','income','fnlwgt','education','occupation','native.country','capital.gain','capital.loss'],axis = 1)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=42, solver='newton-cg',max_iter = 200).fit(features_train, label_train)
cross_val_score(LR,features_train, label_train,cv=10).mean()
from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(solver='lbfgs', random_state=0).fit(features_train, label_train)
cross_val_score(NN,features_train, label_train,cv=10).mean()
from sklearn.ensemble import AdaBoostClassifier
ADA = AdaBoostClassifier(n_estimators=100).fit(features_train, label_train)
cross_val_score(ADA,features_train, label_train,cv=10).mean()
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=25).fit(features_train, label_train)
cross_val_score(KNN,features_train, label_train,cv=10).mean()
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=7)
cross_val_score(clf,features_train, label_train,cv=10).mean()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
LDA=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
cross_val_score(LDA,features_train, label_train,cv=10).mean()
from sklearn.ensemble import BaggingClassifier
Bag= BaggingClassifier(base_estimator=None,n_estimators=150,random_state=42,bootstrap=True, bootstrap_features=True,oob_score=True)
cross_val_score(Bag,features_train, label_train,cv=10).mean()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
cross_val_score(gnb,features_train, label_train,cv=10).mean()
from sklearn.ensemble import GradientBoostingClassifier
gbc= GradientBoostingClassifier(max_depth=2,learning_rate=1)
cross_val_score(gbc,features_train, label_train,cv=10).mean()
testi = pd.read_csv('../input/test_data.csv',na_values='?')
test = testi.dropna()
num_test = test.apply(preprocessing.LabelEncoder().fit_transform)
features_test = num_test.drop(['Id','fnlwgt','education','occupation','native.country','capital.gain','capital.loss'],axis = 1)
features_test = num_test.drop(['Id','fnlwgt','education','occupation','native.country','capital.gain','capital.loss'],axis = 1)
x_val_test = features_test

gbc.fit(features_train, label_train)  
RFpredict = gbc.predict(x_val_test)

y_val_test = gbc.predict(x_val_test)

# criacao do data frame
dfSave = pd.DataFrame(data={"Id" : num_test["Id"], "income" : y_val_test})
dfSave['Id'] = dfSave['Id'].astype(int)
dfSave["income"] = dfSave["income"].map({0:'<=50K', 1:'>50K'})
pd.DataFrame(dfSave[["Id", "income"]], columns = ["Id", "income"])

#extensao do data frame para que tenha a quantidade de linhas sufuciente
YtPred = pd.DataFrame(index=range(0, len(testi)),columns= ["Id", "income"])
YtPred['income'] = dfSave['income'] 

# ajuste da culuna 'income' colocando que a pessoa recebe '>50K' caso esteja zerado
for i in range(0, len(testi)):
    YtPred['Id'][i]=i
    if (YtPred['income'][i] != '<=50K' and YtPred['income'][i] != '>50K'):
        YtPred['income'][i] = '<=50K'
YtPred 
YtPred.to_csv("entrega4.csv", index = False)