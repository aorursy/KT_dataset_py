import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
exemplo=pd.read_csv('/kaggle/titanic/gender_submission.csv')
data=pd.read_csv('/home/manu/Programacion/kaggle/titanic/train.csv',index_col='PassengerId')
dataTest=pd.read_csv('/home/manu/Programacion/kaggle/titanic/test.csv',index_col='PassengerId')
data.head(5)
data.describe(), data.info()
data.drop(['Cabin'],axis=1,inplace=True)
dataTest.drop(['Cabin'],axis=1,inplace=True)
data.drop(['Ticket'],axis=1,inplace=True)
dataTest.drop(['Ticket'],axis=1,inplace=True)
data.drop(['Name'],axis=1,inplace=True)
dataTest.drop(['Name'],axis=1,inplace=True)
g = sns.FacetGrid(data, col='Pclass',row='Sex',hue='Survived')
g.map(plt.hist, 'Age', alpha=0.5)
plt.legend()
data.Sex=data.Sex.map(lambda x:1 if x=='female' else 0)
dataTest.Sex=dataTest.Sex.map(lambda x:1 if x=='female' else 0)
#para train_df
for sexo in range(2):
    for clase in range(3):
        idade=data[(data.Sex==sexo) & (data.Pclass==clase+1)].Age.dropna()
        mediana=idade.median()
        print("\n sexo: "+str(sexo)+" clase= "+str(clase+1)+" -> Mediana:"+str(mediana))
        data.loc[(data.Age.isnull()) & (data.Sex == sexo) 
                    & (data.Pclass == clase+1),'Age']=mediana
data.Age=data.Age.astype(int)

print("_________________________")
#para test_df
for sexo in range(2):
    for clase in range(3):
        idade=dataTest[(dataTest.Sex==sexo) & (dataTest.Pclass==clase+1)].Age.dropna()
        mediana=idade.median()
        print("\n sexo: "+str(sexo)+" clase= "+str(clase+1)+" -> Mediana:"+str(mediana))
        dataTest.loc[(dataTest.Age.isnull()) & (dataTest.Sex == sexo) 
                    & (dataTest.Pclass == clase+1),'Age']=mediana                
dataTest.Age=dataTest.Age.astype(int)
data['AgeBand'] = pd.cut(data['Age'], 5)
data[['AgeBand', 'Survived']].groupby(['AgeBand'], 
                              as_index=False).mean().sort_values(by='AgeBand', ascending=True)

data.Age=data.Age.map(lambda x:0 if x<=16 else 1 if x<=32 
                      else 2 if x<=48 else 3 if x <= 64 else 4)
data.drop(['AgeBand'],axis=1,inplace=True)

dataTest['AgeBand'] = pd.cut(dataTest['Age'], 5)
dataTest.Age=dataTest.Age.map(lambda x:0 if x<=16 else 1 if x<=32 
                      else 2 if x<=48 else 3 if x <= 64 else 4)
dataTest.drop(['AgeBand'],axis=1,inplace=True)
data.head()
data['ViaxaSo']=data['SibSp']+data['Parch']
data.ViaxaSo=data.ViaxaSo.map(lambda x:1 if x==0 else 0)
dataTest['ViaxaSo']=dataTest['SibSp']+dataTest['Parch']
dataTest.ViaxaSo=dataTest.ViaxaSo.map(lambda x:1 if x==0 else 0)
data.drop(['SibSp','Parch'],axis=1,inplace=True)
dataTest.drop(['SibSp','Parch'],axis=1,inplace=True)
g = sns.FacetGrid(data, col='ViaxaSo')
g.map(plt.hist, 'Survived', alpha=0.5)

g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Embarked', alpha=0.5)

data.loc[data.Embarked.isnull(),'Embarked']='C'
dataTest.loc[dataTest.Embarked.isnull(),'Embarked']='C'
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
dataTest['Embarked'] = dataTest['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare']=data['Fare'].fillna(data['Fare'].dropna().median())
data['FareBands']=pd.qcut(data['Fare'],4)
dataTest['Fare']=dataTest['Fare'].fillna(dataTest['Fare'].dropna().median())
dataTest['FareBands']=pd.qcut(dataTest['Fare'],4)
data.head(5)
data.FareBands.value_counts()
data.Fare=data.Fare.map(lambda x:0 if x<=7.91 else 1 if x<=14.454 
                      else 2 if x<=31.0 else 3 )
data.drop(['FareBands'],axis=1,inplace=True)

dataTest.Fare=dataTest.Fare.map(lambda x:0 if x<=7.91 else 1 if x<=14.454 
                      else 2 if x<=31.0 else 3 )
dataTest.drop(['FareBands'],axis=1,inplace=True)
data.head(5)
g=sns.FacetGrid(data,col='Survived',row='Pclass')
g.map(plt.hist,'Fare')
X_train = data.drop("Survived", axis=1)
Y_train = data["Survived"]
X_test  = dataTest
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
coeff_df = pd.DataFrame(data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=1000, max_depth=20)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submit=pd.DataFrame({'Survived':Y_pred}, index=dataTest.index)
submit.to_csv('/home/manu/Programacion/kaggle/titanic/TreeRandomForestFinal.csv')