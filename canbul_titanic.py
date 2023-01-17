#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import numpy as np
import pandas as pd

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv('C:\\Users\\John\\Desktop\\Titanic\\train2.csv')
test_data = pd.read_csv('C:\\Users\\John\\Desktop\\Titanic\\test.csv')
all_data = [train_data, test_data]
count, bin_edges = np.histogram(train_data['Age'])
train_data['Age'].plot(kind='hist', xticks = bin_edges)


plt.title('Histogram')
plt.ylabel('No of Countries')
plt.xlabel('No of Immigrants')

plt.show()
from sklearn.preprocessing import LabelEncoder #datayı dönüştürmek için kullanabiliriz.
from sklearn.preprocessing import StandardScaler
#train_data['Sex'] = LabelEncoder().fit_transform(train_data['Sex'])
#train_data['Sex'].head()
#data['Name'] = data['Name'].apply(lambda x: replacement.get(x)) replacementı dictionary olarak verebiliriz.
#data['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))
train_data['Cabin'].fillna('U', inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])

test_data['Cabin'].fillna('U', inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

train_data['Cabin'] = train_data['Cabin'].apply(lambda x: replacement.get(x))
train_data['Cabin'] = StandardScaler().fit_transform(train_data['Cabin'].values.reshape(-1, 1))

test_data['Cabin'] = test_data['Cabin'].apply(lambda x: replacement.get(x))
test_data['Cabin'] = StandardScaler().fit_transform(test_data['Cabin'].values.reshape(-1, 1))

Cabin = train_data['Cabin']
train_data['Title']= train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title']= test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data=test_data.drop(['Name','Ticket', 'PassengerId'], axis=1)
train_data=train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1)

train_data['Family']=train_data['SibSp']+ train_data['Parch']+1
test_data['Family'] = test_data['SibSp']+test_data['Parch']+1
train_data['Isalone'] = 'Yes'
test_data['Isalone'] = 'Yes'
train_data['Isalone'].loc[train_data['Family']>1] = 'No'
test_data['Isalone'].loc[test_data['Family']>1] = 'No'
train_data

train_data['Age'] = train_data.apply(
    lambda row: 4.57 if np.isnan(row['Age']) and row["Title"]=="Master" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 21.84 if np.isnan(row['Age']) and row["Title"]=="Miss" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 32.36 if np.isnan(row['Age']) and row["Title"]=="Mr" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 35.78 if np.isnan(row['Age']) and row["Title"]=="Mrs" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 45.54 if np.isnan(row['Age']) and row["Title"]=="Other" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 4.57 if np.isnan(row['Age']) and row["Title"]=="Master" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 21.84 if np.isnan(row['Age']) and row["Title"]=="Miss" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 32.36 if np.isnan(row['Age']) and row["Title"]=="Mr" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 35.78 if np.isnan(row['Age']) and row["Title"]=="Mrs" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 45.54 if np.isnan(row['Age']) and row["Title"]=="Other" else row['Age'],   axis=1 )
a=int(10)
test_data['Fare'] = test_data['Fare'].fillna(a)
train_data['Fare'] = train_data['Fare'].fillna(a)
train_data.head()
Can_deneme = train_data[['Sex', 'Parch']]

norm_train = train_data[['Age', 'Fare']]
norm_test = test_data[['Age', 'Fare']]

norm_test.head()
train_data_mean=list(norm_train.mean())
train_data_div=list(norm_train.max()-norm_train.min())
train_data_mean
norm_train.head()
norm_train= (norm_train-train_data_mean) / train_data_div
norm_test=(norm_test-train_data_mean) / train_data_div


train_data['Newage']= norm_train['Age']
train_data['Newfare']= norm_train['Fare']
test_data['Newage']= norm_test['Age']
test_data['Newfare']= norm_test['Fare']

train_data = train_data.drop(['Age', 'Fare'], axis=1)
test_data = test_data.drop(['Age', 'Fare'], axis=1)

data1_x = ['Sex','Pclass', 'Embarked', 'SibSp', 'Parch', 'Title', 'Family', 'Isalone','Newage', 'Newfare', 'Cabin' ]
data1_dummy = pd.get_dummies(train_data[data1_x])
data2_dummy = pd.get_dummies(test_data[data1_x])
data1_dummy.head(10)
X_train = data1_dummy
Y_train = train_data['Survived']
X_test  = data2_dummy
X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_nb = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_lsvm = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dtree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rfor = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
acc_mlp
prediction = pd.read_csv('C:\\Users\\John\\Desktop\\Titanic\\gender_submission.csv')

i=5

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_log
prediction.to_csv('submission_'+str(i)+'log.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_svm
prediction.to_csv('submission_'+str(i)+'sVm.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_lsvm
prediction.to_csv('submission_'+str(i)+'lsVm.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_knn
prediction.to_csv('submission_'+str(i)+'knn.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_dtree
prediction.to_csv('submission_'+str(i)+'dtree.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_rfor
prediction.to_csv('submission_'+str(i)+'rfor.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_nb
prediction.to_csv('submission_'+str(i)+'nb.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_sgd
prediction.to_csv('submission_'+str(i)+'sgd.csv', encoding='utf-8', index=False)

prediction = prediction.drop(["Survived"], axis=1)
prediction['Survived']= Y_pred_mlp
prediction.to_csv('submission_'+str(i)+'mlp.csv', encoding='utf-8', index=False)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Stochastic Gradient Descent',  
              'Linear SVC', 
              'Decision Tree', 'Neural Net'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,  
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_mlp]})
models.sort_values(by='Score', ascending=False)

