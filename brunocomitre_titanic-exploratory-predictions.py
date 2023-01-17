import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format

%matplotlib inline
trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
dataset = trainset.append(testset , ignore_index = True )
titanic = dataset[ :891 ]
trainset.head(2)
testset.head(2)
dataset.head(2)
titanic.head(2)
print('This DataSet has rows:', titanic.shape[0])
print('This DataSet has columns:', titanic.shape[1])
titanic.describe()
titanic.info()
titanic.dtypes
titanic.count()
titanic.mean()
titanic.std()
titanic.sem()
titanic.isnull()
titanic.isnull().sum()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic.head(5)
titanic.fillna(0, inplace = True)
titanic.head(5)
titanic.isnull().sum()
titanic.groupby(['Sex']).size().reset_index(name='Quantity')
titanic.groupby(['Survived']).size().reset_index(name='Quantity')
titanic['Survived'].value_counts().plot.pie(colors=('tab:red', 'tab:green'), 
                                       title='Percentage of Surviving and Non-surviving Persons', 
                                       fontsize=12, shadow=True, startangle=90, autopct='%1.1f%%', 
                                       labels=('Not Survived','Survived')).set_ylabel('')
titanic['Not Survived'] = titanic['Survived'].map({0:1,1:0})
titanic.head(2)
titanic[titanic['Sex'] == 'female'].groupby('Sex')['Not Survived'].apply(lambda x: np.sum(x == 1))
df_survive = titanic[titanic['Survived'] == 1].groupby('Sex')[['Survived']].count()
df_survive
plot = df_survive.apply(lambda x: (x / x.sum(axis=0))*100)['Survived']
plot
plot.plot.pie(colors=('tab:red', 'tab:green'), 
                                       title='Percentage of passengers by Sex', 
                                       fontsize=12, shadow=True, startangle=90, autopct='%1.1f%%', 
                                       labels=('Woman','Men')).set_ylabel('')
print('Passengers without age filled:',titanic['Age'].isnull().sum())
print('Passengers with full age:',(~titanic['Age'].isnull()).sum())
titanic['Age'].value_counts().sort_values(ascending = [False]).nlargest(1)
plt.figure();
titanic.hist(column='Age', color=('green'), alpha=0.5, bins=10)
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')
df_hist = pd.DataFrame({'Total': titanic['Age'],
                           'Not Survived': titanic[titanic['Not Survived'] == 1]['Age'], 
                           'Survived':titanic[titanic['Survived'] == 1]['Age']},                       
                    
                          columns=['Total','Not Survived', 'Survived'])

plt.figure();

df_hist.plot.hist(bins=10, alpha=0.5, figsize=(10,5), color=('red','tab:blue','green'), 
                     title='Histograms (Total, Survived and Not Survived) by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
ax = sns.kdeplot(df_hist['Not Survived'], shade=True, color="r")
ax = sns.kdeplot(df_hist['Survived'], shade=True, color="g")
plt.title('Density of Survived and Not Survived by Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.xticks((0, 10, 20, 30, 40, 50, 60, 70, 80))
titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 1)].groupby('Sex').mean()['Age']
df_priority = (titanic['Age'] <= 15) & (titanic['Age'] > 0) | (titanic['Sex'] == 0)
df_priority = titanic[df_priority]
df_priority.head(2)
df_priority.groupby('Sex')['Survived'].apply(lambda x: np.mean(x ==  1)*100)
df_priority[df_priority['Age'] <= 15].groupby('Sex')['Survived'].apply(lambda x: np.mean(x == 1)*100)
pd.pivot_table(titanic, values='Name', index='Pclass', aggfunc='count')
pd.crosstab(titanic['Pclass'],titanic['Survived'])[[1]].apply(lambda x: (x / x.sum(axis=0))*100)
titanic.pivot_table(index='Pclass',  values='Name', aggfunc='count').plot(kind='bar', legend=None,
                                                                     title='Number of people per Class', 
                                                                     color='blue', rot=0).set_xlabel('Class')
plt.ylabel('Quantity')
titanic[titanic['Survived'] == 1].groupby('Pclass').sum()['Survived'].plot(kind='bar',
                                                      title='Number of survivors per Class', rot=0).set_xlabel('Classe')
plt.ylabel('Quantity')
titanic.pivot_table('Survived', ["Pclass"], 'Sex', aggfunc='count')
titanic.pivot_table('Not Survived', ["Sex","Pclass"], 'Survived', aggfunc='count')
titanic.pivot_table('PassengerId', ['Pclass'], 'Sex', aggfunc='count').sort_index().plot(kind='barh', stacked=True, 
                                            title='Number of Men and Women by Class').legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Quantity')
titanic[titanic['Survived'] == 1].pivot_table('PassengerId', ['Pclass'], 'Sex', aggfunc='count').plot(kind='barh', 
                                                              title='Number of Survivors Men and Women by Class')\
                                                              .legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Quantity')
sns.heatmap(titanic.corr(),annot=True,cmap=sns.diverging_palette(220, 10, as_cmap = True),linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
titanic['Sex'] = titanic['Sex'].map({'female': 0,'male': 1})
titanic.head(2)
sns.countplot(x = 'Sex', hue ='Survived',data = titanic, palette = 'viridis'); 
sns.countplot(x = 'Pclass', hue ='Survived',data = titanic, palette = 'viridis');
sns.countplot(x = 'Pclass', hue ='Sex',data = titanic, palette = 'viridis');
embark = pd.get_dummies(dataset.Embarked , prefix='Embarked')
embark.head(2)
classify = pd.get_dummies(dataset.Pclass , prefix='Pclass')
classify.head(2)
gender = pd.Series(np.where(dataset.Sex == 'male' , 1 , 0) , name = 'Sex')
gender.head(2)
booth = pd.DataFrame()
booth['Cabin'] = dataset.Cabin.fillna('U')
booth['Cabin'] = booth['Cabin'].map(lambda c : c[0])
booth = pd.get_dummies(booth['Cabin'] , prefix = 'Cabin')
booth.head(2)
entry = pd.DataFrame()
entry['Age'] = dataset.Age.fillna(dataset.Age.mean())
entry['Fare'] = dataset.Fare.fillna(dataset.Fare.mean())
entry['SibSp'] = dataset.SibSp.fillna(dataset.SibSp.mean())
entry['Parch'] = dataset.Parch.fillna(dataset.Parch.mean())
entry.head(2)
featured_data = pd.concat([entry , embark , classify , gender, booth], axis=1)
featured_data.tail(2)
from sklearn.model_selection import train_test_split

featured_data_final = featured_data.apply(lambda x:(x - np.mean(x)) / (np.max(x) - np.min(x)))
featured_data['Age'] = featured_data_final['Age']
featured_data['Fare'] = featured_data_final['Fare']
featured_data['SibSp'] = featured_data_final['SibSp']
featured_data['Parch'] = featured_data_final['Parch']
training_data_final = featured_data[0:891]
training_data_valid = titanic.Survived
featuring_data_test = featured_data[891:]
train_data, test_data, train_labels, test_labels = train_test_split(training_data_final, training_data_valid, train_size=.7)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(train_data, train_labels)
y_pred = gaussian.predict(test_data)
acc_gaussian = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_gaussian)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(train_data, train_labels)
y_pred = logreg.predict(test_data)
acc_logreg = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_logreg)
from sklearn.svm import SVC

svc = SVC()
svc.fit(train_data, train_labels)
y_pred = svc.predict(test_data)
acc_svc = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_svc)
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(train_data, train_labels)
y_pred = linear_svc.predict(test_data)
acc_linear_svc = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_linear_svc)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(train_data, train_labels)
y_pred = perceptron.predict(test_data)
acc_perceptron = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_perceptron)
from sklearn.neural_network import MLPClassifier

mlperceptron = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(6, 2), random_state=1)
mlperceptron.fit(train_data, train_labels)
y_pred = mlperceptron.predict(test_data)
acc_mlperceptron = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_mlperceptron)
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(train_data, train_labels)
y_pred = decisiontree.predict(test_data)
acc_decisiontree = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_decisiontree)
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(train_data, train_labels)
y_pred = adaboost.predict(test_data)
acc_adaboost = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_adaboost)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(train_data, train_labels)
y_pred = randomforest.predict(test_data)
acc_randomforest = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_randomforest)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)
y_pred = knn.predict(test_data)
acc_knn = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_knn)
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

baggedknn = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
baggedknn.fit(train_data, train_labels)
y_pred = baggedknn.predict(test_data)
result = accuracy_score(y_pred, test_labels)
cross = cross_val_score(baggedknn,train_data,train_labels,cv=10,scoring='accuracy')
acc_baggedknn = (cross.mean() * 100)
print(acc_baggedknn)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(train_data, train_labels)
y_pred = sgd.predict(test_data)
acc_sgd = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_sgd)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(train_data, train_labels)
y_pred = gbk.predict(test_data)
acc_gbk = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_gbk)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Bagged KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Multi-layer Perception', 'Linear SVC', 
              'Decision Tree', 'Adaboost', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_baggedknn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron, acc_mlperceptron,
              acc_linear_svc, acc_decisiontree, acc_adaboost,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
test_final = featuring_data_test.as_matrix()

predictions  = gbk.predict(test_final)
predictions  = predictions.flatten().round().astype(int)
passenger_id = dataset[891:].PassengerId
output = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions })
output.shape
output.head()
output.to_csv('submission.csv', index = False)
pd.show_versions()