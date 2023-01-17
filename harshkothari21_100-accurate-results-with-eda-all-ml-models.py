import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import re

import warnings

from statistics import mode

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
target = train.Survived
train.head()
print(f'Unique Values in Pclass :{train.Pclass.unique()}')
print(f'Unique Values in SibSp :{train.SibSp.unique()}')
print(f'Unique Values in Embarked :{train.Embarked.unique()}')
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(train.Survived)

plt.title('Number of passenger Survived');



plt.subplot(1,2,2)

sns.countplot(x="Survived", hue="Sex", data=train)

plt.title('Number of passenger Survived');
plt.style.use('seaborn')

plt.figure(figsize=(10,5))

sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma')

plt.title('Null Values in Training Set');
plt.figure(figsize=(15,5))

plt.style.use('fivethirtyeight')



plt.subplot(1,2,1)

sns.countplot(train['Pclass'])

plt.title('Count Plot for PClass');



plt.subplot(1,2,2)

sns.countplot(x="Survived", hue="Pclass", data=train)

plt.title('Number of passenger Survived');
pclass1 = train[train.Pclass == 1]['Survived'].value_counts(normalize=True).values[0]*100

pclass2 = train[train.Pclass == 2]['Survived'].value_counts(normalize=True).values[1]*100

pclass3 = train[train.Pclass == 3]['Survived'].value_counts(normalize=True).values[1]*100



print("Lets look at some satistical data!\n")

print("Pclaas-1: {:.1f}% People Survived".format(pclass1))

print("Pclaas-2: {:.1f}% People Survived".format(pclass2))

print("Pclaas-3: {:.1f}% People Survived".format(pclass3))
train['Age'].plot(kind='hist')
train['Age'].hist(bins=40)

plt.title('Age Distribution');
# set plot size

plt.figure(figsize=(15, 3))



# plot a univariate distribution of Age observations 

sns.distplot(train[(train["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)



# set titles and labels

plt.title('Distrubution of passengers age',fontsize= 14)

plt.xlabel('Age')

plt.ylabel('Frequency')

# clean layout

plt.tight_layout()
plt.figure(figsize=(15, 3))



# Draw a box plot to show Age distributions with respect to survival status.

sns.boxplot(y = 'Survived', x = 'Age', data = train,

     palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(y = 'Survived', x = 'Age', data = train,

     linewidth = 0.6, palette=["#3f3e6fd1", "#85c6a9"], orient = 'h')



plt.yticks( np.arange(2), ['drowned', 'survived'])

plt.title('Age distribution grouped by surviving status (train data)',fontsize= 14)

plt.ylabel('Passenger status after the tragedy')

plt.tight_layout()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(train['SibSp'])

plt.title('Number of siblings/spouses aboard');



plt.subplot(1,2,2)

sns.countplot(x="Survived", hue="SibSp", data=train)

plt.legend(loc='right')

plt.title('Number of passenger Survived');
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(train['Embarked'])

plt.title('Number of Port of embarkation');



plt.subplot(1,2,2)

sns.countplot(x="Survived", hue="Embarked", data=train)

plt.legend(loc='right')

plt.title('Number of passenger Survived');
sns.heatmap(train.corr(), annot=True)

plt.title('Corelation Matrix');
corr = train.corr()

sns.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1)], annot=True, linewidths=.5, fmt= '.2f')

plt.title('Configured Corelation Matrix');
sns.catplot(x="Embarked", y="Fare", kind="violin", inner=None,

            data=train, height = 6, order = ['C', 'Q', 'S'])

plt.title('Distribution of Fare by Embarked')

plt.tight_layout()
sns.catplot(x="Pclass", y="Fare", kind="swarm", data=train, height = 6)



plt.tight_layout()
sns.catplot(x="Pclass", y="Fare",  hue = "Survived", kind="swarm", data=train, 

                                    palette=["#3f3e6fd1", "#85c6a9"], height = 6)

plt.tight_layout()
train['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare', color = ['#C62D42', '#FE6F5E']);

plt.xlabel('Index')

plt.ylabel('Fare');
train['Age'].nlargest(10).plot(kind='bar', color = ['#5946B2','#9C51B6']);

plt.title('10 largest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
train['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])

plt.title('10 smallest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.corr(), annot=True)
train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')





#Same thing for test set

test.loc[test.Age.isnull(), 'Age'] = test.groupby("Pclass").Age.transform('median')
train.Embarked.value_counts()
train['Embarked'] = train['Embarked'].fillna(mode(train['Embarked']))



#Applying the same technique for test set

test['Embarked'] = test['Embarked'].fillna(mode(test['Embarked']))
train['Fare']  = train.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

test['Fare']  = test.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
train.Cabin.value_counts()
train['Cabin'] = train['Cabin'].fillna('U')

test['Cabin'] = test['Cabin'].fillna('U')
train.Sex.unique()
train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1



test['Sex'][test['Sex'] == 'male'] = 0

test['Sex'][test['Sex'] == 'female'] = 1
train.Embarked.unique()
from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder()

temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

train = train.join(temp)

train.drop(columns='Embarked', inplace=True)



temp = pd.DataFrame(encoder.transform(test[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

test = test.join(temp)

test.drop(columns='Embarked', inplace=True)
train.columns
train.Cabin.tolist()[0:20]
train['Cabin'] = train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())

test['Cabin'] = test['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
train.Cabin.unique()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

train['Cabin'] = train['Cabin'].map(cabin_category)

test['Cabin'] = test['Cabin'].map(cabin_category)
train.Name
train['Name'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

test['Name'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
train['Name'].unique().tolist()
train.rename(columns={'Name' : 'Title'}, inplace=True)

train['Title'] = train['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

                                      

test.rename(columns={'Name' : 'Title'}, inplace=True)

test['Title'] = test['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
train['Title'].value_counts(normalize = True) * 100
encoder = OneHotEncoder()

temp = pd.DataFrame(encoder.fit_transform(train[['Title']]).toarray())

train = train.join(temp)

train.drop(columns='Title', inplace=True)



temp = pd.DataFrame(encoder.transform(test[['Title']]).toarray())

test = test.join(temp)

test.drop(columns='Title', inplace=True)
train['familySize'] = train['SibSp'] + train['Parch'] + 1

test['familySize'] = test['SibSp'] + test['Parch'] + 1
fig = plt.figure(figsize = (15,4))



ax1 = fig.add_subplot(121)

ax = sns.countplot(train['familySize'], ax = ax1)



# calculate passengers for each category

labels = (train['familySize'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+6, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    

plt.title('Passengers distribution by family size')

plt.ylabel('Number of passengers')



ax2 = fig.add_subplot(122)

d = train.groupby('familySize')['Survived'].value_counts(normalize = True).unstack()

d.plot(kind='bar', color=["#3f3e6fd1", "#85c6a9"], stacked='True', ax = ax2)

plt.title('Proportion of survived/drowned passengers by family size (train data)')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

plt.xticks(rotation = False)



plt.tight_layout()
# Drop redundant features

train = train.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)

test = test.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)
train.head()
columns = train.columns[2:]

from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(train.drop(columns=["PassengerId","Survived"]))



new_df = pd.DataFrame(X_train, columns=columns)
from sklearn.decomposition import PCA



pca = PCA(n_components = 2)

df_pca = pca.fit_transform(new_df)
plt.figure(figsize =(8, 6))

plt.scatter(df_pca[:, 0], df_pca[:, 1], c = target, cmap ='plasma')

# labeling x and y axes

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component');
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(linreg.score(X_train, y_train)))

print("R-Squared for test set: {:.3f}" .format(linreg.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=10000, C=50)

logreg.fit(X_train, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test, y_test)))
print(logreg.intercept_)

print(logreg.coef_)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)



# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train_scaled, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test)))
from sklearn.neighbors import KNeighborsClassifier



knnclf = KNeighborsClassifier(n_neighbors=7)



# Train the model using the training sets

knnclf.fit(X_train, y_train)

y_pred = knnclf.predict(X_test)
from sklearn.metrics import accuracy_score



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(y_test, y_pred))
knnclf = KNeighborsClassifier(n_neighbors=7)



# Train the model using the scaled training sets

knnclf.fit(X_train_scaled, y_train)

y_pred = knnclf.predict(X_test_scaled)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(y_test, y_pred))
from sklearn.svm import LinearSVC



svmclf = LinearSVC(C=50)

svmclf.fit(X_train, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svmclf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svmclf.score(X_test, y_test)))
svmclf = LinearSVC()

svmclf.fit(X_train_scaled, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svmclf.score(X_train_scaled, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svmclf.score(X_test_scaled, y_test)))
from sklearn.svm import SVC



svcclf = SVC(gamma=0.1)

svcclf.fit(X_train, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svcclf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svcclf.score(X_test, y_test)))
svcclf = SVC(gamma=50)

svcclf.fit(X_train_scaled, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svcclf.score(X_train_scaled, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svcclf.score(X_test_scaled, y_test)))
from sklearn.tree import DecisionTreeClassifier



dtclf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(dtclf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(dtclf.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(random_state=2)
# Set our parameter grid

param_grid = { 

    'criterion' : ['gini', 'entropy'],

    'n_estimators': [100, 300, 500],

    'max_features': ['auto', 'log2'],

    'max_depth' : [3, 5, 7]    

}
from sklearn.model_selection import GridSearchCV



randomForest_CV = GridSearchCV(estimator = rfclf, param_grid = param_grid, cv = 5)

randomForest_CV.fit(X_train, y_train)
randomForest_CV.best_params_
rf_clf = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 100)



rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, predictions) * 100
#Linear Model

print("Linear Model R-Squared for Train set: {:.3f}".format(linreg.score(X_train, y_train)))

print("Linear Model R-Squared for test set: {:.3f}" .format(linreg.score(X_test, y_test)))

print()



#Logistic Regression

print("Logistic Regression R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train)))

print("Logistic Regression R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test)))

print()



#KNN Classifier

print("KNN Classifier Accuracy:",accuracy_score(y_test, y_pred))

print()



#SVM

print('SVM Accuracy on training set: {:.2f}'

     .format(svmclf.score(X_train_scaled, y_train)))

print('SVM Accuracy on test set: {:.2f}'

     .format(svmclf.score(X_test_scaled, y_test)))

print()



#Kerelize SVM

print('SVC Accuracy on training set: {:.2f}'

     .format(svcclf.score(X_train_scaled, y_train)))

print('Accuracy on test set: {:.2f}'

     .format(svcclf.score(X_test_scaled, y_test)))

print()



#Decision Tree

print('Accuracy of Decision Tree on training set: {:.2f}'

     .format(dtclf.score(X_train, y_train)))

print('Accuracy of Decision Tree on test set: {:.2f}'

     .format(dtclf.score(X_test, y_test)))

print()



#Random Forest

print('Random Forest Accuracy:{:.3f}'.format(accuracy_score(y_test, predictions) * 100))
scaler = MinMaxScaler()



train_conv = scaler.fit_transform(train.drop(['Survived', 'PassengerId'], axis=1))

test_conv = scaler.transform(test.drop(['PassengerId'], axis = 1))
svcclf = SVC(gamma=50)

svcclf.fit(train_conv, train['Survived'])
test['Survived'] = svcclf.predict(test_conv)
test[['PassengerId', 'Survived']].to_csv('MySubmission1.csv', index = False)