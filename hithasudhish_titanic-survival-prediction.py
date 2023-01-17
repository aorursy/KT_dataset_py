import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
df_train = pd.read_csv("../input/titanic/train.csv")
df_test =pd.read_csv("../input/titanic/test.csv") 
df_train.head()
df_test.head()
df_train.shape
df_train.info()
#Describing the columns with mean, min and max
df_train.describe(include='all')
df_test.shape
df_test.info()
#Describing the columns with mean, min and max
df_test.describe(include='all')
# make sure no of duplicate rows are zero
sum(df_train.duplicated(subset = "PassengerId")) == 0
# check for NA values
na_columns = (df_train.isnull().sum())
na_columns = (df_train.isnull().sum()*100/df_train.shape[0])
na_columns = na_columns[na_columns>0]
na_columns.sort_values(inplace=True, ascending=False)
na_columns
#Selecting the  columns which has 70% missing value
drop_columns = na_columns[(na_columns > 70) == True].index
drop_columns
#Dropping the columns with 70% missing values
df_train = df_train.drop(drop_columns, axis = 1) 
df_train.columns
#Dropping 'PassengerId' 'Name' 'Fare' and 'Ticket' as they are not contributing directly to survival
df_train = df_train.drop(['PassengerId','Name','Fare','Ticket'], axis=1) 
df_train.columns
#Replacing the null values in 'Age' colum using mean
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
#Checking the NA values and confirming the number is zero
df_train['Age'].isnull().sum()
#Replacing the null values of 'Embarked' column with most frequent  value
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].value_counts().index[0])
#Checking the NA values and confirming the number is zero
df_train['Embarked'].isnull().sum()
# make sure no of duplicate rows are zero
sum(df_test.duplicated(subset = "PassengerId")) == 0
# check for NA values in test data
na_test = (df_test.isnull().sum())
na_test = (df_test.isnull().sum()*100/df_test.shape[0])
na_test = na_test[na_test>0]
na_test.sort_values(inplace=True, ascending=False)
na_test
#Selecting the  columns which has 70% missing value
test_drop_columns = na_test[(na_test > 70) == True].index
test_drop_columns
#Dropping the columns with 70% missing values
df_test = df_test.drop(test_drop_columns, axis = 1) 
df_test.columns
PassengerId = df_test['PassengerId']
PassengerId.head(10)
#Dropping 'PassengerId' 'Name' 'Fare' and 'Ticket' from test data as they are not contributing directly to survival
df_test = df_test.drop(['PassengerId','Name','Fare','Ticket'], axis=1) 
df_test.columns
#Replacing the null values in 'Age' column using mean
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
#Checking the NA values and confirming the number is zero
df_test['Age'].isnull().sum()
#Deriving new feature 'family_size' in train data by combining 'SibSp' and 'Parch'
df_train['family_size'] = df_train['SibSp'] + df_train['Parch']
df_train[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Creating age bins in train data
df_train['age_bins'] = pd.cut(x=df_train['Age'], bins=[0,10,60,80], labels = [10, 60,80])
df_train['age_bins'].value_counts()
#Deriving new feature 'family_size' in test data by combining 'SibSp' and 'Parch'
df_test['family_size'] = df_test['SibSp'] + df_test['Parch']
# Creating age bins in test data
df_test['age_bins'] = pd.cut(x=df_test['Age'], bins=[0,10,60,80], labels = [10, 60,80])
df_test['age_bins'].value_counts()
sns.barplot(x='Sex', y='Survived', data = df_train)
sns.barplot(x='Pclass', y='family_size', data=df_train)
sns.barplot(x='Pclass', y='Survived', data=df_train)
sns.barplot(x="Survived", y="Pclass", hue="Sex",data=df_train,orient='h')
Embarked_grid = sns.FacetGrid(df_train, row='Embarked', height=4.5, aspect=1.6)
Embarked_grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
Embarked_grid.add_legend()
grid = sns.FacetGrid(df_train,col="Pclass", row="Survived",hue="Sex", aspect=1.6)
grid.map(plt.hist, "Age",bins=15)
sns.barplot(x="family_size", y="Survived", hue="Sex",data=df_train)
with sns.axes_style(style='ticks'):
    g = sns.catplot("Pclass","family_size", "Survived", data=df_train, kind="box")
    g.set_axis_labels( "Pclass", "family_size");
sns.heatmap(df_train.loc[:, ['Survived','Pclass', 'Sex', 'family_size', 'Age', ]].corr(),annot=True)
# Converting train data objects and floats to integers- Sex,Embarked, Age
label_encoder = preprocessing.LabelEncoder()
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
df_train['Sex'].head(10)
label_encoder = preprocessing.LabelEncoder()
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'])
df_train['Embarked'].tail(10)
df_train['Age'] = df_train['Age'].astype(int)
df_train['age_bins'] = df_train['age_bins'].astype(int)
df_train.info()
# Converting test data objects and floats to integers- Age,Sex,Embarked
label_encoder = preprocessing.LabelEncoder()
df_test['Sex'] = label_encoder.fit_transform(df_test['Sex'])
df_test['Sex'].head(10)
label_encoder = preprocessing.LabelEncoder()
df_test['Embarked'] = label_encoder.fit_transform(df_test['Embarked'])
df_test['Embarked'].head(10)
df_test['Age'] = df_test['Age'].astype(int)
df_test['age_bins'] = df_test['age_bins'].astype(int)
df_test.info()
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test
X_train.shape, Y_train.shape, X_test.shape
X_train.info()
X_test.info()
# Logistic regression
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_train)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)
# random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Perceptron
perceptron = Perceptron(max_iter=100)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# Linear svc
linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
models = pd.DataFrame({
    'models': [ 'Logistic Regression','Random Forest', 'Decision Tree',
               'Perceptron','Gaussian Naive Bayes',  
              'Stochastic Gradient Decent',  
              'Linear SVC','Support Vector Machines'],
    'Score': [acc_log, acc_random_forest, acc_decision_tree
              , acc_perceptron, acc_gaussian,
              acc_sgd, acc_linear_svc, acc_svc,]})
models.sort_values(by='Score', ascending=False)
models
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)

submission = pd.DataFrame({"PassengerId": PassengerId,"Survived": Y_pred})
submission.to_csv('submission.csv', index=False)