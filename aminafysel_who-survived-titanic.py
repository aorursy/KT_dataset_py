# data processing

import pandas as pd



## linear algebra

import numpy as np



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

 

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix
df=pd.read_csv('../input/titanic-data/titanic.csv')
df.head()
df.shape
df.describe()
df.info()
missing=df.isnull().sum().sort_values(ascending=False)

missing.head()
percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)

missing_percentage=pd.concat([missing, percent], axis=1, keys=['Missing','Percent'])
missing_percentage.head(15)
drop_column = ['Body','Cabin',]

df.drop(drop_column, axis= 1, inplace = True)
df.head(5)

drop_column = ['Lifeboat']

df.drop(drop_column, axis= 1, inplace = True)
df['Age'].fillna(df['Age'].median(), inplace = True)

df['Survived'].fillna(df['Survived'].mode()[0], inplace = True)
df.isnull().sum()
df1 = df.dropna()

print(df1)
df1.isnull().sum()
sns.set(style="darkgrid")

plt.figure(figsize = (8, 5))

graph= sns.countplot(x='Survived', hue="Survived", data=df1)

plt.figure(figsize = (8, 5))

graph  = sns.countplot(x ="Sex", hue ="Survived", data = df1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4)) 

x = sns.countplot(df1['Pclass'], ax=ax[0])

y = sns.countplot(df1['Embarked'], ax=ax[1])



fig.show()
drop_column = ['Embarked']

df1.drop(drop_column, axis=1, inplace = True)
df1.head()
plt.figure(figsize = (8, 5))

graph  = sns.countplot(x ="Pclass", hue ="Survived", data = df1)

plt.figure(figsize = (8, 5))

sns.barplot(x='Pclass', y='Survived', data=df1)
axes = sns.factorplot('SibSp','Survived', 

                      data=df1, aspect = 2.5, )
axes = sns.factorplot('Parch','Survived', 

                      data=df1, aspect = 2.5, )
df1['Age_bin'] = pd.cut(df1['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

    

plt.figure(figsize = (8, 5))

sns.barplot(x='Age_bin', y='Survived', data=df1)
plt.figure(figsize = (8, 5))

sns.countplot(x='Age_bin', hue='Survived', data=df1)

df1['Fare_bin'] = pd.cut(df1['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])

plt.figure(figsize = (8, 5))

sns.countplot(x='Pclass', hue='Fare_bin', data=df1)

sns.barplot(x='Fare_bin', y='Survived', data=df1)
df1.head()
pd.DataFrame(abs(df1.corr()['Survived']).sort_values(ascending = False))
f, ax = plt.subplots(figsize=(10, 8))

corr = df1.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax,annot=True)
df1.info()
# Convert ‘Sex’ feature into numeric.

genders = {"male": 0, "female": 1}



df1['Sex'] = df1['Sex'].map(genders)

df1['Sex'].value_counts()
drop_column = ['Age_bin','Fare','Name','Ticket', 'PassengerId','WikiId','Name_wiki','Age_wiki','Hometown','Boarded','Destination','Fare_bin']

df1.drop(drop_column, axis=1, inplace = True)
df1.head()
all_features = df1.drop("Survived",axis=1)

Targete = df1["Survived"]

X_train,X_test,y_train,y_test = train_test_split(all_features,Targete,test_size=0.3,random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
# standard scaling 

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 22)

logreg.fit(X_train, y_train)



# Support Vector Classifier Algorithm

from sklearn.svm import SVC

svc = SVC(kernel = 'linear', random_state = 22)

svc.fit(X_train, y_train)
# Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

# Decision tree Algorithm

from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 22)

dectree.fit(X_train, y_train)

# Random forest Algorithm

from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 22)

ranfor.fit(X_train, y_train)
# Making predictions on test dataset

Y_pred_logreg = logreg.predict(X_test)

#Y_pred_knn = knn.predict(X_test)

Y_pred_svc = svc.predict(X_test)

Y_pred_nb = nb.predict(X_test)

Y_pred_dectree = dectree.predict(X_test)

Y_pred_ranfor = ranfor.predict(X_test)
# Evaluating using accuracy_score metric

from sklearn.metrics import accuracy_score

accuracy_logreg = accuracy_score(y_test, Y_pred_logreg)

#accuracy_knn = accuracy_score(y_test, Y_pred_knn)

accuracy_svc = accuracy_score(y_test, Y_pred_svc)

accuracy_nb = accuracy_score(y_test, Y_pred_nb)

accuracy_dectree = accuracy_score(y_test, Y_pred_dectree)

accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)
# Accuracy on test set

print("Logistic Regression: " + str(accuracy_logreg * 100))

#print("K Nearest neighbors: " + str(accuracy_knn * 100))

print("Support Vector Classifier: " + str(accuracy_svc * 100))

print("Naive Bayes: " + str(accuracy_nb * 100))

print("Decision tree: " + str(accuracy_dectree * 100))

print("Random Forest: " + str(accuracy_ranfor * 100))
# Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, Y_pred_logreg)

cm
# Heatmap of Confusion matrix

sns.heatmap(pd.DataFrame(cm), annot=True)
# Classification report

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred_ranfor))