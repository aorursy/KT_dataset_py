import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
titanic = pd.read_csv("../input/titanic-dataset/titanic.csv",sep=",")
titanic.info()
titanic.shape
titanic.ndim
titanic.head()
titanic.dtypes
titanic.count()
titanic.head()
titanic.describe()
pd.isnull(titanic).sum()
titanic["Pclass"].dropna(axis=0, inplace=True)
titanic["Survived"].dropna(axis=0, inplace=True)
print('The percentage of missing values ​​in the "Cabin" column is %.2f%%' %((titanic['Cabin'].isnull().sum()/titanic.shape[0])*100))
titanic.drop('Cabin',axis=1,inplace=True)
titanic.head()
titanic_temp = titanic
titanic_temp["Age"].dropna(axis=0, inplace=True)
titanic_temp.head()
plt.figure(figsize=(15,8))
ax = sns.kdeplot(titanic_temp["Age"][titanic_temp.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(titanic_temp["Age"][titanic_temp.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass',y='Age',data=titanic)
titanic.head()
titanic.info()
type(titanic)
titanic["Name"].dropna(axis=0,inplace=True)
titanic['Name']
titanic['Title'] = titanic['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
titanic['Title'].unique()
Standardized_Titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
titanic.Title = titanic.Title.map(Standardized_Titles)
print(titanic.Title.value_counts())
grouped = titanic.groupby(['Sex','Pclass', 'Title']) 
grouped.Age.median()
titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
titanic.Age.unique()
titanic.count()
titanic.info()
titanic["Fare"].dropna(axis=0,inplace=True)
titanic.info()
plt.figure(figsize=(15,8))
ax = sns.kdeplot(titanic["Fare"][titanic.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(titanic["Fare"][titanic.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density ')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()
print('Percent of missing "embarked" records is %.2f%%' %((titanic['Embarked'].isnull().sum()/titanic.shape[0])*100))
titanic["Embarked"].dropna(axis=0, inplace=True)
titanic['Embarked'].isnull().sum()
print('Boarded passengers grouped by port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(titanic['Embarked'].value_counts())
sns.countplot(x='Embarked', data=titanic, palette='Set2')
plt.show()
sns.barplot('Embarked', 'Survived', data=titanic, color="teal")
plt.show()
sns.countplot(x='Embarked', data=titanic, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()
titanic.head(3)
titanic.info()
titanic.head(3)
titanic['Embarked'].unique()
titanic.dtypes
sns.barplot('SibSp', 'Survived', data=titanic, color="mediumturquoise")
plt.show()
sns.catplot(x='SibSp',col='Survived', data=titanic, kind="count");
sns.catplot(x='Parch',col='Survived', data=titanic, kind="count");
titanic.dropna(axis=0, inplace=True)
pd.isnull(titanic).sum()
sns.countplot(x='Survived',data=titanic)
sns.barplot('Pclass', 'Survived', data=titanic, color="darkturquoise")
plt.show()
titanic.pivot_table('Survived', index='Sex', columns='Pclass')
age = pd.cut(titanic['Age'], [0, 18, 80])
titanic.pivot_table('Survived', ['Sex', age], 'Pclass')
def titanic_corr(data):
    correlation = data.corr()
titanic_corr(titanic)
Pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")
Pclass_dummies.head()
Embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")
Embarked_dummies.head()
titanic.Sex = titanic.Sex.map({"Male": 0, "Female":1})
titanic.head(3)
titanic.loc[(titanic['Sex'] == "Male", "Sex")]==1
titanic.loc[(titanic['Sex'] == "Female", "Sex")]==0
titanic.head(3)
Title_dummies = pd.get_dummies(titanic.Title, prefix="Title")
Title_dummies.head()
print('The mean of "Age" is %.2f' %(titanic["Age"].mean(skipna=True)))
print('The median of "Age" is %.2f' %(titanic["Age"].median(skipna=True)))
ax = titanic["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
titanic["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
max(titanic["Age"])
titanic["cat_Age"] = pd.cut(titanic["Age"],bins=[0,6,18,60,80], labels=[1,2,3,4])

titanic[["Age", "cat_Age"]].head
titanic.corr()
titanic.drop(['Age'],axis=1,inplace=True)
titanic.drop(['Sex'],axis=1,inplace=True)
titanic.drop(['Name'],axis=1,inplace=True)
titanic.drop(['Ticket'],axis=1,inplace=True)
titanic.info()
titanic_dummies = pd.concat([titanic, Pclass_dummies, Title_dummies, Embarked_dummies], axis=1)
titanic_dummies.drop(['Pclass', 'Title', 'Embarked'], axis=1, inplace=True)
titanic_dummies.info()
titanic_dummies.head()
titanic_dummies.drop(['Pclass_1', 'Embarked_C', 'Title_Master'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
titanic_dummies["Survived"].dropna(axis=0, inplace=True)
titanic_dummies.Survived = titanic_dummies.Survived.astype(int)
X = titanic_dummies.drop('Survived',axis=1)
X_train, X_test, y_train, y_test = train_test_split(titanic_dummies.drop('Survived', axis=1), titanic_dummies['Survived'], test_size=0.2, random_state=0)
X_train.head(4)
X_train.info()
y_train.head()
type(y_train)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 80, 2)
acc_log
from sklearn.metrics import f1_score
f1_LG = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_LG)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 80, 2)
acc_knn
from sklearn.metrics import f1_score
f1_KNN = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_KNN)
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 80, 2)
acc_decision_tree
from sklearn.metrics import f1_score
f1_DT = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_DT)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 80, 2)
acc_random_forest
from sklearn.metrics import f1_score
f1_RF = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_RF)
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 80, 2)
acc_gaussian
from sklearn.metrics import f1_score
f1_GNB = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_GNB)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 80, 2)
acc_svc
from sklearn.metrics import f1_score
f1_SVC = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_SVC)
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 80, 2)
acc_linear_svc
from sklearn.metrics import f1_score
f1_LSVC = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_LSVC)
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 80, 2)
acc_perceptron
from sklearn.metrics import f1_score
f1_PRCP = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1_PRCP)
from sklearn.metrics import accuracy_score
acc = (accuracy_score(y_test,Y_pred))
print('Accuracy score: %f' % acc)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
from sklearn.metrics import f1_score
f1 = f1_score(y_test, Y_pred)
print('F1 score: %f' % f1)

labels = ['Logistic Regression','kNN','Decision Tree','Random Forest','Gaussian','SVC','Linear SVM','Perceptron']
fscores= [f1_LG, f1_KNN ,f1_DT, f1_RF ,f1_GNB,f1_SVC, f1_LSVC,f1_PRCP]

x = np.arange(len(labels))  
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, fscores, width, label='F1-Scores')

ax.set_ylabel('F1-Score Value')
ax.set_title('F1-Score by Specific Algorithm')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
plt.show()
labels = ['Logistic Regression','kNN','Decision Tree','Random Forest','Gaussian','SVC','Linear SVM','Perceptron']
Accuracies = [acc_log, acc_knn ,acc_decision_tree, acc_random_forest ,acc_gaussian,acc_svc, acc_linear_svc,acc_perceptron]
#women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Accuracies, width, label='Accuracies')

ax.set_ylabel('Accuracy Value')
ax.set_title('Accuracy by Specific Algorithm')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
plt.show()
