import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
train = pd.read_csv('E:\\R files\\raj python files\\logistic regression\\titanic_data.csv')
train.head()
train.info()
train.describe()
train.isnull().sum()
#EDA
sns.countplot(x='Survived',data=train)

sns.countplot(x='Survived',hue='Sex',data=train)

sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),bins=30)

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))

sns.boxplot(x='Pclass',y='Age',data=train)
sns.heatmap(train.isnull())
sns.heatmap(train.corr())
train['Age']=train['Age'].fillna(train['Age'].mean())
train['Embarked']=train['Embarked'].fillna('S')
X = train.iloc[:, [2, 4, 5, 6, 7, 9,11]]
y = train.iloc[:, 1]
X.head()
y.head()
sex = pd.get_dummies(X['Sex'], prefix = 'Sex')
embark = pd.get_dummies(X['Embarked'], prefix = 'Embarked')
passenger_class = pd.get_dummies(X['Pclass'], prefix = 'Pclass')
X = pd.concat([X,sex,embark, passenger_class],axis=1)
X.head()
sns.boxplot(data= X).set_title("Outlier Box Plot")
X.columns
X=X.drop(['Sex','Embarked','Pclass'],axis=1)
X.head()
X['travel_alone']=np.where((X['SibSp']+X['Parch'])>0,1,0)
X.corr()
X.head()
X=X.drop(['SibSp','Parch','Sex_male'],axis=1)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,1]] = sc.fit_transform(X_train.iloc[:, [0,1]])
X_test.iloc[:, [0,1]] = sc.transform(X_test.iloc[:, [0,1]])
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)
classifier.score(X_train,y_train)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
from sklearn.feature_selection import RFE

rfe = RFE(classifier, rfecv.n_features_)
rfe = rfe.fit(X_train, y_train)
print(list(X.columns[rfe.support_]))

x=X.drop(['Fare','Embarked_C','Pclass_2','travel_alone'],axis=1)
x.head()

y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test,y_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()
model_accuracy,model_standard_deviation
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
confusion_matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
list(X.columns[rfe.support_])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

