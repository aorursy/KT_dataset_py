import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
# This function will describe the basic statistics about data
train_data.describe()
# This will return the correlation values of each feature with each other
train_data.corr()
#Passengers who survived
People_Survived = train_data[train_data['Survived']==1]
People_Survived.head()
# Let's know how many male and female have been survived
Gender_Survived = People_Survived[['Sex','Survived']].groupby('Sex').count()
Gender_Survived['Gender'] = Gender_Survived.index
plt.bar(Gender_Survived.iloc[:,1].values, Gender_Survived.iloc[:,0], color='red')
plt.xlabel("Gender")
plt.ylabel("No. of Male and Female Survived")
plt.show()
#Let's know is there any affect on survival due to Pclass
Pclass_survived = People_Survived[['Pclass','Survived']].groupby('Pclass').count()
Pclass_survived['pclass'] = Pclass_survived.index
plt.bar(Pclass_survived.iloc[:,1].values, Pclass_survived.iloc[:,0], color='blue')
plt.xlabel('Number of Classes')
plt.ylabel('Number of people survived in each class')
plt.xticks(np.arange(1,4))
plt.show()
#Let's check the number of passengers who has been survived is from which port ?
Embarked_No = People_Survived[['Embarked','PassengerId']].groupby('Embarked').count()
Embarked_No['embarked'] = Embarked_No.index
plt.bar(Embarked_No.iloc[:,1].values, Embarked_No.iloc[:,0], color='yellow')
plt.xlabel('Ports from where passengers embarked')
plt.ylabel('Total passenger survived from each port')
plt.show()
#Let's see the distribution of the age (range) for the passengers who have been survived and not survived
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)
train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
train_data.head()
train_data.isnull().sum()
#Scikit Learn provides a built in imputation class under preprocessing module
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data.iloc[:,3].values.reshape(-1,1))
train_data.head(10) # You can see the median value has been updated to Null values. 
train_data.isnull().sum()
train_data = train_data.dropna()
train_data.isnull().sum()
test_data = test_data.drop(['Name','Ticket','Cabin'],axis=1)
test_data.head()
test_data.isnull().sum()
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
test_data['Age'] = imputer.fit_transform(test_data.iloc[:,3].values.reshape(-1,1))
test_data.isnull().sum()
test_data['Fare'] = imputer.transform(test_data.iloc[:,6].values.reshape(-1,1))
test_data.isnull().sum()
test_df = test_data.drop('PassengerId',axis=1)
#Pandas has built in method named: get_dummies(), which will take care of One Hot Encoding
train_data = pd.get_dummies(train_data)
test_df = pd.get_dummies(test_df)
train_data.head()
test_df.head()
column_to_normalize = ['Pclass','Age','Fare']
train_data[column_to_normalize] = train_data[column_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))  #Min-Max Normalization
train_data.head()
test_df[column_to_normalize] = test_df[column_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))  #Min-Max Normalization
test_df.head()
#Let's first seprate our features and labels
X_train = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']
# These are the list of algorithms that we are going to use to train our model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score
#Logistic Regression
clf = LogisticRegression()
clf.fit(X_train,y)
y_pred_LR = clf.predict(test_df)
log_reg_acc = clf.score(X_train, y)
#KNN K-Nearest Neighbors
clf = KNeighborsClassifier()
clf.fit(X_train,y)
y_pred_KNN = clf.predict(test_df)
knn_acc = clf.score(X_train, y)
#Support Vector Machine (SVM)
clf = SVC()
clf.fit(X_train,y)
y_pred_svm = clf.predict(test_df)
svm_acc = clf.score(X_train, y)
#Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y)
y_pred_DT = clf.predict(test_df)
decision_tree_acc = clf.score(X_train, y)
#Random Forest
clf = RandomForestClassifier()
clf.fit(X_train,y)
y_pred_RF = clf.predict(test_df)
random_forest_acc = clf.score(X_train, y)
#Gradient Boosting
clf = GradientBoostingClassifier()
clf.fit(X_train,y)
y_pred_GB = clf.predict(test_df)
gradient_boosting_acc = clf.score(X_train, y)
#Ada Boost 
clf = AdaBoostClassifier()
clf.fit(X_train,y)
y_pred_ada = clf.predict(test_df)
ada_boost_acc = clf.score(X_train, y)
Accuracy_df = pd.DataFrame({"Models": ['Logistic Regression','KNN','SVM','Decision Tree','Random Forest','Gradient Boosting', 'Ada Boost'], 
                            "Accuracy":[log_reg_acc,knn_acc,svm_acc,decision_tree_acc,random_forest_acc, gradient_boosting_acc, ada_boost_acc ]})
Accuracy_df.sort_values(by='Accuracy',ascending=False)