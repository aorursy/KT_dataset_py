import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.figure(figsize=(16,5))

# load train and test data

train_data = pd.read_csv("../input/titanic/train.csv", index_col='PassengerId')

test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')

# see columns names, types and missing values and head of test_data

print(train_data.info())

print("___________________________________")

train_data.head()
# print description of numerical data

train_data.describe()
# print number of missing values in train and test dataFrame

print(train_data.isnull().sum())

print("________________________")

print(test_data.isnull().sum())

# plot heatmap with numeric features

plt.figure(figsize=(16,5))

sns.heatmap(data=train_data.corr(), vmin=-1, vmax=1, cmap='YlGnBu', annot=True)

plt.show()
#PClass plot barplot



sns.barplot(x="Pclass", y="Survived", data=train_data)

plt.show()



# print values

train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
# SEX plot barplot



sns.barplot(x="Sex", y="Survived", data=train_data)

plt.show()



# print values

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
# SibSp plot barplot



sns.barplot(x="SibSp", y="Survived", data=train_data)

plt.show()



# print values

train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)

# Parch plot barplot



sns.barplot(x="Parch", y="Survived", data=train_data)

plt.show()



# print values

train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
# Embarked plot barplot



sns.barplot(x="Embarked", y="Survived", data=train_data)



# print values

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
#sort the ages into logical categories



bins = [-1, 0, 5, 12, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager-Adult', 'Senior']

for df in [train_data, test_data]:

    df["Age"] = df["Age"].fillna(-0.5)

    df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train_data)

plt.xticks(np.linspace(0,6,7), labels, rotation=45, ha="right")

plt.xlim(-0.6,4.6)

plt.show()



# print values

train_data[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
#sort the ages into logical categories



bins = [-1, 8, 15, 30, np.inf]

labels = ['<8', '8-15', '15-31', '>31']

for df in [train_data, test_data]:

    df["Fare"] = df["Fare"].fillna(-0.5)

    df['FareGroup'] = pd.cut(df["Fare"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="FareGroup", y="Survived", data=train_data)

plt.xticks(np.linspace(0,5,6), labels, rotation=45, ha="right")

plt.xlim(-0.6,3.6)

plt.show()



# print values

train_data[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(

    by='Survived', ascending=False)
# create Name Title



for df in [train_data, test_data]:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')



train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# join SibSp and Parch as FamilySize



for df in [train_data, test_data]:

    df['FamilySize'] = (df['SibSp'] + df['Parch'] + 1)

    df.loc[df['FamilySize'] > 4, 'FamilySize'] = 5



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(

        by='Survived', ascending=False)
# fill missing Embarked with mode



for df in [train_data, test_data]:

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# map each Sex value to a numerical value



sex_mapping = {"male": 0, "female": 1}

for df in [train_data, test_data]:

    df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
# map each Age value to a numerical value and fill missing values



age_group_mapping = {'Baby' : 0, 'Child' : 1, 'Teenager-Adult' : 2, 'Senior' : 3}

for df in [train_data, test_data]:

    # Fill missing values based on Title    

    df['AgeGroup'] = df['AgeGroup'].replace(['Unknown'], [None])

    mr_age = df[df["Title"] == "Mr"]["AgeGroup"].mode()[0] 

    miss_age = df[df["Title"] == "Miss"]["AgeGroup"].mode()[0]

    mrs_age = df[df["Title"] == "Mrs"]["AgeGroup"].mode()[0] 

    master_age = df[df["Title"] == "Master"]["AgeGroup"].mode()[0]

    rare_age = df[df["Title"] == "Rare"]["AgeGroup"].mode()[0]

    title_age_mapping = {"Mr": mr_age, "Miss": miss_age, "Mrs": mrs_age, "Master": master_age, "Rare": rare_age}

    df['AgeGroup'].fillna(df['Title'].map(title_age_mapping), inplace=True)

    

    # map strings to int

    df['AgeGroup'] = df['AgeGroup'].map(age_group_mapping).astype('int')

# map each FareGroup value to a numerical value



fare_mapping = {'<8' : 0, '8-15' : 1, '15-31' : 2, '>31' : 3}

for df in [train_data, test_data]:

    df['FareGroup'] = df['FareGroup'].map(fare_mapping).astype('int')
# create CabinBool feature, that show if the passanger have or not a Cabin



for df in [train_data, test_data]:

    df["CabinBool"] = df["Cabin"].notnull().astype('bool')
# drop unused data



for df in [train_data, test_data]:

    df.drop(['Name'], axis = 1, inplace=True)

    df.drop(['SibSp'], axis = 1, inplace=True)

    df.drop(['Parch'], axis = 1, inplace=True)

    df.drop(['Age'], axis = 1, inplace=True)

    df.drop(['Cabin'], axis = 1, inplace=True)

    df.drop(['Fare'], axis = 1, inplace=True)

    df.drop(['Ticket'], axis = 1, inplace=True)
# Checking the data in train and test DataFrames.



train_data.info()

print("********************************************")

test_data.info()
# prepare data to be used in the models



from sklearn.model_selection import cross_val_score

train_data = pd.get_dummies(train_data, columns=['Embarked', 'Title'], drop_first=True)

y = train_data["Survived"]

X = train_data.drop(['Survived'], axis = 1)
# Gaussian Naive Bayes



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

scores = cross_val_score(gaussian, X, y, cv=5)

acc_gaussian = round(scores.mean() * 100, 2)

print(acc_gaussian)
# Logistic Regression



from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver='lbfgs')

scores = cross_val_score(logreg, X, y, cv=5)

acc_logreg = round(scores.mean() * 100, 2)

print(acc_logreg)
# Support Vector Machines



from sklearn.svm import SVC



svc = SVC(gamma='auto')

scores = cross_val_score(svc, X, y, cv=5)

acc_svc = round(scores.mean() * 100, 2)

print(acc_svc)
# Linear SVC



from sklearn.svm import LinearSVC



linear_svc = LinearSVC(max_iter=3000)

scores = cross_val_score(linear_svc, X, y, cv=5)

acc_linear_svc = round(scores.mean() * 100, 2)

print(acc_linear_svc)
# Perceptron



from sklearn.linear_model import Perceptron



perceptron = Perceptron()

scores = cross_val_score(perceptron, X, y, cv=5)

acc_perceptron = round(scores.mean() * 100, 2)

print(acc_perceptron)
#Decision Tree



from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

scores = cross_val_score(decisiontree, X, y, cv=5)

acc_decisiontree = round(scores.mean() * 100, 2)

print(acc_decisiontree)
# Random Forest



from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier(max_depth=4, n_estimators=600)

scores = cross_val_score(randomforest, X, y, cv=5)

acc_randomforest = round(scores.mean() * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, X, y, cv=5)

acc_knn = round(scores.mean() * 100, 2)

print(acc_knn)
# Stochastic Gradient Descent



from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

scores = cross_val_score(sgd, X, y, cv=5)

acc_sgd = round(scores.mean() * 100, 2)

print(acc_sgd)
# Gradient Boosting Classifier



from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier(n_estimators=500, learning_rate=0.11)

scores = cross_val_score(gbk, X, y, cv=5)

acc_gbk = round(scores.mean() * 100, 2)

print(acc_gbk)
# Comparison of Models



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, acc_randomforest, acc_gaussian, acc_perceptron, 

              acc_linear_svc, acc_decisiontree, acc_sgd, acc_gbk]})

models = models.sort_values(by='Score', ascending=False)



print(models)
randomforest.fit(X, y)

dfFit = pd.DataFrame(randomforest.feature_importances_, train_data.drop(['Survived'], axis = 1).columns, 

                     columns=['Coefficient']).sort_values('Coefficient') 

dfFit.sort_values(by='Coefficient', ascending=False)
# print classification report



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

randomforest = RandomForestClassifier(max_depth=4, n_estimators=600)

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_test)



print(classification_report(y_test,y_pred))
# plot confusion matrix



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test,y_pred)

cmNorm = [[cm[0,0]/(cm[0,0]+cm[0,1]), cm[0,1]/(cm[0,0]+cm[0,1])],

         [cm[1,0]/(cm[1,0]+cm[1,1]), cm[1,1]/(cm[1,0]+cm[1,1])]]

df_cm = pd.DataFrame(cmNorm, index=['Real True', 'Real False'], columns=['Predict True', 'Predict False'])

plt.figure(figsize = (6,3))

plt.title("Normalized Confusion Matrix")

sns.heatmap(df_cm, annot=True, vmin=0, vmax=1, cmap='binary', fmt = ".3f")

plt.show()
# plot roc_curve and auc



from sklearn.metrics import roc_curve, auc



y_pred_proba = randomforest.predict_proba(X_test)

y_pred_proba = y_pred_proba[:, 1] 

FPR, TPR, _ = roc_curve(y_test, y_pred_proba)

ROC_AUC = auc(FPR, TPR)

print ("Area Under ROC Curve (AUC):", ROC_AUC)



plt.figure(figsize =[8,7])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Titanic survivors', fontsize= 18)

plt.show()
# plot precision_recall_curve



from sklearn.metrics import precision_recall_curve



precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

PR_AUC = auc(recall, precision)



plt.figure(figsize=[8,7])

plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.xlabel('Recall', fontsize=18)

plt.ylabel('Precision', fontsize=18)

plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
X_test = pd.get_dummies(test_data, columns=['Embarked', 'Title'], drop_first=True)



randomforest = RandomForestClassifier(max_depth=4, n_estimators=600)

randomforest.fit(X, y)

predictions = randomforest.predict(X_test)



output = pd.DataFrame({'PassengerId': X_test.index.values, 'Survived': predictions})

output.to_csv('submission.csv', index=False)

print("Submission successfully")