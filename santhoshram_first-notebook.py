# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # data visualization
%matplotlib inline
import seaborn as sns # data visualization
sns.set()
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.describe()
# Show null values count for each column in train data
train_df.isnull().sum()
# Show null values count for each column in test data
test_df.isnull().sum()
train_df.sample(10)
survived = train_df[train_df.Survived == 1]
not_survived = train_df[train_df.Survived == 0]
print('Survived: %i (%.1f%%)' %(len(survived), float(100*len(survived)/(len(survived)+len(not_survived)))))
print('Did not Survive: %i (%.1f%%)' %(len(not_survived), float(100*len(not_survived)/(len(survived)+len(not_survived)))))
test_df.sample(10)
cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(train_df[cols[i]], hue=train_df["Survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend() 
        
plt.tight_layout() 
bins = np.arange(0, 80, 5)
g = sns.FacetGrid(train_df, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()
null_columns = train_df.columns[train_df.isnull().any()]
print(null_columns)
#pd.options.display.mpl_style = 'default'
labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(train_df[col].isnull().sum())
print(np.arange(len(labels)))
print(values)
print(labels)
ind = np.arange(len(labels))
width=1
fig, ax = plt.subplots(figsize=(6,5))
#ax.plot(ind,np.array(values))
rects = ax.barh( ind, np.array(values), color='green')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");
sns.barplot(ind,values)
print("Count of sex")
print(train_df.Sex.value_counts())
# Prints the number of male and female
print("Count of sex who survived and not")
print(train_df.groupby('Sex').Survived.value_counts())
print("Mean of sex who survived and not")
print(train_df.groupby('Sex').Survived.mean())
sns.barplot(x="Sex", y="Survived", data=train_df)
sns.barplot(x="Pclass", y="Survived", data=train_df)
train_df.sample(5)
sns.barplot(x='Embarked', y= 'Survived', data= train_df)
train_df.Age.describe()
age = train_df.Age.dropna()
sns.distplot(age, bins = 25, kde = False)
train = train_df[~train_df.Age.isnull()]
train.head()
train = train_df[~train_df.Age.isnull()]
train['AgeBand'] = np.where(train.Age <= 16, 'Children', 
                            np.where((train.Age > 16) & (train.Age <= 32), 'Young Adult', 
                                     np.where((train.Age > 32) & (train.Age <= 48), 'Adult', 
                                              np.where((train.Age > 48) & (train.Age <= 64), 'Senior Adult', 
                                                      np.where((train.Age > 64) & (train.Age <= 80), 'Senior', False)))))
print(train.AgeBand.value_counts())
print(train.groupby('AgeBand').Survived.value_counts())
sns.barplot(x='AgeBand', y='Survived', data=train)
bins = [-1, 0, 5, 16, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
#test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
train.head()
sns.barplot(x='AgeGroup', y='Survived', data=train)
from sklearn.impute import SimpleImputer

train['FamilySize'] = train.SibSp + train.Parch
train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)

train.sample(5)
train["IsCabin"] = train_df["Cabin"].notnull().astype('int')
#calculate percentages of IsCabin vs. survived
print("Percentage of Cabin holders who survived:", train["Survived"][train["IsCabin"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of non cabiners who survived:", train["Survived"][train["IsCabin"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="IsCabin", y="Survived", data=train)
plt.show()
cm_surv = ["darkgrey" , "lightgreen"]
fig, ax = plt.subplots(figsize=(13,7))
sns.swarmplot(x='Pclass', y='Age', hue='Survived', split=True, data=train_df , palette=cm_surv, size=7, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

train = train.drop(cols_to_drop, axis=1)
X_test = test_df.drop(cols_to_drop, axis=1)

train_data = pd.get_dummies(train)
train_data.head()

X_test = pd.get_dummies(X_test)

X_train = train_data.drop('Survived', axis=1)
y_train = train_data.Survived

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)
print(X_test)
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, random_state = 0)

my_svm_model = svm.SVC(kernel='linear')
my_svm_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_svm_model, X_train, y_train, cv=kfold)
print("SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_forest_model = RandomForestClassifier(n_estimators=50)
my_forest_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_forest_model, X_train, y_train, cv=kfold)
print("Random Forest Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_knn_model = KNeighborsClassifier(n_neighbors=4)
my_knn_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_knn_model, X_train, y_train, cv=kfold)
print("Knn Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_gnb_model = GaussianNB()
my_gnb_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_gnb_model, X_train, y_train, cv=kfold)
print("GNB Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_logit_model = LogisticRegression()
my_logit_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_logit_model, X_train, y_train, cv=kfold)
print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_xgb_model = XGBClassifier(n_estimators = 1000, learning_rate = 0.65)
my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_xgb_model, X_train, y_train, cv=kfold)
print("XGBTree Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(X_test)
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['FamilySize'] = train.SibSp + train.Parch
train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)

cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

train.Pclass = train.Pclass.astype(str)
train = train.drop(cols_to_drop, axis=1)
test.Pclass = test.Pclass.astype(str)
X_test = test.drop(cols_to_drop, axis=1).copy()

X_test['FamilySize'] = X_test.SibSp + X_test.Parch
X_test['logFare'] = np.where(X_test.Fare != 0, np.log(X_test.Fare), X_test.Fare)

train_data = pd.get_dummies(train)
X_test = pd.get_dummies(X_test)

X_train = train_data.drop('Survived', axis=1)
y_train = train_data.Survived

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.25, random_state = 0)

my_xgb_model = XGBClassifier(n_estimators = 150, 
                             learning_rate = 0.05, 
                             max_depth = 3)
my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_xgb_model, X_train, y_train, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

my_predictions = my_xgb_model.predict(X_test)

jcleme_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": my_predictions})

jcleme_submission.to_csv('new_jcleme_xgb_submission.csv', index = False)

print(jcleme_submission)
