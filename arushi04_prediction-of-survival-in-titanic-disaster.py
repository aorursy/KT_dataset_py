import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

print("train data shape : ", train_data.shape)
print("test data shape : ", test_data.shape)
train_data.head()
train_data.info()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
# Combining train and test data to do feature engineering

train_test = [train_data, test_data]

for data in train_test:
    data['Title'] = data['Name'].str.split(', ').str[1].str.split('.').str[0]
train_data['Title'].value_counts()
# We mainly want the first three titles for our analysis as they are in majority

for dataset in train_test:
    top_3 = [x for x in dataset.Title.value_counts().sort_values(ascending=False).head(3).index]
    for label in top_3:
        dataset[label] = np.where(dataset['Title']==label,1,0)
train_data.head()
# Function to delete unnecessary feature from dataset

def drop_columns(df, col):
    df.drop(col, axis=1, inplace=True)
for dataset in train_test:
    drop_columns(dataset, 'Name')
    drop_columns(dataset, 'Title')
train_data.head()
#Converting and concatenating sex to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Sex'], prefix='gender')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Sex'], prefix='gender')],axis=1)

train_test = [train_data, test_data]
for dataset in train_test:
    drop_columns(dataset, 'Sex')
    drop_columns(dataset, 'gender_male')
test_data.head()
# Replacing missing values with the median age grouped by title : 177, 86

train_data['Age'].fillna(train_data.groupby("Mr")["Age"].transform("median"), inplace=True)
train_data['Age'].fillna(train_data.groupby("Mrs")["Age"].transform("median"), inplace=True)
train_data['Age'].fillna(train_data.groupby("Miss")["Age"].transform("median"), inplace=True)

test_data['Age'].fillna(test_data.groupby("Mr")["Age"].transform("median"), inplace=True)
test_data['Age'].fillna(test_data.groupby("Mrs")["Age"].transform("median"), inplace=True)
test_data['Age'].fillna(test_data.groupby("Miss")["Age"].transform("median"), inplace=True)

Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

# Based on above observation , we can conveniently replace the missing values of embarked with S.

train_data['Embarked'] = train_data['Embarked'].fillna('S')
#Converting and concatenating embarked to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Embarked'], prefix='em')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Embarked'], prefix='em')],axis=1)

drop_columns(train_data, 'em_Q')
drop_columns(test_data, 'em_Q')
drop_columns(train_data, 'Embarked')
drop_columns(test_data, 'Embarked')
test_data.head()
# replacing missing Fare with median fare for each Pclass
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train_data.Cabin.value_counts()
#Getting the first alphabet of each cabin
train_test = [train_data, test_data]
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train_data[train_data['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
train_data['Cabin'].value_counts()
#Dropping cabin. Need to decide how to replace NaN values

drop_columns(train_data, 'Cabin')
drop_columns(test_data, 'Cabin')
#Converting and concatenating Pclass to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Pclass'], prefix='class')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Pclass'], prefix='class')],axis=1)
drop_columns(train_data, 'Pclass')
drop_columns(test_data, 'Pclass')
train_data.head()
#Adding all the parents, children, spouse and siblings to count the no of members in the family on board

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1

#Dropping the unnecessary features

features_drop = ['Ticket', 'SibSp', 'Parch']
train_data = train_data.drop(features_drop, axis=1)
test_data = test_data.drop(features_drop, axis=1)
train_data = train_data.drop('PassengerId', axis=1)
train_data.head()
#Checking corelation matrix

train_data.corr()
#Segregating features and label

y = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)
train_data.head()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt



#Baseline Model : Probability of not surviving which is the majority class

Survival_prob = (y==0).sum() / len(train_data)
Survival_prob
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model = GaussianNB()
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
model = tree.DecisionTreeClassifier(random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
model = KNeighborsClassifier(n_neighbors = 13)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
model = svm.SVC(kernel='linear', random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
model = LogisticRegression(random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)
print(train_data.shape)
print(test.shape)
print(test_data.shape)
model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
model.fit(train_data, y)

test = test_data.drop("PassengerId", axis=1).copy()

prediction = model.predict(test)
print(test)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
submission = pd.read_csv('submission.csv')
submission.head()
#Rough

conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)
train_accuracy = accuracy_score(y.to_numpy(), y_pred)
print(train_accuracy)

precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
print(precision)
print(recall)


#average_precision = average_precision_score(y, y_score)
#disp = plot_precision_recall_curve(model, X, y)
#disp.ax_.set_title('Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))


