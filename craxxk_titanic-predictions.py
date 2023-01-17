import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt

path = '../input/train.csv'
data = pd.read_csv(path)
data =data.drop(columns=['Ticket', 'Cabin'])
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_data = test_data.drop(columns=['Ticket', 'Cabin'])
data.head()
data.info()
test_data.info()
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

data['Fare'].fillna(data['Fare'].median(), inplace = True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)

data['Age'].fillna(data['Age'].median(), inplace = True)
test_data['Age'].fillna(test_data['Age'].median(), inplace = True)
data.info()
test_data.info()
#Combine train and test data
all_data = [data, test_data]
#Create a new column representing family size
for dataset in all_data:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
import re
def get_title(name):
    title_search = re.search(' ([a-zA-Z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

#Create a new feature 'Title'
for row in all_data:
    row['Title'] = row['Name'].apply(get_title)

#Group all non-common titles in one named "Rare"

for row in all_data:
    row['Title'] = row['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    row['Title'] = row['Title'].replace('Mlle', 'Miss')
    row['Title'] = row['Title'].replace('Ms', 'Miss')
    row['Title'] = row['Title'].replace('Mme', 'Miss')

data['Title'].value_counts().plot.bar()
data['Age'].sample(100).value_counts().sort_index().plot.bar()
#Create a bin for all ages
for row in all_data:
    row['Age_bin'] = pd.cut(row['Age'], bins = [0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
data['Age_bin'].value_counts().plot.bar()
#Create a bin for all ages
for row in all_data:
    row['Fare_bin'] = pd.cut(row['Fare'], bins = [0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])
data['Fare_bin'].value_counts().plot.bar()
for row in all_data:
    drop_column = ['Age', 'Fare', 'Name']
    row.drop(drop_column, axis = 1, inplace=True)
data.drop('PassengerId', axis = 1, inplace=True)
testdata_Id = test_data['PassengerId']
test_data.drop('PassengerId', axis = 1, inplace=True)
data.head()
data = pd.get_dummies(data, columns = ["Sex","Title","Age_bin","Fare_bin", "Embarked"],
                             prefix=["Sex","Title","Age_type","Fare_type", "Embarked"])
test_data = pd.get_dummies(test_data, columns = ["Sex","Title","Age_bin","Fare_bin", "Embarked"],
                             prefix=["Sex","Title","Age_type","Fare_type", "Embarked"])
data.head()

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
#data.corr()-->corellation matrix
fig = plt.gcf()
fig.set_size_inches(20, 12)
plt.show
X = data.drop("Survived",axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

init_model = SVC()
param_grid = {'kernel': ['rbf','linear'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

modelsvm = GridSearchCV(init_model,param_grid = param_grid, cv=10, scoring="accuracy", n_jobs= -1, verbose = 1)

modelsvm.fit(X_train,y_train)

print(modelsvm.best_estimator_)

# Best score
print(modelsvm.best_score_)
model = modelsvm.best_estimator_
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is : ',round(accuracy_score(predictions,y_test)*100,2))
kfold = KFold(n_splits=10,shuffle = True, random_state=22) # k=10, split the data into 10 equal parts
scores = cross_val_score(model, X, y, cv=kfold, scoring = "accuracy")
print('The cross validated score for Random Forest Classifier is:',round(scores.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=kfold)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)


# # make predictions which we will submit. 
test_preds = model.predict(test_data)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'PassengerId': testdata_Id,
                       'Survived': test_preds})

output.to_csv('submission7.csv', index=False)

output