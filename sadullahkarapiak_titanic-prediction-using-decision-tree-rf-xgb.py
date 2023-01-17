import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
sns.set() #make the graphs prettier
train_data=pd.read_csv('../input/titanic/train.csv')
test_data=pd.read_csv('../input/titanic/test.csv')
frames=[train_data,test_data]
full_data=pd.concat(frames, sort=True)
full_data.reindex()
#full_data = full_data.loc[:,~full_data.columns.duplicated()]
full_data
train_data
test_data
train_data.describe()
train_data.isnull().sum()
train_data.groupby('Embarked').count()
#print(train_data['Sex'].unique()())
print(train_data[['Sex', 'Survived']].groupby('Sex').count())
print(train_data[['Pclass', 'Survived']].groupby('Pclass').count())
print(train_data[['Fare', 'Survived']].groupby('Fare').count())
print(train_data[['Sex','Pclass', 'Survived']].groupby('Survived').count())
train_data[['Fare','Pclass']]
sns.countplot(train_data['Survived'])
corr=train_data.corr()
corr
fig, ax = plt.subplots()
#fig.set_size_inches(15, 10)
sns.heatmap(corr,cmap='coolwarm',annot=True,linewidths=2)
sns.scatterplot(x=train_data['Pclass'], y=train_data['Survived'])
pd.value_counts(train_data['Fare'])
train_title=[]
for i in train_data['Name'].str.split(','):
    train_title.append(i[1].split('.')[0].lstrip())
test_title=[]
for i in test_data['Name'].str.split(','):
    test_title.append(i[1].split('.')[0].lstrip())
full_title=[]
for i in full_data['Name'].str.split(','):
    full_title.append(i[1].split('.')[0].lstrip())

train_data['Title']=train_title
test_data['Title']=test_title
full_data['Title']=full_title
new_index=[i for i in range(len(full_data))]
full_data.index= new_index
#Age filli
train_data["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data["Fare"].fillna(full_data.groupby("Pclass")["Fare"].transform("mean"), inplace=True)
train_data.isnull().sum()
train_data["Embarked"] = train_data["Embarked"].astype('category')
train_data["Embarked"] = train_data["Embarked"].cat.codes


test_data["Embarked"] = test_data["Embarked"].astype('category')
test_data["Embarked"] = test_data["Embarked"].cat.codes

full_data["Embarked"] = full_data["Embarked"].astype('category')
full_data["Embarked"] = full_data["Embarked"].cat.codes
#Embarked Imputation
train_data= train_data.dropna(axis=0, subset=['Embarked'])
test_data= test_data.dropna(axis=0, subset=['Embarked'])
cols_to_transform=['Pclass']
train_data = pd.get_dummies( train_data, columns = cols_to_transform)
test_data = pd.get_dummies( test_data, columns = cols_to_transform)
'''
train_data["Pclass"] = train_data["Pclass"].astype('category')
train_data["Pclass"] = train_data["Pclass"].cat.codes


test_data["Pclass"] = test_data["Pclass"].astype('category')
test_data["Pclass"] = test_data["Pclass"].cat.codes

full_data["Pclass"] = full_data["Pclass"].astype('category')
full_data["Pclass"] = full_data["Pclass"].cat.codes'''
train_data
train_data["Sex"] = train_data["Sex"].astype('category')
train_data["Sex"] = train_data["Sex"].cat.codes

test_data["Sex"] = test_data["Sex"].astype('category')
test_data["Sex"] = test_data["Sex"].cat.codes

#data["Sex"] = data["Sex"].cat.codes
pd.value_counts(train_data['Sex'])
train_data["Cabin"] = train_data["Cabin"].astype('category')
train_data["Cabin"] = train_data["Cabin"].cat.codes

test_data["Cabin"] = test_data["Cabin"].astype('category')
test_data["Cabin"] = test_data["Cabin"].cat.codes

train_data['Cabin'].unique()
train_data
X = train_data[train_data.columns.difference(['Survived','Name','Ticket','Title','PassengerId'])]## Select all columns except 'Survived','Name','Ticket','Title'
print(X.shape)
y = train_data['Survived']
print(y.shape)
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('y_test',y_test.shape)
X_train.head()
X_train.to_csv('Xtrain.csv')
y_train.to_csv('ytrain.csv')
from sklearn import tree
dt1_gini = tree.DecisionTreeClassifier()   ## Instantiating DecisionTree-Classifier
dt1_gini.fit(X_train, y_train)             ## Training Model
print('Train Accuracy =',dt1_gini.score(X_train, y_train))
print('Test Accuracy =',dt1_gini.score(X_test, y_test))
from sklearn.metrics import confusion_matrix


print("Train - Confusion Matrix")
print(confusion_matrix(y_train,dt1_gini.predict(X_train)))

print("Test - Confusion Matrix")
print(confusion_matrix(y_test,dt1_gini.predict(X_test)))
features = X_train.columns
importances = dt1_gini.feature_importances_
indices = np.argsort(importances)[::-1]
pd.DataFrame([X_train.columns[indices],np.sort(importances)[::-1]])
importances
fig, ax = plt.subplots()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='black')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
test_data1=test_data[test_data.columns.difference(['Name','Ticket','Title','PassengerId'])]
test_data1.isnull().sum()
predicted_PID=list(zip(test_data['PassengerId'].tolist(),dt1_gini.predict(test_data1)))
passenger=[]
for i in test_data['PassengerId']:
    for j in predicted_PID:
        if i==j[0]:
            passenger.append(j[1])
test_data['Survived']=passenger
test_data[['PassengerId','Survived']]
dt2_entropy = tree.DecisionTreeClassifier(criterion='entropy',max_depth =5)
dt2_entropy.fit(X_train, y_train)
print('Train Accuracy =',dt2_entropy.score(X_train, y_train))
print('Test Accuracy =',dt2_entropy.score(X_test, y_test))
dt3_fraction15 = tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=15)
dt3_fraction15.fit(X_train, y_train)
print('Train Accuracy =',dt3_fraction15.score(X_train, y_train))
print('Test Accuracy =',dt3_fraction15.score(X_test, y_test))
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree

dt = tree.DecisionTreeClassifier() 

param_grid = {'criterion':['gini','entropy'],
             'max_leaf_nodes': np.arange(5,30,1),
             'max_depth':np.arange(3,15,1),
             }


rsearch_acc = RandomizedSearchCV(estimator=dt, param_distributions=param_grid,n_iter=500)
rsearch_acc.fit(X_train, y_train)

print(rsearch_acc.best_estimator_)
print('Train Accuracy =',rsearch_acc.best_score_)
print('Test Accuracy =',rsearch_acc.score(X_test, y_test))

print("Train - Confusion Matrix")
print(confusion_matrix(y_train,rsearch_acc.predict(X_train)))
print("Test - Confusion Matrix")
print(confusion_matrix(y_test,rsearch_acc.predict(X_test)))

predicted_PID=list(zip(test_data['PassengerId'].tolist(),dt3_fraction15.predict(test_data1)))

passenger=[]
for i in test_data['PassengerId']:
    for j in predicted_PID:
        if i==j[0]:
            passenger.append(j[1])
test_data['Survived_entropy15']=passenger

test_data[['PassengerId','Survived_entropy15']].to_csv('Predict_Top_Accuracy.csv')
sns.countplot(test_data['Survived'])
test_data.plot.hist(subplots=True,figsize=(15, 15), bins=20, by='Survived')
from sklearn.tree import export_graphviz
export_graphviz(dt3_fraction15, out_file='tree_limited.dot', feature_names = X_test.columns,
                class_names = dt3_fraction15.predict(X_train).astype(str),
                rounded = True, proportion = False, precision = 2, filled = True)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
from IPython.display import Image
Image(filename = 'tree_limited.png')
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
## Predict
train_predictions = rfc.predict(X_train)
test_predictions = rfc.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score


print("TRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=0))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=1))

### Test data accuracy
print("\n\n--------------------------------------\n\n")
print("TEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=0))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=1))

predicted_PID=list(zip(test_data['PassengerId'].tolist(),rfc.predict(test_data1)))

passenger=[]
for i in test_data['PassengerId']:
    for j in predicted_PID:
        if i==j[0]:
            passenger.append(j[1])
test_data['Survived_random_forest']=passenger

test_data[['PassengerId','Survived_random_forest']].to_csv('Prediction_rf_81.csv',index=False)
rfc.feature_importances_
feat_importances = pd.Series(rfc.feature_importances_, index = X_train.columns)
feat_importances.plot(kind='bar')
feat_importances_ordered = feat_importances.nlargest(10)
feat_importances_ordered.plot(kind='bar')
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

## n_jobs = -1 uses all cores of processor
## max_features is the maximum number of attributes to select for each tree
rfc_grid = RandomForestClassifier(n_jobs=-1, max_features='sqrt', class_weight='balanced_subsample')
 
# Use a grid over parameters of interest
## n_estimators is the number of trees in the forest
## max_depth is how deep each tree can be
## min_sample_leaf is the minimum samples required in each leaf node for the root node to split
## "A node will only be split if in each of it's leaf nodes there should be min_sample_leaf"

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}
 
rfc_cv_grid = RandomizedSearchCV(estimator = rfc_grid, 
                                 param_distributions = param_grid, 
                                 cv = 3, n_iter=10)
rfc_cv_grid.fit(X_train, y_train)
rfc_cv_grid.best_params_
#rfc_cv_grid.best_estimator_
## Predict
train_predictions = rfc_cv_grid.predict(X_train)
test_predictions = rfc_cv_grid.predict(X_test)

print("TRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=0))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=1))

### Test data accuracy
print("\n\n--------------------------------------\n\n")
print("TEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=0))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=1))
# Feature Impotances
rfc_cv_grid.best_estimator_.feature_importances_
## Get important Features
feat_importances = pd.Series(rfc_cv_grid.best_estimator_.feature_importances_, index = X_train.columns)
feat_importances_ordered = feat_importances.nlargest(10)
feat_importances_ordered.plot(kind='bar')

predicted_PID=list(zip(test_data['PassengerId'].tolist(),rfc.predict(test_data1)))

passenger=[]
for i in test_data['PassengerId']:
    for j in predicted_PID:
        if i==j[0]:
            passenger.append(j[1])
test_data['Survived_random_forest_grid']=passenger

test_data[['PassengerId','Survived_random_forest_grid']].to_csv('Prediction_rfg_80.csv',index=False)
from xgboost.sklearn import XGBClassifier

xgb_grid = XGBClassifier()
 

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
 
xgb_cv_grid = RandomizedSearchCV(estimator = xgb_grid, 
                                 param_distributions = params, 
                                 cv = 4, n_iter=10)
xgb_cv_grid.fit(X_train, y_train)
## Predict
train_predictions = xgb_cv_grid.predict(X_train)
test_predictions = xgb_cv_grid.predict(X_test)

print("TRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=0))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=1))

### Test data accuracy
print("\n\n--------------------------------------\n\n")
print("TEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=0))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=1))

predicted_PID=list(zip(test_data['PassengerId'].tolist(),xgb_cv_grid.predict(test_data1)))

passenger=[]
for i in test_data['PassengerId']:
    for j in predicted_PID:
        if i==j[0]:
            passenger.append(j[1])
test_data['Survived_random_xgb']=passenger

test_data[['PassengerId','Survived_random_xgb']].to_csv('Prediction_xgb_817.csv',index=False)
# Feature Impotances
xgb_cv_grid.best_estimator_.feature_importances_
## Get important Features
feat_importances = pd.Series(xgb_cv_grid.best_estimator_.feature_importances_, index = X_train.columns)
feat_importances_ordered = feat_importances.nlargest(10)
feat_importances_ordered.plot(kind='bar')