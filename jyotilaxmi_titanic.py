import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas_profiling
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from os import system

plt.style.use('ggplot')
pd.options.display.float_format = '{:,.2f}'.format
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
titanic_df = pd.read_csv("train.csv")
titanic_df.head(20)
titanic_df.shape
titanic_df.dtypes
titanic_df.isnull().values.any()
titanic_df.isnull().sum()
titanic_df.drop(['Cabin'], axis = 1, inplace=True)
titanic_df.isnull().sum()
titanic_df.loc[titanic_df['Embarked'] == 'C']['Embarked'].count()
titanic_df.loc[titanic_df['Embarked'] == 'S']['Embarked'].count()
titanic_df.loc[titanic_df['Embarked'] == 'Q']['Embarked'].count()
titanic_df['Embarked'] = titanic_df['Embarked'].replace(np.nan,'S')
titanic_df.isnull().sum()
result = titanic_df.loc[(titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 0)]['Age'].mean()
print(result)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 0) & (titanic_df['Age'].isnull()) 
                             , 32, titanic_df['Age'])
titanic_df.isnull().sum()
result1 = titanic_df.loc[(titanic_df['Age'].isnull())][['Age','SibSp','Parch']]
print(result1)
result = titanic_df.loc[(titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 0)]['Age'].mean()
print(result)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 0) & (titanic_df['Age'].isnull()) 
                             , 32, titanic_df['Age'])
titanic_df.isnull().sum()
result = titanic_df.loc[(titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 1)]['Age'].mean()
print(result)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 1) & (titanic_df['Age'].isnull()) 
                             , 27, titanic_df['Age'])
titanic_df.isnull().sum()
result1 = titanic_df.loc[(titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 2)]['Age'].mean()
print(result1)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 1) & (titanic_df['Parch'] == 2) & (titanic_df['Age'].isnull()) 
                             , 20, titanic_df['Age'])
result2 = titanic_df.loc[(titanic_df['SibSp'] == 2) & (titanic_df['Parch'] == 0)]['Age'].mean()
print(result2)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 2) & (titanic_df['Parch'] == 0) & (titanic_df['Age'].isnull()) 
                             , result2, titanic_df['Age'])
result3 = titanic_df.loc[(titanic_df['SibSp'] == 3) & (titanic_df['Parch'] == 1)]['Age'].mean()
print(result3)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 3) & (titanic_df['Parch'] == 1) & (titanic_df['Age'].isnull()) 
                             , 4, titanic_df['Age'])
result3 = titanic_df.loc[(titanic_df['SibSp'] == 8) & (titanic_df['Parch'] == 2)]['Age'].mean()
print(result3)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 8) & (titanic_df['Parch'] == 2) & (titanic_df['Age'].isnull()) 
                             , titanic_df.loc[(titanic_df['SibSp'] == 8) & (titanic_df['Parch'] == 2)]['Age'].mean()
                             , titanic_df['Age'])
titanic_df.isnull().sum()
df = titanic_df.loc[(titanic_df['Age'].isnull())][[
'PassengerId', 'Survived' ,'Pclass','Name','Sex','Age','SibSp','Parch','Ticket']]
print(df)
result3 = titanic_df.loc[(titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 1)]['Age'].mean()
print(result3)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 1) & (titanic_df['Age'].isnull()) 
                             , titanic_df.loc[(titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 1)]['Age'].mean()
                             , titanic_df['Age'])
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 2) & (titanic_df['Age'].isnull()) 
                             , titanic_df.loc[(titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 2)]['Age'].mean()
                             , titanic_df['Age'])
resultmean = titanic_df['Age'].mean()
print(resultmean)
titanic_df['Age'] = np.where((titanic_df['SibSp'] == 8) & (titanic_df['Parch'] == 2) & (titanic_df['Age'].isnull()) 
                             , 29
                             , titanic_df['Age'])
titanic_df.isnull().sum()
titanic_df.head(20)
sns.distplot(titanic_df['Age'])
plt.show()
sns.distplot(titanic_df['Fare'])
plt.show()
plt.figure(figsize=(16,9)) 
ax=sns.countplot(x=titanic_df['Survived'],hue=titanic_df['Sex']) 
plt.show()
sns.pairplot(
    titanic_df,
    x_vars=["PassengerId", "Survived", "Pclass", "Age","SibSp","Parch","Fare"],
    y_vars=["PassengerId","Survived","Pclass","Age","SibSp","Parch"],
    hue="Sex"
)
titanic_df.drop(['Name','Ticket'], inplace=True, axis=1)
titanic_df.head(20)
titanic_df['Sex'] = titanic_df['Sex'].map({'female':1, 'male':0})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'S':1, 'C':2, 'Q':3})
titanic_df.head(20)
titanic_df.describe().transpose()
X = titanic_df.drop("Survived" , axis=1)
y = titanic_df["Survived"] 
titanic_df_test = pd.read_csv("test.csv")
titanic_df_test.head(20)
titanic_df_test.shape
titanic_df.shape
titanic_df_test.isnull().any()
titanic_df_test.drop(['Ticket'], axis = 1, inplace=True)
titanic_df_test['Fare'].fillna((titanic_df_test['Fare'].mean()), inplace=True)
result1 = titanic_df_test.loc[(titanic_df_test['Age'].isnull())][['Age','SibSp','Parch']]
print(result1)
res_test = titanic_df_test.loc[(titanic_df['SibSp'] == 0) & (titanic_df['Parch'] == 0)]['Age'].mean()
print(res_test)
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 0) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 0)]['Age'].mean()
                             , titanic_df_test['Age'])
titanic_df_test.isnull().sum()
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 4) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 0)]['Age'].mean()
                             , titanic_df_test['Age'])
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 2) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 0) & (titanic_df_test['Parch'] == 0)]['Age'].mean()
                             , titanic_df_test['Age'])
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 8) & (titanic_df_test['Parch'] == 2) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 8) & (titanic_df_test['Parch'] == 2)]['Age'].mean()
                             , titanic_df_test['Age'])
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 2) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 2)]['Age'].mean()
                             , titanic_df_test['Age'])
result1 = titanic_df_test.loc[(titanic_df_test['Age'].isnull())][['Age','SibSp','Parch']]
print(result1)
titanic_df_test['Age'] = np.where((titanic_df_test['SibSp'] == 1) & (titanic_df_test['Age'].isnull()) 
                             , titanic_df_test.loc[(titanic_df_test['SibSp'] == 1)]['Age'].mean()
                             , titanic_df_test['Age'])
titanic_df_test.head(20)
titanic_df.head(20)
titanic_df_test.drop(['Name'], axis = 1, inplace=True)
titanic_df_test.head(20)
titanic_df_test['Sex'] = titanic_df_test['Sex'].map({'female':1, 'male':0})
titanic_df_test['Embarked'] = titanic_df_test['Embarked'].map({'S':1, 'C':2, 'Q':3})
titanic_df_test.head(20)
X_test = titanic_df_test
from sklearn.model_selection import train_test_split
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train.shape,X_test.shape,X_test_1.shape
algo= []
tr = []
te = []
recall = []
precision = []
roc = []
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score
model = LogisticRegression(random_state=7)

model.fit(X_train, y_train)

algo.append('Logistic Regression')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test_1, y_test_1))
recall.append(recall_score(y_test_1,model.predict(X_test_1)))
precision.append(precision_score(y_test_1,model.predict(X_test_1)))
roc.append(roc_auc_score(y_test_1,model.predict(X_test_1)))
from sklearn.tree import DecisionTreeClassifier
#instantiating decision tree as the default model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
#y_pred=dt_model.predict(X_test)
#training acuracy
dt_model.score(X_train, y_train)
#testing acuracy
dt_model.score(X_test_1, y_test_1)
clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 7, max_depth=3, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)
#training acuracy
clf_pruned.score(X_train, y_train)
#testing acuracy
clf_pruned.score(X_test_1, y_test_1)
## Calculating feature importance
feature_cols = X_train.columns

feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)


feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.sort_values(by=0, ascending=False)[0:10] #Top 10 features
preds_pruned = clf_pruned.predict(X_test_1)
preds_pruned_train = clf_pruned.predict(X_train)

print("Training Accuracy:",accuracy_score(y_train, preds_pruned_train))
print()
print("Training Accuracy:",accuracy_score(y_test_1, preds_pruned))
print()
print("Recall:",recall_score(y_test_1, preds_pruned, average="binary", pos_label=1))
# Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=7, n_estimators=50)

model.fit(X_train, y_train)

algo.append('Random Forest')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test_1, y_test_1))
recall.append(recall_score(y_test_1,model.predict(X_test_1)))
precision.append(precision_score(y_test_1,model.predict(X_test_1)))
roc.append(roc_auc_score(y_test_1,model.predict(X_test_1)))
# Bagging
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(random_state=7,n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True)

model.fit(X_train, y_train)

algo.append('Bagging')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test_1, y_test_1))
recall.append(recall_score(y_test_1,model.predict(X_test_1)))
precision.append(precision_score(y_test_1,model.predict(X_test_1)))
roc.append(roc_auc_score(y_test_1,model.predict(X_test_1)))
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=7,n_estimators= 200, learning_rate=0.1)

model.fit(X_train, y_train)

algo.append('AdaBoost')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test_1, y_test_1))
recall.append(recall_score(y_test_1,model.predict(X_test_1)))
precision.append(precision_score(y_test_1,model.predict(X_test_1)))
roc.append(roc_auc_score(y_test_1,model.predict(X_test_1)))
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=7, n_estimators=200,)

model.fit(X_train, y_train)

algo.append('Gradient Boosting')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test_1, y_test_1))
recall.append(recall_score(y_test_1,model.predict(X_test_1)))
precision.append(precision_score(y_test_1,model.predict(X_test_1)))
roc.append(roc_auc_score(y_test_1,model.predict(X_test_1)))
# DataFrame to compare results.

results = pd.DataFrame()
results['Model'] = algo
results['Training Score'] = tr
results['Testing Score'] = te
results['Recall'] = recall
results['Precision'] = precision
results['ROC AUC Score'] = roc
#results = results.set_index('Model')
results
y_test = model.predict(X_test)
print(y_test)
titanic_df_test['Survived'] = y_test
titanic_df_test.shape
df = titanic_df_test[['PassengerId','Survived']]
df.shape
df.to_csv('titanic-pred.csv',index=False)
