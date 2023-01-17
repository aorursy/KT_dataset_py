import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
gender_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')
train_data = pd.read_csv('../input/titanic/train.csv')
train_data.head()
test_data.head(8)
train_data.shape
train_data.isnull().sum()
train_data.dtypes
object_culumns =train_data.columns[train_data.dtypes==object]
object_culumns
train_data['Sex'].value_counts()
train_data['Cabin'].value_counts()
train_data['Embarked'].value_counts()
categrile_number = {'Embarked':{'S':1,'C':2,'Q':0},'Sex':{'male':0,'female':1}}
train_data.replace(categrile_number, inplace=True)
test_data.replace(categrile_number, inplace=True)
train_data['Embarked'].value_counts()
train_data['Cabin'].unique()
train_data['Ticket'].unique()
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
fig1 = train_data.boxplot(column='Age')
fig1.set_title('Age')

plt.subplot(1,2,2)
fig2 = train_data.boxplot(column='Fare')
fig2.set_title('Fare')
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
fig1 = train_data.Age.hist(bins=30)
fig1.set_xlabel('Age')
fig1.set_ylabel('number of passengers')


plt.subplot(1,2,2)
fig2 = train_data.Fare.hist(bins=30)
fig1.set_xlabel('Fare')
fig1.set_ylabel('number of passengers')
print(train_data.Age.mean() + 3 * train_data.Age.std())
print(train_data.Fare.quantile(0.75) + 3 *(train_data.Fare.quantile(0.75) - train_data.Fare.quantile(0.25)))
train_data['Age'] = np.where(train_data['Age'] > 73, 73, train_data['Age'])
train_data['Fare'] = np.where(train_data['Fare'] > 73, 73, train_data['Fare'])
test_data['Age'] = np.where(test_data['Age'] > 73, 73, test_data['Age'])
test_data['Fare'] = np.where(test_data['Fare'] > 73, 73, test_data['Fare'])
train_data.corr()
train_data['Family'] = train_data['Name'].str.split(',').str[0]
test_data['Family'] = train_data['Name'].str.split(',').str[0]
train_data['Title']=train_data['Name'].str.split(', ').str[1].str.split('.').str[0]
test_data['Title']=test_data['Name'].str.split(', ').str[1].str.split('.').str[0]
train_data['Title'].unique()
train_data['Title'] =train_data['Title'].replace(['Ms','Mlle'], 'Miss')
train_data['Title'] = train_data['Title'].replace(['Mme','Dona','the Countess','Lady'], 'Mrs')
train_data['Title'] =train_data['Title'].replace(['Rev','Mlle','Jonkheer','Dr','Capt','Don','Col','Major','Sir'], 'Mr')

test_data['Title'] =test_data['Title'].replace(['Ms','Mlle'], 'Miss')
test_data['Title'] = test_data['Title'].replace(['Mme','Dona','the Countess','Lady'], 'Mrs')
test_data['Title'] =test_data['Title'].replace(['Rev','Mlle','Jonkheer','Dr','Capt','Don','Col','Major','Sir'], 'Mr')
cleanup_nums = { "Title": {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3 } }
train_data.replace(cleanup_nums, inplace=True)
test_data.replace(cleanup_nums, inplace=True)
train_data['Age'].fillna((train_data['Age'].mean()), inplace=True) 
test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)

bins = [0, 2, 18, 35, 65, np.inf]
names = ['<2', '2-18', '18-35', '35-65', '65+']

train_data['AgeRange'] = pd.cut(train_data['Age'], bins, labels=names)
test_data['AgeRange'] = pd.cut(test_data['Age'], bins, labels=names)

NumberedAgeCategories = {'<2':0 , '2-18':1, '18-35':2, '35-65':3, '65+':4}
train_data['AgeRange']=train_data['AgeRange'].map(NumberedAgeCategories)  
train_data['AgeRange']=pd.to_numeric(train_data['AgeRange'])
test_data['AgeRange']=test_data['AgeRange'].map(NumberedAgeCategories)  
test_data['AgeRange']=pd.to_numeric(test_data['AgeRange'])
train_data
train_data['FamilySize']= train_data['SibSp']+train_data['Parch']+1
test_data['FamilySize']= test_data['SibSp']+test_data['Parch']+1
train_data
cabin_only = train_data[["Cabin"]].copy()
cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x)
cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)
cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only[cabin_only["Cabin_Data"]]
cabin_only
cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
# 用‘N'来代表Deck
cabin_only["Deck"] = cabin_only["Deck"].fillna("N") 
# 用room的平均数表示room空值
cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean()) 
cabin_only=cabin_only.join(pd.get_dummies(cabin_only['Deck'], prefix='Deck'))
cabin_only=cabin_only.drop(['Deck'], axis=1)
train_data=pd.concat([train_data,cabin_only],axis=1)
cabin_only_test = test_data[["Cabin"]].copy()
cabin_only_test["Cabin_Data"] = cabin_only_test["Cabin"].isnull().apply(lambda x: not x) # extract rows that do not contain null Cabin data.
cabin_only_test["Deck"] = cabin_only_test["Cabin"].str.slice(0,1)
cabin_only_test["Room"] = cabin_only_test["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only_test[cabin_only_test["Cabin_Data"]]
cabin_only_test.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
cabin_only_test["Deck"] = cabin_only_test["Deck"].fillna("N") # assign 'N' for the deck name of the null Cabin value. 
cabin_only_test["Room"] = cabin_only_test["Room"].fillna(cabin_only_test["Room"].mean()) # use mean to fill null Room values.
cabin_only_test=cabin_only_test.join(pd.get_dummies(cabin_only_test['Deck'], prefix='Deck'))
cabin_only_test=cabin_only_test.drop(['Deck'], axis=1)
test_data=pd.concat([test_data,cabin_only_test],axis=1)

train_data['Ticket_numerical'] = train_data.Ticket.apply(lambda s: s.split()[-1])
train_data['Ticket_numerical'] = np.where(train_data.Ticket_numerical.str.isdigit(), train_data.Ticket_numerical, np.nan)
train_data['Ticket_numerical'] = train_data['Ticket_numerical'].astype('float')
train_data["Ticket_numerical"] = train_data["Ticket_numerical"].fillna(0) 


test_data['Ticket_numerical'] = test_data.Ticket.apply(lambda s: s.split()[-1])
test_data['Ticket_numerical'] = np.where(test_data.Ticket_numerical.str.isdigit(), test_data.Ticket_numerical, np.nan)
test_data['Ticket_numerical'] = test_data['Ticket_numerical'].astype('float')
test_data["Ticket_numerical"] = test_data["Ticket_numerical"].fillna(0) 


train_data['Ticket_categorical'] = train_data.Ticket.apply(lambda s: s.split()[0])
train_data['Ticket_categorical'] = np.where(train_data.Ticket_categorical.str.isdigit(), np.nan, train_data.Ticket_categorical)
train_data["Ticket_categorical"] = train_data["Ticket_categorical"].fillna("NONE")
train_data['Ticket_numerical'].tolist()
 
test_data['Ticket_categorical'] = test_data.Ticket.apply(lambda s: s.split()[0])
test_data['Ticket_categorical'] = np.where(test_data.Ticket_categorical.str.isdigit(), np.nan, test_data.Ticket_categorical)
test_data["Ticket_categorical"] = test_data["Ticket_categorical"].fillna("NONE")
test_data['Ticket_numerical'].tolist()
    
train_data[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()
train_data.isnull().sum()
# test_data.isnull().sum()
train_data['Fare'].fillna(train_data.Fare.mean,inplace=True)
test_data['Fare'].fillna(train_data.Fare.mean,inplace=True)
train_data = train_data.drop(['Cabin'],axis=1)
test_data = test_data.drop(['Cabin'],axis=1)
train_data = train_data.drop(['Deck_T'],axis=1)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

print(test_data.columns)
print(train_data.columns)
label_encoder = LabelEncoder()
for col in train_data.columns[train_data.dtypes == "object"]:
    train_data[col] = label_encoder.fit_transform(train_data[col].astype('str'))

for col in test_data.columns[test_data.dtypes == "object"]:
    test_data[col] = label_encoder.fit_transform(test_data[col].astype('str'))
    

train_data.dropna(inplace=True)

X = train_data.drop('Survived', axis=1)


y = train_data['Survived']


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
def featureSelection(label):    
    clf = DecisionTreeClassifier(random_state=0)
    if(label=='Decision Tree'):
        clf = DecisionTreeClassifier(random_state=0)
    if(label=='Random Forest'):
        clf = RandomForestClassifier(random_state=0)
    if(label=='XGBoost'):
        clf = XGBClassifier(random_state=0)  
    if(label=='Extra Trees'):
        clf = ExtraTreesClassifier(random_state=0)  
        
    clf= clf.fit(X_train, y_train)
    
    arr= dict(zip(X_train.columns, clf.feature_importances_))
    
    data= pd.DataFrame.from_dict(arr,orient='index', columns=['importance'])
    
    return data.sort_values(['importance'], ascending=False)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,10)) # one row, three columns
r=featureSelection("Decision Tree")
v=featureSelection("Random Forest")
s=featureSelection("XGBoost")
t=featureSelection("Extra Trees")
r.plot.bar(y="importance", rot=70, title="Decision Tree Features with their corresponding importance values",ax=ax1)
v.plot.bar(y="importance", rot=70, title="Random Forest Features with their corresponding importance values", ax=ax2)
s.plot.bar(y="importance", rot=70, title="XGBoost Features with their corresponding importance values", ax=ax3)
t.plot.bar(y="importance", rot=70, title="Extra Trees Features with their corresponding importance values", ax=ax4)
plt.tight_layout() 
logit_model = LogisticRegression(max_iter=10000)
logit_model.fit(X_train, y_train)
 
importance = pd.Series(np.abs(logit_model.coef_.ravel()))
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))
def get_best_model_and_accuracy(model, params, X, y):
    grid_clf_auc = GridSearchCV(model,param_grid=params,error_score=0.,scoring = 'roc_auc')
    # 使用这个选择的模型和参数
    grid_clf_auc.fit(X, y) 
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)
dt=DecisionTreeClassifier(random_state=0)
param_grid = {"max_depth": [3,7,10,50,100],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
              "criterion": ["gini", "entropy"]}  
print("Decision Tree train:")
get_best_model_and_accuracy(dt, param_grid, X_train, y_train)
print("Decision Tree test:")
get_best_model_and_accuracy(dt, param_grid, X_test, y_test)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
LR=LogisticRegression(max_iter=100000,random_state=0)

print("Logistic Regression train:")
get_best_model_and_accuracy(LR, param_grid, X_train, y_train)
print("Logistic Regression test:")
get_best_model_and_accuracy(LR, param_grid, X_test, y_test)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
}
rf=RandomForestClassifier(random_state=0)
print("Random Forest train:")
get_best_model_and_accuracy(rf, param_grid, X_train, y_train)
print("Random Forest test:")
get_best_model_and_accuracy(rf, param_grid, X_test, y_test)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
LR=LogisticRegression(max_iter=100000,random_state=0)
print("Logistic Regression train:")
get_best_model_and_accuracy(LR, param_grid, X_train, y_train)
print("Logistic Regression test:")
get_best_model_and_accuracy(LR, param_grid, X_test, y_test)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
}
rf=RandomForestClassifier(random_state=0)
print("Random Forest train:")
get_best_model_and_accuracy(rf, param_grid, X_train, y_train)
print("Random Forest test:")
get_best_model_and_accuracy(rf, param_grid, X_test, y_test)
param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'max_depth': [3, 4, 5]
        }
XGB= XGBClassifier(random_state=0)
print("XGBoost train:")
get_best_model_and_accuracy(XGB, param_grid, X_train, y_train)
print("XGBoost test:")
get_best_model_and_accuracy(XGB, param_grid, X_test, y_test)
rf= RandomForestClassifier(random_state=0,max_depth= 7, max_features= 'sqrt', n_estimators= 500)
rf.fit(X_train,y_train)
predictions = rf.predict(test_data)
result_csv = pd.DataFrame(test_data.PassengerId)
result_csv['Survived'] = predictions
result_csv.to_csv('sumbsition',index=False)
