def splitfeatures(X):
  X_num=X.describe().columns
  X_cat=X.columns.difference(X_num)
  X_num=X_num.tolist()
  X_cat=X_cat.tolist()
  print("Numeric features:      ",X_num)
  print("Categorical features:  ",X_cat)

def numfeatures(X):
    return(X.describe().columns.tolist())

def catfeatures(X):
    X_num=X.describe().columns
    return(X.columns.difference(X_num).tolist())

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

def percent_value_counts(df, feature):
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)

def unique_values_in_column(data,feature):
    unique_val=pd.Series(data.loc[:,feature].unique())
    return pd.concat([unique_val],axis=1,keys=['Unique Values'])
#Data Processing
import numpy as np 
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set(style="whitegrid")
%matplotlib inline
        
#Warnings
import warnings
warnings.filterwarnings("ignore")
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
train.sample(5)
splitfeatures(train)
print("="*100)
splitfeatures(test)
train.info()
print("="*100)
print("="*100)
test.info()
train.describe(include="all")
missing_percentage(train)
test.describe(include="all")
missing_percentage(test)
print("Numeric features: ",numfeatures(train))
df_num=train[numfeatures(train)]
corr=df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12,6))
cmap = sns.color_palette("PRGn",10)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
train_num=train[numfeatures(train)]
sns.set(style="white")
sns.pairplot(train_num,corner=True)
train.sample(5)
sns.set(style="whitegrid")
fig, ax = pyplot.subplots(figsize=(12,6))
colors=["#80CEE1","#FFB6C1"]
customPalette=sns.set_palette(sns.color_palette(colors))
print(train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean())
sns.barplot(x="Sex",y="Survived",data=train,ax=ax,palette=customPalette)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig, ax = pyplot.subplots(figsize=(12,6))
print(train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean())
sns.barplot(x="Pclass",y="Survived",data=train,hue="Sex",ax=ax,palette=customPalette)
colors=["#80CEE1","#FFB6C1"]
customPalette=sns.set_palette(sns.color_palette(colors))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
percent_value_counts(train,"Name")
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# Apply get_title function
train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)

percent_value_counts(train,"Title")
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

fig, ax = pyplot.subplots(figsize=(12,6))
print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
sns.barplot(x="Title", y="Survived", data=train,hue="Sex",ax=ax)
colors=["#80CEE1","#FFB6C1"]
customPalette=sns.set_palette(sns.color_palette(colors))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)
train.sample(5)
print(train[["Age","Survived"]].groupby(["Age"],as_index=False).mean())
fig, ax = pyplot.subplots(figsize=(12,6))
sns.distplot(train.Age,ax=ax,color="blue")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train["Age"] = train["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test["Age"] = test["Age"].fillna(-0.5)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="AgeGroup", y="Survived", data=train,hue="Sex",ax=ax)

colors=["#80CEE1","#FFB6C1"]
customPalette=sns.set_palette(sns.color_palette(colors))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train.drop(['Age'], axis=1, inplace=True)
test.drop(['Age'], axis=1, inplace=True)
train.sample(5)
fig, ax = pyplot.subplots(figsize=(12,6))
print(train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean())
sns.barplot(x="SibSp",y="Survived",data=train,hue="Sex",ax=ax)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig, ax = pyplot.subplots(figsize=(12,6))
print(train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean())
sns.barplot(x="Parch",y="Survived",data=train,hue="Sex",ax=ax)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
fig, ax = pyplot.subplots(figsize=(12,6))
print(train[["FamilySize","Survived"]].groupby(["FamilySize"],as_index=False).mean())
sns.barplot(x="FamilySize",y="Survived",data=train,ax=ax,hue="Sex")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train["IsAlone"]=0
train.loc[train["FamilySize"] == 1,"IsAlone"]= 1

test["IsAlone"]=0
test.loc[test["FamilySize"] == 1,"IsAlone"]= 1

print(train[["IsAlone","Survived"]].groupby(["IsAlone"],as_index=False).mean())
fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="IsAlone",y="Survived",data=train,ax=ax,hue="Sex")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)
train.sample(5)
print(train[["Fare","Survived"]].groupby(["Fare"],as_index=False).mean())
fig, ax = pyplot.subplots(figsize=(12,6))
sns.distplot(train.Fare,ax=ax,color="Blue")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train['FareGroup'] = pd.qcut(train['Fare'], 4, labels=['A', 'B', 'C', 'D'])
test['FareGroup'] = pd.qcut(test['Fare'], 4, labels=['A', 'B', 'C', 'D'])
print(train[['FareGroup','Survived']].groupby(['FareGroup'], as_index=False).mean())
print(percent_value_counts(train,"FareGroup"))
print(percent_value_counts(test,"FareGroup"))

fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="FareGroup",y="Survived",data=train,ax=ax,color="salmon")
train.drop(['Fare'], axis=1, inplace=True)
test.drop(['Fare'], axis=1, inplace=True)
train.sample(5)
test["FareGroup"].fillna(("B"),inplace=True)
missing_percentage(test)
percent_value_counts(train,"Cabin")
pd.unique(train['Cabin'])
train["Cabin_Data"] = train["Cabin"].isnull().apply(lambda x: not x)
test["Cabin_Data"] = test["Cabin"].isnull().apply(lambda x: not x)

train["Deck"] = train["Cabin"].str.slice(0,1)
train["Room"] = train["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
train["Deck"] = train["Deck"].fillna("N")
train["Room"] = round(train["Room"].fillna(train["Room"].mean()),0).astype("int")

test["Deck"] = test["Cabin"].str.slice(0,1)
test["Room"] = test["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
test["Deck"] = test["Deck"].fillna("N")
test["Room"] = round(test["Room"].fillna(test["Room"].mean()),0).astype("int")

train.sample(5)
percent_value_counts(train,"Deck")
fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="Deck", y="Survived", data=train,ax=ax,color="Salmon")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train['Room'].describe()
bins = [0, 50, 75, 100,np.inf]
labels = ["g1","g2","g3","g4"]
train["RoomGroup"] = pd.cut(train["Room"], bins, labels = labels)
test["RoomGroup"] = pd.cut(test["Room"], bins, labels = labels)
train.sample(5)
fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="RoomGroup", y="Survived", data=train,color="Salmon")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train.drop(["Cabin", "Cabin_Data", "Room"], axis=1, inplace=True)
test.drop(["Cabin", "Cabin_Data", "Room"], axis=1, inplace=True)
train.sample(5)
print(percent_value_counts(train,"Embarked"))
print(train[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean())

fig, ax = pyplot.subplots(figsize=(12,6))
sns.barplot(x="Embarked",y="Survived",data=train,ax=ax,hue="Sex")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
train.Embarked.fillna("S", inplace=True)
missing_percentage(train)
missing_percentage(test)
train_backup=train
test_backup=test

train.drop("SibSp",axis=1,inplace=True)
test.drop("SibSp",axis=1,inplace=True)
train.drop("Parch",axis=1,inplace=True)
test.drop("Parch",axis=1,inplace=True)

ids=test["PassengerId"]

train.drop("PassengerId",axis=1,inplace=True)
test.drop("PassengerId",axis=1,inplace=True)

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test
X_train.shape, y_train.shape, X_test.shape
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
print(X_train.columns)
print("="*100)
print(X_test.columns)
X_train.drop("Deck_T",axis=1,inplace=True)
print(X_train.columns)
print("="*100)
print(X_test.columns)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.fit_transform(X_test)

np.savez_compressed("np_savez_comp", X=X_train, y=y_train)
data = np.load("np_savez_comp.npz")

X = data["X"]
y = data["y"]
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.2, random_state=0)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

gnb = GaussianNB()
gnb.fit(X_train, y_train)

with open('Naive Bayes.pickle', mode='wb') as fp:
    pickle.dump(gnb, fp)
    
score = gnb.score(X_valid, y_valid)
print('NB score: {}' .format(score))

predicted=gnb.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(X_train, y_train)

with open('Logistic Regression.pickle', mode='wb') as fp:
    pickle.dump(logistic_regression, fp)

score = logistic_regression.score(X_valid, y_valid)
print('LR score: {}' .format(score))

predicted=logistic_regression.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)

with open('Support Vector Machine.pickle', mode='wb') as fp:
    pickle.dump(svm, fp)
    
score = svm.score(X_valid, y_valid)
print('SVC linear score: {}' .format(score))


svm2 = SVC(kernel='rbf', C=1.0, random_state=0)
svm2.fit(X_train, y_train)
score = svm2.score(X_valid, y_valid)
print('SVC rbf score: {}' .format(score))

predicted=svm.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

with open('Nearest Neighbors.pickle', mode='wb') as fp:
    pickle.dump(knn, fp)
    
score = knn.score(X_valid, y_valid)
print('KNN score: {}' .format(score))

predicted=knn.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
decision_tree.fit(X_train, y_train)

with open('Decision Tree.pickle', mode='wb') as fp:
    pickle.dump(decision_tree, fp)
    
score = decision_tree.score(X_valid, y_valid)
print('DT score: {}' .format(score))

predicted=decision_tree.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)

with open('Random Forest.pickle', mode='wb') as fp:
    pickle.dump(random_forest, fp)
    
score = random_forest.score(X_valid, y_valid)
print('RF score: {}' .format(score))

predicted=random_forest.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
xgb.fit(X_train, y_train)

with open('Gradient Boosting.pickle', mode='wb') as fp:
    pickle.dump(xgb, fp)
    
score = xgb.score(X_valid, y_valid)
print('score: {}' .format(score))

predicted=xgb.predict(X_valid)
matrix = confusion_matrix(y_valid, predicted)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt
names = ["Support Vector Machine", "Logistic Regression", "Nearest Neighbors",
         "Decision Tree","Random Forest", "Naive Bayes","Gradient Boosting"]

result = []
print("For guessing the Survived feature we used the following models:")
print(" ")
for name in names:
    with open(name + '.pickle', 'rb') as fp:
        clf = pickle.load(fp)
    
    clf.fit(X_train, y_train)
    score1 = clf.score(X_train, y_train)
    score2 = clf.score(X_valid, y_valid)
    result.append([score1, score2])
    
    print(name)

df_result = pd.DataFrame(result, columns=['Training', 'Validation'], index = names)
df_result.sort_values("Validation",ascending=False)
from sklearn.model_selection import GridSearchCV
params = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf=GridSearchCV(logistic_regression, params, cv=10, return_train_score=True)
best_clf = clf.fit(X_train,y_train)
best_clf

means=clf.cv_results_["mean_test_score"]
stds=clf.cv_results_["std_test_score"]
params=clf.cv_results_["params"]

for m,s,p in zip(means,stds,params):
    print("%0.3f (+/-%0.3f) for %r"%(m,2*s,p))
    
print("="*100)
print("="*100)

print('best score: {:0.3f}'.format(clf.score(X, y)))
print('best params: {}'.format(clf.best_params_))
print('best val score:  {:0.3f}'.format(clf.best_score_))
params = {
    'n_estimators': [100],
    'max_depth': [2,3,5,10,None],
    'gamma':[0,.01,.1,1,10,100],
    'min_child_weight':[0,.01,0.1,1,10,100],
    'sampling_method': ['uniform', 'gradient_based']
}

clf = GridSearchCV(xgb, params, cv=5, return_train_score=True)
best_clf = clf.fit(X_train,y_train)
best_clf

means=clf.cv_results_["mean_test_score"]
stds=clf.cv_results_["std_test_score"]
params=clf.cv_results_["params"]

for m,s,p in zip(means,stds,params):
    print("%0.3f (+/-%0.3f) for %r"%(m,2*s,p))
    
print("="*100)
print("="*100)

print('best score: {:0.3f}'.format(clf.score(X, y)))
print('best params: {}'.format(clf.best_params_))
print('best val score:  {:0.3f}'.format(clf.best_score_))
temp=pd.DataFrame(X_train,columns=['Pclass', 'FamilySize', 'IsAlone', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'AgeGroup_Unknown',
       'AgeGroup_Baby', 'AgeGroup_Child', 'AgeGroup_Teenager',
       'AgeGroup_Student', 'AgeGroup_Young Adult', 'AgeGroup_Adult',
       'AgeGroup_Senior', 'FareGroup_A', 'FareGroup_B', 'FareGroup_C',
       'FareGroup_D', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
       'Deck_F', 'Deck_G', 'Deck_N', 'RoomGroup_g1', 'RoomGroup_g2',
       'RoomGroup_g3', 'RoomGroup_g4'])
best_rf = best_clf.best_estimator_.fit(X_train,y_train)
feat_importances = pd.Series(best_rf.feature_importances_,index=temp.columns)
feat_importances.nlargest(20).plot(kind='barh',color="salmon")
params = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100]}]

clf = GridSearchCV(svm, params, cv=10, return_train_score=True)
best_clf = clf.fit(X_train,y_train)
best_clf

means=clf.cv_results_["mean_test_score"]
stds=clf.cv_results_["std_test_score"]
params=clf.cv_results_["params"]

for m,s,p in zip(means,stds,params):
    print("%0.3f (+/-%0.3f) for %r"%(m,2*s,p))
    
print("="*100)
print("="*100)

print('best score: {:0.3f}'.format(clf.score(X, y)))
print('best params: {}'.format(clf.best_params_))
print('best val score:  {:0.3f}'.format(clf.best_score_))
params = {'n_estimators': [100],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}

clf = GridSearchCV(random_forest, params, cv=10, return_train_score=True)
best_clf = clf.fit(X_train,y_train)
best_clf

means=clf.cv_results_["mean_test_score"]
stds=clf.cv_results_["std_test_score"]
params=clf.cv_results_["params"]

for m,s,p in zip(means,stds,params):
    print("%0.3f (+/-%0.3f) for %r"%(m,2*s,p))
    
print("="*100)
print("="*100)

print('best score: {:0.3f}'.format(clf.score(X, y)))
print('best params: {}'.format(clf.best_params_))
print('best val score:  {:0.3f}'.format(clf.best_score_))
temp=pd.DataFrame(X_train,columns=['Pclass', 'FamilySize', 'IsAlone', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'AgeGroup_Unknown',
       'AgeGroup_Baby', 'AgeGroup_Child', 'AgeGroup_Teenager',
       'AgeGroup_Student', 'AgeGroup_Young Adult', 'AgeGroup_Adult',
       'AgeGroup_Senior', 'FareGroup_A', 'FareGroup_B', 'FareGroup_C',
       'FareGroup_D', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
       'Deck_F', 'Deck_G', 'Deck_N', 'RoomGroup_g1', 'RoomGroup_g2',
       'RoomGroup_g3', 'RoomGroup_g4'])
best_rf = best_clf.best_estimator_.fit(X_train,y_train)
feat_importances = pd.Series(best_rf.feature_importances_,index=temp.columns)
feat_importances.nlargest(20).plot(kind='barh',color="salmon")
best_model=RandomForestClassifier(n_estimators=100,bootstrap=True,criterion="gini",max_depth=15,max_features="auto",min_samples_leaf=2,min_samples_split=2)
best_model.fit(X_train,y_train)
preds=best_model.predict(X_test)
output = pd.DataFrame({ "PassengerId" : ids, "Survived": preds })
output.to_csv("submission.csv", index=False)