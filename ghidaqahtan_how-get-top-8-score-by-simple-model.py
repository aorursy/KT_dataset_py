import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

# To ignore unwanted warnings
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')
gender_submission = pd.read_csv('./datasets/gender_submission.csv') 
train.head()
test.head()
train.info()
print('_'*40)
test.info()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
# Train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')
# Test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');
#missing amount for train set
missing= train.isnull().sum().sort_values(ascending=False)
percentage = (train.isnull().sum()/ train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_data.head(3)
#missing amount for test set
missing= test.isnull().sum().sort_values(ascending=False)
percentage = (test.isnull().sum()/ test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_data.head(3)
train.Embarked.fillna(value='S', inplace=True)
train['Embarked'].value_counts()
isn = pd.isnull(test['Fare'])
test[isn]
average_of_fare= test.groupby('Pclass')['Fare'].mean()
print('The mean fare for the Pclass (for missing fare data) is:',average_of_fare[3])
# filling the missing by mean
test.Fare.fillna(value=average_of_fare[3], inplace=True)
mean_age = train.groupby('Pclass')[['Age']].mean()
mean_age
#defining a function 'impute_age'
def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']
    # Passing age_pclass[0] which is 'Age' to variable 'Age'
    Age = age_pclass[0]
    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'
    Pclass = age_pclass[1]
    #applying condition based on the Age and filling the missing data respectively 
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age
#train data
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#test data
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
# train
train['Cabin']=train['Cabin'].notnull().astype('int')
train['Cabin'].unique()
# test
test['Cabin']=test['Cabin'].notnull().astype('int')
test['Cabin'].unique()
# Sex & Age
g = sns.FacetGrid(train, hue = 'Survived', col = 'Sex', height = 3, aspect = 2)
g.map(plt.hist, 'Age', alpha = .5, bins = 20)
g.add_legend()
plt.show()
#Change the data types
train['Age'] = train['Age'].astype(int)
test['Age'] = train['Age'].astype(int)
def age_range(df):
    df['Age'].loc[df['Age'] <= 16 ] = 0
    df['Age'].loc[(df['Age'] > 16) & (df['Age'] <= 32)] = 1
    df['Age'].loc[(df['Age'] > 32) & (df['Age'] <= 48)] = 2
    df['Age'].loc[(df['Age'] > 48) & (df['Age'] <= 64)] = 3
    df['Age'].loc[df['Age'] > 64] = 4   
age_range(train)
age_range(test)
# Creating title dictionary in train data
titles = set()
for name in train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())  
# Mapping Titles
train['Title'] = train.Title.map(Title_Dictionary)
# Creating Title dictionary in test data
titles = set()
for name in test['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary_test = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())    
# Mapping Titles
test['Title'] = test.Title.map(Title_Dictionary_test)
# Missing values
test[test['Title'].isnull()]
# Filling missing values in title
test['Title'].fillna(value='Mr', inplace=True)
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['FamilySize'] = train['FamilySize'].astype(int)
test['FamilySize'] = train['FamilySize'].astype(int)
def family_range(df):
    df['FamilySize'].loc[df['FamilySize'] <= 1 ] = 0
    df['FamilySize'].loc[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)] = 1
    df['FamilySize'].loc[df['FamilySize'] >= 5] = 2   
family_range(train)
family_range(test)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
# Train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')
# Test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');
# Train Data
train = pd.get_dummies(train, columns=['Sex','Embarked','Title'],drop_first=True)
# Test Data
test= pd.get_dummies(test, columns=['Sex','Embarked','Title'],drop_first=True)
test['Title_Royalty'] = 0    # adding Title_Royalty column to match columns in the train df
fig=plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(train.corr(), annot=True,ax=ax, cmap=plt.cm.YlGnBu)
ax.set_title('The correlations between all numeric features')
palette =sns.diverging_palette(80, 110, n=146)
plt.show
# correlation with the target
corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
g = sns.factorplot('Survived',data=train,kind='count',hue='Pclass')
g._legend.set_title('Pclass')
# replace labels
new_labels = ['1st class', '2nd class', '3rd class']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
g = sns.factorplot('Pclass',data=train,hue='Sex_male',kind='count')
g._legend.set_title('Sex')
# replace labels
new_labels = ['Female', 'Male']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
g = sns.factorplot('Survived',data=train,kind='count',hue='FamilySize')
g._legend.set_title('Family Size')
# replace labels
new_labels = ['Small', 'Single', 'Large']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Train data
features_drop = ['PassengerId','Name', 'Ticket', 'Survived','SibSp','Parch']
selected_features = [x for x in train.columns if x not in features_drop]
# Test data
features_drop_test = ['PassengerId','Name', 'Ticket','SibSp','Parch']
selected_features_test = [x for x in test.columns if x not in features_drop_test]
# Train data
X = train[selected_features]
y = train['Survived']
# Test data
testing = test[selected_features_test]
ss = StandardScaler()
Xs =ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3,random_state=55, stratify=y) 
tree= DecisionTreeClassifier()
tree.fit(X_train, y_train)
print('test score' , tree.score(X_train, y_train))
print('test score' , tree.score(X_test, y_test))
y_pred =tree.predict(testing)
dt = DecisionTreeClassifier()
dt_en = BaggingClassifier(base_estimator=dt, n_estimators=100, max_features=10)
dt_en.fit(X_train, y_train)
print('test score' , dt_en.score(X_train, y_train))
print('test score' , dt_en.score(X_test, y_test))
y_pred = dt_en.predict(testing) 
param = { 'max_features': [0.3, 0.6, 1],
        'n_estimators': [50, 150, 200], 
         'base_estimator__max_depth': [3, 5, 20]}
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True)
model_gs = GridSearchCV(model,param, cv=6, verbose=1, n_jobs=-1 )
model_gs.fit(X_train, y_train)
model_gs.best_params_
model_gs.best_estimator_.oob_score_
randomF = RandomForestClassifier(max_depth=350, n_estimators=9, max_features=11, random_state=14, min_samples_split=3)
randomF.fit(X_train, y_train)
print('Train score :',randomF.score(X_train, y_train))
print('Ttest score :',randomF.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(randomF, X, y, cv=cv).mean()
y_pred=randomF.predict(testing)
et = ExtraTreesClassifier(n_estimators=66, min_samples_split=7)
et.fit(X_train, y_train)
print('Train score :',et.score(X_train, y_train))
print('Ttest score :',et.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(et, X, y, cv=cv).mean()
y_pred =et.predict(testing)
knn_classifier = KNeighborsClassifier(n_neighbors=7, leaf_size=48, weights='uniform',p=1)  
knn_classifier.fit(X_train, y_train)
print(knn_classifier.score(X_train, y_train))
print (knn_classifier.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(knn_classifier, X, y, cv=cv).mean()
y_pred = knn_classifier.predict(testing) 
knn = KNeighborsClassifier()
knn_en = BaggingClassifier(base_estimator=knn, n_estimators=45, oob_score=True, max_features=9, random_state=99)
knn_en.fit(X_train, y_train)

print(knn_en.score(X_train, y_train))
print(knn_en.score(X_test, y_test))
y_pred = knn_en.predict(testing) 
knn_en.estimators_[12]
svm_l = svm.SVC(kernel='linear', C=33)
svm_l.fit(X_train, y_train)
print('Train : ', svm_l.score(X_train, y_train))
print('Test: ', svm_l.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_l, Xs, y, cv=cv).mean()
cross_val_score(randomF, X, y, cv=cv)
#main Features importances
sfm = SelectFromModel(randomF, threshold=0.15,prefit=True)

feat_labels = X.columns

for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
y_pred = svm_l.predict(testing) 
svm_p = svm.SVC(kernel='poly', C=3)
svm_p.fit(X_train, y_train)
print(svm_p.score(X_train, y_train))
print(svm_p.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_p, Xs, y, cv=cv).mean()
y_pred = svm_p.predict(testing) 
svm_rbf = svm.SVC(kernel='rbf', C=4)
svm_rbf.fit(X_train, y_train)
print(svm_rbf.score(X_train, y_train))
print(svm_rbf.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_rbf, Xs, y, cv=cv).mean()
y_pred = svm_rbf.predict(testing) 
logreg = LogisticRegression(max_iter=300)
logreg.fit(X_train, y_train)
print('train score' , logreg.score(X_train, y_train))
print('test score' , logreg.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(logreg, X, y, cv=cv).mean()
cross_val_score(randomF, X, y, cv=cv)
#main Features importances
sfm = SelectFromModel(randomF, threshold=0.15,prefit=True)

feat_labels = X.columns

for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
y_pred = logreg.predict(testing) 
adaboost = AdaBoostClassifier(n_estimators=67)
adaboost.fit(X_train, y_train)
print('Train accuracy:', adaboost.score(X_train, y_train))
print('Test accuracy:',adaboost.score(X_test, y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(adaboost, X, y, cv=cv).mean()
y_pred = adaboost.predict(testing) 
thesubmission = gender_submission.copy()
thesubmission['Survived'] = y_pred
thesubmission['Survived'].head()
thesubmission.to_csv('thesubmission.csv', index=False)
list_of_Scores = list()
# Decision Tree Classifier
results = {'Model':'Decision Tree Classifier',
           'Train Score':tree.score(X_train, y_train),
           'Test Score':tree.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# Bagging Classifier with Decision Tree 
results = {'Model':'Bagging with Decision Tree ',
           'Train Score':dt_en.score(X_train, y_train),
           'Test Score':dt_en.score(X_test, y_test),
           'Kaggle Score':0.75598}
list_of_Scores.append(results)

# Random Forest Classifier
results = {'Model':'Random Forest Classifier',
           'Train Score': randomF.score(X_train, y_train),
           'Test Score':randomF.score(X_test, y_test),
           'Kaggle Score':0.77990
}
list_of_Scores.append(results)

# Extra Trees Classifier
results = {'Model':'Extra Trees Classifier',
           'Train Score':et.score(X_train, y_train),
           'Test Score': et.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# KNeighbors Classifier
results = {'Model':'KNeighbors Classifier',
           'Train Score':knn_classifier.score(X_train, y_train),
           'Test Score':knn_classifier.score(X_test, y_test),
           'Kaggle Score':0.77511}
list_of_Scores.append(results)

# Bagging Classifier with a Knn 
results = {'Model':'Bagging Classifier with Knn ',
           'Train Score': knn_en.score(X_train, y_train),
           'Test Score':knn_en.score(X_test, y_test),
           'Kaggle Score':0.66507}
list_of_Scores.append(results)

# SVM with Linear
results = {'Model':'SVM with Linear',
           'Train Score': svm_l.score(X_train, y_train),
           'Test Score':svm_l.score(X_test, y_test),
           'Kaggle Score':0.80382}
list_of_Scores.append(results)


# SVM with Poly
results = {'Model':'SVM with Poly',
           'Train Score':svm_p.score(X_train, y_train),
           'Test Score':svm_p.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# SVM with Rbf 
results = {'Model':"SVM with Rbf",
           'Train Score':svm_rbf.score(X_train, y_train),
           'Test Score':svm_rbf.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results) 


# Logistic Regression
results = {'Model':'Logistic Regression',
           'Train Score':logreg.score(X_train, y_train),
           'Test Score':logreg.score(X_test, y_test),
           'Kaggle Score':0.80382}
list_of_Scores.append(results)

# AdaBoost Classifier
results = {'Model':'AdaBoost Classifier ',
           'Train Score':adaboost.score(X_train, y_train),
           'Test Score':adaboost.score(X_test, y_test),
           'Kaggle Score':0.77511}
list_of_Scores.append(results)
df_results = pd.DataFrame(list_of_Scores)
df_results