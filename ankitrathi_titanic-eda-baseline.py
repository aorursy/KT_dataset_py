# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# import & display data
train_df = pd.read_csv("../input/train.csv")
#test_df    = pd.read_csv("../input/test.csv")

# preview the data
train_df.head()
# check missing data
train_df.info()
# statistical information (numerical columns)
train_df.describe()
# statistical information (categorical columns)
train_df.describe(include = ['O'])
# missing values in data-set
train_df.isnull().sum()
# response variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.countplot('Survived',data=train_df, ax=ax[0])
ax[0].set_title('Survived count')
sns.barplot(y="Survived", data=train_df, ax=ax[1])
ax[1].set_title('Survived ratio')
plt.show()
# Pclass variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Pclass')
sns.countplot('Pclass',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
# Pclass variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
# SibSp variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs SibSp')
sns.countplot('SibSp',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('SibSp:Survived vs Dead')
plt.show()
# Parch variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Parch')
sns.countplot('Parch',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Parch:Survived vs Dead')
plt.show()
# Embarked variable analysis
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Embarked')
sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Embarked:Survived vs Dead')
plt.show()
# Age variable analysis
f,ax=plt.subplots(1,2,figsize=(20,10))
train_df.Age.plot.hist(ax=ax[0],bins=20,edgecolor='black')
ax[0].set_title('Age')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
sns.violinplot("Survived","Age",  data=train_df,split=True,ax=ax[1])
ax[1].set_title('Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
# Fare variable analysis
f,ax=plt.subplots(1,2,figsize=(20,10))
train_df.Fare.plot.hist(ax=ax[0],bins=20,edgecolor='black')
ax[0].set_title('Fare')
x1=list(range(0,600,50))
ax[0].set_xticks(x1)
sns.violinplot("Survived","Fare",  data=train_df,split=True,ax=ax[1])
ax[1].set_title('Fare vs Survived')
ax[1].set_yticks(range(0,600,50))
plt.show()
# Fare Vs Embarked analysis
f,ax=plt.subplots(1,2,figsize=(20,10))
train_df.Fare.plot.hist(ax=ax[0],bins=20,edgecolor='black')
ax[0].set_title('Fare')
x1=list(range(0,600,50))
ax[0].set_xticks(x1)
sns.violinplot("Embarked","Fare",  data=train_df,split=True,ax=ax[1])
ax[1].set_title('Fare vs Embarked')
ax[1].set_yticks(range(0,600,50))
plt.show()
# Fare Vs Age analysis
g = sns.FacetGrid(train_df, hue="Survived", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

# Fare Vs Age on Pclass analysis
g = sns.FacetGrid(train_df, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
data = train_df.drop(['Name', 'Ticket', 'Cabin','PassengerId', 'Age', 'Pclass'],axis=1)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data["Embarked"] = data["Embarked"].fillna('C') # impute missing data
data['Embarked'].replace(['C','S', 'Q'],[1,2,3],inplace=True)

data.head()
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn import svm #support vector Machine
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
import warnings
warnings.filterwarnings("ignore")

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
predictionRF=model.predict(test_X)
print('Accuracy for Random Forests is',metrics.accuracy_score(predictionRF,test_Y))

model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
predictionSVC=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(predictionSVC,test_Y))

model = LogisticRegression()
model.fit(train_X,train_Y)
predictionLR=model.predict(test_X)
print('Accuracy for Logistic Regression is',metrics.accuracy_score(predictionLR,test_Y))
# cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Logistic Regression','Random Forest']
models=[svm.SVC(kernel='linear'),LogisticRegression(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe
# drop unnecessary features
data = train_df.drop([ 'Ticket', 'PassengerId', 'Cabin'],axis=1)
# Split train & valid data
train,valid=train_test_split(data, test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
valid_X=valid[valid.columns[1:]]
valid_Y=valid[valid.columns[:1]]
# handle missing values
#complete missing age with median
train_X['Age'].fillna(train_X['Age'].median(), inplace = True)
valid_X['Age'].fillna(valid_X['Age'].median(), inplace = True)

#complete embarked with mode
train_X['Embarked'].fillna(train_X['Embarked'].mode()[0], inplace = True)
valid_X['Embarked'].fillna(valid_X['Embarked'].mode()[0], inplace = True)
# feature engineering

# FamilySize feature
train_X['FamilySize'] = train_X ['SibSp'] + train_X['Parch'] + 1
valid_X['FamilySize'] = valid_X ['SibSp'] + valid_X['Parch'] + 1

# IsAlone feature
train_X['IsAlone'] = 1 #initialize to yes/1 is alone
train_X['IsAlone'].loc[train_X['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
valid_X['IsAlone'] = 1 #initialize to yes/1 is alone
valid_X['IsAlone'].loc[valid_X['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

# split title from name
train_X['Title'] = train_X['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
valid_X['Title'] = valid_X['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

# fare bins
train_X['FareBin'] = pd.qcut(train_X['Fare'], 4)
valid_X['FareBin'] = pd.qcut(valid_X['Fare'], 4)

# age bins
train_X['AgeBin'] = pd.cut(train_X['Age'].astype(int), 5)
valid_X['AgeBin'] = pd.cut(valid_X['Age'].astype(int), 5)

# clean rare titles
stat_min = 10
title_names = (train_X['Title'].value_counts() < stat_min)
train_X['Title'] = train_X['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

title_names = (valid_X['Title'].value_counts() < stat_min)
valid_X['Title'] = valid_X['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# convert formats

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label = LabelEncoder()
   
train_X['Sex_Code'] = label.fit_transform(train_X['Sex'])
train_X['Embarked_Code'] = label.fit_transform(train_X['Embarked'])
train_X['Title_Code'] = label.fit_transform(train_X['Title'])
train_X['AgeBin_Code'] = label.fit_transform(train_X['AgeBin'])
train_X['FareBin_Code'] = label.fit_transform(train_X['FareBin'])

valid_X['Sex_Code'] = label.fit_transform(valid_X['Sex'])
valid_X['Embarked_Code'] = label.fit_transform(valid_X['Embarked'])
valid_X['Title_Code'] = label.fit_transform(valid_X['Title'])
valid_X['AgeBin_Code'] = label.fit_transform(valid_X['AgeBin'])
valid_X['FareBin_Code'] = label.fit_transform(valid_X['FareBin'])
# drop redundant columns
train_X = train_X.drop([ 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FareBin', 'AgeBin'],axis=1)
valid_X = valid_X.drop([ 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FareBin', 'AgeBin'],axis=1)
# modeling
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
predictionRF=model.predict(valid_X)
print('Accuracy for Random Forests is',metrics.accuracy_score(predictionRF,valid_Y))

model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
predictionSVC=model.predict(valid_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(predictionSVC,valid_Y))

model = LogisticRegression()
model.fit(train_X,train_Y)
predictionLR=model.predict(valid_X)
print('Accuracy for Logistic Regression is',metrics.accuracy_score(predictionLR,valid_Y))
# feature importance
model=RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(train_X,train_Y)
pd.Series(model.feature_importances_,train_X.columns).sort_values(ascending=True).plot.barh(width=0.8)
plt.title('Feature Importance in Random Forests')