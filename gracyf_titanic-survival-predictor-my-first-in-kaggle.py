%matplotlib notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape

test.shape
train.info()
test.info()
#Combine test and train data sets
titanic = pd.concat([train,test],axis=0)
#Fill the test set values of Survived with 0 
titanic['Survived'].fillna(0,inplace=True)
titanic.shape
titanic.info()
titanic.describe()
titanic.dtypes
x = titanic['PassengerId'].unique()
x.shape 
train.describe(include=['O'])
titanic.corr()
#To find if any row is a duplicate
titanic[titanic.nunique(axis=1) == 1] == True
#Duplicate rows are verified again.
titanic[titanic.duplicated() == True]
#Plot the scatter matrix from train data set.
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
#Density plots of the all columns in 
fig = plt.figure(figsize = (10,15))
ax = fig.gca()
train.plot(kind='density', subplots=True, layout=(3,3), sharex=False,ax = ax)
plt.show()
fig = plt.figure(figsize = (8,12))
ax = fig.gca()
train.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,ax=ax)
plt.show()
train.groupby('Survived').hist();
import seaborn as sns
sns.pairplot(train, hue='Survived', diag_kind='kde', size=2);
# absolute numbers
train["Survived"].value_counts()


# percentages
train["Survived"].value_counts(normalize = True)
#Percentage of people survived w.r.t age
plt.figure()
train['Age'].groupby(train['Survived']).hist(alpha=0.6,bins=50)
plt.legend()
train.groupby('Survived').Age.value_counts().unstack()
plt.figure()
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train);
plt.figure()
sns.barplot(x="Sex", y="Survived", hue="Embarked", data=train);
train.groupby('Survived').Sex.value_counts(normalize = True).unstack()
import seaborn as sns
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.distplot(train['Age'].dropna())
plt.show()
#Age distribution follows a right skewed distribution. This has to be converted to a normal distribution.

#Compare Age and Pclass
g = sns.FacetGrid(train, row='Survived', col='Pclass')
g.map(sns.distplot, "Age")
plt.show()
category_group=train.groupby(['Survived','Pclass']).count()['PassengerId']
category_group.unstack().head()
g = sns.FacetGrid(train, row='Survived', col='Sex')
g.map(sns.distplot, "Age")
plt.show()
train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)
train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)
category_group=train.groupby(['Survived','SibSp']).count()['PassengerId']
category_group.unstack().head()
category_group=train.groupby(['Survived','Parch']).count()['PassengerId']
category_group.unstack().head()
plt.figure()
sns.heatmap(train.corr(), annot=True, fmt=".2f")
category_group=train.groupby(['Survived','Embarked']).count()['PassengerId']
category_group.unstack().head()
category_group.unstack().plot(kind='bar',stacked=True,title="Survival As per Embarked Region")
# Find potential outliers in values array
# and visualize them on a plot

def is_outlier(value, p25, p75):
    """Check if value is an outlier
    """
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return value <= lower or value >= upper
 
 
def get_indices_of_outliers(values):
    """Get outlier indices (if any)
    """
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
     
    indices_of_outliers = []
    for ind, value in enumerate(values):
        if is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)
    return indices_of_outliers
 


indices_of_outliers = get_indices_of_outliers(train['Age'])
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train['Age'], 'b-', label='distances')
ax.plot(
    indices_of_outliers,
    train['Age'][indices_of_outliers],
    'ro',
    markersize = 7,
    label='outliers')
ax.legend(loc='best')
#Use Label encoding for categorical variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(titanic['Sex'])
titanic['Sex'] = le.transform(titanic['Sex'])

#To include the Embarked feature in knn fit to predict age..there are very few say 2 missing values in Embarked.. using the most
#occuring value of embarked column for this missing values

titanic['Embarked'] = titanic['Embarked'].fillna("S")
le.fit(titanic['Embarked'])
titanic['Embarked'] = le.transform(titanic['Embarked'])
#Fare contains one null value..replacec it with mean of Fare value
titanic[titanic['Fare'].isnull() == True]
titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].mean())
#Impute Age
df = titanic[['PassengerId','Pclass', 'Sex', 'SibSp','Parch','Fare','Embarked','Age']]
train_df = df[df['Age'].isnull() == False]
test_df = df[df['Age'].isnull() == True]

 
X_train_df = train_df[['PassengerId','Pclass', 'Sex', 'SibSp','Parch','Fare','Embarked']]
y_train_df = train_df['Age']
X_test_df = test_df[['PassengerId','Pclass', 'Sex', 'SibSp','Parch','Fare','Embarked']]
y_test_df = test_df['Age']

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 5)

knn.fit(X_train_df , y_train_df)

pred_values = knn.predict(X_test_df)

test_df['Age'] = pred_values
df_complete = pd.concat([train_df, test_df])
df_complete.sort_values(['PassengerId'],inplace=True)
titanic['Age'] = df_complete['Age']
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.distplot(df_complete['Age'].dropna())
plt.show()
#After filling up the missing values still Age distribution follows a right skewed distribution. 
# We will have to scale it
#MinMax Scalar for numeric columns...
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(titanic['Fare'].reshape(-1, 1))
titanic['Fare_Scaled'] = df_scaled
titanic['AgeCategory'] = titanic['Age'].apply(lambda x: 'Infant' if x < 1.0 else('Child' if x < 10.0 else('Adult' if x < 75 else 'OldAged')))
le.fit(titanic['AgeCategory'])
titanic['AgeCategory'] = le.transform(titanic['AgeCategory'])
titanic['FamilyCount'] = titanic['SibSp'] + titanic['Parch'] + 1
##So categories familycount into singles midsizedfamily largefamily >  & labelencode 0,1,2
titanic['FamilyCategory'] = titanic['FamilyCount'].apply(lambda x: 0 if x == 1 else( 2 if x > 4  else 1))

#work on cabin level ...if Nan put as unknown 'U' else the first letter of the cabin...all cabins in same level has starts with same letter
titanic['Cabin'] = titanic['Cabin'].fillna(value="U")

titanic['CabinLevel'] =   titanic['Cabin'].apply( lambda x : x[:1])

titanic['CabinLevel'] =   titanic['CabinLevel'].apply( lambda x : 1 if x == 'A' else( 3 if (x == "F" or x == "G")  else (0 if x == "U" else 2) ))
#Title Can be separated from Name
titanic['Title'] = titanic['Name'].apply(lambda x :(x.split(", ")[1]).split(".")[0] )
#100 % survival rate for Lady, Mlle, Mme, Sir and The Countess...so might be high ranked people 
# Can replace Lady, Mlle , Mme , the Countess - Lady
# Capt, Col, Jonkheer, Rev  - Ranked Personal
# Miss and Ms to - Miss
titanic['Title'] = titanic['Title'].apply(lambda x: 'Lady' if ( x == "Mlle" or x == "Mme" or x == "the Countess" ) else ('Ranked' if (x == "Capt" or x == "Col" or x == "Rev") else ( "Miss" if (x == "Ms") else x)) )
le.fit(titanic['Title'])
titanic['Title'] = le.transform(titanic['Title'])


#Pclass and Sex can be combined as depending on pclass and sex the survival rate looks dependent
titanic['Pclass_Sex'] = titanic['Pclass'] * titanic['Sex']

titanic['Embarked_Sex'] = titanic['Embarked'] * titanic['Sex']

#Using target to generate features..how many people in each embarked survived..mean of that value 
train.sort_values(['Embarked'])
#total survived for each embarked , total passengers in each embarked
total_passengers_Emb_C = train[train['Embarked'] == 'C'].count()['PassengerId']
total_passengers_survived_Emb_C = train[ (train['Embarked'] == 'C') & ( train['Survived'] == 1)].count()['PassengerId']
#df[df['Survived'] == 1 ].count()['PassengerId']
Emb_C_mean = (total_passengers_survived_Emb_C * 1.0) / total_passengers_Emb_C
Emb_C_mean

total_passengers_Emb_S = train[train['Embarked'] == 'S'].count()['PassengerId']
total_passengers_survived_Emb_S = train[ (train['Embarked'] == 'S') & ( train['Survived'] == 1)].count()['PassengerId']
#df[df['Survived'] == 1 ].count()['PassengerId']
Emb_S_mean = (total_passengers_survived_Emb_S * 1.0) / total_passengers_Emb_S
Emb_S_mean

total_passengers_Emb_Q = train[train['Embarked'] == 'Q'].count()['PassengerId']
total_passengers_survived_Emb_Q = train[ (train['Embarked'] == 'Q') & ( train['Survived'] == 1)].count()['PassengerId']
#df[df['Survived'] == 1 ].count()['PassengerId']
Emb_Q_mean = (total_passengers_survived_Emb_Q * 1.0) / total_passengers_Emb_Q

titanic['Emb_Surivived_mean'] = titanic['Embarked'].apply(lambda x: 0.55 if x == "C" else( 0.39 if x == "Q" else 0.34))
titanic['Emb_label_encoding'] = titanic['Embarked'].apply(lambda x: 1 if x == "C" else( 3 if x == "Q" else 2))
g = sns.PairGrid(titanic,
                 x_vars=["Embarked", "AgeCategory", "CabinLevel"],
                 y_vars=["Pclass", "Sex"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel");
titanic.sort_values('PassengerId',inplace=True)
#Rearrange columns and drop the unnecessary ones
titanic = titanic[['PassengerId','Embarked','Pclass','Sex','Fare_Scaled','AgeCategory','FamilyCount','FamilyCategory','CabinLevel','Title','Pclass_Sex','Embarked_Sex','Emb_label_encoding','Survived']]
X_train = titanic.iloc[:,0:13]
y_train = titanic.iloc[:,13:14]
X_train.head()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier().fit(X_train, np.ravel(y_train))
feature_df = (X_train.columns)
feature_df = feature_df.tolist()
feature_df

feature_imp = clf.feature_importances_.tolist()
feature_imp


feat = pd.DataFrame({'feat':feature_df})
feat['ImpVal'] = feature_imp
feat.sort_values('ImpVal',inplace=True,ascending=False)
feat
#Drop the column with least value for importance and proceed iwth rest as there are very few columns in this data set.
titanic = titanic[['PassengerId','Embarked','Pclass','Sex','Fare_Scaled','AgeCategory','FamilyCount','FamilyCategory','CabinLevel','Title','Pclass_Sex','Survived']]
#Now we can split train and test and apply other ML algo
titanic.Survived = titanic.Survived.astype(int)
trainSet = titanic[titanic['PassengerId'].isin(range(1,892))]
testSet =  titanic[titanic['PassengerId'].isin(range(892,1310))]
#titanic.info()
#Random Forest Predictor
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(trainSet.iloc[:,1:11], trainSet.iloc[:,11:12], random_state = 3)
clf = RandomForestClassifier().fit(X_train, np.ravel(y_train))

print('Accuracy of RandomForest classifier on training set: {:.2f}'
     .format(clf.score(X_train, np.ravel(y_train))))
print('Accuracy of RandomForest classifier on validation set: {:.2f}'
     .format(clf.score(X_val, np.ravel(y_val))))
#To get the submission file for kaggle 
pred_values = clf.predict(testSet.iloc[:,1:11])
results = test['PassengerId']
results = results.to_frame()
results['Survived'] = pred_values
results.to_csv("results.csv", sep=',',index=False) 
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,np.ravel(y_train))
print(model.score(X_train,np.ravel(y_train)))
print(model.score(X_val,np.ravel(y_val)))
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
model.fit(X_train,np.ravel(y_train))
print(model.score(X_train,np.ravel(y_train)))
print(model.score(X_val,np.ravel(y_val)))
#Gradient Boost 
from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0).fit(X_train, np.ravel(y_train))
print('Accuracy of Gradient Boost Decision Tree classifier on training set: {:.2f}'
     .format(clf_gb.score(X_train, np.ravel(y_train))))
print('Accuracy of Gradient Boost Decision Tree classifier on validation set: {:.2f}'
     .format(clf_gb.score(X_val, np.ravel(y_val))))
from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators': [50, 100, 200, 300,500] , 'learning_rate' : [0.1,0.5,1.0] , 'max_depth' : [1,2,3,4,5] }
clf_gb_grid = GradientBoostingClassifier()

grid_clf_gb_acc = GridSearchCV(clf_gb_grid, param_grid = grid_values)
grid_clf_gb_acc.fit(X_train,np.ravel( y_train))
y_decision_fn_scores_acc = grid_clf_gb_acc.decision_function(X_val) 

print('Grid best parameter (max. accuracy): ', grid_clf_gb_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_gb_acc.best_score_)
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(X_train, np.ravel(y_train))

print('Accuracy of XGBOOST classifier on training set: {:.2f}'
     .format(xgb1.score(X_train, np.ravel(y_train))))
print('Accuracy of XGBOOST classifier on validation set: {:.2f}'
     .format(xgb1.score(X_val, np.ravel(y_val))))
grid_values = {'n_estimators': [300] , 'learning_rate' : [0.05] , 'max_depth' : [5] , 'min_child_weight' : [1],
              'colsample_bytree': [0.8] , 'subsample' : [0.6], 'gamma': [0]}
clf_xgb_grid = XGBClassifier(seed=2,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1)

clf_xgb_grid_acc = GridSearchCV(clf_xgb_grid, param_grid = grid_values)
clf_xgb_grid_acc.fit(X_train, np.ravel(y_train))
#y_decision_fn_scores_acc = clf_xgb_grid_acc.decision_function(X_val) 

print('Grid best parameter (max. accuracy): ', clf_xgb_grid_acc.best_params_)
print('Grid best score (accuracy): ', clf_xgb_grid_acc.best_score_)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
#X_train, X_test, y_train, y_test = train_test_split(trainSet.iloc[:,1:11], trainSet.iloc[:,11:12], 
#    test_size = 0.2, random_state = 0)

X_train =  trainSet.iloc[:,1:11]
y_train =  trainSet.iloc[:,11:12]
X_test =   testSet.iloc[:,1:11]
y_test =   testSet.iloc[:,11:12]

# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1st level models.
models = [
    ExtraTreesClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBClassifier(seed = 0,  learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    
# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = accuracy_score, n_folds = 4, 
    stratified = True, shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = XGBClassifier(seed = 0,  learning_rate = 0.1, 
    n_estimators = 100, max_depth = 3)
    
# Fit 2nd level model
model = model.fit(S_train, np.ravel(y_train))

# Predict
y_pred = model.predict(S_test)

# Final prediction score
print('Final prediction score: [%.8f]' % accuracy_score(np.ravel(y_test), np.ravel(y_pred)))
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED,shuffle=False)
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        #params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def score(self,x,y):
        return self.clf.score(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

        
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'random_state' : 0,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'random_state' : 0,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'random_state' : 0,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
    'random_state' : 0,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'random_state' : 0,
    'C' : 0.025
    }

xgboost_params = {
    'learning_rate'  : 0.1,
    'seed' : 0,
    'n_estimators' : 1000,
    'max_depth' : 5,
    'min_child_weight' : 1,
    'gamma' : 0,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'objective' : 'binary:logistic',
}

binomialnb_params = {
    'alpha' : .01
}

multinomialnb_params = {
    'alpha' : .01
}

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
xgb = SklearnHelper(clf=XGBClassifier, seed=SEED, params=xgboost_params)
bernb = SklearnHelper(clf=BernoulliNB, seed=SEED, params=binomialnb_params)
multinb = SklearnHelper(clf=MultinomialNB, seed=SEED, params=multinomialnb_params)

def get_oof(clf, X_train, y_train, X_val,y_val,trainSet,testSet):
    clf.train(X_train,y_train)
    print('Accuracy of The classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
    print('Accuracy of The classifier on validation set: {:.2f}'
     .format(clf.score(X_val, y_val)))
    return clf.predict(trainSet),clf.predict(testSet)
train_pred_val_rf, test_pred_val_rf = get_oof(rf,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) # Random Forest
train_pred_val_et,test_pred_val_et =  get_oof(et, X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) #Extra Trees
train_pred_val_ada,test_pred_val_ada = get_oof(ada, X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) # AdaBoost 
train_pred_val_gb, test_pred_val_gb = get_oof(gb,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) # Gradient Boost
train_pred_val_svc, test_pred_val_svc = get_oof(svc,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) # Support Vector Classifier
train_pred_val_xgb,test_pred_val_xgb = get_oof(xgb,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) #XGBoost
train_pred_val_bernb,test_pred_val_bernb = get_oof(bernb,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) #Bernouli Naive Bayes
train_pred_val_multinb,test_pred_val_multinb = get_oof(multinb,X_train, np.ravel(y_train), X_val,y_val,trainSet.iloc[:,1:11],testSet.iloc[:,1:11]) #Multinomial Naive bayes
#Removed highly correlated models and selecting only three for futhre processing
base_predictions_train = pd.DataFrame( {
     'RandomForest': train_pred_val_rf.ravel(),
    # 'ExtraTrees' : train_pred_val_et.ravel(),
    # 'AdaBoost': train_pred_val_ada.ravel(),
     'GradientBoost': train_pred_val_gb.ravel(),
     # 'SupportVector' : train_pred_val_svc.ravel(),
     # 'XGBoost' :   train_pred_val_xgb.ravel() , 
     # 'BernouliNB' : train_pred_val_bernb.ravel(),
      'MultinomialNB' : train_pred_val_multinb.ravel()
    })
base_predictions_train.head()
base_predictions_test = pd.DataFrame( {
     'RandomForest': test_pred_val_rf.ravel(),
     #'ExtraTrees' : test_pred_val_et.ravel(),
     #'AdaBoost': test_pred_val_ada.ravel(),
      'GradientBoost': test_pred_val_gb.ravel(),
     # 'SupportVector' : test_pred_val_svc.ravel(),
     # 'XGBoost' :   test_pred_val_xgb.ravel(),
     # 'BernouliNB' : test_pred_val_bernb.ravel(),
      'MultinomialNB' : test_pred_val_multinb.ravel()
    })
base_predictions_test.head()
base_predictions_test.corr()
import seaborn as sns
plt.figure(figsize = (8,8))
sns.heatmap(base_predictions_test.corr(), annot=True, fmt=".2f")
y_train = np.ravel(trainSet['Survived'])
#Second level model XGBoost
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(base_predictions_train,y_train)
predictions = xgb1.predict(base_predictions_test)
xgb1.score(base_predictions_train,y_train)
results['Survived'] = predictions
results.to_csv("results_ensemble.csv", sep=',',index=False) #).77 kaggle only, might be tuning params is required.
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(trainSet.iloc[:,1:11], trainSet.iloc[:,11:12], random_state = 3)

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X_train, np.ravel(y_train), cv=kfold)
print(results.mean())
ensemble.fit(X_train, y_train)
pred_values = ensemble.predict(testSet.iloc[:,1:11])