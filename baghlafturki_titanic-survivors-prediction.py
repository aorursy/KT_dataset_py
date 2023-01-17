# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import boxcox
from sklearn.model_selection import StratifiedKFold
# this is not to skip some columns when showing the dataframe
pd.set_option('display.max_columns', None)

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head(2)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head(2)
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
# This code allocates the passengers who survived and those who didn't into 2 different variables.
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]
# This code prints the number of survived and dead passengers along with the percentage.
print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))
#we will use scater plot to see outlier .. 
fig, ax = plt.subplots()
ax.scatter(x = train['Age'], y = train['Survived'])
plt.ylabel('Survived', fontsize=13)
plt.xlabel('Age', fontsize=13)
plt.show()
# dropping that one old grandma from the boat
train = train.drop(train[(train['Age']>79) & (train['Survived']>0.8)].index)
fig, ax = plt.subplots()
ax.scatter(x = train['Fare'], y = train['Survived'])
plt.ylabel('Survived', fontsize=13)
plt.xlabel('Fare', fontsize=13)
plt.show()
train = train.drop(train[(train['Fare']>400) & (train['Survived']>0.8)].index)
target = train['Survived']
#concatinating the two dataframes
whole_df = pd.concat([train,test],keys=[0,1], sort = False)
# Sanity check to ensure the total length matches 
whole_df.shape[0] == train.shape[0]+test.shape[0]
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))

# This is a plot to show the missing values in the merged dataset
sns.heatmap(whole_df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='copper')
ax.set_title('Dataset')

sns.barplot(x='Pclass', y='Survived', data=whole_df)
# This code shows the number of survived and dead passengers based on their class.
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=whole_df,order=None,hue_order=None)
whole_df['Fare'].fillna(whole_df[whole_df['Pclass'] == 3]['Fare'].mean(), inplace=True)
plt.figure(figsize=(15,10))
sns.heatmap(whole_df.drop('PassengerId',axis=1).corr(), vmax=0.6, annot=True)
sns.boxplot(data=whole_df, y='Age', x='Pclass')
sns.boxplot(data=whole_df, y='Age', x='SibSp')
sns.boxplot(data=whole_df, y='Age', x='Parch')

def adjust_age(df,age_s):
    #convert age feature to a numpy array
    age_array = age_s.to_numpy()
    result = []
    
    # in case cannot find similar group we use default to fill
    default = int(df["Age"].median())
    # iterate every row
    for i,val in enumerate(age_array):
        # if empty
        if np.isnan(val):
            # get the median age of the group with similar attributes
            median_age = df[(df['Pclass']==df.iloc[i]['Pclass'])&(df['SibSp']==df.iloc[i]['SibSp'])&(df['Parch']==df.iloc[i]['Parch'])]['Age'].median()
            
            try:
                # age will be nan if no similar group found
                result.append(int(median_age))
            except:
                # append the default if no median_age found
                result.append(default)
        else:
            result.append(int(val))

    return result

# filling age using our function        
whole_df['Age'] = adjust_age(whole_df,whole_df['Age'])



# Rechecking the condition of our dataset
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))
sns.heatmap(whole_df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='copper')
ax.set_title('Whole dataset')

whole_df['Cabin'].fillna(value="NA",inplace=True)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))

# train data 
sns.heatmap(whole_df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='copper')
ax.set_title('Train data')

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
whole_df['Fare'] = pd.qcut(whole_df['Fare'], 13)

sns.countplot(data=whole_df, y='Fare',hue='Survived')
# This plots the count of people from different ages who survived and died
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
sns.countplot(data=whole_df[(whole_df['Age'] >= 0)&(whole_df['Age'] <= 70)], y='Age', hue='Survived',ax=ax)

def categorize_age(val):
    if val < 1:
        return "infant"
    # Child case
    if val >= 1 and val <= 14:
        # classify as child
        return "child"
    #youth case
    elif val >= 15 and val <= 24:
        return "youth"
    # limited to 63 to keep seniors pure
    elif val >= 25 and val <= 63:
        return "adult"
    # seniors case (all dead)
    elif val >= 64:
        return "senior"
    print("Something is wrong!")

# applying the new function to age
whole_df['Age'] = whole_df['Age'].apply(categorize_age)
# plotting the survival of each age category
sns.countplot(data=whole_df, y='Age',hue='Survived')
# +1 is because im adding the person as well. 1 means alone
whole_df['famsize'] = whole_df['Parch']+whole_df['SibSp']+1

#whole_df['famsize'] = np.log1p(whole_df['famsize'])
sns.distplot(whole_df['famsize'])
whole_df['famsize'].skew()
#sns.countplot(data=train, y='famsize',hue='Survived')
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 8))
sns.barplot(data=whole_df, y='Survived',x='famsize',hue='Age',ax=ax)
sns.countplot(data=whole_df, y='famsize',hue='Survived')
def classify_fam_size(famsize):
    # highest count and high death rate
    if famsize == 1:
        return "solo"
    # higher survival rate than death rate
    elif famsize in [2,3,4]:
        return "small-family"
    # noticably higher death rate compared to the last groups
    elif famsize in [5,6]:
        return "mid-family"
    elif famsize > 6:
        return "large-family"

whole_df['famsize'] = whole_df['famsize'].apply(classify_fam_size)
whole_df['Cabin'].unique()
def take_section(code):
        return code[0]
whole_df['Cabin'] = whole_df['Cabin'].apply(take_section)
whole_df['Cabin'].unique()
sns.countplot(data=whole_df, x='Cabin',hue='Pclass')
sns.barplot(data=whole_df, x='Cabin',y='Survived',hue='Pclass')
#grouping cabins based on class
def group_cabin(code):
    # adding T because there is 1 person in T and has a class same as ppl
    # in A,C and B
    if code in ['A','C','B','T']:
        return "ACBT"
    elif code in ["F","G"]:
        return "FG"
    elif code in ['E','D']:
        return 'DE'
    else:
        return code
    
whole_df['Cabin'] = whole_df['Cabin'].apply(group_cabin)
sns.barplot(data=whole_df, x='Cabin',y='Survived')
# Getting the title of each person
whole_df['title'] = whole_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
whole_df['title'].unique()
sns.countplot(data=whole_df, y='title',hue='Survived')
sns.barplot(data=whole_df, y='title',x='Survived', hue='Sex')
def standardise_names(name):
    if name in ['Mrs','Miss','Ms', 'Lady','Mlle','the Countess','Dona']:
        return "female_titles"
    elif name in ['Mr','Master']:
        return name
    else:
        return 'others'
whole_df['title'] = whole_df['title'].apply(standardise_names)
sns.countplot(data=whole_df, y='title',hue='Survived')
tickets = {}
# grouping training set by ticket and finding the survival ratio
for ticket, df in whole_df.xs(0).groupby('Ticket'):
    # not putting high survival rate for solo ppl
    if df.shape[0]>1:
        tickets[ticket] = df['Survived'].sum()/ df.shape[0]
# setting the calculated survival rate based on the ticket
# if ticket not precalculated then set 0.5 (idk)
default = 0.5
whole_df['ticket_sr'] = whole_df['Ticket']
whole_df['ticket_sr'] = whole_df['ticket_sr'].apply(lambda x: tickets[x] if x in tickets.keys() else default)
sns.countplot(data=whole_df, hue='famsize',y='ticket_sr',orient='h')
sns.barplot(data=whole_df, y='ticket_sr',x='Survived',orient='h')
whole_df.drop(labels= ['Name','Ticket','SibSp','Parch'], axis=1,inplace=True)
whole_df = pd.get_dummies(whole_df,columns=['Pclass','Sex',"Age",'Fare','Cabin','Embarked','title','famsize'],drop_first=True)
# global setting to show all the features
pd.set_option('display.max_columns', None)
whole_df.head(3)
corr = whole_df.drop('PassengerId',axis=1).corr()
corr = corr[(corr['Survived'] > 0.1) | (corr['Survived'] < -0.1)]
corr['Survived'].sort_values(ascending=False).drop('Survived').plot(kind='bar')
# separate the sets back
train,test = whole_df.xs(0),whole_df.xs(1)
# preparing X and target
y = train['Survived'].astype(int)
X = train.drop(labels=['Survived','PassengerId'], axis=1)
# This one was used during testing the different models but they were all removed to make the notebook cleaner
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=24)
# prepping the test set to submit to kaggle
X_final,IDs =  test.drop(labels=['PassengerId','Survived'], axis=1), test['PassengerId']
# special libraries for ML
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold, KFold
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
thresh = 0.1
corr = train.drop('PassengerId',axis=1).corr()
corr = corr[(corr['Survived'] > thresh) | (corr['Survived'] < -thresh)].drop('Survived')
selected_features = corr.sort_values('Survived').index
selected_features
# make visualization of the score of different models
def make_benchmark(X,y, scoring='accuracy', cv = 5):
    # using stratified because we have more deaths than survivors
    folds = StratifiedKFold(n_splits=cv)
    # different models for benchmark
    models = [SVC(gamma='auto'),
            DecisionTreeClassifier(),KNeighborsClassifier(),
            ExtraTreeClassifier(),LogisticRegression(max_iter=10000,solver='lbfgs'),
            RidgeClassifier(),AdaBoostClassifier(),
            BaggingClassifier(),RandomForestClassifier()]
    # names of the models
    names = ["SVM","Decision Tree","KNN","Extra Trees","Logistic Regression",
             "Ridge classifier","AdaBoost","Bagging Classifier","Random Forest"]
    # cross val mean score for each model
    scores = [np.mean(cross_val_score(model, X, y, cv=folds, scoring=scoring)) for model in models]
    

    return pd.DataFrame({"model_name":names,
                         "model": models,
                         scoring+"_score":scores})
# we will use this later on gridsearch
folds = StratifiedKFold(n_splits=5)
# make benchmark
bench = make_benchmark(X,y,cv=5)
#plot benchmark
sns.barplot(x='accuracy_score',y='model_name',data=bench.sort_values('accuracy_score',ascending=False)).set_title("Models Score Benchmark")
# do benchmark but using features selected previously (has corr > 0.1)

bench = make_benchmark(X[selected_features],y,cv=5)
sns.barplot(x='accuracy_score',y='model_name',data=bench.sort_values('accuracy_score',ascending=False)).set_title("Models Score Benchmark")
## WARNING: this will take more than 10 minutes
## IF you still want to do so please uncomment the block

'''rf = RandomForestClassifier(random_state=24)
params = {'criterion':['gini','entropy'],
          'max_features':['auto'],
          'n_estimators':[150,200,300],
          'max_depth': [3,5,7,10,14],
          'class_weight':[ "balanced", "balanced_subsample"],
          'min_samples_split':[20,15,10]}

gsrf = GridSearchCV(rf , params,cv=KFold(5),verbose=2)
gsrf.fit(X,y)
gsrf.best_estimator_'''
# different numbers of estimators effect on random forest
folds = StratifiedKFold(n_splits=5)
x_axis, y_axis = [],[]
for i in range(250,800,50):
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                           criterion='entropy', max_depth=7, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=15,
                           min_weight_fraction_leaf=0.0, n_estimators=i,
                           n_jobs=None, oob_score=False, random_state=24, verbose=0,
                           warm_start=False)
    
    y_axis.append(np.mean(cross_val_score(rf, X, y, cv=folds,scoring='accuracy')))
    x_axis.append(i)

sns.lineplot(x=x_axis, y=y_axis)
#effect of  different depth on RandomForest

folds = StratifiedKFold(n_splits=5)
x_axis, y_axis = [],[]
for i in range(3,30,1):
    #our model
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                           criterion='entropy', max_depth=i, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=15,
                           min_weight_fraction_leaf=0.0, n_estimators=700,
                           n_jobs=None, oob_score=False, random_state=24, verbose=0,
                           warm_start=False)
    # add score for plotting
    y_axis.append(np.mean(cross_val_score(rf, X, y, cv=folds,scoring='accuracy')))
    x_axis.append(i)

sns.lineplot(x=x_axis, y=y_axis)
#effect of  different depth on RandomForest

folds = StratifiedKFold(n_splits=5)
x_axis, y_axis = [],[]
for i in np.linspace(0.1,1.,10):
    #our model
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                           criterion='entropy', max_depth=9, max_features=i,
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=15,
                           min_weight_fraction_leaf=0.0, n_estimators=700,
                           n_jobs=None, oob_score=False, random_state=24, verbose=0,
                           warm_start=False)
    # add score for plotting
    y_axis.append(np.mean(cross_val_score(rf, X, y, cv=folds,scoring='accuracy')))
    x_axis.append(i)

sns.lineplot(x=x_axis, y=y_axis)
folds = StratifiedKFold(n_splits=5)
x_axis, y_axis = [],[]
for i in np.linspace(0.0001,0.3,15):
    #our model
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=i, class_weight='balanced',
                           criterion='entropy', max_depth=9, max_features=.8,
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=15,
                           min_weight_fraction_leaf=0.0, n_estimators=700,
                           n_jobs=None, oob_score=False, random_state=24, verbose=0,
                           warm_start=False)
    # add score for plotting
    y_axis.append(np.mean(cross_val_score(rf, X, y, cv=folds,scoring='accuracy')))
    x_axis.append(i)

sns.lineplot(x=x_axis, y=y_axis)
folds = StratifiedKFold(n_splits=5)
x_axis, y_axis = [],[]
for i in range(1,50,3):
    #our model
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.02, class_weight='balanced',
                           criterion='entropy', max_depth=9, max_features=.8,
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=i, min_samples_split=15,
                           min_weight_fraction_leaf=0.0, n_estimators=700,
                           n_jobs=None, oob_score=False, random_state=24, verbose=0,
                           warm_start=False)
    # add score for plotting
    y_axis.append(np.mean(cross_val_score(rf, X, y, cv=folds,scoring='accuracy')))
    x_axis.append(i)

sns.lineplot(x=x_axis, y=y_axis)
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.00,
                       class_weight='balanced_subsample', criterion='gini',
                       max_depth=9, max_features=0.2, min_samples_leaf=4,
                       min_samples_split=15,n_estimators=700,
                       random_state=24, verbose=0)
np.mean(cross_val_score(rf, X, y, cv=folds, scoring='accuracy')) 
rf.fit(X,y)
y_hat = rf.predict(X_final)
submission_df = pd.DataFrame({"PassengerId":IDs,
                              "Survived": y_hat})
submission_df.to_csv('submit.csv',index=False)