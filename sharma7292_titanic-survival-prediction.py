### Ignore Deprecation and Future Warnings
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning) 
warnings.filterwarnings('ignore', category = FutureWarning) 

### Standard Inputs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
plt.style.use('bmh')                    # Use bmh's style for plotting

from collections import Counter

### Sklearn Imports

# Standards

from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV


### Load Data

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
IDtest = test["PassengerId"]

train.shape
# Visualising Train and Test data

train.head(5)
test.head(5) 
# Outlier Detection (Interquartile Range)

def IQR_outlier(df,n,features):
  
    outlier_indices=[]
    
    #Iterating over features(columns)
    
    for col in features:
        Q1=np.percentile(df[col],25)
        Q3=np.percentile(df[col],75)
        
        # Interquartile range
        IQR=Q3-Q1
        
        # outlier step
        outlier_step=1.5*IQR
        
        outlier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        print('Total Number Outliers of',col,' : ',len(outlier_list_col))
        print('Percentage of Outliers of',col,' : ',np.round(len(outlier_list_col)/len(df[col])*100),'%')
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(k for k, v in outlier_indices.items() if v>n)
    
    return multiple_outliers

num_features=['Age','SibSp','Parch','Fare']

Outliers_to_drop=IQR_outlier(train,2,num_features)

print('Total number of Outlier indices : ', len(Outliers_to_drop))

train.loc[Outliers_to_drop,num_features]
# Visualing Age, SibSp, Parch and Fare data with and without outliers

num_features=['Age','SibSp','Parch','Fare']

for feature in num_features:
    plt.figure()
    g=sns.boxplot(x=feature,data=train)

print('Skewness :')
print(train[num_features].skew())

### Dropping outliers
train_temp=train.copy()
train_temp=train_temp.drop(Outliers_to_drop,axis=0).reset_index(drop=True)

print('Skewness in Data without outliers :')
print(train_temp[num_features].skew())

for feature in num_features:
    plt.figure()
    g=sns.boxplot(x=feature,data=train_temp)
print('Skewness in Data with outliers :')
print(train[num_features].skew())

print('Skewness in Data without outliers :')
print(train_temp[num_features].skew())

## Joining Train and Test set to get same number of features during categorical conversion

train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
 
print('Dataset shape :',dataset.shape)

# Fill empty and NaNs with NaN        

dataset=dataset.fillna(np.nan)

dataset.isnull().sum()
# Information

train.info()
train.dtypes

# Data Summary

train.describe()
### Correlation matrix in numerical values

num_features.insert(0,'Survived')
g=sns.heatmap(train[num_features].corr(),annot=True, fmt = ".2f")

print('Skew:',train.Age.skew())
train.Age.describe()
# Explore Age vs Survived

# Plotting the distribution of Age amongst passengers who survived and wthose who didn't.
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

# Overlapping the two plots
plt.figure()
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])

# Explore SibSp feature vs Survived

g = sns.catplot(x="SibSp",y="Survived",data=train,kind="bar", height = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Parch feature vs Survived
g=sns.catplot(x='Parch',y='Survived',data=train,kind='bar',height=6,palette='muted')
g.despine(left=True)
g = g.set_ylabels("survival probability")

dataset["Fare"].isnull().sum()
# Imputing the 1 missing value with median value of the combined dataset
dataset["Fare"]=dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution
g=sns.distplot(dataset['Fare'],label='Skewness: %2f'%(dataset['Fare'].skew()))
g.legend(loc='best')
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g=sns.distplot(dataset['Fare'],label='Skewness: %2f'%(dataset['Fare'].skew()))
g.legend(loc='best')
### Sex

g=sns.catplot(x='Sex',y='Survived',data=train,kind='bar')
train[['Sex','Survived']].groupby('Sex').mean()

# Converting sex to categorical values 0: male, 1:female

dataset['Sex']=dataset['Sex'].map({'male':0,'female':1})
### Pclass

g=sns.catplot('Pclass','Survived',hue='Sex',data=train,kind='bar')
g=sns.catplot('Pclass','Survived',data=train,kind='bar')

print('Number of Null entries: ',dataset['Embarked'].isnull().sum())
print('Most common dock: ',dataset.Embarked.mode()[0])
# Filling Embarked with most common dock 

dataset['Embarked']=dataset['Embarked'].fillna('S')

# Exploring Embarked

g=sns.catplot(x='Embarked',y='Survived',data=train,kind='bar')

# Exploring embarked with Pclass

g=sns.catplot(x='Embarked',y='Survived',hue='Pclass',data=train,kind='bar')
g=sns.catplot(x='Pclass',col='Embarked',data=train,kind='count')
dataset.Name.head(5)
# Visualising Cabin data
dataset.Cabin.head(5)
display(dataset.Cabin.shape)
print('Number of null values : ', dataset.Cabin.isnull().sum())
print('Percentage of null values : ', round(dataset.Cabin.isnull().sum()/dataset.Cabin.shape[0]*100),'%')
dataset.Ticket.head(10)
# Identifying missing values in the dataset

dataset.isnull().sum()
# Features most correlated with age
g=sns.heatmap(dataset[['Age','Fare','Parch','Pclass','Sex','SibSp']].corr(),annot=True,fmt='.2f')
g=sns.catplot(y='Age',x='SibSp',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Sex',data=dataset,kind='box')
g=sns.catplot(y='Age',x='Sex',hue='Pclass',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Parch',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Pclass',data=dataset,kind='box')
plt.figure()
g=sns.lineplot(x='Pclass',y='Fare',data=dataset)
plt.figure()
g=sns.lineplot(x='Age',y='Fare',data=dataset)
## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows
index_NaN_age = dataset["Age"][dataset.Age.isnull()].index

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset['SibSp'][i]) & (dataset['Parch'] == dataset['Parch'][i]) & (dataset['Pclass'] == dataset['Pclass'][i]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
        
dataset.isnull().sum()
dataset.Cabin[dataset.Cabin.notna()].head(10)
## Replacing each Cabin entry with the first letter, and replacing the missing cabin entries with 'X'

dataset.Cabin=pd.Series(['X' if pd.isnull(i) else i[0] for i in dataset.Cabin])                        
dataset.head()
print('Mean : ', dataset[['Cabin','Survived']].groupby('Cabin').mean())
print('Count : ', dataset[['Cabin','Survived']].groupby('Cabin').count())
g=sns.catplot(x='Cabin',y='Survived',data=dataset,kind='bar', order=['A','B','C','D','E','F','G','T','X']) 
dataset.Name.head()
# The title is mentioned as the 2 word in the string

dataset['Title']=pd.Series([i.split(',')[1].split('.')[0].strip() for i in dataset.Name])
dataset.head()
g=sns.countplot(x='Title',data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].replace(['Ms', 'Mlle'], 'Miss')
dataset["Title"] = dataset["Title"].replace(['Mme'], 'Mrs')
dataset[['Name','Title']].head(5)
dataset['Ticket'].head(10)
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.

Ticket=[]
for i in list(dataset['Ticket']):
    if i.isdigit():
        Ticket.append('X')
        
    else:
        Ticket.append(i.split(' ')[0])

dataset['Ticket']=Ticket
dataset.Ticket.head(10)
dataset.Ticket.unique()
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip()) #Take prefix
    else:
        Ticket.append(i)

dataset['Ticket']=Ticket
dataset.Ticket.unique()
# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")
dataset["Fsize"].replace(to_replace = [1], value = 'Single', inplace = True)
dataset["Fsize"].replace(to_replace = [2], value = 'Small', inplace = True)
dataset["Fsize"].replace(to_replace = [3,4], value = 'Medium', inplace = True)
dataset["Fsize"].replace(to_replace = [5,6,7,8,11], value = 'Large', inplace = True)

g=sns.catplot(x='Fsize',y='Survived',data=dataset,kind='bar',order=['Single','Small','Medium','Large'] ) 
dataset.head(5)
# Converting categorical data to usable form

dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cab")
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Fsize"], prefix="Fam") 
dataset = pd.get_dummies(dataset, columns = ["Pclass"], prefix="Pc")
dataset = pd.get_dummies(dataset, columns = ["Title"], prefix="Title") 
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="Tick")
# Dropping the unnecessary variables
# labels : Name, Passenger Id
dataset.drop(labels = ['Name','PassengerId'], axis = 1, inplace = True)
dataset.head(2)
dataset.shape
from sklearn.decomposition import PCA
dataset1=dataset.copy()

dataset1.drop(labels='Survived',axis=1,inplace=True)
pca=PCA(0.999,whiten=True).fit(dataset1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
data=pca.transform(dataset1)

data.shape

#train=dataset[:train_len]
#test=dataset[train_len:]

#test.drop(labels='Survived',axis=1,inplace=True)

#train['Survived']=train['Survived'].astype(int)

#Y_train=train['Survived']

#X_train=train.drop(labels='Survived',axis=1)

Y_train=dataset[:train_len]['Survived']
X_train=data[:train_len]
test=data[train_len:]



#train=dataset[:train_len]
#test=dataset[train_len:]

#test.drop(labels='Survived',axis=1,inplace=True)

#train['Survived']=train['Survived'].astype(int)

#Y_train=train['Survived']

#X_train=train.drop(labels='Survived',axis=1)

#X_train.head()
#Y_train.head()
### Model Imports

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10) 
# Modeling step to test differents algorithms 

random_state = 42
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))

cv_results = [] 
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","RandomForest","KNeighbours","LogisticRegression"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


cv_res.head()
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 'gamma': [ 0.001, 0.01, 0.1, 1],'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_
DTC = DecisionTreeClassifier()

dt_param_grid = {'max_features': ['auto', 'sqrt', 'log2'],'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                 'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],'random_state':[42]}

gsDTC = GridSearchCV(DTC,param_grid = dt_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsDTC.fit(X_train,Y_train)

DTC_best = gsDTC.best_estimator_

# Best score
gsDTC.best_score_
RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],"max_features": [1, 3, 10],"min_samples_split": [2, 3, 10],"min_samples_leaf": [1, 3, 10],
                 "bootstrap": [False],"n_estimators" :[100,300],"criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
LRC = LogisticRegression() 

lr_param_grid = {'penalty':['l1', 'l2'],'C': np.logspace(0, 4, 10)}

gsLRC=GridSearchCV(LRC,param_grid=lr_param_grid,cv=kfold,scoring='accuracy', n_jobs= 4, verbose = 1)

gsLRC.fit(X_train,Y_train)

LRC_best = gsLRC.best_estimator_

# Best score
gsLRC.best_score_
KNNC = KNeighborsClassifier()
knn_param_grid = {'n_neighbors':[3, 4, 5, 6, 7, 8],'leaf_size':[1, 2, 3, 5],
              'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree','kd_tree','brute']}

gsKNNC=GridSearchCV(KNNC,param_grid=knn_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)

gsKNNC.fit(X_train,Y_train)

KNN_best=gsKNNC.best_estimator_

# Best score
gsKNNC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsDTC.best_estimator_,"DT learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsLRC.best_estimator_,"LR learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsKNNC.best_estimator_,"KNN learning curves",X_train,Y_train,cv=kfold)

'''
names_classifiers = [("RandomForest",RFC_best),("Desicion Tree",DTC_best)]

for nclassifier in range(2):
    name = names_classifiers[nclassifier][0]
    classifier = names_classifiers[nclassifier][1] 
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    plt.figure(figsize=(8,8))
    g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " : feature importance")
'''    

test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_DTC = pd.Series(DTC_best.predict(test), name="DTC")
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_LRC = pd.Series(LRC_best.predict(test), name="LRC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_SVMC,test_Survived_DTC,test_Survived_RFC,test_Survived_LRC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[('svc', SVMC_best),('rfc', RFC_best), ('lrc', LRC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)
'''
test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("Titanic_test_set_prediction.csv",index=False)

'''

test_Survived = votingC.predict(test).astype(int)
submission = pd.DataFrame({
        "PassengerId": IDtest,
        "Survived": test_Survived
    })
submission.to_csv('Titanic_test_prediction_V9.csv', index=False)
accuracy_score(Y_train,votingC.predict(X_train))