# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , VotingClassifier , AdaBoostClassifier , ExtraTreesClassifier , GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score , StratifiedKFold , GridSearchCV , learning_curve
from xgboost.sklearn import XGBClassifier
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_len=len(train_df)
train_len

train_df.head(1)
test_df.head(1)
dataset=pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
#Id Variable : PassengerId
IDtest = test_df["PassengerId"]
#Target : Survived
print(train_df.describe(include=['O']))
# Name , Sex , Ticket , Cabin , Embarked are Character/Object Type

print(train_df.describe())
# PassengerId , Survived(D) , Pclass(D) , Age(C) ,SibSp(D), Parch(D) and Fare(C) are Numeric : Discrete(D),Continuous(C)


print("Pclass");print(len(train_df.Pclass.unique()))
# Sex , Embarked , Pclass are Categorical

# Age , Fare are Continuous
train_df.isnull().sum()
dataset.isnull().sum()
# Name / Title
# Some persons with distinguished titles might be given preference over others hence Deriving TITLE and dropping NAME 
# could be a good feature
dataset['Name'].head()
dataset_title=[i.split(",")[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title']=pd.Series(dataset_title)
dataset.head()
g=sns.countplot(x='Title',data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 
# We can replace many titles with a more common name or classify them as Rare.
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col',\
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Sex-Survived
# Sex=female > 74 % . Hence 'Sex' should be included in the Model.
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)

## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
dataset["Ticket"] = Ticket
dataset.head()
# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])

dataset.head()
# Embarked
freq_port=train_df.Embarked.dropna().mode()[0]
freq_port

dataset['Embarked']=dataset['Embarked'].fillna(freq_port)
dataset.head()
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
sns.heatmap(train_df[["Pclass","SibSp","Parch","Fare","Age","Survived"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# Pclass(D)-Survived
# Significant correlation with Pclass=1 > 62 %. Should be included in our Model.
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# SibSp(D)-Survived
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Parch(D)-Survived
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
  

dataset.head()
# Fare(C)-Survived
sns.FacetGrid(train_df,col='Survived').map(sns.barplot,'Sex','Fare')
# Higher Fare paying passengers survived , hence it can be included in the model.Since its continuous numeric , a
# new feature FareRange needs to be created for analysis

# Since Fare is Continuous we will plot distribution plot next
# Explore Fare distribution 
g = sns.distplot(train_df["Fare"], color="m", label="Skewness : %.2f"%(train_df["Fare"].skew()))
g = g.legend(loc="best")

# This highly skewed plot should be transformed using log t oreduce the skewness
# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# Plot after reducing skewness
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")
dataset.head()
dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].dropna().median())
#train_df['Fare']=train_df['Fare'].fillna(train_df['Fare'].dropna().median(),inplace=True)

dataset.head()
dataset['FareBand']=pd.qcut(dataset['Fare'],4)
dataset[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean()
dataset.loc[dataset['Fare'] <=2.06,'Fare']=0
dataset.loc[(dataset['Fare'] <=2.67) & (dataset['Fare'] > 2.06)  ,'Fare']=1
dataset.loc[(dataset['Fare'] <=3.44) & (dataset['Fare'] > 2.67)  ,'Fare']=2
dataset.loc[(dataset['Fare'] <=6.2) & (dataset['Fare'] > 3.44)  ,'Fare']=3
dataset.loc[dataset['Fare'] > 6.2  ,'Fare']=4
dataset['Fare']=dataset['Fare'].astype(int)
dataset.head()
# Age
# Explore Age distibution
g = sns.FacetGrid(train_df,col='Survived')
g.map(sns.distplot,'Age')

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) &
                               (dataset['Parch'] == dataset.iloc[i]["Parch"]) &
                               (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        #print(i)
        dataset['Age'].iloc[i] = age_med
        #print(train_df.iloc[i])
dataset['AgeBand']=pd.cut(dataset['Age'],5)

dataset[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean()
dataset.loc[dataset['Age'] <=16 , 'Age' ] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32) , 'Age' ] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <=48) , 'Age' ] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <=64) , 'Age' ] = 3
dataset.loc[(dataset['Age'] > 64) , 'Age' ] = 4
dataset['Age']=dataset['Age'].astype(int)

dataset.head()
dataset.head(1)
# DROP
dataset=dataset.drop(['Name','Parch','PassengerId','SibSp','Fsize','FareBand','AgeBand'],axis=1)
dataset.head(1)
# ENCODING
dataset=pd.get_dummies(dataset)

dataset.columns
dataset=pd.get_dummies(dataset,prefix=['Pclass','Age','Fare'],columns=['Pclass','Age','Fare'])
dataset.columns
## Separate train dataset and test dataset

train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 
train["Survived"] = train["Survived"].astype(int)
X_train = train.drop(labels = ["Survived"],axis = 1)
Y_train = train["Survived"]
X_test=test
print(X_train.shape)
print(X_test.shape)

# Applying k-Fold Cross Validation

kfold= StratifiedKFold(n_splits=10)
random_state=2
classifiers=[]
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=kfold,scoring='accuracy',n_jobs=-1))
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
# Cross Validation Results
cv_res=pd.DataFrame({'CrossValMeans':cv_means , 'CrossValErrors':cv_std , 'Algorithm':['SVC','DTC','RFC','KNN','LR','LDA','ADA','XT','GBC','XGB']})

sns.barplot('CrossValMeans','Algorithm',data=cv_res)

# SVC
svc_classifier=SVC(probability=True)
svc_param_grid = [{'C': [1], 'kernel': ['rbf'],
                   'gamma': [0.1],
                  'cache_size':[100],
                  'coef0':[0.1],
                  'degree':[1],
                  'tol':[0.001]}]

gs_SVC = GridSearchCV(estimator = svc_classifier,
                           param_grid = svc_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_SVC = gs_SVC.fit(X_train, Y_train)
svc_best_params = gs_SVC.best_params_
svc_best_score = gs_SVC.best_score_
svc_best=gs_SVC.best_estimator_
svc_best_score

svc_best
# DTC
dtc_classifier=DecisionTreeClassifier()
dtc_param_grid = [{'criterion': ['gini'], 
                   "min_samples_split": [2],
              "max_depth": [None],
              "min_samples_leaf": [5],
              "max_leaf_nodes": [10],
                   'splitter': ['best']}]
gs_DTC = GridSearchCV(estimator = dtc_classifier,
                           param_grid = dtc_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_DTC = gs_DTC.fit(X_train, Y_train)
dtc_best_params = gs_DTC.best_params_
dtc_best_score = gs_DTC.best_score_
dtc_best=gs_DTC.best_estimator_
dtc_best_score
dtc_best
# RFC
rfc_classifier=RandomForestClassifier()
rfc_param_grid = [{'n_estimators':[1200] ,'criterion': ['entropy'],
                   'max_features':['auto'] ,'min_samples_split':[9],'min_samples_leaf':[2],
                  'bootstrap' : [True], 'n_jobs':[-1] ,'oob_score':[True]}]
gs_RFC = GridSearchCV(estimator = rfc_classifier,
                           param_grid = rfc_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_RFC = gs_RFC.fit(X_train, Y_train)
rfc_best_params = gs_RFC.best_params_
rfc_best_score = gs_RFC.best_score_
rfc_best=gs_RFC.best_estimator_
rfc_best_score
rfc_best
# KNN
knn_classifier=KNeighborsClassifier()
knn_param_grid = [{'n_neighbors':[10],'weights':['uniform'],'algorithm':['brute'],
                  }]
gs_KNN = GridSearchCV(estimator = knn_classifier,
                           param_grid = knn_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_KNN = gs_KNN.fit(X_train, Y_train)
knn_best_params = gs_KNN.best_params_
knn_best_score = gs_KNN.best_score_
knn_best=gs_KNN.best_estimator_
knn_best_score
knn_best
# LR
lr_classifier=LogisticRegression()
lr_param_grid = [{'penalty':['l1','l2'] , 'C':[1]}]
gs_LR = GridSearchCV(estimator = lr_classifier,
                           param_grid = lr_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_LR = gs_LR.fit(X_train, Y_train)
lr_best_params = gs_LR.best_params_
lr_best_score = gs_LR.best_score_
lr_best=gs_LR.best_estimator_
lr_best_score
lr_best
# # LDA
# lda_classifier=DecisionTreeClassifier()
# lda_param_grid = [{'criterion': ['gini','entropy'], 'splitter': ['best','random']}]
# gs_LDA = GridSearchCV(estimator = lda_classifier,
#                            param_grid = lda_param_grid,
#                            scoring = 'accuracy',
#                            cv = kfold,
#                            n_jobs = -1)
# gs_LDA = gs_LDA.fit(X_train, Y_train)
# lda_best_params = gs_LDA.best_params_
# lda_best_score = gs_LDA.best_score_
# lda_best_score
# ADA - DTC
adadtc_classifier=AdaBoostClassifier(dtc_best)
adadtc_param_grid = [{'n_estimators':[500], 
                      "learning_rate":  [0.1],
                      "algorithm" : ["SAMME"],
                     }]
gs_ADADTC = GridSearchCV(estimator = adadtc_classifier,
                           param_grid = adadtc_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_ADADTC = gs_ADADTC.fit(X_train, Y_train)
adadtc_best_params = gs_ADADTC.best_params_
adadtc_best_score = gs_ADADTC.best_score_
adadtc_best=gs_ADADTC.best_estimator_
adadtc_best_score
adadtc_best
# XT
xt_classifier=ExtraTreesClassifier()
xt_param_grid = [{'n_estimators':[100],
                 'criterion':['entropy'],
                 'max_features':[None],
                 'max_depth':[None],
                 'min_samples_split':[2],
                 'min_samples_leaf':[10]
                 }]
gs_XT = GridSearchCV(estimator = xt_classifier,
                           param_grid = xt_param_grid,
                           scoring = 'accuracy',
                           cv = kfold,
                           n_jobs = -1)
gs_XT = gs_XT.fit(X_train, Y_train)
xt_best_params = gs_XT.best_params_
xt_best_score = gs_XT.best_score_
xt_best=gs_XT.best_estimator_
xt_best_score
xt_best
# # XGB
# xgb_classifier=XGBClassifier()
# xgb_param_grid = [{'booster':['gbtree', 'gblinear','dart'] ,
#                    'min_child_weight':[0.01,0.5,1,1.5] ,
#                   'max_depth' : [3,5,8,10] ,
#                   'gamma':[0,0.5,1,2] ,
#                   'subsample': [0.5,0.75,1] ,
#                   'colsample_bytree' : [0.5,0.75,1] ,
#                   'reg_lambda' : [0.5,1,1.5] ,
#                   'scale_pos_weight':[1,2,5]}]
# gs_XGB = GridSearchCV(estimator = xgb_classifier,
#                            param_grid = xgb_param_grid,
#                            scoring = 'accuracy',
#                            cv = kfold,
#                            n_jobs = -1)
# gs_XGB = gs_XGB.fit(X_train, Y_train)
# xgb_best_params = gs_XGB.best_params_
# xgb_best_score = gs_XGB.best_score_
# xgb_best=gs_XGB.best_estimator_
# xgb_best_score
#xgb_classifier.get_params().keys()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g=plot_learning_curve(gs_SVC.best_estimator_,"SVC Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_DTC.best_estimator_,"DTC Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_RFC.best_estimator_,"RFC Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_KNN.best_estimator_,"KNN Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_LR.best_estimator_,"LR Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_ADADTC.best_estimator_,"ADA DTC Learning Curve",X_train,Y_train,cv=kfold)
g=plot_learning_curve(gs_XT.best_estimator_,"XT Learning Curve",X_train,Y_train,cv=kfold)
votingC = VotingClassifier(estimators=[('svc', svc_best), ('dtc', dtc_best),
('rfc', rfc_best), ('knn',knn_best),('lr',lr_best),('adadtc',adadtc_best),('xt',xt_best)], voting='soft', n_jobs=-1)

votingC = votingC.fit(X_train, Y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)



