# Common imports

import numpy as np

import pandas as pd



# for data preprocessing

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer



# to make this notebook's output stable across runs

np.random.seed(42)



# model and related

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier, plot_importance 

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Saving train & test shapes

ntrain = train.shape[0]

ntest = test.shape[0]



# Creating y_train variable

train_y = train['Survived']
# New all encompassing dataset

all_data = pd.concat((train, test),sort=False).reset_index(drop=True)

train_ID=train['PassengerId']

test_ID=test['PassengerId']



# Dropping the target

all_data.drop(['Survived'], axis=1, inplace=True)
type(all_data)
# feature management

def fill_cat(all_data):

    all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].value_counts().index[0], inplace=False)

    return all_data



def fill_num(all_data):

    all_data['Fare'] = all_data.groupby(['Pclass','Sex'])['Fare'].apply(lambda x: x.fillna(x.mean()))

    return all_data



def fill_age(all_data):

    all_data['Age'] = all_data.groupby(['Title','Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.mean()))

    return all_data



def add_title(all_data):    

    titles = list()

    for name in all_data['Name']:

        titles.append(name.split(',')[1].split('.')[0].strip())

    all_data['Title']=titles

       

    # normalize the titles

    normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

    }

    all_data['Title'] = all_data['Title'].map(normalized_titles)

    return all_data





def log_transform(all_data, cols):

    for col in cols:

        all_data['LogT'+col] = np.rint(np.log1p(all_data[col]))

    return all_data



def binning_family(all_data):    

    all_data['family_size'] = np.array(all_data["SibSp"] + all_data["Parch"] + 1)

    bin_ranges = [0, 1, 4, 100]

    bin_names = [1, 2, 3]

    

    all_data['family_size_range'] = pd.cut(np.array(all_data['family_size']), bins=bin_ranges)

    all_data['family_size_label'] = pd.cut(np.array(all_data['family_size']), bins=bin_ranges, labels=bin_names)

    return all_data 



def binning_age(all_data): 

    bin_ranges = [0, 5, 10, 20, 40, 60, 100]

    bin_names = [1, 2, 3, 4, 5, 6]

    

    all_data['Age_range'] = pd.cut(np.array(all_data['Age']), bins=bin_ranges)

    all_data['Age_label'] = pd.cut(np.array(all_data['Age']), bins=bin_ranges, labels=bin_names)

    return all_data



def add_ageclass(all_data):

    all_data['AgeClass'] = all_data['Age'] * all_data['Pclass'] 

    all_data['AgeClass'] = np.ceil(all_data['AgeClass']/15)

    return all_data
(all_data.pipe(fill_num)

         .pipe(fill_cat)

         .pipe(add_title) 

         .pipe(fill_age)

         .pipe(log_transform, cols = list(['Fare']))

         .pipe(binning_family)

         .pipe(binning_age)  

         .pipe(add_ageclass)

)

# Dropping some features

drop_cols=['PassengerId','Name', 'Age','SibSp','Parch','Ticket','Fare','Cabin','family_size', 'family_size_range','Age','Age_range']

all_data.drop(drop_cols, axis=1, inplace=True)

all_data.info()
# change number to object

all_data['Pclass'] = all_data['Pclass'].astype('category')

all_data['AgeClass'] = all_data['AgeClass'].astype('category')
data_num = all_data.select_dtypes(include=[np.number])

data_cat = all_data.select_dtypes(exclude=[np.number])

        

n_cat = data_cat.shape[1]



np_num_scaled = StandardScaler().fit_transform(data_num)

#np_num_scaled = data_num

df_num_scaled = pd.DataFrame(np_num_scaled, columns=list(data_num))



df_cat_encoded = pd.get_dummies(data_cat, prefix=data_cat.columns.values)

       

data_prepared =  pd.concat([df_num_scaled, df_cat_encoded], axis=1, sort=False)

data_prepared
train_data = data_prepared[:ntrain]

test_data = data_prepared[ntrain:]
# select the best features

from sklearn.feature_selection import RFE

from sklearn.svm import SVR



estimator = SVR(kernel="linear")

selector = RFE(estimator, 15, step=1)

selector = selector.fit(train_data, train_y)

selector.support_ 

selector.ranking_



zipped = sorted(zip(train_data.columns, selector.ranking_),key = lambda t: t[1])



for name, rank in zipped:

    print(name,rank)
train_sel = train_data.iloc[:,selector.ranking_==1]

test_sel = test_data.iloc[:,selector.ranking_==1]
# quick fit

clf_rf = RandomForestClassifier()

clf_et = ExtraTreesClassifier()

clf_bc = BaggingClassifier()

clf_ada = AdaBoostClassifier()

clf_dt = DecisionTreeClassifier()

clf_xg = XGBClassifier()

clf_lr = LogisticRegression()

clf_svm = SVC()

 

Classifiers = ['RandomForest','ExtraTrees','Bagging','AdaBoost','DecisionTree','XGBoost','LogisticRegression','SVM']

scores = []

models = [clf_rf, clf_et, clf_bc, clf_ada, clf_dt, clf_xg, clf_lr, clf_svm]

for model in models:

    score = cross_val_score(model, train_sel, train_y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()

    scores.append(score)

    

mode = pd.DataFrame(scores, index = Classifiers, columns = ['score_best_features']).sort_values(by = 'score_best_features',

             ascending = False)

mode
scores = []

for model in models:

    score = cross_val_score(model, train_data, train_y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()

    scores.append(score)

    

mode = pd.DataFrame(scores, index = Classifiers, columns = ['score']).sort_values(by = 'score',

             ascending = False)

mode
# For illustration only. Sklearn has train_test_split()

def split_train_test(data, y, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices,], data.iloc[test_indices,], y.iloc[train_indices,], y.iloc[test_indices,]



X_train, X_test, y_train, y_test = split_train_test(train_data, train_y, 0.01)

X_train.shape, X_test.shape
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



# LR tunning



lr_grid_param = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge



lr_gs = GridSearchCV(LogisticRegression(), lr_grid_param, scoring='accuracy', cv=10)



param_grid = {'C': np.arange(1e-05, 3, 0.1)}



lr_gs = GridSearchCV(LogisticRegression(), return_train_score=True,

                  param_grid=param_grid,  cv=10, refit='Accuracy')



lr_gs.fit(X_train, y_train)



#rf_gs.fit(X_train, y_train)

print(lr_gs.best_params_)

# Best score

print(lr_gs.best_score_)

# keep the model

lr_best = lr_gs.best_estimator_
# RFC Parameters tunning 



rf = RandomForestClassifier(max_features=1)



param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [2, 3, 4, 5],

               "n_estimators": [20, 50, 75, 100, 200]}



rf_gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)



rf_gs.fit(X_train, y_train)

#rf_gs.fit(X_train, y_train)

print(rf_gs.best_params_)

# Best score

print(rf_gs.best_score_)

# keep the model

rf_best = rf_gs.best_estimator_
#  SVM Parameters tunning 

svm = SVC(probability=True)

parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}

svm_gs = GridSearchCV(svm, parameters, cv=10, n_jobs=-1)

svm_gs.fit(X_train, y_train)



print(svm_gs.best_params_)

# Best score

print(svm_gs.best_score_)



# keep the model for predicting

svm_best = svm_gs.best_estimator_
# #ExtraTrees tunning



## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





ET_gs = GridSearchCV(clf_et,param_grid = ex_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)



ET_gs.fit(X_train, y_train)



ET_gs_best = ET_gs.best_estimator_



# Best score

ET_gs.best_score_
# Adaboost tunning

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "n_estimators": [1, 2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]

             }



DTC = DecisionTreeClassifier(random_state = 11)



ada = AdaBoostClassifier(base_estimator = DTC)



ada_gs = GridSearchCV(ada, param_grid = ada_param_grid, scoring="accuracy", cv = 10)



ada_gs.fit(X_train, y_train)



print(ada_gs.best_score_)



ada_best = ada_gs.best_estimator_
from sklearn.model_selection import RandomizedSearchCV

# XGboost tunning

gbm_param_grid = {

    'n_estimators': range(8, 20),

    'max_depth': range(6, 10),

    'learning_rate': [.4, .45, .5, .55, .6],

    'colsample_bytree': [.6, .7, .8, .9, 1]

}



# Instantiate the regressor: gbm

gbm = XGBClassifier(n_estimators=10)



# Perform random search: grid_mse

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 

                                    estimator = gbm, scoring = "accuracy", 

                                    verbose = 1, n_iter = 50, cv = 10)





# Fit randomized_mse to the data

xgb_random.fit(X_train, y_train)



xgb_best = xgb_random.best_estimator_



# Print the best parameters and lowest RMSE

print("Best parameters found: ", xgb_random.best_params_)

print("Best accuracy found: ", xgb_random.best_score_)
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Plot learning curves

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



g = plot_learning_curve(rf_gs.best_estimator_,"RF mearning curves",X_train, y_train,cv=10)

g = plot_learning_curve(ET_gs.best_estimator_,"ExtraTrees learning curves",X_train, y_train,cv=10)

g = plot_learning_curve(svm_gs.best_estimator_,"SVC learning curves",X_train, y_train,cv=10)

g = plot_learning_curve(ada_gs.best_estimator_,"AdaBoost learning curves",X_train, y_train,cv=10)

g = plot_learning_curve(xgb_random.best_estimator_,"XGBoosting learning curves",X_train, y_train,cv=10)

g = plot_learning_curve(lr_gs.best_estimator_,"Logistic Regression learning curves",X_train, y_train,cv=10)
#votingC = VotingClassifier(estimators=[('rf', rf_best), ('ExTree', ET_gs_best),

#('svc', svm_best), ('LR',lr_best),('xgb',xgb_best), ('ada',ada_best),], voting='soft', n_jobs=4)

votingC = VotingClassifier(estimators=[('rf', rf_best), ('svc', svm_best), ('LR',lr_best),], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)



votingC_pred = votingC.predict(test_data)

#accuracy_score(y_test, votingC_pred)
import seaborn as sns

test_lr = pd.Series(lr_best.predict(test_data), name="LR")

test_rf = pd.Series(rf_best.predict(test_data), name="RF")

test_svm = pd.Series(svm_best.predict(test_data), name="SVC")

test_ada = pd.Series(ada_best.predict(test_data), name="Ada")

test_xgb = pd.Series(xgb_best.predict(test_data), name="XGB")

test_et = pd.Series(ET_gs_best.predict(test_data), name="ExTree")

test_voting = pd.Series(votingC_pred, name="VotingC")



# Concatenate all classifier results

ensemble_results = pd.concat([test_rf,test_et,test_ada,test_lr,test_svm, test_xgb, test_voting],axis=1)

#ensemble_results = pd.concat([test_rf,test_et,test_ada,test_lr,test_svm, test_xgb, test_voting, y_test.reset_index(drop=True)],axis=1)

g= sns.heatmap(ensemble_results.corr(),annot=True)

#accuracy_score(y_test, emsemble_results[:,1])
Titan=votingC.predict(test_data)

sub = pd.DataFrame()

sub['PassengerId'] = test_ID

sub['Survived'] = Titan

sub.to_csv('votingSoft3.csv',index=False)
RF_pred=rf_best.predict(test_data)

sub = pd.DataFrame()

sub['PassengerId'] = test_ID

sub['Survived'] = RF_pred

sub.to_csv('RF_prediction.csv',index=False)
LR_pred=lr_best.predict(test_data)

sub = pd.DataFrame()

sub['PassengerId'] = test_ID

sub['Survived'] = LR_pred

sub.to_csv('LR_prediction.csv',index=False)