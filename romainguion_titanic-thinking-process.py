import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os ; os.environ['OMP_NUM_THREADS'] = '4'
titanic = pd.read_csv('../input/train.csv') # labelled survival 
titanic.head()
titanic.describe()
# we join the train and test datasets to full diversity of features. No predictions are made at this stage.
test = pd.read_csv('../input/test.csv')
titanic_comb = pd.concat([titanic.drop('Survived',axis=1, inplace=False), test])
sns.heatmap(titanic_comb.isnull(),yticklabels=False,cbar=False,cmap='plasma')
print('fraction of values missing')
print(titanic_comb['Age'].isnull().sum() / titanic_comb['PassengerId'].count())
print()
print('how many lines?')
print(titanic_comb.shape)
sns.heatmap(titanic.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#---------- build a quick and dirty dataframe ---------- 
titanic_dirty = titanic.copy()

titanic_dirty.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'],axis=1,inplace=True) # most likely to create spurious correlations than causal ones

#categorical variables transformed into simple several binary columns e.g. male = 1 female = 0 type of thing
sex = pd.get_dummies(titanic_dirty['Sex'],drop_first=True) # probably very relevant, not that hard
embark = pd.get_dummies(titanic_dirty['Embarked'],drop_first=True)

titanic_dirty.drop(['Sex', 'Embarked'],axis=1,inplace=True) # drop categorical columns now that we've converted them
titanic_dirty = pd.concat([titanic_dirty,sex,embark],axis=1)

#deal with na
titanic_dirty.dropna(inplace=True) # we drop values for lines in which Age is missing for this first model (quicker)
titanic_dirty.head()
# one fundamental step in supervised learning is be able to assess the efficacy of our models on "unseen data"
# here we'll do this by randomly spliting our dataset into two parts: 70% for training set and 30% for test set
from sklearn.model_selection import train_test_split

# features are stored in X, and survival in y. Train corresponds to training.
X_train, X_test, y_train, y_test = train_test_split(titanic_dirty.drop('Survived',axis=1), 
                                                    titanic_dirty['Survived'], test_size=0.30, random_state = 2)
# here we train our random forest classifier on the training set, and predict survival on the test set
from sklearn.ensemble import RandomForestClassifier
random_state = 2 # handy to have repeatable results (and I change it when I want)
RFC = RandomForestClassifier(random_state = 2) 
RFC.fit(X_train, y_train) # this single line trains our machine learning model! (sklearn offers a pretty amazing interface)
y_test_pred = RFC.predict(X_test)
# here we assess the prediction's accuracy
# Kaggle chose to use accuracy as a metric, so this is what we will focus on
print('Accuracy: ',(y_test_pred == y_test).mean())

# However, I believe it is a good habit to look beyond accuracy, as precision, recall or F1-score are sometimes closer to business objectives, and this can flag odd behaviors
# this can be done easily with the  following functions: from sklearn.metrics import confusion_matrix, classification_report
print(1 - y_train.mean())
print(1 - y_test.mean())
# let's try to see what numerical variables only yield
RFC_num = RandomForestClassifier(random_state = 2) 
RFC_num.fit(X_train.drop(['male', 'Q', 'S'],axis=1), y_train)
y_test_num_pred = RFC_num.predict(X_test.drop(['male', 'Q', 'S'],axis=1))

print('Test Accuracy (num variables only): ',(y_test_num_pred == y_test).mean())
y_train_pred = RFC.predict(X_train)
print('Train Accuracy: ',(y_train_pred == y_train).mean())
print('Test Accuracy: ',(y_test_pred == y_test).mean())
# data overview
g = sns.PairGrid(titanic.drop('Pclass',axis=1).dropna(),hue='Sex') # we drop NA and Pclass to quickly get data overview
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot).add_legend()

titanic.head(1)
# ------- title -------
# Title could be a predictor of status, age and sex. Not sure at this point whether it adds any information.

# notice how the title is in between a comma "," and a dot "." We use split() to isolate that
# there may also be some gap - strip() can deal with this 
titanic_comb['title'] = titanic_comb['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())

#plot
plt.figure(figsize=(16,3))
sns.countplot(titanic_comb['title'])
plt.tight_layout()
plt.show()
titanic['title'] = titanic['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
plt.figure(figsize=(16,3))
sns.barplot(data=titanic, x='title', y='Survived')
plt.tight_layout()
plt.show()
g = sns.barplot(data=titanic[(titanic['title']=='Miss') | (titanic['title']=='Mrs')], x='title', y='Survived', hue='Pclass')
titanic_comb.groupby('title')['Age'].median().plot.bar()
titanic_comb[titanic_comb['Age'].isnull()]['title'].value_counts()
titanic_comb[titanic_comb['Age'].isnull()]['Pclass'].value_counts()
def title_preprocessing(df):
    df['title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    
    # grouping for age prediction
    df['title_age'] = df['title'].replace(['Ms'], 'Mrs')
    df['title_age'] = df['title_age'].replace(['Don', 'Rev', 'Mme', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'], 'irrelevant_here')
    
    # grouping for status prediction
    df['title_status'] = df['title'].replace(['Capt', 'Don', 'Jonkheer', 'Rev'], 'Mr')
    df['title_status'] = df['title_status'].replace(['Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'the Countess', 'Dona'], 'Mrs')
    df['title_status'] = df['title_status'].replace(['Miss'], 'Mrs')
    
    return df

def age_replacement_table(titanic_comb):
    replacement_table = pd.pivot_table(titanic_comb, values='Age', index='Pclass', columns='title_age',aggfunc=np.median)
    replacement_table['Master'][2] = titanic_comb[titanic_comb['title_age']=='Master']['Age'].median() # we replace this value that looks spurious
    return replacement_table

replacement_table = age_replacement_table(title_preprocessing(titanic_comb))
replacement_table
#the values above look reasonable, apart from Master's age perhaps.
# so at the risk of using values from too small categories, we'll use this table for the replacement

def input_age(cols):
    '''takes Age, title_age and Pclass as input
    returns 'Age' with null values replaced by smart ones'''
    # grab inputs
    Age = cols[0]
    title_age = cols[1]
    Pclass = cols[2]
    
    # correct outputs
    if pd.isnull(Age):
        return replacement_table[title_age][Pclass]
    else:
        return Age
input_age([None,'Mr',1])
def title_and_age_processing(df):
    df = title_preprocessing(df)
    df['Age'] = df[['Age','title_age','Pclass']].apply(input_age,axis=1)
    return df

# process for age and title combined, train and test datasets
titanic_comb = title_and_age_processing(titanic_comb)
titanic = title_and_age_processing(titanic)
test = title_and_age_processing(test)
g = sns.barplot(data=titanic,x='title_status', y='Survived', hue='Pclass')
# ------- familly size -------
titanic['familly_size'] = titanic['SibSp'] + titanic['Parch'] + 1
test['familly_size'] = test['SibSp'] + test['Parch'] + 1
titanic_comb['familly_size'] = titanic_comb['SibSp'] + titanic_comb['Parch'] + 1

graph = sns.factorplot(data=titanic, x='familly_size',y='Survived')
graph.set_ylabels('Survival probability')
graph = sns.factorplot(data=titanic, x='familly_size',y='Survived', col='Pclass')
graph.set_ylabels('Survival probability')
graph = sns.factorplot(data=titanic, x='familly_size',y='Survived', col='title_status')
graph.set_ylabels('Survival probability')
titanic.tail(3)
# first, test has one missing  value for Fare
# this is a passenger in third class, travelling alone. Getting the median fare price should get him covered.
test = test.fillna(titanic_comb['Fare'].median())
sns.distplot(titanic['Fare'])
titanic['log_fare'] = np.log(titanic['Fare']+1)
test['log_fare'] = np.log(test['Fare']+1)
titanic_comb['log_fare'] = np.log(titanic_comb['Fare']+1)

sns.distplot(titanic['log_fare'])
g = sns.factorplot(data=titanic, x='title_status', y='Survived', hue='Pclass', col='Embarked',kind='bar')
print(((titanic['title_status'] == 'Mrs') & (titanic['Pclass']==1) & (titanic['Embarked']=='Q')).sum())
print(((titanic['title_status'] == 'Master') & (titanic['Pclass']==3) & (titanic['Embarked']=='S')).sum())
titanic_2 = titanic.copy() #for convenience, copy the dataset before dropping variables, so I can keep playing with the old dataframe
titanic_2 = pd.get_dummies(titanic_2, columns=['Sex'], drop_first=True)
titanic_2 = pd.get_dummies(titanic_2, columns=['Embarked'], prefix='Em',drop_first=True)
titanic_2 = pd.get_dummies(titanic_2, columns=['title_status'],drop_first=True)

test_2 = test.copy()
test_2 = pd.get_dummies(test_2, columns=['Sex'], drop_first=True)
test_2 = pd.get_dummies(test_2, columns=['Embarked'], prefix='Em', drop_first=True)
test_2 = pd.get_dummies(test_2, columns=['title_status'], drop_first=True)

# ----- to try later?
# create categorical variable from numerical ones?
# Pclass
# Fare
# Familly size
# get rid of variables we won't use
titanic_2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'title', 'title_age'], axis=1, inplace=True)
test_2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'title', 'title_age'], axis=1, inplace=True)
titanic_2.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_2.drop('Survived',axis=1), 
                                                    titanic_2['Survived'], test_size=0.30, random_state = 2)
# Let's import a range of classifiers that have different strengths and weaknesses and seee how they do
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
random_state = 2

models = [
    RandomForestClassifier(random_state = random_state),
    AdaBoostClassifier(random_state = random_state),
    DecisionTreeClassifier(random_state = random_state),
    LogisticRegression(random_state = random_state),
    SVC(random_state = random_state, gamma='auto'),
    KNeighborsClassifier(),
    MLPClassifier(random_state = random_state),
    LinearDiscriminantAnalysis()
]

model_names = [
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "DecisionTreeClassifier",
    "LogisticRegression",
    "SVC",
    "KNeighborsClassifier",
    "MLPClassifier",
    "LinearDiscriminantAnalysis"
]
i=0
for model in models:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(model_names[i])
    print((y_test == y_test_pred).mean())
    print('')
    i=i+1
ada = AdaBoostClassifier(random_state = random_state)
ada.fit(X_train, y_train)
y_pred_Adaboost = ada.predict(test_2)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_Adaboost})
data_to_submit.to_csv('ada_to_submit.csv', index = False)

RF = RandomForestClassifier(random_state = random_state)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(test_2)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_RF})
data_to_submit.to_csv('RF_to_submit.csv', index = False)
from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)
# compute test score several times on different folds for each model
cv_results = []
for model in models:
    cv_results.append(cross_val_score(model, X_train, y=y_train, scoring="accuracy", cv=kfold, n_jobs=4))
    
# from these different scores, assess mean and standard deviation
cv_means = []
cv_std = []
for cv_results in cv_results:
    cv_means.append(cv_results.mean())
    cv_std.append(cv_results.std())                 
# put results together and plot
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":model_names})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="coolwarm",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")     
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    # I came across this very nice representation of learning curves on a kernel by Yassine Ghouzam
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

g = plot_learning_curve(RandomForestClassifier(random_state = random_state),"RandomForestClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(DecisionTreeClassifier(random_state = random_state),"DecisionTreeClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(AdaBoostClassifier(random_state = random_state),"AdaBoostClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(LogisticRegression(random_state = random_state),"LogisticRegression learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(SVC(random_state = random_state, gamma='auto'),"SVC learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(MLPClassifier(random_state = random_state, gamma='auto'),"MLPClassifier learning curves",X_train,y_train,cv=kfold)

forest = AdaBoostClassifier()
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking and contribution
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f],test_2.columns.tolist()[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking and contribution
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f],test_2.columns.tolist()[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
titanic_3 = titanic_2.copy()
titanic_3.drop(['Sex_male', 'SibSp', 'Parch', 'Em_Q', 'Em_S', 'Fare'], axis=1, inplace=True)

test_3 = test_2.copy()
test_3.drop(['Sex_male', 'SibSp', 'Parch', 'Em_Q', 'Em_S', 'Fare'], axis=1, inplace=True)
titanic_3.head(1)
X_train, X_test, y_train, y_test = train_test_split(titanic_3.drop('Survived',axis=1), 
                                                    titanic_3['Survived'], test_size=0.20, random_state = 2)

from sklearn.model_selection import GridSearchCV
# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2,5,10],
              "learning_rate":  [0.0001, 0.01, 0.1,0.15, 0.2,0.25, 0.3, 2]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,y_train)

ada_best = gsadaDTC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsadaDTC.best_score_)
print('Test accuracy: ', (ada_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsadaDTC.best_params_)
# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "bootstrap": [False],
              "n_estimators" :[100,300,500],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsRFC.best_score_)
print('Test accuracy: ', (RFC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsRFC.best_params_)
### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1, 3],
                  'C': [0.01, 1, 10, 50, 100, 200, 300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,y_train)

SVMC_best = gsSVMC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsSVMC.best_score_)
print('Test accuracy: ', (SVMC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsSVMC.best_params_)

### MLPClassifier
neural = MLPClassifier()
neural_param_grid = {'solver': ['lbfgs'], 'max_iter': [500, 1000, 1500], 
                     'alpha': [0.0001, 0.0005, 0.001, 0.003, 0.01], 
                     'hidden_layer_sizes':[6,7,8], 
                     'random_state':[0,2,8]
    
}

gsneural = GridSearchCV(neural,param_grid = neural_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsneural.fit(X_train,y_train)

neural_best = gsneural.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsneural.best_score_)
print('Test accuracy: ', (neural_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsneural.best_params_)
### LogisticRegression
logistic = LogisticRegression()
logistic_param_grid = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 10, 30, 100, 300, 1000]
    
}

gslogistic = GridSearchCV(logistic,param_grid = logistic_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gslogistic.fit(X_train,y_train)

logistic_best = gslogistic.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gslogistic.best_score_)
print('Test accuracy: ', (logistic_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gslogistic.best_params_)
y_pred_Adaboost = ada_best.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_Adaboost})
data_to_submit.to_csv('ada_opt_to_submit.csv', index = False)


y_pred_RF = RFC_best.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_RF})
data_to_submit.to_csv('RF_opt_to_submit.csv', index = False)
g = plot_learning_curve(ada_best,"AdaBoostClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(RFC_best,"RandomForestClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(logistic_best,"LogisticRegression learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(SVMC_best,"SVC learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(neural_best,"MLPClassifier learning curves",X_train,y_train,cv=kfold)
y_test_ada = pd.Series(ada_best.predict(X_test), name='Ada')
y_test_RFC = pd.Series(RFC_best.predict(X_test), name='RF')
y_test_SVMC = pd.Series(SVMC_best.predict(X_test), name='SVM')
y_test_log = pd.Series(logistic_best.predict(X_test), name='logistic')
y_test_neur = pd.Series(neural_best.predict(X_test), name='neural')

ensemble_test_predictions = pd.concat([y_test_ada,y_test_RFC,y_test_SVMC,y_test_log, y_test_neur],axis=1)

g = sns.heatmap(ensemble_test_predictions.corr(),annot=True, cmap="coolwarm")
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('rfc', RFC_best), ('logistic', logistic_best),
('svc', SVMC_best), ('ada',ada_best),('neural',neural_best)], voting='soft', n_jobs=4)

voting = voting.fit(X_train, y_train)
y_test_voting = voting.predict(X_test)
print('Test accuracy: ', (y_test_voting == y_test).mean())
y_pred_vote = voting.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_vote})
data_to_submit.to_csv('ensemble_to_submit.csv', index = False)
# DTC Parameters tunning 
DTC = DecisionTreeClassifier()

## Search grid for optimal parameters
DTC_param_grid = {"max_depth": [1,2,3,4,5,6,7,8,9,10,20,30,50],
              "max_features": [1, 3, 5, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "criterion": ["gini", "entropy"],
                "random_state": [0, 2, 4]}

gsDTC = GridSearchCV(DTC,param_grid = DTC_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsDTC.fit(X_train,y_train)

DTC_best = gsDTC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsDTC.best_score_)
print('Test accuracy: ', (DTC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsDTC.best_params_)
import graphviz
from sklearn import tree
# in line function for small trees
dot_data = tree.export_graphviz(DTC_best, out_file=None, 
                         feature_names=test_3.columns.tolist(),
                                class_names=["Died","Survived"],
                         filled=True, rounded=False, precision=2, label='root', impurity=False,
                         special_characters=True, max_depth=2) 
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("titanic_tree")
graph
# in line function for small trees
dot_data = tree.export_graphviz(DTC_best, out_file=None, 
                         feature_names=test_3.columns.tolist(),
                                class_names=["Died","Survived"],
                         filled=True, rounded=False, precision=2, label='root', impurity=False,
                         special_characters=True, max_depth=6) 
graph = graphviz.Source(dot_data)
graph
