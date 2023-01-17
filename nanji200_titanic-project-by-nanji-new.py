# This kernel borrows heavily from https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8

# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Model Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

import xgboost



# Model evaluations

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



pd.set_option('display.max_colwidth', 0)



hypertuning = 0
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")
def exploreData(df,dfName):

    #Set display

    #pd.options.display.max_columns=15

    #pd.options.display.max_rows=892



    print('SNAPSHOT OF '+ dfName)

    df.info()

    print('\n')



    print('BASIC DESCRIPTION')

    print(df.describe())

    print('\n')



    print('SNAPSHOT OF FIRST 8 RECORDS')

    print(df.head(8))

    print('\n')



    # List missing values

    print('MISSING VALUE SUMMARY')

    total = df.isnull().sum().sort_values(ascending=False)

    percent_1 = df.isnull().sum()/df.isnull().count()*100

    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

    print(missing_data.head(5))   
 # Describe features

des='''DESCRIPTION OF FEATURES:

survival:    Survival 

PassengerId: Unique Id of a passenger. 

pclass:    Ticket class     

sex:    Sex     

Age:    Age in years     

sibsp:    # of siblings / spouses aboard the Titanic     

parch:    # of parents / children aboard the Titanic     

ticket:    Ticket number     

fare:    Passenger fare     

cabin:    Cabin number     

embarked:    Port of Embarkation'''

print(des)
exploreData(train_df,'TRAIN Dataset')
exploreData(test_df,'TEST Dataset')
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

genders = {"male": 0, "female": 1}

ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

#add new column "relatives"

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)    

#Since the Embarked feature has only 2 missing values, we will just fill these with the most common one

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Converting “Fare” from float to int64, using the “astype()” function pandas provides    

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)

# extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#update age column

    titles_list = ['Mr','Miss','Mrs','Master','Rare']

    # Replacing missing age by median/title 

    for title in titles_list:

        age_to_impute = dataset.groupby('Title')['Age'].mean()[titles_list.index(title)]

        dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title), 'Age'] = age_to_impute

    dataset["Age"] = dataset["Age"].astype(int)    

# convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

# filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

#Convert ‘Sex’ feature into numeric    

    dataset['Sex'] = dataset['Sex'].map(genders)

#Convert ‘Embarked’ feature into numeric    

    dataset['Embarked'] = dataset['Embarked'].map(ports)

#create age categories    

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

#create fare categories 

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

#Age times Class    

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

#Fare per Person    

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

#Create new column IsThereCabinData

    dataset.loc[dataset['Cabin'].isna(), 'IsThereCabinData'] = 0

    dataset.loc[dataset['Cabin'].notna(), 'IsThereCabinData'] = 1

    dataset['IsThereCabinData'] = dataset['IsThereCabinData'].astype(int)

#train_df['not_alone'].value_counts()

    

#create new column "deck"

import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]

for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['DeckLetter'] = dataset['Deck']

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(8)

    dataset['Deck'] = dataset['Deck'].astype(int)
sns.barplot(x='DeckLetter', y='Survived', data=train_df)
#drop columns

#First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons survival probability. I will not drop it from the test set, since it is required there for the submission.

#train_df = train_df.drop(['PassengerId'], axis=1)

# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

train_df = train_df.drop(['DeckLetter'], axis=1)

test_df = test_df.drop(['DeckLetter'], axis=1)
list=['Pclass','Sex', 'Age', 'Fare','Embarked','relatives','Title','Age_Class','Fare_Per_Person','Deck']



train_df_copy = train_df[list].copy() 

X_enc_train = pd.get_dummies(train_df_copy, columns=list)

train_df = train_df.drop(list,axis=1)

train_df = pd.concat([train_df,X_enc_train], axis=1) 



test_df_copy = test_df[list].copy() 

X_enc_test = pd.get_dummies(test_df_copy, columns=list)

test_df = test_df.drop(list,axis=1)

test_df = pd.concat([test_df,X_enc_test], axis=1) 
'''train_df = train_df.drop(['Title'], axis=1)

test_df = test_df.drop(['Title'], axis=1)

train_df = train_df.drop(['Sex'], axis=1)

test_df = test_df.drop(['Sex'], axis=1)

train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)

train_df = train_df.drop(['Fare_Per_Person'], axis=1)

test_df = test_df.drop(['Fare_Per_Person'], axis=1)

train_df = train_df.drop(['SibSp'], axis=1)

test_df = test_df.drop(['SibSp'], axis=1)

train_df = train_df.drop(['Parch'], axis=1)

test_df = test_df.drop(['Parch'], axis=1)

train_df = train_df.drop(['not_alone'], axis=1)

test_df = test_df.drop(['not_alone'], axis=1)'''
train_df = train_df.drop(['PassengerId'], axis=1)

X_train = train_df.drop("Survived", axis=1) #Remove target variable from the training set

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
#Stochastic Gradient Descent (SGD):

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)



#Random Forest:

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



#Logistic Regression:

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



#K Nearest Neighbor:

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



#Gaussian Naive Bayes:

gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



#Perceptron:

perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)



#Linear Support Vector Machine:

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)



#Decision Tree

decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)



results = pd.DataFrame({

          'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

          'Random Forest', 'Naive Bayes', 'Perceptron', 

          'Stochastic Gradient Decent', 

          'Decision Tree'],

          'Accuracy Score': [acc_linear_svc, acc_knn, acc_log, 

          acc_random_forest, acc_gaussian, acc_perceptron, 

          acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Accuracy Score', ascending=False)

result_df = result_df.set_index('Accuracy Score')

result_df
def findKFoldCrossValScoreForOneModel(Model_Obj, Model_Name, X_train, Y_train):

    scores = cross_val_score(Model_Obj, X_train, Y_train, cv=10, scoring = "accuracy")

    result_df = pd.DataFrame({

              'Model': [Model_Name],

              'Cross Val Score': [np.around(scores,decimals=2)],

              'Cross Val Mean': [scores.mean()],

              'Cross Val STD': [scores.std()]})

    return(result_df)     
Model_Obj = RandomForestClassifier(n_estimators=100)

Model_Name = 'Random Forest Classifier'

findKFoldCrossValScoreForOneModel(Model_Obj, Model_Name, X_train, Y_train)
Model_Obj = DecisionTreeClassifier()

Model_Name = 'Decision Tree Classifier'

findKFoldCrossValScoreForOneModel(Model_Obj, Model_Name, X_train, Y_train)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances
#XGBoost

xgb_model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, random_state=42, eval_metric=["auc", "error", "error@0.6"])

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

xgb_model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)])

#y_pred = xgb_model.predict(X_test)
Model_Name='XGBoost'

findKFoldCrossValScoreForOneModel(xgb_model, Model_Name, X_train, Y_train)
xgboost.plot_importance(xgb_model)
X_train  = X_train.drop("not_alone", axis=1)

X_test  = X_test.drop("not_alone", axis=1)

X_train  = X_train.drop("Parch", axis=1)

X_test  = X_test.drop("Parch", axis=1)

X_train  = X_train.drop("IsThereCabinData", axis=1)

X_test  = X_test.drop("IsThereCabinData", axis=1)
#Run Random Forest again:

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
print("random_forest oob score:", round(random_forest.oob_score_*100, 4), "%")
if hypertuning == 1:

    param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 

                  "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}



    rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

    clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)

    clf.fit(X_train, Y_train)

    clf.best_params_
# Random Forest

random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=400, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)



random_forest.fit(X_train, Y_train)

Y_prediction_rf = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("score:",round(random_forest.score(X_train, Y_train) * 100, 2))

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
#xgb_model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, random_state=42, eval_metric=["auc", "error", "error@0.6"])

xgb_model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, random_state=42, eval_metric=["auc"])

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

xgb_model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)])

#y_pred = xgb_model.predict(X_test)

#print('xgb_model.best_score=', xgb_model.best_score, 'xgb_model.best_iteration=', xgb_model.best_iteration)
if hypertuning == 1:

    params = {

            'min_child_weight': [1, 5, 10],

            'gamma': [0.5, 1, 1.5, 2, 5],

            'subsample': [0.6, 0.8, 1.0],

            'learning_rate': [0.04, 0.05, 0.06],

            'colsample_bytree': [0.6, 0.8, 1.0],

            'max_depth': [3, 4, 5]

            }

    xgb_model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, random_state=42, eval_metric=["auc"])

    clf = GridSearchCV(estimator=xgb_model, param_grid=params, n_jobs=-1)

    clf.fit(X_train, Y_train)

    clf.best_params_
xgb_model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, random_state=42, gamma=1.5, learning_rate=0.06, subsample=0.8,  eval_metric=["auc"])

xgb_model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)])

Y_prediction_XGBoost =  xgb_model.predict(X_test)
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)

print(confusion_matrix(Y_train, predictions))

print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))

print("F1 score:",f1_score(Y_train, predictions))
y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]



# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
r_a_score = roc_auc_score(Y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
#this submission is random forest based - received a score of 0.77990

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_prediction_rf

    })



#this submission is XGBoost based - received a score of ?

'''submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_prediction_XGBoost

    })'''



submission.to_csv('submission.csv', index=False)
train_df.columns.values
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df
train_df['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, )