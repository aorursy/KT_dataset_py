#About RMS Titanic

# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean 

# in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to 

# New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of 

# modern history's deadliest peacetime commercial marine disasters. 

# Wikipedia link: https://en.wikipedia.org/wiki/RMS_Titanic
# Import Packages

import pandas as pd                                                       # Pandas package for reading csv files

import numpy as np                                                        # Numpy package for computing

import matplotlib.pyplot as plt                                           # Visualization package

%matplotlib inline

import seaborn as sns                                                     # Visualization package
import os

#Reading Titanic Test given Data Set.

import os

for dirname, _, filenames in os.walk('kaggle/input/titanic_test.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import os

#Reading Titanic Train given Data Set.

import os

for dirname, _, filenames in os.walk('kaggle/input/titanic_train.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load given titanic train and test data set.

titanic_train = pd.read_csv('/kaggle/input/titanic_train.csv')                             # Reading data using simple Pandas

titanic_test = pd.read_csv('/kaggle/input/titanic_test.csv')                             # Reading data using simple Pandas
#Checking the first 5 rows

titanic_train.head(2)
titanic_test.head(2)
 # Checking the shape of the train dataframe

titanic_train.shape
 # Checking the shape of the test dataframe

titanic_test.shape
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

titanic_train.describe()
titanic_test.describe()
# Info method is used to get a concise summary of the dataframe.

titanic_train.info()
# this information shows null value percentage

titanic_train.isnull().sum()/1309*100
# we have 418 null values for survived column and those are related to test data set which can ignore it.

# Rest of other columns Age,Cabin,Embarked and Fare column we need to fill those values.

titanic_train.isnull().sum()
# Verify 'Embarked' column data count

titanic_train['Embarked'].value_counts()
# Verify 'Embarked' column data count

titanic_test['Embarked'].value_counts()
# Picking the mode value of Embarked column

titanic_train['Embarked'].mode()
titanic_test['Embarked'].mode()
# Filling with mode value to the missing value of Embarked column

titanic_train['Embarked'] = titanic_train['Embarked'].fillna('S')

titanic_train['Embarked'].value_counts()
# Filling with mode value to the missing value of Embarked column

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_test['Embarked'].value_counts()
# Verifying the mean Age from the given dataset who were traveling by Pclass.

titanic_train.groupby('Pclass').mean()['Age']
# Verifying the mean Age from the given dataset who were traveling by Pclass.

titanic_test.groupby('Pclass').mean()['Age']
# Verifying the mean Fare from the given dataset who were traveling by Pclass.

titanic_train.groupby('Pclass').mean()['Fare']
# Verifying the mean Fare from the given dataset who were traveling by Pclass.

titanic_test.groupby('Pclass').mean()['Fare']
# Verify 'Sex' column data count

titanic_train['Sex'].value_counts()
# Verify 'Sex' column data count

titanic_test['Sex'].value_counts()
# Verifying the mean value of Sex column

titanic_train.groupby('Sex').mean()
# Verifying the mean value of Sex column

titanic_test.groupby('Sex').mean()
# Verify the title assinged to each and every passenger.

titanic_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False).unique()
# Verify the title assinged to each and every passenger.

titanic_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False).unique()
# Introduce new column named as 'Title' and perform one-hot encoding by filling with nominal values.

data = [titanic_train]

titles = {"Mr": 1, "Mrs": 2, "Miss": 3, "Master": 4}



for dataset in data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Mme',\

       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Jonkheer'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].map(titles)

    dataset['Title'] = dataset['Title'].fillna(5)  # Unknown persons



titanic_train = titanic_train.drop(['Name'], axis=1)

titanic_train.head(2)
# Introduce new column named as 'Title' and perform one-hot encoding by filling with nominal values.

data = [titanic_test]

titles = {"Mr": 1, "Mrs": 2, "Miss": 3, "Master": 4}



for dataset in data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Mme',\

       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Jonkheer'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].map(titles)

    dataset['Title'] = dataset['Title'].fillna(5)  # Unknown persons



titanic_test = titanic_test.drop(['Name'], axis=1)

titanic_test.head(2)
# Convert Title datatype from float to int.

titanic_train['Title'] = titanic_train['Title'].apply(np.int64)

titanic_train.head(2)
# Convert Title datatype from float to int.

titanic_test['Title'] = titanic_test['Title'].apply(np.int64)

titanic_test.head(2)
# Verify Sex (male and female) count based on Title.

titanic_train.groupby(by=['Title','Sex']).size()
# Verify Sex (male and female) count based on Title.

titanic_test.groupby(by=['Title','Sex']).size()
# Verify Age count based on Title.

titanic_train.groupby('Title').count()['Age']
# Verify Age count based on Title.

titanic_test.groupby('Title').count()['Age']
# Verify mean Age value based on Title and Sex.

titanic_train.groupby(by=['Title', 'Sex'])['Age'].mean()
# total number of missing Age value count is 263.

titanic_train['Age'].isnull().sum()
# Below apply_age function is used to assign missing age value calculate w.r.t to mean age value based on Title and Sex.

# To maintain accuracy of missing Age value. In real time it is very difficult to calculate missing Age value in case of

# DateOfBirth value column details were missing in the given dataset. To maintain consistency written below function.

def apply_age(title,sex):

    if(title==1 and sex=='male'):

        age=32

    elif (title==2 and sex=='female'):

        age=36

    elif (title==3 and sex=='female'):

        age=22

    elif (title==4 and sex=='male'):

        age=5

    elif (title==5 and sex=='male'):

        age=46

    elif (title==5 and sex=='female'):

        age=34

    else:

        age=30 # mean age considered from describe()

    return age



#print(apply_age(1,'male'))

        

# Filling missing values of age column.

age_nulldata=titanic_train[titanic_train['Age'].isnull()]

age_nulldata['Age'] = age_nulldata.apply(lambda row : apply_age(row['Title'],row['Sex']), axis = 1) 

#age_nulldata['Age']

titanic_train['Age'].fillna(value=age_nulldata['Age'],inplace=True)
#Verify whether is their any null values exists in Age column or not.

titanic_train['Age'].isnull().sum()
# Verify Age count based on Title.

titanic_train.groupby('Title').count()['Age']
#Checking the first 5 rows

titanic_train.head(2)
# Verify mean Age value based on Title and Sex.

titanic_test.groupby(by=['Title', 'Sex'])['Age'].mean()
# total number of missing Age value count is 263.

titanic_test['Age'].isnull().sum()
# Below apply_age function is used to assign missing age value calculate w.r.t to mean age value based on Title and Sex.

# To maintain accuracy of missing Age value. In real time it is very difficult to calculate missing Age value in case of

# DateOfBirth value column details were missing in the given dataset. To maintain consistency written below function.

def apply_age_test(title,sex):

    if(title==1 and sex=='male'):

        age=32

    elif (title==2 and sex=='female'):

        age=39

    elif (title==3 and sex=='female'):

        age=22

    elif (title==4 and sex=='male'):

        age=7

    elif (title==5 and sex=='male'):

        age=45

    elif (title==5 and sex=='female'):

        age=39

    else:

        age=30 # mean age considered from describe()

    return age



#print(apply_age(1,'male'))

        
# Filling missing values of age column.

age_nulldata1=titanic_test[titanic_test['Age'].isnull()]

age_nulldata1['Age'] = age_nulldata1.apply(lambda row : apply_age_test(row['Title'],row['Sex']), axis = 1) 

#age_nulldata['Age']

titanic_test['Age'].fillna(value=age_nulldata1['Age'],inplace=True)
#Verify whether is their any null values exists in Age column or not.

titanic_test['Age'].isnull().sum()
titanic_train['Embarked'].value_counts()
# perform one-hot encoding to Embarked column by filling with nominal values.

data = [titanic_train]

embark_titles = {"Q": 1, "C": 2, "S": 3}



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(embark_titles)



titanic_train.head(2)
# perform one-hot encoding to Embarked column by filling with nominal values.

data = [titanic_test]

embark_titles = {"Q": 1, "C": 2, "S": 3}



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(embark_titles)



titanic_test.head(2)
# Drop unused columns 'Ticket' and 'Cabin' which were not required to trian the model.

titanic_train = titanic_train.drop(['Ticket'], axis=1)

titanic_train = titanic_train.drop(['Cabin'], axis=1)
# Drop unused columns 'Ticket' and 'Cabin' which were not required to trian the model.

titanic_test = titanic_test.drop(['Ticket'], axis=1)

titanic_test = titanic_test.drop(['Cabin'], axis=1)
# Identified one missing value in 'Fare' column and filled with taking median value based on PClass to calculate accurately.

titanic_train['Fare'] = titanic_train.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
# Identified one missing value in 'Fare' column and filled with taking median value based on PClass to calculate accurately.

titanic_test['Fare'] = titanic_test.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
# Convert Fare and Age Column datatype from float to int datattype.

titanic_train['Fare'] = titanic_train['Fare'].astype(int)

titanic_train['Age'] = titanic_train['Age'].astype(int)

# Convert Fare and Age Column datatype from float to int datattype.

titanic_test['Fare'] = titanic_test['Fare'].astype(int)

titanic_test['Age'] = titanic_test['Age'].astype(int)
# perform one-hot encoding to Sex column by filling with nominal values.

data = [titanic_train]

sex_titles = {"male": 0, "female": 1}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(sex_titles)



titanic_train.head(2)
# perform one-hot encoding to Sex column by filling with nominal values.

data = [titanic_test]

sex_titles = {"male": 0, "female": 1}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(sex_titles)



titanic_test.head(2)
#family size

titanic_train['Family_Size'] = titanic_train['SibSp'] + titanic_train['Parch'] +1

titanic_train.head(2)
#family size

titanic_test['Family_Size'] = titanic_test['SibSp'] + titanic_test['Parch'] +1

titanic_test.head(2)
for dataset in [titanic_train]:

    dataset.loc[dataset['Age']<=16,'Age']=0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age']=3

    dataset.loc[(dataset['Age']>64) & (dataset['Age']<=80),'Age']=4
# Drop unused columns 'Ticket' and 'Cabin' which were not required to trian the model.

titanic_train = titanic_train.drop(['PassengerId'], axis=1)

titanic_train = titanic_train.drop(['SibSp'], axis=1)

titanic_train = titanic_train.drop(['Parch'], axis=1)
titanic_train.head(5)
titanic_train.shape
for dataset in [titanic_test]:

    dataset.loc[dataset['Age']<=16,'Age']=0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age']=3

    dataset.loc[(dataset['Age']>64) & (dataset['Age']<=80),'Age']=4
# Drop unused columns 'Ticket' and 'Cabin' which were not required to trian the model.

titanic_test = titanic_test.drop(['PassengerId'], axis=1)

titanic_test = titanic_test.drop(['SibSp'], axis=1)

titanic_test = titanic_test.drop(['Parch'], axis=1)
# Checking the shape of the final test dataframe

titanic_test.shape
# Checking the shape of the final test dataframe

titanic_train.shape
#Import basic packages



import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from subprocess import check_output

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, log_loss

%matplotlib inline



# Importing Models

from sklearn import model_selection, metrics                                    

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

#from xgboost import XGBClassifier

#from lightgbm import LGBMClassifier



# Importing other tools

from sklearn import model_selection

from sklearn.metrics import confusion_matrix, classification_report, make_scorer

from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.preprocessing import StandardScaler

# Seperate target variable and prepare X and y data to train your model on training dataset.

X = titanic_train.drop(['Survived'],axis=1)

Y = titanic_train['Survived']


from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import classification_report


clf_xgb = XGBClassifier()



params = {"n_estimators" : [500, 700, 1000],

          "learning_rate" : [0.005, 0.1],

          "max_depth" : [5, 7],

          "max_features" : [3, 5], 

          "gamma" : [0.5, 0.6, 0.7]}



gsc_xgb = GridSearchCV(clf_xgb, params, cv = StratifiedShuffleSplit(n_splits = 5, 

                                                                    test_size = 0.3, 

                                                                    random_state = 15)) 

gsc_xgb = gsc_xgb.fit(X, Y)



print(gsc_xgb.best_estimator_)

clf_xgb = gsc_xgb.best_estimator_

clf_xgb.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_xgb.score(X, Y), 4)))



Y_predicted_xgb = clf_xgb.predict_proba(X)[:, 1]


clf_boost = GradientBoostingClassifier()



params = {"n_estimators" : [300, 500, 700],

          "learning_rate" : [0.002, 0.01, 0.05],

          "max_depth" : [3, 5],

          "max_features" : [7, 9]}



gsc_boost = GridSearchCV(clf_boost, params, cv = StratifiedShuffleSplit(n_splits = 5, 

                                                                        test_size = 0.3, 

                                                                        random_state = 15)) 

gsc_boost = gsc_boost.fit(X, Y)



print(gsc_boost.best_estimator_)

clf_boost = gsc_boost.best_estimator_

clf_boost.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_boost.score(X, Y), 4)))



Y_predicted_boost = clf_boost.predict_proba(X)[:, 1]
clf_log = LogisticRegression()



params = {"C" : [0.001, 0.01, 0.1, 1, 1.1, 10],

          "max_iter" : [10000],

          "solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}



gsc_log = GridSearchCV(clf_log, params, cv = StratifiedShuffleSplit(n_splits = 10, 

                                                                    test_size = 0.3, 

                                                                    random_state = 15)) 

gsc_log = gsc_log.fit(X, Y)



print(gsc_log.best_estimator_)

clf_log = gsc_log.best_estimator_

clf_log.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_log.score(X, Y), 4)))



Y_predicted_log = clf_log.predict_proba(X)[:, 1]
clf_rf = RandomForestClassifier(criterion = "gini", 

                                max_features = "auto")



params = {"max_depth" : [1, 3, 5, 7, 9],

          "n_estimators" : [150, 200, 250, 300]}



gsc_rf = GridSearchCV(clf_rf, params, cv = StratifiedShuffleSplit(n_splits = 5, 

                                                                  test_size = 0.3, 

                                                                  random_state = 15))

gsc_rf = gsc_rf.fit(X, Y)



print(gsc_rf.best_estimator_)

clf_rf = gsc_rf.best_estimator_

clf_rf.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_rf.score(X, Y), 4)))



Y_predicted_rf = clf_rf.predict_proba(X)[:, 1]


pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability = True, kernel = "rbf"))])



params = {"svm__C" : [0.01, 0.1, 1],

          "svm__gamma" : [0.01, 0.1, 1]}



gsc_svm = GridSearchCV(pipeline, param_grid = params, cv = StratifiedShuffleSplit(n_splits = 10, 

                                                                                  test_size = 0.3, 

                                                                                  random_state = 15)) 

gsc_svm = gsc_svm.fit(X, Y)



print(gsc_svm.best_estimator_)

clf_svm = gsc_svm.best_estimator_

clf_svm.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_svm.score(X, Y), 4)))



Y_predicted_svm = clf_svm.predict_proba(X)[:, 1]


clf_bag = BaggingClassifier()



params = {"n_estimators" : [30, 50, 70, 100],

          "max_features" : [3, 5, 7, 9],

          "max_samples" : [3, 5, 7, 9]}



gsc_bag = GridSearchCV(clf_bag, 

                       params, 

                       cv = StratifiedShuffleSplit(n_splits = 5, 

                                                   test_size = 0.3, 

                                                   random_state = 15))



gsc_bag = gsc_bag.fit(X, Y)



print(gsc_bag.best_estimator_)

clf_bag = gsc_bag.best_estimator_

clf_bag.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_bag.score(X, Y), 4)))



Y_predicted_bag = clf_bag.predict_proba(X)[:, 1]


clf_xt = ExtraTreesClassifier(criterion = "gini", 

                              max_features = "auto")



params = {"max_depth" : [5, 7, 9],

          "n_estimators" : [300, 500, 700]}



gsc_xt = GridSearchCV(clf_xt, params, cv = StratifiedShuffleSplit(n_splits = 10, 

                                                                  test_size = 0.3, 

                                                                  random_state = 15))

gsc_xt = gsc_xt.fit(X, Y)



print(gsc_xt.best_estimator_)

clf_xt = gsc_xt.best_estimator_

clf_xt.fit(X, Y)



print("")

print("Accuracy Score: " + str(round(clf_xt.score(X, Y), 4)))



Y_predicted_xt = clf_xt.predict_proba(X)[:, 1]
from sklearn.metrics import roc_curve, roc_auc_score
log_fpr, log_tpr, log_treshholds = roc_curve(Y, Y_predicted_log)

boost_fpr, boost_tpr, boost_treshholds = roc_curve(Y, Y_predicted_boost) 

svm_fpr, svm_tpr, svm_treshholds = roc_curve(Y, Y_predicted_svm)

rf_fpr, rf_tpr, rf_treshholds = roc_curve(Y, Y_predicted_rf)

xgb_fpr, xgb_tpr, xgb_treshholds = roc_curve(Y, Y_predicted_xgb)

bag_fpr, bag_tpr, bag_treshholds = roc_curve(Y, Y_predicted_bag)

xt_fpr, xt_tpr, xt_treshholds = roc_curve(Y, Y_predicted_xt)



auc_score_log = roc_auc_score(Y, Y_predicted_log)

auc_score_boost = roc_auc_score(Y, Y_predicted_boost)

auc_score_svm = roc_auc_score(Y, Y_predicted_svm)

auc_score_rf = roc_auc_score(Y, Y_predicted_rf)

auc_score_xgb = roc_auc_score(Y, Y_predicted_xgb)

auc_score_bag = roc_auc_score(Y, Y_predicted_bag)

auc_score_xt = roc_auc_score(Y, Y_predicted_xt)



plt.figure(figsize = (12,6))

plt.plot([0,1], [0,1])

plt.plot(log_fpr, log_tpr, label = "Logistic Regression (AUC-Score: " + str(round(auc_score_log, 2)) + ")")

plt.plot(boost_fpr, boost_tpr, label = "Gradient Boosting (AUC-Score: " + str(round(auc_score_boost, 2)) + ")")

plt.plot(svm_fpr, svm_tpr, label = "SVM (AUC-Score: " + str(round(auc_score_svm, 2)) + ")")

plt.plot(rf_fpr, rf_tpr, label = "Random Forest (AUC-Score: " + str(round(auc_score_rf, 2)) + ")")

plt.plot(xgb_fpr, xgb_tpr, label = "XGBoost (AUC-Score: " + str(round(auc_score_xgb, 2)) + ")")

plt.plot(bag_fpr, bag_tpr, label = "Bagging Classifier (AUC-Score: " + str(round(auc_score_bag, 2)) + ")")

plt.plot(xt_fpr, xt_tpr, label = "Extra Trees Clasifier (AUC-Score: " + str(round(auc_score_xt, 2)) + ")")

plt.title("ROC-Curve")

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.legend()

plt.show()
titanic_test.head(1)


vote = VotingClassifier(estimators = [("XGBoost", clf_xgb), 

                                      ("GradientBoosting", clf_boost),

                                      ("RandomForest", clf_rf), 

                                      ("Logistic Regression", clf_log), 

                                      ("SVM", clf_svm),  

                                      ("Extra Trees Classifier", clf_xt)], 

                                      voting = "soft")



vote.fit(X, Y)

print(round(vote.score(X, Y), 4))



Y_predicted_stack = vote.predict(titanic_test)