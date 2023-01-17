# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # regex

import matplotlib.pyplot as plt

%matplotlib inline



import sklearn

import xgboost as xgb

import seaborn as sns



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

passenger_ids = test["PassengerId"]

test.head(3)
# Data preprocessing, add new features

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



# feautures from Sina

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+).', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



full_data = [train, test]



for dataset in full_data:

    # normalize name length

    dataset['Name_length'] = dataset['Name'].apply(len)



    # normalize sex

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1} ).astype(int)



    # normilize age

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    

    # normilize title

    dataset['Title'] = dataset['Name'].apply(get_title)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # has cabin

    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    

    # normilize embarque

    # 	C = Cherbourg, Q = Queenstown, S = Southampton

    dataset['Embarked'] = dataset['Embarked'].fillna('S') # replace null with 'S'

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ) # .astype(int)

    

    # drop unnecessary columns

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

    dataset.drop(drop_elements, axis=1, inplace=True)

    

     # Mapping Fare

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] ;

    



train.head(3)
# check null values

train.isnull().values.any()



# train['Sex'].describe()
# print corelation map

colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.values
# try SVM

from sklearn.svm import SVC

svc_classifier = SVC(kernel = 'linear', C = 0.025)

svc_classifier.fit(X_train, y_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = svc_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())



svc_classifier.score(X_train, y_train)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', warm_start = True, max_depth= 6, min_samples_leaf = 2, max_features = 'sqrt', verbose= 0)

rf_classifier.fit(X_train, y_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = rf_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())

rf_classifier.score(X_train, y_train)
# GradientBoostingClassifier 

from sklearn.ensemble import GradientBoostingClassifier

gb_classifier = GradientBoostingClassifier(n_estimators = 2000, max_depth = 4, min_samples_leaf = 1, verbose = 0)

gb_classifier.fit(X_train, y_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = gb_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())



gb_classifier.score(X_train, y_train)
# AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

ada_classifier = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.75)

ada_classifier.fit(X_train, y_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = ada_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())



ada_classifier.score(X_train, y_train)
from sklearn.ensemble import ExtraTreesClassifier

et_classifier = ExtraTreesClassifier()

et_classifier.fit(X_train, y_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = et_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())
# try staking models



from sklearn.cross_validation import KFold;



# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et_classifier, X_train, y_train, X_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf_classifier, X_train, y_train, X_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada_classifier, X_train, y_train, X_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb_classifier,X_train, y_train, X_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc_classifier,X_train, y_train, X_test) # Support Vector Classifier





x_train_ens = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test_ens = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
X_test.head()
# Generate Submission File 

submission = pd.DataFrame({ 'PassengerId': passenger_ids,

                            'Survived': predictions })

submission.to_csv("xgb_submission.csv", index=False)
from subprocess import check_output

# Any results you write to the current directory are saved as output.

print(check_output(["ls", "."]).decode("utf8"))
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(X_train, y_train)

predictions = gbm.predict(test)