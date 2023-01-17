# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



##visual imports

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



##Missing data

from sklearn.impute import SimpleImputer



##Categorical Encoding

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



##Feature Scaling

from sklearn.preprocessing import StandardScaler



##Splitting data

from sklearn.model_selection import train_test_split



#Splitting Data

from sklearn.model_selection import train_test_split



# Feature Scaling

from sklearn.preprocessing import StandardScaler



#Confusion Matrix & accuracy



from sklearn.metrics import confusion_matrix, accuracy_score



######################

### Classification ###

######################



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/iris/Iris.csv')
train_data.head()
train_data.info()
train_data.describe()
train_data.describe(include = ["O"])
train_data["Species"].value_counts()
train_data.drop(["Id"], axis = 1,inplace = True)
correl = train_data.corr()

sns.heatmap(correl, annot = True)
sns.pairplot(train_data, hue = "Species")
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Species',axis=1), 

                                                    train_data['Species'], test_size=0.30, 

                                                    random_state=101)
classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("_"*20)

print(classification_report(y_test,y_pred))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_validate



MLA = [LogisticRegression(max_iter=500),KNeighborsClassifier(), SVC(), GaussianNB(),DecisionTreeClassifier(), RandomForestClassifier()]

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Accuracy Mean', 'MLA Test Accuracy 3*STD']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#create table to compare MLA predictions

MLA_predict = pd.DataFrame() 

MLA_predict["Actual Results"] = y_test



cv_split = ShuffleSplit(n_splits = 10, test_size = 0.3, train_size = 0.6, random_state = 0 )
def MLA_Runner(MLA_Algo, MLA_Compare):

    #index through MLA and save performance to table

    row_index = 0

    for alg in MLA:



        

        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())



        cv_results = cross_val_score(estimator = alg, X = X_train, y = y_train, cv = cv_split)



        MLA_compare.loc[row_index, 'MLA Accuracy Mean'] = cv_results.mean()*100

        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results.std()*3



        alg.fit(X_train, y_train)

        MLA_predict[MLA_name] = alg.predict(X_test)



        row_index +=1



    MLA_compare.sort_values(by = ["MLA Accuracy Mean"], ascending = False, inplace = True)

    MLA_compare

    

    plt.title("MLA Accuracy Rank")

    sns.barplot(x = "MLA Accuracy Mean", y = "MLA Name", data = MLA_compare)

MLA_Runner(MLA, MLA_compare)
svc = SVC()

base_result = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = cv_split)

svc.fit(X_train, y_train)

print("Basic SVC Accuracy :{:.2f} %".format(base_result.mean()*100))

print("_"*30)



from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = svc,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = cv_split,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Tuned SVC Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best SVC Parameters:", best_parameters)
KNC = KNeighborsClassifier()

base_result = cross_val_score(estimator = KNC, X = X_train, y = y_train, cv = cv_split)

svc.fit(X_train, y_train)

print("Basic KNeighborsClassifier Accuracy:{:.2f} %".format(base_result.mean()*100))

print("_"*30)



parameters = [{'n_neighbors': [3, 5, 7], 'weights': ['uniform','distance'], "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"], "leaf_size" :[20,30,40], "p" :[1,2] }]

grid_search = GridSearchCV(estimator = KNC,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = cv_split,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best KNeighborsClassifier Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best KNeighborsClassifier Parameters:", best_parameters)
LR = LogisticRegression(max_iter=500)

base_result = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = cv_split)

svc.fit(X_train, y_train)

print("Basic LogisticRegression Accuracy:{:.2f} %".format(base_result.mean()*100))

print("_"*30)



parameters = [{"penalty" : ["l1", "l2", "elasticnet","none"], "C" : [1,5,10,15], "dual": [True, False], 'fit_intercept': [True, False],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], "max_iter" : [200,250,300]}]

grid_search = GridSearchCV(estimator = LR,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = cv_split,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best LogisticRegression Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best LogisticRegression Parameters:", best_parameters)
RFC = RandomForestClassifier()

base_result = cross_val_score(estimator = RFC, X = X_train, y = y_train, cv = cv_split)

print("Basic RandomForestClassifier Accuracy:{:.2f} %".format(base_result.mean()*100))

print("_"*30)



parameters = [{"n_estimators":[10,50,100,200], "criterion":["gini","entropy"], "max_features" :["auto", "sqrt", "log2"],

            'max_depth': [2, 4, 6, 8, 10, None],

            "oob_score": [True,False]}]

grid_search = GridSearchCV(estimator = RFC,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = cv_split,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best RandomForestClassifier Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best RandomForestClassifier Parameters:", best_parameters)
DTC = DecisionTreeClassifier()

base_result = cross_val_score(estimator = DTC, X = X_train, y = y_train, cv = cv_split)

print("Basic DecisionTreeClassifier Accuracy:{:.2f} %".format(base_result.mean()*100))

print("_"*30)



parameters = [{"criterion":["gini","entropy"], "max_depth":[5,10,None],

               "max_features" :["auto", "sqrt", "log2"]}]

grid_search = GridSearchCV(estimator = RFC,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = cv_split,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best DecisionTreeClassifier: {:.2f} %".format(best_accuracy*100))

print("Best DecisionTreeClassifier Parameters:", best_parameters)
Optimal_MLA = MLA = [LogisticRegression(C= 1, dual = False, fit_intercept = True, max_iter = 200, penalty= 'l2', solver= 'sag'),

                     KNeighborsClassifier(algorithm = 'auto', leaf_size = 20, n_neighbors = 7, p = 1, weights ='uniform'), 

                     SVC(C = 1, kernel ='linear'), GaussianNB(),

                     DecisionTreeClassifier(criterion = 'gini', max_depth = 5, max_features = 'auto'), 

                     RandomForestClassifier(criterion='gini', max_depth= 2, max_features='sqrt', n_estimators = 200, oob_score = True)]
MLA_Runner(Optimal_MLA, MLA_compare)
MLA_compare