#Importing all the required libraries

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 50)
audit_risk = pd.read_csv("../input/audit-data/audit_risk.csv")

trial = pd.read_csv("../input/audit-data/trial.csv")
audit_risk.describe()
trial.describe()
#Renaming columns

trial.columns = ['Sector_score','LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B',

       'Score_B',  'TOTAL', 'numbers', 'Marks',

       'Money_Value', 'MONEY_Marks', 'District',

       'Loss', 'LOSS_SCORE', 'History', 'History_score', 'Score', 'Risk_trial' ]
trial['Score_A'] = trial['Score_A']/10

trial['Score_B'] = trial['Score_B']/10
same_columns = np.intersect1d(audit_risk.columns, trial.columns)

same_columns
# Merge two Dataframes  on common columns  using outer join

merged_df = pd.merge(audit_risk, trial, how='outer', on = ['History', 'LOCATION_ID', 'Money_Value', 'PARA_A', 'PARA_B',

       'Score', 'Score_A', 'Score_B', 'Sector_score', 'TOTAL', 'numbers'])

merged_df.columns
df = merged_df.drop(['Risk_trial'], axis = 1)
df.info()
#Replacing the missing value by the median of the column

df['Money_Value'] = df['Money_Value'].fillna(df['Money_Value'].median())
df.describe()
df = df.drop(['Detection_Risk', 'Risk_F'], axis = 1) 

df.info()
#Unique values in LOCATION_ID column

df["LOCATION_ID"].unique()
print("These are the number of non-numeric values in LOCATION_ID: ", len(df[(df["LOCATION_ID"] == 'LOHARU') | (df["LOCATION_ID"] ==  'NUH') | (df["LOCATION_ID"] == 'SAFIDON')]))
df = df[(df.LOCATION_ID != 'LOHARU')]

df = df[(df.LOCATION_ID != 'NUH')]

df = df[(df.LOCATION_ID != 'SAFIDON')]

df = df.astype(float)

print("Updated number of rows in the dataset: ",len(df))
df = df.drop_duplicates(keep = 'first')

print("Updated number of rows in the dataset: ",len(df))
#Number of unique values in each columns

for i in range(0, len(df.columns)):

    print(df.columns[i], ":", df.iloc[:,i].nunique())
import seaborn as sns

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')

# 'RdBu_r' & 'BrBG' are other good diverging colormaps

cm = sns.diverging_palette(220, 20, sep=20, as_cmap=True) 

corr.style.background_gradient(cmap=cm).set_precision(2)
#Keeping just the columns that are correlated with the target variable and not with other independent variables.

df = df[['Risk_A', 'Risk_B', 'Risk_C', 'Risk_D','RiSk_E', 'Prob', 'Score', 'CONTROL_RISK',

        'Audit_Risk', 'Risk', 'MONEY_Marks', 'Loss']]

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')

# 'RdBu_r' & 'BrBG' are other good diverging colormaps

cm = sns.diverging_palette(220, 20, sep=20, as_cmap=True) 

corr.style.background_gradient(cmap=cm).set_precision(2)
#Creating a new dataframe for classification by deleting the Audit_Risk column.

class_df = df.drop("Audit_Risk", axis = 1)
classification_X = class_df.drop(["Risk"], axis = 1)

classification_y = class_df["Risk"]
from sklearn.model_selection import train_test_split



X_train_org, X_test_org, y_train, y_test = train_test_split(classification_X, classification_y, 

                                                            test_size = 0.25, random_state = 0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train = scaler.fit_transform(X_train_org)

X_test  = scaler.transform(X_test_org)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



lr_hard = LogisticRegression()

lr_hard.fit(X_train, y_train)

knn_hard = KNeighborsClassifier(7)

knn_hard.fit(X_train, y_train)

svc_hard = SVC(C = 10, probability = True)

svc_hard.fit(X_train, y_train)



voting_clf_hard = VotingClassifier(estimators=[('lr', lr_hard), ('knn', knn_hard), ('svc', svc_hard)], voting='hard')

voting_clf_hard.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in (lr_hard, knn_hard, svc_hard, voting_clf_hard):

    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    y_train_pred = clf.predict(X_train)

    print(clf.__class__.__name__, accuracy_score(y_test, y_test_pred))
report_table_1 = ['Hard Voting Classifier 1', '',accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)]
knn_hard = KNeighborsClassifier(3)

knn_hard.fit(X_train, y_train)

svc_hard = SVC(kernel='rbf', random_state= 0)

svc_hard.fit(X_train, y_train)

dt_hard = DecisionTreeClassifier(max_depth = 5, random_state= 0)

dt_hard.fit(X_train, y_train)



voting_clf_hard = VotingClassifier(estimators=[('knn', knn_hard), ('svc', svc_hard), ('dt', dt_hard)], voting='hard')

voting_clf_hard.fit(X_train, y_train)
for clf in (knn_hard, svc_hard, dt_hard, voting_clf_hard):

    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    y_train_pred = clf.predict(X_train)

    print(clf.__class__.__name__, accuracy_score(y_test, y_test_pred))
report_table_2 = ['Hard Voting Classifier 2', '',accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)]
lr_soft = LogisticRegression()

lr_soft.fit(X_train, y_train)

knn_soft = KNeighborsClassifier(7)

knn_soft.fit(X_train, y_train)

svc_soft = SVC(C = 10, probability = True)

svc_soft.fit(X_train, y_train)



voting_clf_soft = VotingClassifier(estimators=[('lr', lr_soft), ('knn', knn_soft), ('svc', svc_soft)], voting='soft')

voting_clf_soft.fit(X_train, y_train)
for clf in (knn_soft, svc_soft, lr_soft, voting_clf_soft):

    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    y_train_pred = clf.predict(X_train)

    print(clf.__class__.__name__, accuracy_score(y_test, y_test_pred))
report_table_3 = ['Soft Voting Classifier 1', '',accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)]
knn_soft = KNeighborsClassifier(5)

knn_soft.fit(X_train, y_train)

svc_soft = SVC(kernel='rbf', random_state= 0, probability= True)

svc_soft.fit(X_train, y_train)

dt_soft = DecisionTreeClassifier(max_depth = 7, random_state= 0)

dt_soft.fit(X_train_org, y_train)



voting_clf_soft = VotingClassifier(estimators=[('knn', knn_soft), ('svc', svc_soft), ('dt', dt_soft)], voting='soft')

voting_clf_soft.fit(X_train, y_train)
for clf in (knn_soft, svc_soft, dt_soft, voting_clf_soft):

    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    y_train_pred = clf.predict(X_train)

    print(clf.__class__.__name__, accuracy_score(y_test, y_test_pred))
report_table_4 = ['Soft Voting Classifier 2', '',accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)]
from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()



knn_param ={'n_neighbors': [3,5,7,11,15]}

knn_grid = GridSearchCV(knn, knn_param,cv = 5, n_jobs= -1)

knn_grid.fit(X_train, y_train)
print("Best Parameters for KNN Classifier: ", knn_grid.best_params_)
knn = KNeighborsClassifier(n_neighbors=3)

bag = BaggingClassifier(knn, bootstrap=True, random_state = 0)

#model param

grid_param = {'n_estimators': [100, 500, 1000],

              'max_samples': [0.1, 0.5, 1.0]}



#grid model

bag_knn_grid = GridSearchCV(bag, grid_param, cv = 5, n_jobs = -1, return_train_score= True)



#train grid model

bag_knn_grid.fit(X_train, y_train)
print("Best Parameters for Bagging Classifier: ", bag_knn_grid.best_params_)
bag = BaggingClassifier(knn, n_estimators=100, max_samples=1.0, n_jobs = -1, bootstrap=True, random_state=0)

bag.fit(X_train, y_train)
print("KNN with Bagging Training Score: ", bag.score(X_train, y_train))

print("KNN with Bagging Testing Score: ", bag.score(X_test, y_test))
report_table_5 = ['KNN with Bagging', 'n_neighbors: 3, max_samples: 1.0, n_estimators: 100',bag.score(X_train, y_train), bag.score(X_test, y_test)]
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

bag = BaggingClassifier(lr, bootstrap=True, random_state = 0)

grid_param = {'n_estimators': [100, 500, 1000],

              'max_samples': [0.1, 0.5, 1.0]}



#grid model

bag_lr_grid = GridSearchCV(bag, grid_param, cv = 5, n_jobs = -1, return_train_score= True)



#train grid model

bag_lr_grid.fit(X_train, y_train)
print("Best Parameters for Bagging Classifier: ", bag_lr_grid.best_params_)
bag = BaggingClassifier(lr, n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=0)

bag.fit(X_train, y_train)
print("Logistic Regression with Bagging Training Score: ", bag.score(X_train, y_train))

print("Logistic Regression with Bagging Testing Score: ", bag.score(X_test, y_test))
report_table_6 = ['Logistic Regression with Bagging', 'max_samples: 1.0, n_estimators: 500',bag.score(X_train, y_train), bag.score(X_test, y_test)]
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state = 0)

grid_param = {'max_depth': [3, 5, 7, 9, 11, 15]}

dt_grid = GridSearchCV(dt, grid_param, cv = 5, n_jobs = -1)

dt_grid.fit(X_train_org, y_train)
print("Best Parameters for Decision Tree Classifier: ", dt_grid.best_params_)
#base model

dt = DecisionTreeClassifier(max_depth = 9, random_state=0)

bag = BaggingClassifier(dt, random_state = 0)

#model param

grid_param = {'n_estimators': [100, 500, 1000],

              'max_samples': [0.1, 0.5, 1.0]}



#grid model

bag_dt_grid = GridSearchCV(bag, grid_param, cv = 5, n_jobs = -1, return_train_score= True)



#train grid model

bag_dt_grid.fit(X_train_org, y_train)
print("Best Parameters for Bagging Classifier with Pasting: ", bag_dt_grid.best_params_)
bag = BaggingClassifier(dt, n_estimators=100, max_samples=1.0, n_jobs = -1, bootstrap=False, random_state=0)

bag.fit(X_train_org, y_train)
print("Decision Tree Classifier with Pasting Training Score: ", bag.score(X_train_org, y_train))

print("Decision Tree Classifier with Pasting Testing Score: ", bag.score(X_test_org, y_test))
report_table_7 = ['Decision Tree Classifier with Pasting', 'max_depth: 9, max_samples: 1.0, n_estimators: 100',bag.score(X_train_org, y_train), bag.score(X_test_org, y_test)]
from sklearn.svm import LinearSVC

svc = LinearSVC(penalty = 'l2', random_state=0)

grid_param = {'C':[1, 10, 100, 1000]}

svc_grid = GridSearchCV(svc, grid_param, cv = 5)

svc_grid.fit(X_train, y_train)
print("Best Parameters for Linear SVC: ", svc_grid.best_params_)
#base model

svc = LinearSVC(C = 100, penalty = 'l2', random_state=0)

bag = BaggingClassifier(svc, random_state = 0)

#model param

grid_param = {'n_estimators': [100, 500, 1000],

              'max_samples': [0.1, 0.5, 1.0]}



#grid model

bag_svc_grid = GridSearchCV(bag, grid_param, cv = 5, n_jobs = -1, return_train_score= True)



#train grid model

bag_svc_grid.fit(X_train, y_train)
print("Best Parameters for Bagging Classifier with Pasting: ", bag_svc_grid.best_params_)
bag = BaggingClassifier(svc, n_estimators= 100, max_samples= 0.1, bootstrap=False, n_jobs=-1, random_state=0)

bag.fit(X_train, y_train)
print("Linear SVC with Pasting Training Score: ", bag.score(X_train, y_train))

print("Linear SVC with Pasting Testing Score: ", bag.score(X_test, y_test))
report_table_8 = ['Linear SVC with Pasting', 'C: 100, max_samples: 0.1, n_estimators: 100',bag.score(X_train, y_train), bag.score(X_test, y_test)]
from sklearn.ensemble import AdaBoostClassifier



dt = DecisionTreeClassifier(max_depth = 9, random_state=0)

ada = AdaBoostClassifier(dt)



param = {'n_estimators' : [100,500,1000],

        'learning_rate': [0.1, 0.5, 1, 10]}



ada_grid = GridSearchCV(ada, param, cv=5, n_jobs= -1)



ada_grid.fit(X_train_org, y_train)
print("Best Parameters for Adaboost Classifier: ", ada_grid.best_params_)
ada = AdaBoostClassifier(dt, n_estimators= 100, learning_rate= 0.1, algorithm="SAMME.R", random_state=0)

ada.fit(X_train_org, y_train)
print("Decision Tree Classifier with Adaboost Training Score: ", ada.score(X_train_org, y_train))

print("Decision Tree Classifier with Adaboost Testing Score: ", ada.score(X_test_org, y_test))
report_table_9 = ['Decision Tree Classifier with Adaboost', 'max_depth = 9, learning_rate = 0.1, n_estimators = 1000',

                  ada.score(X_train_org, y_train), ada.score(X_test_org, y_test)]
lr = LogisticRegression()

ada = AdaBoostClassifier(lr)



param = {'n_estimators' : [100,500,1000],

        'learning_rate': [0.1, 0.5, 1]}



ada_grid = GridSearchCV(ada, param, cv=5, n_jobs= -1)



ada_grid.fit(X_train, y_train)
print("Best Parameters for Adaboost Classifier: ", ada_grid.best_params_)
ada = AdaBoostClassifier(dt, n_estimators= 1000, learning_rate= 1, algorithm="SAMME.R", random_state=0)

ada.fit(X_train, y_train)
print("Logistic Regression with Adaboost Training Score: ", ada.score(X_train, y_train))

print("Logistic Regression with Adaboost Testing Score: ", ada.score(X_test, y_test))
report_table_10 = ['Logistic Regression with Adaboost', 'learning_rate = 1, n_estimators = 1000',

                  ada.score(X_train, y_train), ada.score(X_test, y_test)]
from  sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(random_state=0)



param = {'max_depth': [4,5,7,9,11,15],

           'n_estimators': [100, 500, 1000],

           'learning_rate': [0.01,0.1, 0.5, 1.0]}

gb_grid = GridSearchCV(gb, param, cv = 5, return_train_score= True)

gb_grid.fit(X_train, y_train)
print("Best Parameters for Gradient Boosting Classifier: ", gb_grid.best_params_)
gb = GradientBoostingClassifier(max_depth=4, n_estimators=100, learning_rate=1.0, random_state=0)

gb.fit(X_train, y_train)
print("Gradient Boosting Classifier Training Score: ", gb.score(X_train, y_train))

print("Gradient Boosting Classifier Testing Score: ", gb.score(X_test, y_test))
report_table_11 = ['Gradient Boosting Classifier', 'learning_rate = 1.0, max_depth = 4, n_estimators = 100',

                  gb.score(X_train, y_train), gb.score(X_test, y_test)]
report_table = pd.DataFrame(list(zip(report_table_1,

             report_table_2,

             report_table_3,

             report_table_4,

             report_table_5,

             report_table_6,

             report_table_7,

             report_table_8,

             report_table_9,

             report_table_10,

             report_table_11))).transpose()
report_table.columns = ['Model Name', 'Model Parameter', 'Training Score', 'Testing Score']

report_table.index = report_table['Model Name']

report_table.head(10)
from sklearn.decomposition import PCA



pca = PCA(n_components=0.95)



X_train_reduced = pca.fit_transform(X_train)

X_test_reduced = pca.transform(X_test)
print("Number of PCA components: ", pca.n_components_)
knn = KNeighborsClassifier()

param_grid = {'n_neighbors':[3, 4, 5, 6, 7, 8, 9, 10, 15]}



grid_knn_clf = GridSearchCV(knn, param_grid=param_grid, cv = 10, scoring='roc_auc')

grid_knn_clf.fit(X_train_reduced, y_train)
print("Best Parameters for KNN Classifier with PCA: ", grid_knn_clf.best_params_)
pca_knn = KNeighborsClassifier(n_neighbors=8)

pca_knn.fit(X_train_reduced, y_train)
print("KNN Classifier with PCA Training Score: ", pca_knn.score(X_train_reduced, y_train))

print("KNN Classifier with PCA Testing Score: ", pca_knn.score(X_test_reduced, y_test))
pca_report_table_1 = ['KNN Classifier with PCA', 'n_neighbors = 8', 

                      pca_knn.score(X_train_reduced, y_train), pca_knn.score(X_test_reduced, y_test)]
svc = SVC()

param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}



grid_svc_clf = GridSearchCV(svc, param_grid, cv = 5, scoring='roc_auc', return_train_score=True)

grid_svc_clf.fit(X_train_reduced, y_train)
print("Best Parameters for LinearSVC with PCA: ", grid_svc_clf.best_params_)
pca_svc = SVC(C= 10, kernel= 'rbf')

pca_svc.fit(X_train_reduced, y_train)
print("SVC with PCA Training Score: ", pca_svc.score(X_train_reduced, y_train))

print("SVC with PCA Testing Score: ", pca_svc.score(X_test_reduced, y_test))
pca_report_table_2 = ['SVC with PCA', 'C =10, kernel= rbf', 

                      pca_svc.score(X_train_reduced, y_train), pca_svc.score(X_test_reduced, y_test)]
pca_lr = LogisticRegression(random_state=0)



param_grid = {'penalty':['l1', 'l2']}



grid_log_clf = GridSearchCV(pca_lr , param_grid, cv = 5, return_train_score=True, scoring='roc_auc')

grid_log_clf.fit(X_train_reduced, y_train)
print("Best Parameters for Logistic Regression with PCA: ", grid_log_clf.best_params_)
pca_lr = LogisticRegression(penalty= 'l1')

pca_lr.fit(X_train_reduced, y_train)
print("Logistic Regression with PCA Training Score: ", pca_lr.score(X_train_reduced, y_train))

print("Logistic Regression with PCA Testing Score: ", pca_lr.score(X_test_reduced, y_test))
pca_report_table_3 = ['Logistic Regression with PCA', 'penalty = l1', 

                      pca_lr.score(X_train_reduced, y_train), pca_lr.score(X_test_reduced, y_test)]
#Base model

dt = DecisionTreeClassifier(random_state = 0)



#model param

grid_param = {'max_depth': [3, 5, 7, 9, 11, 15]}



#grid model

dt_grid = GridSearchCV(dt, grid_param, cv = 5, n_jobs = -1)



#train grid model

dt_grid.fit(X_train_reduced, y_train)
print("Best Parameters for Decision Tree Classifier: ", dt_grid.best_params_)
pca_dt = DecisionTreeClassifier(max_depth= 5, random_state= 0)

pca_dt.fit(X_train_reduced, y_train)
print("Decision Tree Classifier with PCA Training Score: ", pca_dt.score(X_train_reduced, y_train))

print("Decision Tree Classifier with PCA Testing Score: ", pca_dt.score(X_test_reduced, y_test))
pca_report_table_4 = ['Decision Tree Classifier with PCA', 'max_depth: 5', 

                      pca_dt.score(X_train_reduced, y_train), pca_dt.score(X_test_reduced, y_test)]
pca_report_table = pd.DataFrame(list(zip(pca_report_table_1,

             pca_report_table_2,

             pca_report_table_3,

             pca_report_table_4))).transpose()
pca_report_table.columns = ['Model Name', 'Model Parameter', 'Training Score', 'Testing Score']

pca_report_table.index = pca_report_table['Model Name']
pca_report_table.head(10)
report_table_without_pca = pd.read_csv('../input/classification-report/Classification Report Table without PCA.csv')

report_table_without_pca.head(5)
import matplotlib.pyplot as plt



ax = pca_report_table[['Training Score','Testing Score']].plot(kind='bar',

            title = "Comparison of Accuracies of Different Models with PCA", figsize=(8, 8), fontsize = 8)

plt.show()
from keras.models import Sequential

from keras.layers import Dense,MaxPooling1D

np.random.seed(0)



model = Sequential()

model.add(Dense(10, input_dim = 10, activation = 'sigmoid'))

model.add(Dense(1))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



model.fit(X_train, y_train, epochs = 40, batch_size = 50)
from sklearn.metrics import accuracy_score

y_train_pred = model.predict(X_train)

y_train_pred = np.where(y_train_pred >= 0.5 , 1, 0)

y_test_pred = model.predict(X_test)

y_test_pred = np.where(y_test_pred >= 0.5 , 1, 0)

print("NN Train Score: ",accuracy_score(y_train, y_train_pred))

print("NN Test Score: ", accuracy_score(y_test, y_test_pred))
print("Logistic Regression with PCA Training Score: ", pca_lr.score(X_train_reduced, y_train))

print("Logistic Regression with PCA Testing Score: ", pca_lr.score(X_test_reduced, y_test))
y_predicted = pca_lr.predict(X_test_reduced)

print("Predicted value for 1st testing row: ", y_predicted[0])

print("Original value for 1st testing row: ", y_test.values[0])

print("")

print("Predicted value for 5th testing row: ", y_predicted[4])

print("Original value for 5th testing row: ", y_test.values[4])