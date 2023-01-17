# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')



# for others analyse save we the original data

data_new = data.copy()



data.head()
# Data infos

data.info()
# Data features scatter plot

sns.set(style="ticks")

sns.pairplot(data,hue='class',diag_kind='hist',palette='husl',markers='D');
# Staitistical infos of Data

data.describe().T
# Feature counts of class

data['class'].value_counts()
# Visualization of class data-value counts

sns.countplot(data['class']);
# NaN's feature of Data

data.isnull().any().sum()
# classification of class (make binary features of 0,1)

data['class'] = [1 if i == 'Abnormal'else 0 for i in data['class']]



# labels or dependet features of Data

y = data[['class']]



# independet features of Data

x_data = data.drop(['class'],axis = 1)



# and normalization of feature

x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Modul import for encode

from sklearn.preprocessing import OneHotEncoder



# 'Abnormal' and 'Normal' features

binary_values = data_new[['class']]



# model

ohe = OneHotEncoder()



binary = ohe.fit_transform(binary_values).toarray()



binary[:10]
# correlations between features of data

data.corr()
# visualization data correlation

sns.heatmap(data.corr(),annot = True,fmt='.2f',linewidths=0.5,linecolor='b')

plt.title('Correlation Heatmap');
# now Correlation analyse with p-Value



# dependet feature

y = data[['class']]



# independet features of Data

x_data = data.drop(['class'],axis = 1)



import statsmodels.api as sm



# model

analyse = sm.OLS(y,x_data).fit()



analyse.summary()
# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# modul imort

from sklearn.linear_model import LogisticRegression



# model

lg = LogisticRegression()



# fit

lg.fit(x_train,y_train)



# predicts

log_predicts = lg.predict(x_test)



# accuracy with Logistic Regression Model

accuracy_log = lg.score(x_test,y_test)



# confusion metrics of Logistic Regression Model

from sklearn.metrics import confusion_matrix



cm_log = confusion_matrix(y_test,log_predicts)



# correlation the predicts values with x_test of data

lg_analyse = sm.OLS(log_predicts,x_test).fit()



lg_analyse.summary()
# Backward elimination of features bigger than p-value (0.05)

# Elimination of features : lumbar_lordosis_angle(0.484) and degree_spondylolisthesis(0.884)



x_train_b = x_train.drop(['lumbar_lordosis_angle','degree_spondylolisthesis'],axis = 1)

x_test_b = x_test.drop(['lumbar_lordosis_angle','degree_spondylolisthesis'],axis = 1)



# model

lg_b = LogisticRegression()



# fit

lg_b.fit(x_train_b,y_train)



# predicts

log_predicts_b = lg_b.predict(x_test_b)



# accuracy with Logistic Regression Model

accuracy_log_b = lg_b.score(x_test_b,y_test)



# confusion metrics of Logistic Regression Model

from sklearn.metrics import confusion_matrix



cm_log_b = confusion_matrix(y_test,log_predicts_b)



# correlation the predicts values with x_test of data

lg_analyse_b = sm.OLS(log_predicts_b,x_test_b).fit()



lg_analyse_b.summary()

# we have now bigger R-squared 
 # Grid Cross Validation



from sklearn.model_selection import GridSearchCV

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge



logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv = 10)

logreg_cv.fit(x_train,y_train)



print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)

print("accuracy: ",logreg_cv.best_score_)
# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
# modul import

from sklearn.neighbors import KNeighborsClassifier



# find best k

k_neighbors = np.arange(1,10)

score_list = []

for i in k_neighbors:

    

    knn_i = KNeighborsClassifier(n_neighbors = i)

    knn_i.fit(x_train,y_train)

    score_list.append(knn_i.score(x_test,y_test))

    

plt.plot(k_neighbors,score_list)

plt.xlabel('k_number')

plt.ylabel('score of k');
# we have the best k = 6

knn = KNeighborsClassifier(n_neighbors = 6)



knn.fit(x_train,y_train)



knn_predicts = knn.predict(x_test)



accuracy_knn = knn.score(x_test,y_test)



from sklearn.metrics import confusion_matrix



cm_knn = confusion_matrix(y_test,knn_predicts)



# correlation the predicts values with x_test of data

knn_analyse = sm.OLS(knn_predicts,x_test).fit()



knn_analyse.summary()
# Backward elimination of features bigger than p-value (0.05)

# Now elimination of features : lumbar_lordosis_angle(0.088)



x_train_b1 = x_train.drop(['lumbar_lordosis_angle'],axis=1)

x_test_b1 = x_test.drop(['lumbar_lordosis_angle'],axis=1)



# we have the best k = 6

knn_b = KNeighborsClassifier(n_neighbors = 6)



knn_b.fit(x_train_b1,y_train)



knn_predicts_b = knn_b.predict(x_test_b1)



accuracy_knn_b = knn_b.score(x_test_b1,y_test)



from sklearn.metrics import confusion_matrix



cm_knn_b = confusion_matrix(y_test,knn_predicts_b)



# correlation the predicts values with x_test of data

knn_analyse_b = sm.OLS(knn_predicts_b,x_test_b).fit()



knn_analyse_b.summary()
# Grid Cross Validation

grid = {"n_neighbors":np.arange(1,50)}

knn= KNeighborsClassifier()



knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV

knn_cv.fit(x_train,y_train)



#%% print hyperparameter KNN 

print("tuned hyperparameter K: ",knn_cv.best_params_)

print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)
# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
# modul import

from sklearn.svm import SVC



# model

svm = SVC(random_state = 42)



# fit

svm.fit(x_train,y_train)



# predicts

svm_predicts = svm.predict(x_test)



# accuracy

accuracy_svm = svm.score(x_test,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_svm = confusion_matrix(y_test,svm_predicts)



# correlation the predicts values with x_test of data

svm_analyse = sm.OLS(svm_predicts,x_test).fit()



svm_analyse.summary()
# Backward elimination of features bigger than p-value (0.05)

# Elimination of features : lumbar_lordosis_angle(0.334)



x_train_b2 = x_train.drop(['lumbar_lordosis_angle'],axis=1)

x_test_b2 = x_test.drop(['lumbar_lordosis_angle'],axis=1)



# modul import

from sklearn.svm import SVC



# model

svm_b = SVC(random_state = 42)



# fit

svm_b.fit(x_train_b2,y_train)



# predicts

svm_predicts_b = svm_b.predict(x_test_b2)



# accuracy

accuracy_svm_b = svm_b.score(x_test_b2,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_svm_b = confusion_matrix(y_test,svm_predicts_b)



# correlation the predicts values with x_test of data

svm_analyse_b = sm.OLS(svm_predicts_b,x_test_b).fit()



svm_analyse_b.summary()
# model

svm = SVC(random_state = 42)



# fit

svm.fit(x_train,y_train)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator = svm, X = x_train, y= y_train, cv = 10)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
# modul import

from sklearn.naive_bayes import GaussianNB



# model

nb = GaussianNB()



# fit

nb.fit(x_train,y_train)



# predicts

nb_predicts = nb.predict(x_test)



# accuracy

accuracy_nb = nb.score(x_test,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_nb = confusion_matrix(y_test,nb_predicts)



# correlation the predicts values with x_test of data

nb_analyse = sm.OLS(nb_predicts,x_test).fit()



nb_analyse.summary()
# model

nb = GaussianNB()



# fit

nb.fit(x_train,y_train)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator = nb, X = x_train, y= y_train, cv = 10)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))
# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
# modul import

from sklearn.tree import DecisionTreeClassifier



# model

d_tree = DecisionTreeClassifier()



# fit

d_tree.fit(x_train,y_train)



# predicts

d_tree_predicts = d_tree.predict(x_test)



# accuracy

accuracy_d_tree = d_tree.score(x_test,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_tree = confusion_matrix(y_test,d_tree_predicts)



# correlation the predicts values with x_test of data

d_tree_analyse = sm.OLS(d_tree_predicts,x_test).fit()



d_tree_analyse.summary()
# Backward elimination of features bigger than p-value (0.05)

# Elimination of features : lumbar_lordosis_angle(0.992),pelvic_radius(0.384)



x_train_b3 = x_train.drop(['lumbar_lordosis_angle','pelvic_radius'],axis=1)

x_test_b3 = x_test.drop(['lumbar_lordosis_angle','pelvic_radius'],axis=1)



# modul import

from sklearn.tree import DecisionTreeClassifier



# model

d_tree_b = DecisionTreeClassifier()



# fit

d_tree_b.fit(x_train_b3,y_train)



# predicts

d_tree_predicts_b = d_tree_b.predict(x_test_b3)



# accuracy

accuracy_d_tree_b = d_tree_b.score(x_test_b3,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_tree_b = confusion_matrix(y_test,d_tree_predicts_b)



# correlation the predicts values with x_test of data

d_tree_analyse_b = sm.OLS(d_tree_predicts_b,x_test_b3).fit()



d_tree_analyse_b.summary()

# model

d_tree = DecisionTreeClassifier()



# fit

d_tree.fit(x_train,y_train)



from sklearn.model_selection import cross_val_score





accuracies = cross_val_score(estimator = d_tree, X = x_train, y= y_train, cv = 10)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))
# for training ours model, splittin as train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.25,random_state = 42)
# modul import

from sklearn.ensemble import RandomForestClassifier



# model

r_forest = RandomForestClassifier(n_estimators = 100,random_state = 42)



# fit

r_forest.fit(x_train,y_train)



# predicts

r_forest_predicts = r_forest.predict(x_test)



# accuracy

accuracy_r_forest = r_forest.score(x_test,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_forest = confusion_matrix(y_test,r_forest_predicts)



# correlation the predicts values with x_test of data

r_forest_analyse = sm.OLS(r_forest_predicts,x_test).fit()



r_forest_analyse.summary()
# Backward elimination of features bigger than p-value (0.05)

# Elimination of features : lumbar_lordosis_angle(0.693)



x_train_b4 = x_train.drop(['lumbar_lordosis_angle'],axis=1)

x_test_b4 = x_test.drop(['lumbar_lordosis_angle'],axis=1)



# modul import

from sklearn.ensemble import RandomForestClassifier



# model

r_forest_b = RandomForestClassifier(n_estimators = 100,random_state = 42)



# fit

r_forest_b.fit(x_train_b4,y_train)



# predicts

r_forest_predicts_b = r_forest_b.predict(x_test_b4)



# accuracy

accuracy_r_forest_b = r_forest_b.score(x_test_b4,y_test)



# confusion metrics

from sklearn.metrics import confusion_matrix



cm_forest_b = confusion_matrix(y_test,r_forest_predicts_b)



# correlation the predicts values with x_test of data

r_forest_analyse_b = sm.OLS(r_forest_predicts_b,x_test_b).fit()



r_forest_analyse_b.summary()
grid = {"n_estimators":np.arange(1,50)}

random_f= RandomForestClassifier()



random_cv = GridSearchCV(random_f, grid, cv = 10)  # GridSearchCV

random_cv.fit(x_train,y_train)



print("tuned hyperparameter K: ",random_cv.best_params_)

print("tuned parametreye gore en iyi accuracy (best score): ",random_cv.best_score_)
accuracys = ({'accuracy_log':accuracy_log,'accuracy_knn':accuracy_knn,

             'accuracy_svm':accuracy_svm,'accuracy_nb':accuracy_nb,

             'accuracy_d_tree':accuracy_d_tree,'accuracy_r_forest':accuracy_r_forest})



#confusions = [cm_log,cm_knn,cm_svm,cm_nb,cm_tree,cm_forest]



accuracy = pd.DataFrame.from_dict(accuracys,orient='index')



plt.figure(figsize = (10,6))



plt.plot(accuracy,color='r')

plt.scatter(accuracy.index,accuracy.values);

             
# confusions of Models

confusions = {'cm_log':cm_log,'cm_knn':cm_knn,'cm_svm':cm_svm,'cm_nb':cm_nb,'cm_tree':cm_tree,'cm_forest':cm_forest}



for i,j in confusions.items():



    sns.heatmap(j,annot=True,fmt='.0f',linewidths=0.5,linecolor='r')

    plt.title('Confusion of {}'.format(i))

    plt.show()
# confusion performans of models



# number of normal features is 21

n_normal = 21



# number of abnormal is 57

n_abnormal = 57



# True predicts of normal is true/21 and for abnormal true/57

# performans is equal : true/21 + true/57



confus_performans = []



for i ,j  in confusions.items():

    

    performans_normal = j[0][0] / n_normal

    

    performans_abnormal = j[1][1] / n_abnormal

    

    total_performans = performans_normal + performans_abnormal

    

    confus_performans.append((i,total_performans))

    

confus_performans

    
# Best cunfusion performans



conf = pd.DataFrame(confus_performans)



conf.set_index(0)



plt.plot(conf[0],conf[1],color = 'r')

plt.scatter(conf[0],conf[1]);

plt.title('True predicts of Normal vs Abnormal')

plt.ylabel('Sum of True predicts');
accuracys_b = ({'accuracy_log_b':accuracy_log_b,'accuracy_knn_b':accuracy_knn_b,

             'accuracy_svm_b':accuracy_svm_b,'accuracy_nb':accuracy_nb,

             'accuracy_d_tree_b':accuracy_d_tree_b,'accuracy_r_forest_b':accuracy_r_forest_b})



accuracy_b = pd.DataFrame.from_dict(accuracys_b,orient='index')



plt.figure(figsize = (16,6))



plt.plot(accuracy_b,color='g')

plt.scatter(accuracy_b.index,accuracy_b.values);



# confusions of Models with Backward Elimination

confusions_b = {'cm_log_b':cm_log_b,'cm_knn_b':cm_knn_b,'cm_svm_b':cm_svm_b,'cm_nb':cm_nb,'cm_tree_b':cm_tree_b,'cm_forest_b':cm_forest_b}



for i,j in confusions_b.items():



    sns.heatmap(j,annot=True,fmt='.0f',linewidths=0.5,linecolor='r',cmap="Greens")

    plt.title('Confusion of {}'.format(i))

    plt.show()
# confusion performans of models with Backwards Elimination



# number of normal features is 21

n_normal_b = 21



# number of abnormal is 57

n_abnormal_b = 57



# True predicts of normal is true/21 and for abnormal true/57

# performans is equal : true/21 + true/57



confus_performans_b = []



for i ,j  in confusions_b.items():

    

    performans_normal_b = j[0][0] / n_normal_b

    

    performans_abnormal_b = j[1][1] / n_abnormal_b

    

    total_performans_b = performans_normal_b + performans_abnormal_b

    

    confus_performans_b.append((i,total_performans_b))

    

confus_performans_b
# Best cunfusion performans with Backward Elimination



conf_b = pd.DataFrame(confus_performans_b)



conf_b.set_index(0)



plt.plot(conf_b[0],conf_b[1],color = 'green')

plt.scatter(conf_b[0],conf_b[1]);

plt.title('True predicts of Normal vs Abnormal')

plt.ylabel('Sum of True predicts');
# Now compare the confusion performans of models vs models with backward elimination



compare = list(zip((confus_performans[:6],confus_performans_b)))



print('The first list is for Models,and second list for Models with Bacward Elimination:\n')

for i in range(len(compare)):

    print('List {} : {}'.format(i+1,compare[i]))

    
# parameters for cros validation

random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(probability=True,random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
# Best parameters

cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = (GridSearchCV(

                       #probability=True,

                       classifier[i],

                       param_grid=classifier_param[i],

                       cv = StratifiedKFold(n_splits = 10),

                       scoring = "accuracy", n_jobs = -1,verbose = 1))

    

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
# Visualisation

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means","ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores");
# the bests

votingC = VotingClassifier(estimators = [("SVM",best_estimators[1]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3]),

                                        ('knn',best_estimators[4])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(x_train, y_train)

y_predict = votingC.predict(x_test)
accuracy_score(y_predict,y_test)