# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold, cross_val_score

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, f1_score, mean_squared_error, log_loss

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB
df = pd.read_csv("../input/train-hers/Train_data.csv")

df_test = pd.read_csv("../input/validationhers/Test_data.csv")
df.columns
df.drop(df.iloc[:,7:12], axis=1, inplace=True)

df_test.drop(df_test.iloc[:,7:12], axis=1, inplace=True)

df.drop(df.iloc[:,8:9], axis=1, inplace=True)

df_test.drop(df_test.iloc[:,8:9], axis=1, inplace=True)
df.columns = ["username","gender","bachelor_degree","type_of_work","10_sci","10_math","10_overall","science_satisfaction_score","physics","chemistry","maths","computer","english","realistic","investigative","artistic","social","enterprising","conventional","personality","mean_career_satisfaction"]

df_test.columns = ["username","gender","bachelor_degree","type_of_work","10_sci","10_math","10_overall","science_satisfaction_score","physics","chemistry","maths","computer","english","realistic","investigative","artistic","social","enterprising","conventional","personality","mean_career_satisfaction"]
df.drop(["username"], axis=1, inplace=True)

df_test.drop(["username"], axis=1, inplace=True)
df.drop(df.index[[0,1,47]], inplace = True)

df_test.drop(df_test.index[[18,80,81]], inplace=True)
degree = df.bachelor_degree.copy()

degree_test = df_test.bachelor_degree.copy()

df.drop("bachelor_degree", axis=1, inplace=True)

df_test.drop("bachelor_degree", axis=1, inplace=True)
df_test["bachelor_degree"] = degree_test.copy()

y_test = df_test.mean_career_satisfaction.copy()

df_test.drop("mean_career_satisfaction", axis=1, inplace=True)
y_test.reset_index(drop=True, inplace =True)
df = pd.get_dummies(df)

df_test = pd.get_dummies(df_test)
smote = SMOTE(k_neighbors = 2, random_state=492)

x_smote, y_smote = smote.fit_resample(df,degree)
x_smote["bachelor_degree"] = y_smote

y = x_smote.mean_career_satisfaction

x_smote.drop("mean_career_satisfaction", axis=1, inplace=True)
x_smote = pd.get_dummies(x_smote)
y_cf_train = y.copy()

for i in range(len(y_cf_train)):

    if y_cf_train[i] >=3:

        y_cf_train[i] = 1

    else:

        y_cf_train[i] = 0





y_cf_test = y_test.copy()

for i in range(len(y_cf_test)):

    if y_cf_test[i] >=3:

        y_cf_test[i] = 1

    else:

        y_cf_test[i] = 0
smote = SMOTE(k_neighbors=5, random_state=492)

x_smote_f, y_cf_f = smote.fit_resample(x_smote, y_cf_train)
x_smote_f.columns
def classification_model(classifier):

    model = classifier

    model.fit(x_smote_f, y_cf_f)

    pred_train = model.predict(x_smote_f)

    pred_test = model.predict(df_test)

    pred_prob_test = model.predict_proba(df_test)[:,1]

    pred_prob_train = model.predict_proba(x_smote_f)[:1]

    print("train_mse",mean_squared_error(y_cf_f,pred_train))

    print("test_mse", mean_squared_error(y_cf_test, pred_test))

    print("train_log-loss",log_loss(y_cf_f,pred_train))

    print("test_log-loss", log_loss(y_cf_test, pred_test))

    print(classifier)

    print("Confusion matrix",'\n', pd.crosstab(y_cf_test, pred_test))

    print(classification_report(y_cf_test, pred_test))

    print(roc_auc_score(y_cf_test, pred_prob_test))

    fpr, tpr, threshold = roc_curve(y_cf_test, pred_prob_test)

    roc_auc = auc(fpr,tpr)

    sns.set()

    plt.figure()

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    return model

classification_model(DecisionTreeClassifier(max_depth=200,min_samples_split=10,min_samples_leaf=10,random_state=10))

classification_model(RandomForestClassifier(random_state=10,bootstrap= True, max_depth= 13, min_samples_leaf= 1, min_samples_split= 5, n_estimators=300))

classification_model(BaggingClassifier(n_estimators=2000,random_state=10))

classification_model(SVC(random_state=10, probability=True))

classification_model(KNeighborsClassifier(n_neighbors=4))

classification_model(XGBClassifier(base_score=0.5,subsample= 0.8, min_child_weight=1, max_depth=10, gamma= 0.4, colsample_bytree= 0.8, learning_rate=0.15, random_state=10 ))

classification_model(GradientBoostingClassifier(n_estimators= 400, min_samples_split= 50, min_samples_leaf=40, max_depth=500, learning_rate= 0.01,random_state=10))

classification_model(LogisticRegression(max_iter=1000,random_state=10))

classification_model(GaussianNB())
# Hyperparameter Training of Decision Tree 

# model = DecisionTreeClassifier(random_state=10)

# max_depth = [200,250,500,1000,1500,2000]

# min_samples_leaf = [10,20,30,40,50,60,70,80,90,100]

# min_samples_split = [10,20,30,40,50,60,70,80,90,100]

# grid = dict(max_depth = max_depth,

# min_samples_leaf = min_samples_leaf,

# min_samples_split = min_samples_split )

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)

# grid_search_dt = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='roc_auc',error_score=0)

# grid_result_dt = grid_search_dt.fit(x_smote_f, y_cf_f)
# print(grid_result_dt.best_params_)

# print(grid_result_dt.best_score_)
# Hyperparameter Training of GradientBoostingClassifier

# model = GradientBoostingClassifier()

# learning_rate = [0.1,0.01,0.001]

# n_estimators = [50,100,150,200,250,300,350]

# max_depth = [500,1000,1500,2000]

# min_samples_leaf = [30,40,50,60,70]

# min_samples_split = [30,40,50,60,70]

# grid = dict(learning_rate = learning_rate,n_estimators=n_estimators,

#     max_depth = max_depth,

# min_samples_leaf = min_samples_leaf,

# min_samples_split = min_samples_split )

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)

# grid_search_gb = RandomizedSearchCV(estimator=model, param_distributions=grid, n_jobs=-1,n_iter=100, scoring='f1',error_score=0)

# grid_result_gb = grid_search_gb.fit(x_smote_f, y_cf_f)
# print(grid_result_gb.best_params_)

# print(grid_result_gb.best_score_)
# Hyperparameter Training RandomForestClassifier

# model = RandomForestClassifier()

# max_depth = [5,7,9,11,13,15]

# min_samples_leaf = [1,2,3,4,5]

# min_samples_split = [4,5,6,7,8,9]

# n_estimators = [10,50,100,150,200,300,350,400,500,600]

# bootstrap = ["True", "False"]

# max_features = ["sqrt","log2"]

# grid = dict(max_depth = max_depth,

# min_samples_leaf = min_samples_leaf,

# min_samples_split = min_samples_split,

#            n_estimators = n_estimators,

#            bootstrap = bootstrap,

#            max_features = max_features)

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)

# grid_search_rf = RandomizedSearchCV(estimator=model, param_distributions=grid, n_jobs=-1, scoring='roc_auc',error_score=0)

# grid_result_rf = grid_search.fit(x_smote_f, y_cf_f)
# print(grid_result_rf.best_params_)

# print(grid_result_rf.best_score_
# Hyperparameter Tuning Bagging Classifier

# model = BaggingClassifier()

# n_estimators = [10, 100, 1000, 2000]

# grid = dict(n_estimators=n_estimators)

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)

# grid_search_bag = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)

# grid_result_bag = grid_search.fit(x_smote_f, y_cf_f)
# print(grid_result_bag.best_params_)

# print(grid_result_bag.best_score_)
# Hyperparameter Tuning XGBoost Classifier

# params = {

#         'min_child_weight': [1, 5, 10],

#         'gamma': [0.5, 1, 1.5, 2, 5],

#         'subsample': [0.6, 0.8, 1.0],

#         'colsample_bytree': [0.6, 0.8, 1.0],

#         'max_depth': [3, 4, 5]

#         }

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)

# model=XGBClassifier(random_state=10)

# grid_search_xgb = RandomizedSearchCV(estimator=model, param_distributions=params,n_iter=100, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)

# grid_result_xgb = grid_search.fit(x_smote_f, y_cf_f)
# print(grid_result_xgb.best_params_)

# print(grid_result_xgb.best_score_)
x_smote.columns
classification_model(AdaBoostClassifier(DecisionTreeClassifier(max_depth=500,min_samples_split=50,min_samples_leaf=40, random_state=10),

                                        n_estimators = 110,

                                        learning_rate=0.001,

                                        random_state=10))
final_data = pd.concat([x_smote_f,df_test],axis = 0,ignore_index=True)

final_y = pd.concat([y_cf_f, y_cf_test], ignore_index=True, axis=0)
#Final Model



final_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=500,min_samples_split=50,min_samples_leaf=40),

                                        n_estimators = 110,

                                        learning_rate=0.001,

                                        random_state=10)

final_model.fit(final_data, final_y)
# test = pd.read_csv("../input/data-collected-hers/hers_1.csv")
# test.English.fillna(value=np.mean(test.English), inplace=True)

# test.Chemistry.fillna(value=np.mean(test.Chemistry), inplace=True)

# test.Physics.fillna(value=np.mean(test.Physics), inplace=True)

# test.Mathematics.fillna(value=np.mean(test.Mathematics), inplace=True)

# test.Computer.fillna(value=np.mean(test.Computer), inplace=True)
# test.columns = ["gender","type_of_work","10_sci","10_math","10_overall","science_satisfaction_score","physics","chemistry","maths","computer","english","realistic","investigative","artistic","social","enterprising","conventional","personality"]
# test.loc[test.gender=="female",['gender']]="Female"

# test.loc[test.gender=="male",['gender']]="Male"



# test.loc[test.type_of_work=="office_work",['type_of_work']]="Office Work"

# test.loc[test.type_of_work=="field_work",['type_of_work']]="Field Work"
# test.drop(test.index[34], inplace=True)


# test = pd.get_dummies(test)

bach_degree = ['bachelor_degree_BOTANY',

       'bachelor_degree_CHEMICAL ENEGINEERING', 'bachelor_degree_CHEMISTRY',

       'bachelor_degree_CIVIL ENGINEERING',

       'bachelor_degree_COMPUTER SCIENCE AND ENGINEERING',

       'bachelor_degree_ELECTRICAL ENEGINEERING',

       'bachelor_degree_ELECTRONICS', 'bachelor_degree_ENVIRONMENTAL SCIENCE',

       'bachelor_degree_GEOGRAPHY', 'bachelor_degree_GEOLOGY',

       'bachelor_degree_MATHEMATICS', 'bachelor_degree_MECHANICAL ENGINEERING',

       'bachelor_degree_METALLURGY AND MATERIALS ENEGINEERING',

       'bachelor_degree_PHYSICS', 'bachelor_degree_STATISTICS',

       'bachelor_degree_TEXTILE ENGINEERING']
# d = {name: pd.DataFrame({'gender':test.physics}) for name in bach_degree}

# print(d.keys())
# for name,df in d.items():

#     for i in range(len(d.keys())):

#         for j in bach_degree:

#             if name!=j:

#                 df[j] = 0

#             else:

#                 df[name] = 1

#     df.drop('gender', axis=1,inplace=True)
# bach_degree_clean = ['BOTANY',

#        'CHEMICAL_ENEGINEERING', 'CHEMISTRY',

#        'CIVIL_ENGINEERING',

#        'COMPUTER_SCIENCE_AND_ENGINEERING',

#        'ELECTRICAL_ENEGINEERING',

#        'ELECTRONICS', 'ENVIRONMENTAL_SCIENCE',

#        'GEOGRAPHY', 'GEOLOGY',

#        'MATHEMATICS', 'MECHANICAL_ENGINEERING',

#        'METALLURGY_AND_MATERIALS_ENEGINEERING',

#        'PHYSICS', 'STATISTICS',

#        'TEXTILE_ENGINEERING']
# final_df = {}

# for i in range(len(bach_degree_clean)):

#     final_df[bach_degree_clean[i]] = pd.concat([test,d.get(bach_degree[i])], axis=1)
# for name in bach_degree_clean:

#     print(name)
# predictions = pd.DataFrame()

# for i in range(len(bach_degree_clean)):

#     pred_prob = final_model.predict_proba(final_df.get(bach_degree_clean[i]))

#     predictions[bach_degree_clean[i]] = pred_prob[:,1]