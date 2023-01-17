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
!pip install pandas-profiling
!pip install missingno
!pip install catboost
!pip install mlens

import numpy as np
import pandas as pd
import pandas_profiling
import missingno as msno

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

%matplotlib inline

# Close warnings
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head(5)
df.tail(5)
df.info()
df.profile_report()
# Unique values for the variables 

for i, col in enumerate(df):
    unique=df.iloc[:,i].unique()
    unique.sort()
    display(col, unique)
# Error correction regarding 0 values those do not make sense

print('0 values in below features are replaced as nan:')
for i, col in enumerate(df):
    if i> 0 and i<6:
        df[col].replace(to_replace=0, value=np.nan, inplace=True)
        display(col)
# Ref <https://www.kaggle.com/kingychiu/home-credit-eda-distributions-and-outliers>

total_nans = df.isna().sum()
nan_precents = (df.isna().sum()/df.isna().count()*100)
feature_overview_df  = pd.concat([total_nans, nan_precents], axis=1, keys=['NaN Count', 'NaN Pencent'])
feature_overview_df['Type'] = [df[c].dtype for c in feature_overview_df.index]
pd.set_option('display.max_rows', None)
display(feature_overview_df)
pd.set_option('display.max_rows', 20)
# get the number of missing data points per column
missing_values_count = df.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
fig, ax = plt.subplots(4, 2, figsize=(16,16))

sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,0])
sns.distplot(df.Glucose, bins = 20, ax=ax[0,1])
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,0])
sns.distplot(df.SkinThickness, bins = 20, ax=ax[1,1])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,0])
sns.distplot(df.BMI, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0])
sns.distplot(df.Age, bins = 20, ax=ax[3,1])
msno.matrix(df)
msno.bar(df)
plt.show()
# Nullity correlation

msno.heatmap(df)
msno.dendrogram(df)
# Making a copy of df with null values

df_with_null = df
# Imputation

my_imputer = SimpleImputer()
df = pd.DataFrame(my_imputer.fit_transform(df))
df.columns = df_with_null.columns
df.isnull().sum()
for x_index, col_1 in enumerate(df):
    for y_index, col_2 in enumerate(df):
        if (x_index != y_index &  y_index != (x_index-1)):
            
            fig = plt.gcf()
            fig.set_size_inches(9, 6)
            sns.scatterplot(x=df.iloc[:, x_index], y=df.iloc[:, y_index], hue=df.iloc[:,-1], data=df);
            plt.show()
df.describe([0.01,0.1,0.25,0.5,0.75,0.99]).T
df.corr()
fig, ax = plt.subplots(figsize=(12,8)) 
sns.heatmap(df.iloc[:,0:len(df)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()
# Features and Target Variable

X = df.drop(['Outcome'], axis=1)
y = df["Outcome"]
# Applying Standard Scaling to get optimized results

scale = StandardScaler()
X = scale.fit_transform(X)
# Split the data into training/testing sets

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=y, stratify=y, random_state = 42)
# Set cross validation 

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# Data with Null Values

X_with_null = df_with_null.drop(['Outcome'], axis=1)
y_with_null = df_with_null["Outcome"]

scale = StandardScaler()
X_with_null = scale.fit_transform(X_with_null)

X_train_with_null, X_test_with_null, y_train_with_null, y_test_with_null= train_test_split(
    X_with_null, y_with_null, test_size=0.2, shuffle=y, stratify=y, random_state = 42)

# Create the model
log_reg = LogisticRegression(random_state=42)

# Fit the model
log_reg.fit(X_train, y_train)

# Predict the model
y_pred = log_reg.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
logit_roc_auc = roc_auc_score(y, log_reg.predict(X))

fpr, tpr, thresholds = roc_curve(y, log_reg.predict_proba(X)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.title('ROC')
plt.show()
#  Model tuning

log_reg_params = {"C":np.logspace(-1, 1, 10),
                  "penalty": ["l1","l2"], 
                  "solver":['lbfgs', 'liblinear', 'sag', 'saga'], 
                  "max_iter":[1000]}
log_reg = LogisticRegression(random_state=42)
log_reg_cv_model = GridSearchCV(log_reg, log_reg_params, cv=sss)
log_reg_cv_model.fit(X_train, y_train)
print("Best score:" + str(log_reg_cv_model.best_score_))
print("Best parameters: " + str(log_reg_cv_model.best_params_))
# Create the model
nb = GaussianNB()

# Fit the model
nb_model = nb.fit(X_train, y_train)

# Predict the model
y_pred = nb_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
#  Model tuning

nb_params = {'var_smoothing': np.logspace(0,-9, num=100)}
nb = GaussianNB()
nb_cv_model = GridSearchCV(nb, nb_params, cv=sss)
nb_cv_model.fit(X_train, y_train)
print("Best score:" + str(nb_cv_model.best_score_))
print("Best parameters: " + str(nb_cv_model.best_params_))
# Create the model
knn = KNeighborsClassifier()

# Fit the model
knn_model = knn.fit(X_train, y_train)

# Predict the model
y_pred = knn_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
#  Model tuning

knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv_model = GridSearchCV(knn, knn_params, cv=sss)
knn_cv_model.fit(X_train, y_train)
print("Best score:" + str(knn_cv_model.best_score_))
print("Best parameters: " + str(knn_cv_model.best_params_))
kernel = "linear"
svm_model = SVC(kernel = kernel,random_state=42).fit(X_train, y_train)
svm_model
y_pred = svm_model.predict(X_test)
print(i, "accuracy score:", accuracy_score(y_test, y_pred))
svc_params = {"C": [0.00001, 0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100, 500,1000],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}
svc = SVC(kernel = kernel)
svc_linear_cv_model = GridSearchCV(svc,svc_params, 
                            cv = sss, 
                            n_jobs = -1, 
                            verbose = 2 )
svc_linear_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(svc_linear_cv_model.best_params_))
kernel = "rbf"
svm_model = SVC(kernel = kernel,random_state=42).fit(X_train, y_train)
svm_model
y_pred = svm_model.predict(X_test)
print(i, "accuracy score:", accuracy_score(y_test, y_pred))
svc_params = {"C": [0.00001, 0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100, 500,1000],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}
svc = SVC(kernel = kernel)
svc_rbf_cv_model = GridSearchCV(svc,svc_params, 
                            cv = sss, 
                            n_jobs = -1, 
                            verbose = 2 )
svc_rbf_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(svc_rbf_cv_model.best_params_))
# Create the model
cart = DecisionTreeClassifier(random_state=42)

# Fit the model
cart_model = cart.fit(X_train, y_train)

# Predict the model
y_pred = cart_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
cart = DecisionTreeClassifier(random_state=42)
cart_cv = GridSearchCV(cart, cart_grid, cv = sss, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
# Create the model
bagging = BaggingClassifier(n_estimators = 500, max_samples = 0.5, max_features = 0.5, random_state=42) 

# Fit the model
bagging_model = bagging.fit(X_train, y_train)

# Predict the model
y_pred = bagging_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

bagging_params = {
 'n_estimators': [10, 100, 500, 1000],
 'max_samples' : [0.05, 0.1, 0.2, 0.5]
}
bagging = BaggingClassifier(random_state=42) 
bagging_cv_model = GridSearchCV(bagging, bagging_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             )
bagging_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(bagging_cv_model.best_params_))
# Create the model
rf = RandomForestClassifier(random_state=42)

# Fit the model
rf_model = rf.fit(X_train, y_train)

# Predict the model
y_pred = rf_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,3,5,7],
            "n_estimators": [10,100,200,500,1000],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = sss, 
                           n_jobs = -1, 
                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
# Create the model
ex_tree = ExtraTreesClassifier(n_estimators=300, random_state=42) 

# Fit the model
ex_tree_model = ex_tree.fit(X_train, y_train)

# Predict the model
y_pred = ex_tree_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

ex_tree_params = {
 'n_estimators': [50, 100, 200, 300],
 'min_samples_leaf' : [5, 20, 50],
 'min_samples_split' : [5, 15, 30]}
ex_tree = ExtraTreesClassifier(random_state=42) 
ex_tree_cv_model = GridSearchCV(ex_tree, ex_tree_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
ex_tree_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(ex_tree_cv_model.best_params_))
# Create the model
gbc = GradientBoostingClassifier(random_state=42)

# Fit the model
gbc_model = gbc.fit(X_train, y_train)

# Predict the model
y_pred = gbc_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

gbc_params = {
        "learning_rate" : [0.001, 0.01, 0.1, 0.05],
        "n_estimators": [100,500,1000],
        "max_depth": [3,5,10],
        "min_samples_split": [2,5,10]}
gbc = GradientBoostingClassifier(random_state=42)
gbc_cv_model = GridSearchCV(gbc, gbc_params, cv = sss, n_jobs = -1, verbose = 2)
gbc_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(gbc_cv_model.best_params_))
# Create the model
xgb = XGBClassifier(random_state=42)

# Fit the model
xgb_model = xgb.fit(X_train, y_train)

# Predict the model
y_pred = xgb_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05]}
xgb = XGBClassifier(random_state=42)
xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = sss, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(xgb_cv_model.best_params_))
# Create the model
xgb = XGBClassifier(random_state=42)

# Fit the model
xgb_model_with_null = xgb.fit(X_train_with_null, y_train_with_null)

# Predict the model
y_pred_with_null = xgb_model_with_null.predict(X_test_with_null)

# Accuracy Score
accuracy_score(y_test_with_null, y_pred_with_null)
# Model Tuning

xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05]}
xgb = XGBClassifier(random_state=42)
xgb_cv_model_with_null = GridSearchCV(xgb, xgb_params, cv = sss, n_jobs = -1, verbose = 2)
xgb_cv_model_with_null.fit(X_train_with_null, y_train_with_null)
print("Best parameters: " + str(xgb_cv_model_with_null.best_params_))
# Create the model
lgbm = LGBMClassifier(random_state=42)

# Fit the model
lgbm_model = lgbm.fit(X_train, y_train)

# Predict the model
y_pred = lgbm_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05],
        "min_child_samples": [5, 10, 20]}
lgbm = LGBMClassifier(random_state=42)
lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(lgbm_cv_model.best_params_))
# Create the model
lgbm = LGBMClassifier(random_state=42)

# Fit the model
lgbm_model = lgbm.fit(X_train_with_null, y_train_with_null)

# Predict the model
y_pred_with_null = lgbm_model.predict(X_test_with_null)

# Accuracy Score
accuracy_score(y_test_with_null, y_pred_with_null)
# Model Tuning

lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05],
        "min_child_samples": [5, 10, 20]}
lgbm = LGBMClassifier(random_state=42)
lgbm_cv_model_with_null = GridSearchCV(lgbm, lgbm_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model_with_null.fit(X_train_with_null, y_train_with_null)
print("Best parameters: " + str(lgbm_cv_model_with_null.best_params_))
# Create the model
catboost = CatBoostClassifier(random_state=42,verbose = False)

# Fit the model
catboost_model = catboost.fit(X_train, y_train)

# Predict the model
y_pred = catboost_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

catboost_params = {
        'depth':[2, 3, 4],
        'loss_function': ['Logloss', 'CrossEntropy'],
        'l2_leaf_reg':np.arange(2,31)}
catboost = CatBoostClassifier(random_state=42,verbose = False)
catboost_cv_model = GridSearchCV(catboost, catboost_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
catboost_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(catboost_cv_model.best_params_))
# Create the model
catboost = CatBoostClassifier(random_state=42,verbose = False)

# Fit the model
catboost_model = catboost.fit(X_train_with_null, y_train_with_null)

# Predict the model
y_pred = catboost_model.predict(X_test_with_null)

# Accuracy Score
accuracy_score(y_test_with_null, y_pred_with_null)
# Model Tuning

catboost_params = {
        'depth':[2, 3, 4],
        'loss_function': ['Logloss', 'CrossEntropy'],
        'l2_leaf_reg':np.arange(2,31)}
catboost = CatBoostClassifier(random_state=42,verbose = False)
catboost_cv_model_with_null = GridSearchCV(catboost, catboost_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
catboost_cv_model_with_null.fit(X_train_with_null, y_train_with_null)
print("Best parameters: " + str(catboost_cv_model_with_null.best_params_))
# Create the model
ada=AdaBoostClassifier(n_estimators=50, random_state=42) 

# Fit the model
ada_model = ada.fit(X_train, y_train)

# Predict the model
y_pred = ada_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)

# Model Tuning

ada_params = {
 'n_estimators': [50, 100, 200, 300],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 }
ada=AdaBoostClassifier(random_state=42)
ada_cv_model = GridSearchCV(ada, ada_params, 
                             cv = sss,
                           n_jobs = -1)
ada_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(ada_cv_model.best_params_))
# Create the model
mlpc = MLPClassifier(random_state=42)

# Fit the model
mlpc_model = mlpc.fit(X_train, y_train)

# Predict the model
y_pred = mlpc_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

mlpc_params = {
        "alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
        "hidden_layer_sizes": [(10,10,10),
                               (100,100,100),
                               (100,100),
                               (3,5), 
                               (5, 3)],             
        "solver" : ["lbfgs","adam","sgd"],"max_iter":[1000]}
mlpc = MLPClassifier(random_state=42)
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
mlpc_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(mlpc_cv_model.best_params_))
# Tuned Logistic Regression Model

param = log_reg_cv_model.best_params_
log_reg = LogisticRegression(**param, random_state=42)
log_reg_tuned = log_reg.fit(X_train, y_train)
y_pred = log_reg_tuned.predict(X_test)
log_reg_final = accuracy_score(y_test, y_pred)
log_reg_final

# Tuned Gaussian Naive Bayes Model

param = nb_cv_model.best_params_
nb = GaussianNB(**param)
nb_tuned = nb.fit(X_train, y_train)
y_pred = nb_tuned.predict(X_test)
nb_final = accuracy_score(y_test, y_pred)
nb_final
# Tuned KNN Model

param = knn_cv_model.best_params_
knn = KNeighborsClassifier(**param)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
knn_final = accuracy_score(y_test, y_pred)
knn_final
# Tuned SVC Linear Model

param = svc_linear_cv_model.best_params_
svc_linear = SVC(**param, kernel = 'linear',random_state=42)
svc_tuned = svc_linear.fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
svc_linear_final = accuracy_score(y_test, y_pred)
svc_linear_final
# Tuned SVC RBF Model

param = svc_rbf_cv_model.best_params_
svc_rbf = SVC(**param, kernel = 'rbf', random_state=42)
svc_tuned = svc_rbf.fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
svc_rbf_final = accuracy_score(y_test, y_pred)
svc_rbf_final
# Tuned CART Model

param = cart_cv_model.best_params_
cart = DecisionTreeClassifier(**param, random_state=42)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
cart_final = accuracy_score(y_test, y_pred)
cart_final
# Tuned Bagging Classifier Model

param = bagging_cv_model.best_params_
bagging = BaggingClassifier(**param, random_state=42) 
bagging_tuned = bagging.fit(X_train, y_train)
y_pred = bagging_tuned.predict(X_test)
bagging_final = accuracy_score(y_test, y_pred)
bagging_final
# Tuned Random Forest Model 

param = rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(**param, random_state=42)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
rf_final = accuracy_score(y_test, y_pred)
rf_final
# Tuned Extra Trees Classifier Model

param = ex_tree_cv_model.best_params_
ex_tree = ExtraTreesClassifier(**param, random_state=42) 
ex_tree_tuned = ex_tree.fit(X_train, y_train)
y_pred = ex_tree_tuned.predict(X_test)
ex_tree_final = accuracy_score(y_test, y_pred)
ex_tree_final
# Tuned GBC Model

param = gbc_cv_model.best_params_
gbc = GradientBoostingClassifier(**param, random_state=42)
gbc_tuned =  gbc.fit(X_train,y_train)
y_pred = gbc_tuned.predict(X_test)
gbc_final = accuracy_score(y_test, y_pred)
gbc_final
# Tuned XGB Model

param = xgb_cv_model.best_params_
xgb = XGBClassifier(**param, random_state=42)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
xgb_final = accuracy_score(y_test, y_pred)
xgb_final
# Tuned XGB - Handling Missing Values Internally

param = xgb_cv_model_with_null.best_params_
xgb_with_null = XGBClassifier(**param, random_state=42)
xgb_tuned =  xgb_with_null.fit(X_train_with_null,y_train_with_null)
y_pred_with_null = xgb_tuned.predict(X_test_with_null)
xgb_final_with_null = accuracy_score(y_test_with_null, y_pred_with_null)
xgb_final_with_null
# Tuned LGBM Model

param = lgbm_cv_model.best_params_
lgbm = LGBMClassifier(**param, random_state=42)
lgbm_tuned = lgbm.fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
lgbm_final = accuracy_score(y_test, y_pred)
lgbm_final
# Tuned LGBM Model - Handling Missing Values Internally

param = lgbm_cv_model_with_null.best_params_
lgbm_with_null = LGBMClassifier(**param, random_state=42)
lgbm_tuned = lgbm_with_null.fit(X_train_with_null, y_train_with_null)
y_pred_with_null = lgbm_tuned.predict(X_test_with_null)
lgbm_final_with_null = accuracy_score(y_test_with_null, y_pred_with_null)
lgbm_final_with_null
# Tuned CatBoost Model

param = catboost_cv_model.best_params_
catboost = CatBoostClassifier(**param, random_state=42)
catboost_tuned = catboost.fit(X_train, y_train)
y_pred = catboost_tuned.predict(X_test)
catboost_final = accuracy_score(y_test, y_pred)
catboost_final
# Tuned CatBoost Model- Handling Missing Values Internally

param = catboost_cv_model_with_null.best_params_
catboost_with_null = CatBoostClassifier(**param, random_state=42)
catboost_tuned = catboost_with_null.fit(X_train_with_null, y_train_with_null)
y_pred = catboost_tuned.predict(X_test_with_null)
catboost_final_with_null = accuracy_score(y_test_with_null, y_pred_with_null)
catboost_final_with_null
# Tuned AdaBoost Model

param = ada_cv_model.best_params_
ada = AdaBoostClassifier(**param,random_state=42)
ada_tuned = ada.fit(X_train, y_train)
y_pred = ada_tuned.predict(X_test)
ada_final = accuracy_score(y_test, y_pred)
ada_final
# Tuned MLPC Model

param = mlpc_cv_model.best_params_
mlpc = MLPClassifier(**param, random_state=42)
mlpc_tuned = mlpc.fit(X_train, y_train)
y_pred = mlpc_tuned.predict(X_test)
mlpc_final = accuracy_score(y_test, y_pred)
mlpc_final
accuracy_scores = {
'log_reg_final': log_reg_final,
'nb_final': nb_final,
'knn_final': knn_final,
'svc_linear_final': svc_linear_final,
'svc_rbf_final': svc_rbf_final,
'cart_final': cart_final,
'bagging': bagging_final,
'ex_tree_final': ex_tree_final,    
'rf_final': rf_final,
'gbc_final': gbc_final,
'xgb_final': xgb_final,
'xgb_final_with_null': xgb_final_with_null,
'lgbm_final': lgbm_final,
'lgbm_final_with_null': lgbm_final_with_null,
'catboost_final': catboost_final,
'catboost_final_with_null': catboost_final_with_null,
'ada_final': ada_final,
'mlpc_final': mlpc_final  
}

accuracy_scores = pd.Series(accuracy_scores).to_frame('Accuracy_Score')
accuracy_scores = accuracy_scores.sort_values(by='Accuracy_Score', ascending=False)
accuracy_scores['rank'] = (accuracy_scores.reset_index().index +1)
accuracy_scores
# Ensembing first 5 models
model_1 = [('Adaboost', ada),
         ('LGBM', lgbm),
         ('XGB', xgb),
         ('KNN', knn),  
         ('Random Forest', rf),
         ('GBC', gbc),
          ]
voting_reg = VotingClassifier(model_1, voting='soft')
voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)
print(f"Voting Classifier's accuracy: {accuracy_score(y_pred, y_test):.4f}")
# Ensembing with null models
model_2 = [('XGB', xgb_with_null),
         ('LGBM', lgbm_with_null),
          ('catboost', catboost_with_null)
         ]
voting_reg = VotingClassifier(model_2, voting='soft')
voting_reg.fit(X_train_with_null, y_train_with_null)
y_pred_with_null = voting_reg.predict(X_test_with_null)
print(f"Voting Classifier's accuracy: {accuracy_score(y_pred_with_null, y_test_with_null):.4f}")