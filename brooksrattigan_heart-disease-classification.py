# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns; sns.set()

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# For statistics, preprocessing and ML
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/heart-disease-cleveland-uci/heart_cleveland_upload.csv")
df = data.copy()
display(df.head())
display(df.tail())
df.info()
df.isnull().sum()
data.describe().T
df.nunique()
df.corr()
df1 = df.drop(['age','trestbps','chol','thalach','oldpeak'],axis=1)

for i, col in enumerate(df1.columns):
    plt.figure(i)
    plt.title(col, color = 'blue',fontsize=15)
    sns.countplot(x=col, data=df1)
sns.catplot(x="sex", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Sex and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="cp", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Chest Pain Type and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="fbs", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Fasting Blood Sugar and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="restecg", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Resting Electrocardiographic Results and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="exang", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Exercise Induced Angina and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="slope", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Slope of the Peak Exercise ST Segment and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="ca", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Number of Major Vessels Colored by Flourosopy and Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.catplot(x="thal", hue="condition", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
plt.title('Thal and Heart Condition', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=df.condition, y=df.age);
plt.xticks(rotation= 0)
plt.xlabel('condition', fontsize=14)
plt.ylabel('age', fontsize=14)
plt.title('Average Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=df.condition, y=df.trestbps);
plt.xticks(rotation= 0)
plt.xlabel('condition', fontsize=14)
plt.ylabel('trestbps', fontsize=14)
plt.title('Average Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=df.condition, y=df.chol);
plt.xticks(rotation= 0)
plt.xlabel('condition', fontsize=14)
plt.ylabel('chol', fontsize=14)
plt.title('Average Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=df.condition, y=df.thalach);
plt.xticks(rotation= 0)
plt.xlabel('condition', fontsize=14)
plt.ylabel('thalach', fontsize=14)
plt.title('Average Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=df.condition, y=df.oldpeak);
plt.xticks(rotation= 0)
plt.xlabel('condition', fontsize=14)
plt.ylabel('oldpeak', fontsize=14)
plt.title('Average ST depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
sns.boxplot(x="condition", y="age", data=df, palette="PRGn")
plt.title('Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="condition", y="trestbps", data=df, palette="PRGn")
plt.title('Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="condition", y="chol", data=df, palette="PRGn")
plt.title('Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="condition", y="thalach", data=df, palette="PRGn")
plt.title('Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="condition", y="oldpeak", data=df, palette="PRGn")
plt.title('ST Depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="condition", y="age", hue='exang' ,data=df)
plt.title('Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="trestbps",hue='exang' ,data=df)
plt.title('Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="chol",hue='exang' ,data=df)
plt.title('Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="thalach",hue='exang' ,data=df)
plt.title('Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="oldpeak",hue='exang' ,data=df)
plt.title('ST depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="condition", y="age", hue='slope' ,data=df)
plt.title('Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="trestbps",hue='slope' ,data=df)
plt.title('Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="chol",hue='slope' ,data=df)
plt.title('Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="thalach",hue='slope' ,data=df)
plt.title('Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="oldpeak",hue='slope' ,data=df)
plt.title('ST depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="condition", y="age", hue='ca' ,data=df)
plt.title('Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="trestbps",hue='ca' ,data=df)
plt.title('Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="chol",hue='ca' ,data=df)
plt.title('Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="thalach",hue='ca' ,data=df)
plt.title('Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="oldpeak",hue='ca' ,data=df)
plt.title('ST depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="condition", y="age", hue='thal' ,data=df)
plt.title('Age by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="trestbps",hue='thal' ,data=df)
plt.title('Resting Blood Pressure by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="chol",hue='thal' ,data=df)
plt.title('Cholesterol by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="thalach",hue='thal' ,data=df)
plt.title('Maximum Heart Rate Achieved by Heart Condition', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="condition", y="oldpeak",hue='thal' ,data=df)
plt.title('ST depression by Heart Condition', color = 'blue', fontsize=15)
plt.show()
f,ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, linecolor="red", fmt= '.2f',ax=ax)
plt.show()
sns.pairplot(df, hue='condition', vars=['age','trestbps','chol','thalach','oldpeak'],kind='reg')
plt.show()
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['restecg'], prefix = "restecg")
c = pd.get_dummies(df['slope'], prefix = "slope")
d = pd.get_dummies(df['thal'], prefix = "thal")
frames = [df, a, b, c, d]
df = pd.concat(frames, axis = 1)
df = df.drop(columns = ['cp','restecg','slope','thal'])
df.head()
from sklearn.preprocessing import MinMaxScaler
X = df.drop(["condition"],axis = 1)
y = df.condition
scaler = MinMaxScaler().fit(X)

X_scaled = scaler.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.3, random_state = 42)
log_reg = LogisticRegression().fit(X_train,y_train)
log_reg
log_reg.intercept_
log_reg.coef_
y_pred = log_reg.predict(X_test)
confusion_matrix(y_test, y_pred)
log_reg.predict_proba(X_test)[0:10]
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
logit_roc_auc = roc_auc_score(y, log_reg.predict(X_scaled))

fpr, tpr, thresholds = roc_curve(y, log_reg.predict_proba(X_scaled)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.title('ROC')
plt.show()
log_reg_final = cross_val_score(log_reg, X_test, y_test, cv = 10).mean()
log_reg_final
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
nb_model.predict(X_test)[0:10]
nb_model.predict_proba(X_test)[0:10]
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
nb_final = cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
nb_final
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
print("Best score:" + str(knn_cv.best_score_))
print("Best parameters: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(n_neighbors = 5)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
knn_final = accuracy_score(y_test, y_pred)
knn_final
svm_model = SVC(kernel = "linear").fit(X_train, y_train)
svm_model
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)
svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}

svc = SVC(kernel = "linear")

svc_cv_model = GridSearchCV(svc,svc_params, 
                            cv = 10, 
                            n_jobs = -1, 
                            verbose = 2 )

svc_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(kernel = "linear", C = 1).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
svc_linear_final = accuracy_score(y_test, y_pred)
svc_linear_final
svc_model = SVC(kernel = "rbf").fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
accuracy_score(y_test, y_pred)
svc_params = {"C": [0.00001, 0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}
svc = SVC(kernel = "rbf")
svc_cv_model = GridSearchCV(svc, svc_params, 
                         cv = 10, 
                         n_jobs = -1,
                         verbose = 2)

svc_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(kernel = "rbf", C = 5, gamma = 0.1).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
svc_rbf_final = accuracy_score(y_test, y_pred)
svc_rbf_final
svc_model = SVC(kernel = "poly").fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
accuracy_score(y_test, y_pred)
svc_params = {"C": [0.00001, 0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}
svc = SVC(kernel = "poly")
svc_cv_model = GridSearchCV(svc, svc_params, 
                         cv = 10, 
                         n_jobs = -1,
                         verbose = 2)

svc_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(kernel = "poly", C = 0.001, gamma = 1).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
svc_poly_final = accuracy_score(y_test, y_pred)
svc_poly_final
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
cart = tree.DecisionTreeClassifier(max_depth = 3, min_samples_split = 18)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
cart_final = accuracy_score(y_test, y_pred)
cart_final
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,3,5,7],
            "n_estimators": [10,100,200,500,1000],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 5, 
                                  max_features = 2, 
                                  min_samples_split = 2,
                                  n_estimators = 200)

rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
rf_final = accuracy_score(y_test, y_pred)
rf_final
pd.DataFrame(X_train).head()
X_train_pd = pd.DataFrame(X_train)
df_x = df.drop(['condition'], axis=1)
X_train_pd.columns = df_x.columns[:22]
X_train_pd.head()
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train_pd.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Independent Variables");
gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
gbm_params = {"learning_rate" : [0.001, 0.01, 0.05, 0.1],
             "n_estimators": [100,500,1000],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()

gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(X_train, y_train)
print("Best parameters: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.05, 
                                 max_depth = 3,
                                min_samples_split = 5,
                                n_estimators = 1000)
gbm_tuned =  gbm.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
gbm_final = accuracy_score(y_test, y_pred)
gbm_final
xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)
xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05]}
xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)
print("Best parameters: " + str(xgb_cv_model.best_params_))
xgb = XGBClassifier(learning_rate = 0.1, 
                    max_depth = 3,
                    n_estimators = 100,
                    subsample = 1.0)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
xgb_final = accuracy_score(y_test, y_pred)
xgb_final
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}
lgbm = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.01, 
                       max_depth = 3,
                       min_child_samples = 10,
                       n_estimators = 500,
                       subsample = 0.6,
                       )
lgbm_tuned = lgbm.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
lgbm_final = accuracy_score(y_test, y_pred)
lgbm_final
cat_model = CatBoostClassifier().fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
accuracy_score(y_test, y_pred)
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.05, 0.1],
    'depth': [3,5,8] }
catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb = CatBoostClassifier(iterations = 200, 
                          learning_rate = 0.01, 
                          depth = 3)

catb_tuned = catb.fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
y_pred = catb_tuned.predict(X_test)
catb_final = accuracy_score(y_test, y_pred)
catb_final
models = {
'log_reg_final': log_reg_final,
'nb_final': nb_final,
'knn_final': knn_final,
'svc_linear_final': svc_linear_final,
'svc_rbf_final': svc_rbf_final,
'svc_poly_final': svc_poly_final,
'cart_final': cart_final,
'rf_final': rf_final,
'gbm_final': gbm_final,
'xgb_final': xgb_final,
'lgbm_final': lgbm_final,
'catb_final': catb_final
}

for model,score in models.items():
    print("-"*28)
    print(model + ":" )
    print("Accuracy: {:.4%}".format(score))
indexes = ["Log","NB","KNN","SVC_Lin","SVC_Rbf", "SVC_Poly", "CART", "RF", "GBM", "XGB", "LGBM", "CATB"]
scores = [
     log_reg_final,
nb_final,
knn_final,
svc_linear_final,
svc_rbf_final,
svc_poly_final,
cart_final,
    rf_final,
    gbm_final,
xgb_final,
lgbm_final,
    catb_final]

plt.figure(figsize=(12,8))
sns.barplot(x=indexes,y=scores)
plt.xticks()
plt.title('Model Comparision',color = 'orange',fontsize=20);