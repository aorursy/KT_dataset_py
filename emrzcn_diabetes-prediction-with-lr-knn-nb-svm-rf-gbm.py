# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid")       

import pandas_profiling as pp 



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/diabetes/diabetes.csv")
df.head()
df.info()
df.isnull().sum()
df.describe().T
import pandas_profiling as pp 

profile_df = pp.ProfileReport(df)
profile_df
df.hist(figsize=(10, 10), bins=50, xlabelsize=5, ylabelsize=5);
sns.catplot(x="Outcome",data=df, kind="count");
df.plot(kind="density", layout=(6,5),subplots=True,sharex=False, sharey=False, figsize=(15,15));

plt.tight_layout() 
sns.pairplot(df, kind = "reg")
df_corr = df.corr()
sns.heatmap(df_corr, linewidths = 1);
sns.pairplot(df_corr, kind = "reg");
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve
X = df.drop(["Outcome"], axis = 1)

y = df["Outcome"]



#or 

#X = df[:,0:8]

#y = df[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.30, 

                                                    random_state = 42)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(solver = "liblinear")

log_model = log.fit(X_train,y_train)

log_model
y_pred = log_model.predict(X_test)


confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

accuracy_score(y_test, log_model.predict(X_test))

cross_val_score(log_model, X_test, y_test, cv = 10).mean()
logit_roc_auc = roc_auc_score(y_test, log_model.predict(X_test))



fpr, tpr, thresholds = roc_curve(y_test, log_model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive ')

plt.ylabel('True Positive ')

plt.title('ROC')

plt.show()
from sklearn.naive_bayes import GaussianNB





nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model

y_pred = nb_model.predict(X_test)

accuracy_score(y_test, y_pred)

cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

knn_model



y_pred = knn_model.predict(X_test)

accuracy_score(y_test, y_pred)



knn_params = {"n_neighbors": np.arange(1,20)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv=10)

knn_cv.fit(X_train, y_train)
print("Best KNN score:" + str(knn_cv.best_score_))

print("Best KNN parameter: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(1)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
d = {'Accuracy in KNN before GridSearchCV ': [0.77], 'Accuracy in KNN After GridSearchCV': [0.95]}

knn_data = pd.DataFrame(data=d)

knn_data
from sklearn.svm import SVC





svm_model = SVC(kernel = "linear").fit(X_train, y_train)



y_pred = svm_model.predict(X_test)

accuracy_score(y_test, y_pred)



svc_params = {"C": np.arange(1,10)}



svc = SVC(kernel = "linear")



svc_cv_model = GridSearchCV(svc,svc_params, 

                            cv = 10, 

                            n_jobs = -1, 

                            verbose = 2 )

svc_cv_model.fit(X_train, y_train)

print("Best Params: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(kernel = "linear", C = 2).fit(X_train, y_train)



y_pred = svc_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
d = {'Accuracy in SVM before GridSearchCV ': [0.7983], 'Accuracy in SVM After GridSearchCV': [0.7933]}

svm_data = pd.DataFrame(data=d)

svm_data
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}



rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 



rf_cv_model.fit(X_train, y_train)
print("Best Params: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 8, 

                                  max_features = 8, 

                                  min_samples_split = 2,

                                  n_estimators = 1000)
rf_tuned.fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r");



d = {'Accuracy in RF before GridSearchCV ': [0.97], 'Accuracy in RF After GridSearchCV': [0.92]}

rf_data = pd.DataFrame(data=d)

rf_data
from sklearn.ensemble import GradientBoostingClassifier



gbm_model = GradientBoostingClassifier().fit(X_train, y_train)



y_pred = gbm_model.predict(X_test)

accuracy_score(y_test, y_pred)
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}



gbm = GradientBoostingClassifier()



gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)

gbm_cv.fit(X_train, y_train)
print("Best Params: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.1, 

                                 max_depth = 10,

                                min_samples_split = 2,

                                n_estimators = 100)



gbm_tuned =  gbm.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r");
d = {'Accuracy in GBM before GridSearchCV ': [0.87], 'Accuracy in GBM After GridSearchCV': [0.95]}

gbm_data = pd.DataFrame(data=d)

gbm_data
models = [

    knn_tuned,

    log_model,

    svc_tuned,

    nb_model,

    rf_tuned,

    gbm_tuned,

    

]





for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("-"*28)

    print(name + ":" )

    print("Accuracy: {:.4%}".format(accuracy))
result = []



results = pd.DataFrame(columns= ["Models","Accuracy"])



for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)    

    result = pd.DataFrame([[name, accuracy*100]], columns= ["Models","Accuracy"])

    results = results.append(result)

    

    

sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")

plt.xlabel('Accuracy %')

plt.title('accuracy rate of models'); 
sns.catplot(x="Outcome",data=df, kind="count");
df["Outcome"].value_counts()
df["Insulin"].value_counts() 
df["BMI"].value_counts()