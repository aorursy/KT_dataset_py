import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from collections import Counter
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
df.head(10)
# There are so many "?" in some features. We will change then with nan. And we will drop these features.
df = df.replace(['?'],[np.nan]) 
df = df.replace(['<'],[np.nan])
df.head()
print(df.info())
# We have a lot of object type features. We convert these features to float.
list = df.columns
for i in list:
    df[i] = df[i].astype(float)
# We will drop these columns because there are only 71 non-null values. We will handle the other null values in part 6.
df.drop(["STDs: Time since first diagnosis","STDs: Time since last diagnosis"], axis = 1, inplace = True)
# These two columns have all 0. For that we drop them too.
df.drop(["STDs:AIDS","STDs:cervical condylomatosis"],axis=1,inplace=True)
df.describe()
df.columns
def bar_plot(variable):
    # get a feature
    var = df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n {}".format(variable,varValue))
category = ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis", \
"STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", \
"STDs:genital herpes", "STDs:molluscum contagiosum",  "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis", \
"Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy"]
for c in category:
    bar_plot(c)
def plot_hist(variable):
    plt.figure(figsize = (15,6))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numeric = ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)", \
           "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs (number)", "STDs: Number of diagnosis"]
for c in numeric:
    plot_hist(c)
#We can also make a simple EDA with "pandasprofiling".
df_profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
df_profile
#from sklearn.cluster import DBSCAN
#outlier_detection = DBSCAN(min_samples = 2, eps = 3)
#clusters = outlier_detection.fit_predict(df)
#list(clusters).count(-1)
def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    
    #multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    #return multiple_outliers
    
    return outlier_indices
df.loc[detect_outlier(df, ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)",
                    "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs (number)",
                    "STDs: Number of diagnosis"])]
df = df.drop(detect_outlier(df, ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
                                 "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)",
                                 "STDs (number)", "STDs: Number of diagnosis"]), axis = 0).reset_index(drop =True)
df_len = len(df)
df.head()
df.columns[df.isnull().any()]
df.isnull().sum()
liste =  ["STDs", "STDs (number)", "STDs:condylomatosis", "STDs:vaginal condylomatosis", \
          "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", \
          "STDs:molluscum contagiosum",  "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV"]
for m in liste:
    df[m] = df[m].fillna(0.0)
df.isnull().sum()
df = df.dropna(subset=['Smokes', 'Hormonal Contraceptives', 'IUD'])
df.isnull().sum()
df["Number of sexual partners"] = df["Number of sexual partners"].fillna(np.mean(df["Number of sexual partners"]))
df["First sexual intercourse"] = df["First sexual intercourse"].fillna(np.mean(df["First sexual intercourse"]))
df["Num of pregnancies"] = df["Num of pregnancies"].fillna(np.mean(df["Num of pregnancies"]))
df.isnull().sum()
df.shape
df.Biopsy.value_counts()
# We have 4 target labels. We will work on "Biopsy" target label
y = df.Biopsy.values
X = df.drop(["Biopsy", "Hinselmann", "Schiller", "Citology"], axis=1)
from imblearn.over_sampling import SMOTE

sm  = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X),columns = X.columns)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
acc_logreg_train = round(logreg.score(x_train, y_train)*100,2)
acc_logreg_test = round(logreg.score(x_test, y_test)*100,2)
print("Training Accuracy: % {}".format(acc_logreg_train))
print("Testing Accuracy: % {}".format(acc_logreg_test))
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
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
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("rfc",best_estimators[2]),
                                        ("knn",best_estimators[4])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))
y_pred = votingC.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
