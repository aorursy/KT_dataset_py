import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.tail()
print("Number of rows:{} \nNumber of columns:{}".format(df.shape[0], df.shape[1]))
df.describe()
# I like somthing like that, so I have a rough insight about the variation in information, type of data (continuous, numerical, categorical), ect.
for i in df:
    print(i)
    print(df[i].unique())
    print("#"*40)
# you can do similar automated things with codes like,
    #from sklearn.feature_selection import VarianceThreshold
    #VarianceThreshold(0.1).fit_transform(X)

df.drop(columns = {"Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"}, inplace = True)
print("Number of variables after VarianceThreshold:", df.shape[1])
# no columns with high ration of missing values so no column to drop or to fill in (:
df.isnull().sum()
# seperate categorical from numerical features

numerical = df.dtypes[df.dtypes != "object"].index
print("numerical features:\n", numerical)
print("+"*80)
categorical = df.dtypes[df.dtypes== "object"].index
print("categorical features:\n", categorical)
#we have a highly imbalanced dependant variable and skewness will be high as well

df["Attrition"].value_counts().plot(kind = "bar", x = df["Attrition"])

# simple Visulazation of the categorical features
Cat = categorical.tolist()
type(Cat)
print(Cat)

for x in Cat:
    if x != "Attrition":
        df[categorical].groupby(["Attrition", x])[x].count().unstack().plot(kind = "bar", stacked = True) 
# The visualization of the numerical features is not as easy as I first thought.
# Some features are in reality true categorical features with no intervall or true zero (Education etc.) which makes it harder to interpret
# Other numerical features like "YearsSinceLastPromotion" are not as conclusive as I would have suspected
# Younger people seem to suffer more under attrition.
# People with greater "DistanceFromHome" is a candidate to influence attrition.
# A higher Attrition-rate seems to be connected to lower "JobLevel" and lower "JobSatisfaction" and lower "MonthlyIncome" and so on.
# The lack on possible StockOptions and a lower "MonthlyIncome" seem to be good indicators, which is quite reasonable.

import seaborn as sns
import matplotlib.pyplot as plt

Num = numerical.tolist()

for j in Num:
    fig, ax = plt.subplots()
    ax = sns.boxplot(y = df[j], x = df["Attrition"], data = df)
    median = df.groupby(['Attrition'])[j].median().values
    median_labels = [str(np.round(s, 2)) for s in median]
    pos = range(len(median))
    for i in pos:
        ax.text(pos[i-1], median[i], median_labels[i], 
        horizontalalignment='center', size='x-small', color='w', weight='semibold')
Education_dict = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}

EnvironmentSatisfaction_dict = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}

JobInvolvement_dict = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}

JobSatisfaction_dict = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}

PerformanceRating_dict = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

RelationshipSatisfaction_dict = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}

WorkLifeBalance_dict = {1:'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}

df["Education"] = df["Education"].map(Education_dict)
df["EnvironmentSatisfaction"] = df["EnvironmentSatisfaction"].map(EnvironmentSatisfaction_dict)
df["JobInvolvement"] = df["JobInvolvement"].map(JobInvolvement_dict)
df["JobSatisfaction"] = df["JobSatisfaction"].map(JobSatisfaction_dict)
df["PerformanceRating"] = df["PerformanceRating"].map(PerformanceRating_dict)
df["RelationshipSatisfaction"] = df["RelationshipSatisfaction"].map(RelationshipSatisfaction_dict)
df["WorkLifeBalance"] = df["WorkLifeBalance"].map(WorkLifeBalance_dict)
df = pd.get_dummies(data = df, columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'OverTime', "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "WorkLifeBalance"], drop_first = True)
df.head()

# creation of the artificial dataset and heatmap

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = "minority", random_state = 10)
X_art, y_art = sm.fit_sample(df.drop(columns = {"Attrition_Yes"}), df["Attrition_Yes"])
artificial_df = pd.concat([pd.DataFrame(X_art), pd.DataFrame(y_art)], axis = 1)
artificial_df.columns = [df.columns]


fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(artificial_df.corr())
ax.set_title("Correlation after SMOTE")
plt.show()
corr = artificial_df.corr()

print("before Multi-check:", corr.shape)
for vars in corr:
    mask = (corr[[vars]] > 0.7) & (corr[[vars]] < 1) | (corr[[vars]] < -0.7) 
    corr[mask] = np.nan
corr.dropna(inplace = True)
print("after Multi-check:", corr.shape)
# 9x rows got eliminated

# dont know why df[corr.index] does not work? any explanation is welcome!
# anyway, now we got a dataframe ready!
df = df[corr.index.get_level_values(0)]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, recall_score)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


X = df.drop(columns = {"Attrition_Yes"})
y = df["Attrition_Yes"]

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state = 7, stratify = y, test_size = 0.15)

X_trainval_over, y_trainval_over = sm.fit_sample(X_trainval, y_trainval)

print("Size of trainval:{}\nSize of test:{}".format(X_trainval_over.shape, X_test.shape))

all_scores = []
learning_rate = [0.025, 0.05, 0.1, 0.15, 0.2]
for lr in learning_rate:
    GBM = GradientBoostingClassifier(random_state = 10, learning_rate = lr, max_features = "sqrt")
    score = cross_val_score(GBM, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(score)
fig = plt.figure()

plt.scatter(x = learning_rate, y = all_scores)
plt.xlabel("learning_rate")
plt.ylabel("Recall-Score")
plt.title("GBM with different learning rate")
all_scores = []
n_estimators = [100, 200, 300, 400, 500]
for est in n_estimators:
    GBM = GradientBoostingClassifier(random_state = 10, n_estimators = est, max_features = "sqrt")
    score = cross_val_score(GBM, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(score)
fig = plt.figure()

plt.scatter(x = n_estimators, y = all_scores)
plt.xlabel("n_estimators")
plt.ylabel("Recall-Score")
plt.title("GBM with different numbers of estimators")
learning_rate = [0.2, 0.15, 0.1, 0.5, 0.025]
n_estimators = [100, 200, 300, 400, 500]
best_score = 0
for est in n_estimators:
    for lr in learning_rate:
        GBM = GradientBoostingClassifier(random_state = 10, n_estimators = est, learning_rate = lr, max_features = "sqrt")
        score = cross_val_score(GBM, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
        if score > best_score:
            best_score = score
            best_parameters = {"n_estimator": est, "learning_rate": lr}
print("best recall:{} with best parameters:{}".format(best_score,best_parameters))
all_scores = []
estimators = range(400,900,100)
for est in estimators:
    RF = RandomForestClassifier(random_state = 10, n_jobs = -1, n_estimators = est)
    score = cross_val_score(RF, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(score)
fig = plt.figure()

plt.scatter(x = estimators, y = all_scores)
plt.xlabel("estimators")
plt.ylabel("Recall-Score")
plt.title("RandomForests with different numbers of estimators")
all_scores = []
max_depth = range(1,10,1)
for depth in max_depth:
    RF = RandomForestClassifier(random_state = 10, n_jobs = -1, max_depth = depth)
    scores = cross_val_score(RF, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(scores)
fig = plt.figure()
plt.scatter (x = max_depth, y = all_scores)
plt.xlabel("max_depth")
plt.ylabel("Recall-Score")
plt.title("RandomForests with different tree depth")
max_depth = range(1,10,1)
estimators = range(400,900,100)
best_score = 0
for depth in max_depth:
    for est in estimators:
        RF = RandomForestClassifier(random_state = 10, max_depth = depth, n_estimators = est, n_jobs = -1)
        scores = cross_val_score(RF, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
        if scores > best_score:
            best_score = scores
            best_parameters = {"depth": depth, "estimators": est}
print("best recall:{} with best parameters:{}".format(best_score,best_parameters))
#the liblinear-solver (defualt solver for LR), can handle both MAE and MSE penalty

penalty = ["l1", "l2"]
all_score = []
for p in penalty:
    LR = LogisticRegression(penalty = p)
    scores = cross_val_score(LR, X_trainval_over, y_trainval_over,cv = 5, scoring = "recall").mean()
    all_score.append(scores)
fig = plt.figure()
plt.scatter(x = penalty, y = all_score)
plt.xlabel("penalty")
plt.ylabel("Recall-Score")
plt.title("LogisticRegression with MAE and MSE as penalty")
C_value = [0.001, 0.01, 0.1, 1, 10, 100]                       
all_score = []
for C in C_value:
    LR = LogisticRegression(C = C, n_jobs = -1)
    scores = cross_val_score(LR, X_trainval_over, y_trainval_over,cv = 5, scoring = "recall").mean()
    all_score.append(scores)
fig = plt.figure()
plt.scatter (x = C_value, y = all_score)
plt.xlabel("C")
plt.ylabel("Recall-Score")
plt.title("LogisticRegression with different regularization Values for C")
C_value = [0.001, 0.01, 0.1, 1, 10, 100] 
penalty = ["l1", "l2"]
best_score = 0

for C in C_value:
    for p in penalty:
        LR = LogisticRegression(C = C, penalty = p, n_jobs = -1)
        scores = cross_val_score(LR, X_trainval_over, y_trainval_over, cv = 5, scoring = "recall").mean()
        if scores > best_score:
            best_score = scores
            best_parameters = {"p": p, "C": C}
print("best recall_score:\n{}".format(best_score))
print("with given parameters:\n{}".format(best_parameters))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_trainval_over)
X_train_scaled = scaler.transform(X_trainval_over)
SVM = SVC(random_state = 10, kernel = "rbf")
score = cross_val_score(SVM, X_train_scaled, y_trainval_over, cv = 5, scoring = "recall").mean()
print("BaseLineScore:", score)
# low values for C means that values far away from a imaginary decision boundry will be included in the calculation
# vice versa, high values for C ignore these far away values and focus only on values near the imaginary decision boundry
all_scores = []
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
for C in C_values:
    SVM = SVC(random_state = 10, C = C, kernel = "rbf")
    score = cross_val_score(SVM, X_train_scaled, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(score)
fig = plt.figure()
plt.scatter(x = C_values, y = all_scores)
plt.xlabel("Parameter C")
plt.ylabel("Recall_score")
plt.title("SVM with different C values")
all_scores = []
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
for g in gamma:
    SVM = SVC(random_state = 10, gamma = g, kernel = "rbf")
    score = cross_val_score(SVM, X_train_scaled, y_trainval_over, cv = 5, scoring = "recall").mean()
    all_scores.append(score)
fig = plt.figure()
plt.scatter(x = gamma, y = all_scores)
plt.xlabel("Parameter Gamma")
plt.ylabel("Recall_score")
plt.title("SVM with different C values")
C_Values = [0.001, 0.01, 0.1, 1, 10, 100]
Gamma = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
for C in C_Values:
    for g in Gamma:
        SVM = SVC(random_state = 10, C = C, gamma=g, kernel = "rbf")
        score = cross_val_score(SVM, X_train_scaled, y_trainval_over, cv = 5, scoring = "recall").mean()
        all_scores.append(score)
        if score > best_score:
            best_score = score
            best_parameters = {"Gamma": g, "C": C}
print("best recall:{} with best parameters:{}".format(best_score,best_parameters))
GBM = GradientBoostingClassifier(random_state = 10, n_estimators = 500, learning_rate = 0.2, max_features = "sqrt")
GBM.fit(X_trainval_over, y_trainval_over)
GBM_predict = GBM.predict(X_test)
print(round(recall_score(y_test, GBM_predict, average = "micro"),2))
print(classification_report(y_test, GBM_predict))
feature_importance_GBM = pd.DataFrame(dict(Column = np.array(X.columns), Importance = GBM.feature_importances_)).sort_values(by = "Importance", ascending = False)
feature_importance_GBM
fig = plt.figure(figsize = (14,4))
plt.bar(x = feature_importance_GBM.iloc[:10, 0], height = feature_importance_GBM.iloc[:10, 1])
plt.xticks(rotation = 75)
RF = RandomForestClassifier(random_state = 10, n_jobs = -1, n_estimators = 400, max_depth = 9)
RF.fit(X_trainval_over, y_trainval_over)
RF_predict = RF.predict(X_test)
print(round(recall_score(y_test, RF_predict, average = "micro"), 2))
print(classification_report(y_test, RF_predict))
#print(RF_predict)
feature_importance_RF = pd.DataFrame(dict(Column = np.array(X.columns), Importance = RF.feature_importances_)).sort_values(by = "Importance", ascending = False)
feature_importance_RF
fig = plt.figure(figsize = (14,4))
plt.bar(x = feature_importance_RF.iloc[:10,0] , height = feature_importance_RF.iloc[:10,1])
plt.xticks(rotation = 75)
LR = LogisticRegression(C = 0.1, penalty = "l2", n_jobs = -1)
LR.fit(X_trainval_over, y_trainval_over)
LR_predict = LR.predict(X_test)
print(round(recall_score(y_test, LR_predict, average = "micro"), 2))
print(classification_report(y_test, LR_predict))
#print(LR_predict)
# nice hack for first n rows and last n rows in a graph
feature_importance_LR = pd.DataFrame(dict(Column = np.hstack(np.array([X.columns])), Importance = np.hstack(LR.coef_))).sort_values(by = "Importance", ascending = False)
feature_importance_LR
fig = plt.figure(figsize = (14,4))
plt.bar(x = pd.concat([feature_importance_LR.iloc[:5,0],feature_importance_LR.iloc[-5:,0]]), height = pd.concat([feature_importance_LR.iloc[:5,1],feature_importance_LR.iloc[-5:,1]]))
plt.xticks(rotation = 75)
SVM = SVC(random_state = 10,gamma = 0.001, C = 0.001, kernel = "rbf")
SVM.fit(X_trainval_over, y_trainval_over)
scaler.fit(X_trainval_over)
X_test_scaled = scaler.transform(X_test)
SVM_predict = SVM.predict(X_test_scaled)
#print(SVM_predict)
clean_score = round(recall_score(y_test, SVM_predict, average = "micro"), 5)
print("Recall without Shuffling:",clean_score)
print(classification_report(y_test, SVM_predict))

all_scores = []
for i in range(X_test.shape[1]):
    X_test_noisy = X_test_scaled.copy()
    np.random.shuffle(X_test_noisy[:, i])                 
    noisy_predict = SVM.predict(X_test_noisy)
    noisy_score = round(recall_score(y_test, noisy_predict, average = "micro"), 5)
    feature_imp = clean_score-noisy_score
    all_scores.append(feature_imp)
#print(all_scores)
feature_importance_SVM = pd.DataFrame(dict(Column = np.array(X.columns), Importance = np.array(all_scores))).sort_values(by = "Importance", ascending = False)
fig = plt.figure(figsize=(16,4))
plt.bar(x = feature_importance_SVM.iloc[:11, 0], height = feature_importance_SVM.iloc[:11, 1])
plt.title("Feature Importance for SVM with rfb Kernel")
plt.xticks(rotation = 75)
