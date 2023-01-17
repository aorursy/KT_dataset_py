import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
train=pd.read_csv('../input/challengearo/TrainData.csv', encoding='ISO-8859-1', sep=';')
test=pd.read_csv('../input/challengearo/TestData.csv', encoding='ISO-8859-1', sep=';')
train.info()
train.head(10)
train.describe()
test.info()
test.head()
"""There are missing values in Tage seit letzter Kampagne column"""
"""Target value shoud be convert to int value"""
train['Zielvariable'] = train['Zielvariable'].str.lower().map({'ja' : 1, 'nein' : 0})
train=pd.get_dummies(train)
"""There are several catrgorical columnd that need to be converted to carry out classification and prediction"""
test=test.drop(["Zielvariable"], axis=1)
test=pd.get_dummies(test)
"""Split train data to X and y"""
X=train.drop("Zielvariable", axis=1)
y=train["Zielvariable"]
"""Drop the column mith massive missing data"""
X=X.drop(["Tage seit letzter Kampagne"],axis=1)
"""As can be seen fron describtion target value is imballansed"""
"""Moreover, values in different columns are in very various range, this issue will be fixed by scale in different classifier models"""
smt = SMOTETomek(ratio="auto")
X,y=smt.fit_sample(X,y)
"""first handling missing data and then scale the train data and dimention reduction """
steps=[("imputation", Imputer(missing_values="NaN", strategy="mean", axis=0)),
       ("scale", StandardScaler()),
       ('pca', PCA()),
       ("log_reg", LogisticRegression())]

pipeline=Pipeline(steps)
"""Create parameters space for tuning"""
parameters={"log_reg__C":np.logspace(-10, 10, 50)}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)
"""Hyperparameters tuning"""
"""Use cross-validation technique for better use of available data"""
log_reg_cv=RandomizedSearchCV(pipeline, parameters)
log_reg_cv.fit(X_train,y_train)
"""Harmonized test DataFrame with train DataFrame"""
test=test.drop(["Tage seit letzter Kampagne"],axis=1)
y_predict=log_reg_cv.predict_proba(X_test)
y_predict=y_predict[:,1]
y_predict_new=log_reg_cv.predict_proba(test)
y_predict_new=y_predict_new[:,1]
"""Measure accuracy of LogisticRegression classifier by AUC"""
auc_score=roc_auc_score(y_test,y_predict)
print("AUC for the LogisticRegression classifier is: ")
print(auc_score)
print("Tuned LogisticRegression: {}".format(log_reg_cv.best_params_))
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label='LogisticRegression, AUC = %0.4f'% auc_score)
plt.legend(loc='lower right')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])

plt.title('Receiver Operating for logistic regression classifier')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""Prediction by logisticRegression classifier"""
Lsung_logisticRegression=pd.DataFrame({"ID":test.loc[:,"Stammnummer"], "Expected":y_predict_new})
Lsung_logisticRegression.head()
"""Save the generated prediction by logisitc regression in Lsun_logisticRegresion.csv file"""
Lsung_logisticRegression.to_csv("Lsung_logsticRegression.csv", index=False)
steps=[("imputation", Imputer(missing_values="NaN", strategy="mean", axis=0)),
       ("scale", StandardScaler()),
       ("ran_forest", RandomForestClassifier())]

pipeline2=Pipeline(steps)
parameters2={"ran_forest__criterion":["gini", "entropy"],
           "ran_forest__class_weight":["balanced", None],
           "ran_forest__max_depth":[10,30,50,70,100, None],
           "ran_forest__max_features":["auto", "sqrt", None],
           "ran_forest__n_jobs":[-1, None],
           "ran_forest__n_estimators":[200,400,600,800,1000]}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)
ran_forest_cv=RandomizedSearchCV(pipeline2, parameters2)
ran_forest_cv.fit(X_train,y_train)
y_predict=ran_forest_cv.predict_proba(X_test)
y_predict=y_predict[:,1]
"""Prediction on unseen data"""
y_predict_new=ran_forest_cv.predict_proba(test)
y_predict_new=y_predict_new[:,1]
auc_score=roc_auc_score(y_test,y_predict)
print("AUC for the RandomForest classifier is: ")
print(auc_score)
print("Tuned RandomForestClassifier: {}".format(ran_forest_cv.best_params_))
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label='RandomForestClassifier, AUC = %0.4f'% auc_score)
plt.legend(loc='lower right')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])

plt.title('Receiver Operating for RandomForest classifier')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
Lsung_RandomForestClassifier=pd.DataFrame({"ID":test.loc[:,"Stammnummer"], "Expected":y_predict_new})
Lsung_RandomForestClassifier.head()
Lsung_RandomForestClassifier.to_csv("Lsung_RandomForestClassifier.csv", index=False)
steps=[("imputation", Imputer(missing_values="NaN", strategy="mean", axis=0)),
       ("scale", StandardScaler()),
       ("pca",PCA()),
       ("dec_tree", DecisionTreeClassifier())]

pipeline=Pipeline(steps)
parameters={"dec_tree__criterion":["gini", "entropy"],
           "dec_tree__class_weight":["balanced", None],
           "dec_tree__max_depth":[3, None],
           "dec_tree__max_features":randint(1, 9),
           "dec_tree__min_samples_leaf":randint(1, 9)}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)
dec_tree_cv=RandomizedSearchCV(pipeline, parameters)
dec_tree_cv.fit(X_train,y_train)
y_predict=dec_tree_cv.predict_proba(X_test)
y_predict=y_predict[:,1]
y_predict_new=dec_tree_cv.predict_proba(test)
y_predict_new=y_predict_new[:,1]
auc_score=roc_auc_score(y_test,y_predict)
print("AUC for the DecisionTree classifier is: ")
print(auc_score)
print("Tuned DecisionTreeClassifier: {}".format(dec_tree_cv.best_params_))
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label='DecisionTreeClassifier, AUC = %0.4f'% auc_score)
plt.legend(loc='lower right')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])

plt.title('Receiver Operating for DecisionTree classifier')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
Lsung_DecisionTreeClassifier=pd.DataFrame({"ID":test.loc[:,"Stammnummer"], "Expected":y_predict_new})
Lsung_DecisionTreeClassifier.head()
Lsung_DecisionTreeClassifier.to_csv("Lsung_DecisionTreeClassifier.csv", index=False)