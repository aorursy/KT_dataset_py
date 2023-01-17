import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler as ss

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV





pd.set_option('display.max_columns', None)



# machine learning

#Trees    

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier



#Ensemble Methods

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting  # explicitly require this experimental feature

from sklearn.ensemble import HistGradientBoostingClassifier # now you can import normally from ensemble

from lightgbm import LGBMClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost

from xgboost import XGBClassifier



#Gaussian Processes

from sklearn.gaussian_process import GaussianProcessClassifier

    

#GLM

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import RidgeClassifierCV

from sklearn.linear_model import Perceptron   

    

#Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier

    

#SVM

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.svm import NuSVC



#Discriminant Analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



 #Navies Bayes

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB



# metrics

from sklearn.metrics import accuracy_score, confusion_matrix



# PCA

from sklearn import decomposition



print("Setup Complete")
# Specify the path of the CSV file to read

train_df_final = pd.read_csv("../input/pumpitup-challenge-dataset/train_df_final.csv")

X_test_final = pd.read_csv("../input/pumpitup-challenge-dataset/X_test_final.csv")
X_test_final.shape
train_df_final.shape
X = train_df_final.drop("label",axis=1)

y = train_df_final["label"]
# Create training and test sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)
X.isnull().values.any()
sc = ss()

X_train = sc.fit_transform(X_train)

X_valid = sc.transform(X_valid)

X_test = sc.transform(X_test_final)
# Make an instance of the Model

pca = decomposition.PCA(.95)
pca.fit(X_train)
pca.n_components_
X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

X_valid_pca = pca.transform(X_valid)
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_valid)



acc_decision_tree = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_decision_tree
# Extra Tree



extra_tree = DecisionTreeClassifier()

extra_tree.fit(X_train, y_train)

y_pred = extra_tree.predict(X_valid)



acc_extra_tree = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_extra_tree
# Random Forest



rfc = RandomForestClassifier(criterion='entropy', n_estimators = 1000,min_samples_split=8,random_state=42,verbose=5)

rfc.fit(X_train, y_train)



y_pred = rfc.predict(X_valid)



acc_rfc = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_rfc
# GradientBoostingClassifier



GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.075, 

                                max_depth=13,max_features=0.5,

                                min_samples_leaf=14, verbose=5)



GB.fit(X_train, y_train)     

y_pred = GB.predict(X_valid)



acc_GB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_GB
# Histogram-based Gradient Boosting Classification Tree.



#This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).





HGB = HistGradientBoostingClassifier(learning_rate=0.075, loss='categorical_crossentropy', 

                                               max_depth=8, min_samples_leaf=15)



HGB = HGB.fit(X_train_pca, y_train)



y_pred = HGB.predict(X_valid_pca)



acc_HGB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_HGB
# LightGBM 



#is another fast tree based gradient boosting algorithm, which supports GPU, and parallel learning.





LGB = LGBMClassifier(objective='multiclass', learning_rate=0.75, num_iterations=100, 

                     num_leaves=50, random_state=123, max_depth=8)



LGB.fit(X_train, y_train)

y_pred = LGB.predict(X_valid)



acc_LGB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_LGB
# AdaBoost classifier



AB = AdaBoostClassifier(n_estimators=100, learning_rate=0.075)

AB.fit(X_train, y_train)     

y_pred = AB.predict(X_valid)



acc_AB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_AB
# BaggingClassifier



BC = BaggingClassifier(n_estimators=100)

BC.fit(X_train_pca, y_train)     

y_pred = BC.predict(X_valid_pca)



acc_BC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_BC
# XGBoost



xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=5)

xgb.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)



y_pred = xgb.predict(X_valid)

acc_xgb = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_xgb
# ExtraTreesClassifier



ETC = ExtraTreesClassifier(n_estimators=100)

ETC.fit(X_train, y_train)     

y_pred = ETC.predict(X_valid)



acc_ETC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_ETC
# Logistic Regression for multilabel classification



# https://acadgild.com/blog/logistic-regression-multiclass-classification

# https://medium.com/@jjw92abhi/is-logistic-regression-a-good-multi-class-classifier-ad20fecf1309



LG = LogisticRegression(solver="lbfgs", multi_class="multinomial")

LG.fit(X_train, y_train)     

y_pred = LG.predict(X_valid)



acc_LG = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_LG
coeff_df = pd.DataFrame(train_df_final.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(LG.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# PassiveAggressiveClassifier



PAC = PassiveAggressiveClassifier()

PAC.fit(X_train, y_train)

y_pred = PAC.predict(X_valid)



acc_PAC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_PAC
# RidgeClassifierCV



RC = RidgeClassifierCV()

RC.fit(X_train, y_train)

y_pred = RC.predict(X_valid)



acc_RC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_RC
# Perceptron



P = Perceptron()

P.fit(X_train, y_train)

y_pred = P.predict(X_valid)



acc_P = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_P
# Stochastic Gradient Descent

# https://scikit-learn.org/stable/modules/sgd.html



SGD = SGDClassifier(shuffle=True,average=True)

SGD.fit(X_train, y_train)

y_pred = SGD.predict(X_valid)



acc_SGD = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_SGD
# KNN



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)



acc_knn = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_knn
# Support Vector Classifier



SVC = SVC(probability=True)

SVC.fit(X_train, y_train)

y_pred = SVC.predict(X_valid)



acc_SVC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_SVC
# Linear SVC



linear_SVC = LinearSVC()

linear_SVC.fit(X_train,y_train)

linear_SVC.predict(X_valid)



acc_linear_SVC = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_linear_SVC
# LinearDiscriminantAnalysis



LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train,y_train)

LDA.predict(X_valid)



acc_LDA = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_LDA
# QuadraticDiscriminantAnalysis



QDA = QuadraticDiscriminantAnalysis()

QDA.fit(X_train,y_train)

QDA.predict(X_valid)



acc_QDA = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_QDA
# BernoulliNB



bernoulliNB = BernoulliNB()

bernoulliNB.fit(X_train,y_train)

bernoulliNB.predict(X_valid)



acc_bernoulliNB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_bernoulliNB
# GaussianNB



gaussianNB = GaussianNB()

gaussianNB.fit(X_train,y_train)

gaussianNB.predict(X_valid)



acc_gaussianNB = round(accuracy_score(y_valid,y_pred) * 100, 2)

acc_gaussianNB
models = pd.DataFrame({

    'Model': ['LightGBM','Decision Tree',"Extra Tree",'Random Forest','Support Vector', 'KNN', 'Logistic Regression', 

              'Stochastic Gradient Decent', 'Linear SVC',"XGBoost", "Ada Boost Classifier", 

              "Bagging Classifier", "Passive Agressive Cl", "Ridge","Perceptron",

              'Gradient Boosting Classifier','Extra Trees',

              "LinearDA","QuadraticDA","BernoulliNB","GaussianNB"],

    'Score': [acc_LGB,acc_decision_tree,acc_extra_tree,acc_rfc, acc_SVC, acc_knn, acc_LG,

              acc_SGD, acc_linear_SVC, acc_xgb, acc_AB, 

              acc_BC, acc_PAC, acc_RC, acc_P,

              acc_GB, acc_ETC,

             acc_LDA, acc_QDA, acc_bernoulliNB, acc_gaussianNB]})

sorted_by_score = models.sort_values(by='Score', ascending=False)
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='Score', y = 'Model', data = sorted_by_score, color = 'g')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score on validation data (%)')

plt.ylabel('Model')
"""

You can combine your best predictors as a VotingClassifier, which can enhance the performance.



"""



estimators = [('RFC', rfc), ('LGB', LGB), ('GB', GB)]



ensemble = VotingClassifier(estimators, voting='soft')



ensemble.fit(X, y)
submission_df = pd.read_csv("../input/pumpitup-challenge-dataset/SubmissionFormat.csv")
X_test = sc.transform(X_test_final)

submission_df['status_group']=rfc.predict(X_test)
vals_to_replace = {2:'functional', 1:'functional needs repair', 0:'non functional'}



submission_df.status_group = submission_df.status_group.replace(vals_to_replace)
submission_df.to_csv("submission_TatianaSwrt_rfc_noretrain_80.csv",sep=',', index=False)