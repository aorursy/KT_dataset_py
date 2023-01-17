######## Base

import numpy as np 

import pandas as pd 



pd.set_option('display.max_columns', None)



######### Warning ##############

import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)





########## Sklearn #############

# Pre-processing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# Metrics

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve

# Models

from sklearn.linear_model import LogisticRegression     # Logistic Regression

from sklearn.naive_bayes import GaussianNB              # Naive Bayes

from sklearn.neighbors import KNeighborsClassifier      # KNN 

from sklearn.svm import SVC                             # SVC 

from sklearn import tree                                # CART - Sınıflandırma ve Regresyon Ağaçları

from sklearn.tree import DecisionTreeClassifier         # CART - Sınıflandırma ve Regresyon Ağaçları

from sklearn.ensemble import BaggingClassifier          # Bagging

from sklearn.ensemble import VotingClassifier           # Voting 

from sklearn.ensemble import RandomForestClassifier     # Random Forest

from sklearn.ensemble import AdaBoostClassifier         # Ada Boost

from sklearn.ensemble import GradientBoostingClassifier # GBM - Gradient Boosting Machine

from xgboost import XGBClassifier                       # XGBoost | !pip install xgboost

from lightgbm import LGBMClassifier                     # LightGBM | !conda install -c conda-forge lightgbm

from catboost import CatBoostClassifier                 # CatBoost | !pip install catboost

!pip install --upgrade nboost                           # NGBoost

!pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git

from ngboost import NGBClassifier

from ngboost.distns import k_categorical, Bernoulli
############ IMPORT

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

tr = train.copy()

ts = test.copy()



############ MISSING VALUE IMPUTATION

train["Age"] = np.where(train.Age.isnull(), train.Age.mean(), train.Age)

test["Age"] = np.where(test.Age.isnull(), test.Age.mean(), test.Age)

test["Fare"] = np.where(test.Fare.isnull(), test.Fare.mean(), test.Fare)



############ DROP VARIABLES

train.drop(["PassengerId", "Name"], axis = 1, inplace = True)

test.drop(["PassengerId", "Name"], axis = 1, inplace = True)



############ LABEL ENCODER

cat = train.select_dtypes(include=["object"]).columns

for col in train[cat].columns:

        

    train[col] = train[col].astype(str)

    test[col] = test[col].astype(str)

        

    le = LabelEncoder()

    le.fit(list(train[col])+list(test[col]))

    train[col] = le.transform(train[col])

    test[col]  = le.transform(test[col])

    

############ TRAIN-TEST SPLIT FOR TRAIN DATA    

X_train, X_test, y_train, y_test = train_test_split(train.drop("Survived", 

                                                               axis = 1),

                                                    train.Survived, 

                                                    test_size = 0.20,

                                                    random_state = 41)
########## ALL MODELS

# Logistic Regression

log = LogisticRegression(solver = "liblinear")

log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)



# Naive Bayes

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)



# KNN

knn = KNeighborsClassifier() # k (n_neighbors) sayısı ön tanımlı değeri 5'tir.

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)



# SVM - Linear

svc = SVC(kernel = "linear", probability=True) 

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)



# SVM - RBF

svc_rbf = SVC(kernel = "rbf",probability=True) 

svc_rbf.fit(X_train, y_train)

y_pred_svc_rbf = svc_rbf.predict(X_test)



# CART

cart = DecisionTreeClassifier()

cart.fit(X_train, y_train)

y_pred_cart = cart.predict(X_test)



# BAGGING

bag = BaggingClassifier()

bag.fit(X_train, y_train)

y_pred_bag = bag.predict(X_test)



# VOTING

clf1 = LogisticRegression(solver = "liblinear")

clf2 = RandomForestClassifier()

clf3 = GaussianNB()

clf4 = KNeighborsClassifier()

vote = VotingClassifier(

    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

vote.fit(X_train, y_train)

y_pred_vote = vote.predict(X_test)



# RANDOM FOREST

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)



# ADABOOST

ada = AdaBoostClassifier()

ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)



# GBM

gbm = GradientBoostingClassifier()

gbm.fit(X_train, y_train)

y_pred_gbm = gbm.predict(X_test)



# XGBOOST

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)



# LGBM

lgb = LGBMClassifier()

lgb.fit(X_train, y_train)

y_pred_lgb = lgb.predict(X_test)



# CATBOOST

cat = CatBoostClassifier()

cat.fit(train.drop("Survived",axis = 1), train.Survived, verbose = 0)

y_pred_cat = cat.predict(X_test) 



# NGBOOST

ngb_cat = NGBClassifier(Dist=k_categorical(2), verbose=False)

ngb_cat.fit(X_train, y_train)

y_pred_ngb = ngb_cat.predict(X_test)
########## RESULTS

models = ["Logistic Regression", "Naive Bayes", "KNN", "Linear SVM", "RBF SVM", "CART", "Bagging", "Voting", 

            "Random Forest", "AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost", "NGBoost"]

test_acc = [

    accuracy_score(y_test, y_pred_log),

    accuracy_score(y_test, y_pred_nb),

    accuracy_score(y_test, y_pred_knn),

    accuracy_score(y_test, y_pred_svc),

    accuracy_score(y_test, y_pred_svc_rbf),

    accuracy_score(y_test, y_pred_cart),

    accuracy_score(y_test, y_pred_bag),

    accuracy_score(y_test, y_pred_vote),

    accuracy_score(y_test, y_pred_rf),

    accuracy_score(y_test, y_pred_ada),

    accuracy_score(y_test, y_pred_gbm),

    accuracy_score(y_test, y_pred_xgb),

    accuracy_score(y_test, y_pred_lgb),

    accuracy_score(y_test, y_pred_cat),

    accuracy_score(y_test, y_pred_ngb)

]



train_acc = [

    

    accuracy_score(y_train, log.predict(X_train)),

    accuracy_score(y_train, nb.predict(X_train)),

    accuracy_score(y_train, knn.predict(X_train)),

    accuracy_score(y_train, svc.predict(X_train)),

    accuracy_score(y_train, svc_rbf.predict(X_train)),

    accuracy_score(y_train, cart.predict(X_train)),

    accuracy_score(y_train, bag.predict(X_train)),

    accuracy_score(y_train, vote.predict(X_train)),

    accuracy_score(y_train, rf.predict(X_train)),

    accuracy_score(y_train, ada.predict(X_train)),

    accuracy_score(y_train, gbm.predict(X_train)),

    accuracy_score(y_train, xgb.predict(X_train)),

    accuracy_score(y_train, lgb.predict(X_train)),

    accuracy_score(y_train, cat.predict(X_train)),

    accuracy_score(y_train, ngb_cat.predict(X_train))

]



pd.DataFrame({

    

    "Model":models,

    "Train Accuracy": train_acc,

    "Test Accuracy": test_acc

    

})
log = LogisticRegression(solver = "liblinear")

log.fit(train.drop("Survived", axis = 1), train.Survived)

ypred = log.predict_proba(test)[:,1]

# Best threshold 0.6

ypred = [1 if i > 0.6 else 0 for i in ypred]

ts["Survived"] = ypred

submissionlog = ts[["PassengerId", "Survived"]]

submissionlog.to_csv("submissionlog.csv",columns = ["PassengerId", "Survived"] , index = None)
########### NAIVE BAYES

nb = GaussianNB()

model = nb.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionnaive = ts[["PassengerId", "Survived"]]

submissionnaive.to_csv("submissionnaive.csv",columns = ["PassengerId", "Survived"] , index = None)



########### KNN

knn = KNeighborsClassifier()

model = knn.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionknn = ts[["PassengerId", "Survived"]]

submissionknn.to_csv("submissionknn.csv",columns = ["PassengerId", "Survived"] , index = None)



########### LINEAR SVM

svc = SVC(kernel = "linear", probability=True) 

model = svc.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.6 else 0 for i in y_predprob]

ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionsvc.csv",columns = ["PassengerId", "Survived"] , index = None)



########### RBF SVM

svc = SVC(kernel = "rbf", probability=True) 

model = svc.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.6 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionsvc.csv",columns = ["PassengerId", "Survived"] , index = None)



########### CART

cart = DecisionTreeClassifier()

model = cart.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissioncart.csv",columns = ["PassengerId", "Survived"] , index = None)



########### BAGGING

bag = BaggingClassifier()

model = bag.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionbagging.csv",columns = ["PassengerId", "Survived"] , index = None)



########### VOTING

clf1 = LogisticRegression(solver = "liblinear")

clf2 = RandomForestClassifier()

clf3 = GaussianNB()

clf4 = KNeighborsClassifier()



vote = VotingClassifier(

    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard'

)

model = vote.fit(train.drop("Survived",axis = 1), train.Survived)

#y_predprob = model.predict_proba(test)[:,1]

#ypred = [1 if i > 0.5 else 0 for i in y_predprob]

ypred = model.predict(test)



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionvoting.csv",columns = ["PassengerId", "Survived"] , index = None)



########### RANDOM FOREST

rf = RandomForestClassifier()

model = rf.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.4 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionrf.csv",columns = ["PassengerId", "Survived"] , index = None)



########### ADABOOST

ada = AdaBoostClassifier()

model = ada.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionada.csv",columns = ["PassengerId", "Survived"] , index = None)



########### GBM

gbm = GradientBoostingClassifier()

model = gbm.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissiongbm.csv",columns = ["PassengerId", "Survived"] , index = None)



########### XGBOOST

xgb = XGBClassifier()

model = xgb.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionxgb.csv",columns = ["PassengerId", "Survived"] , index = None)



########### LGBM

lgb = LGBMClassifier()

model = lgb.fit(train.drop("Survived",axis = 1), train.Survived)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.6 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissionlgb.csv",columns = ["PassengerId", "Survived"] , index = None)



########### CATBOOST

cat = CatBoostClassifier()

model = cat.fit(train.drop("Survived",axis = 1), train.Survived, verbose = 0)

y_predprob = model.predict_proba(test)[:,1]

ypred = [1 if i > 0.5 else 0 for i in y_predprob]



ts["Survived"] = ypred

submissionsvc = ts[["PassengerId", "Survived"]]

submissionsvc.to_csv("submissioncat.csv",columns = ["PassengerId", "Survived"] , index = None)



########### NGBOOST

ngb_cat = NGBClassifier(Dist=k_categorical(2), verbose=False) # tell ngboost that there are 3 possible outcomes

ngb_cat.fit(train.drop("Survived",axis = 1), train.Survived)

ts["Survived"] = ngb_cat.predict(test)

submissionngb = ts[["PassengerId", "Survived"]]

submissionngb.to_csv("submissionngb.csv",columns = ["PassengerId", "Survived"] , index = None)
models = ["Logistic Regression", "Naive Bayes", "KNN", "Linear SVM", "RBF SVM", "CART", "Bagging", "Voting", "Random Forest", "AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost", "NGBoost"]

kaggle_scores = [0.78468, 0.73205, 0.63636, 0.77511, 0.66985, 0.73205, 0.80382, 0.76076, 0.74641, 0.74162,0.76555, 0.77990 ,0.77033 ,0.76076, 0.76555]



pd.DataFrame({"Model": models, "Kaggle Score": kaggle_scores}).sort_values("Kaggle Score", ascending=False).reset_index(drop = True)


