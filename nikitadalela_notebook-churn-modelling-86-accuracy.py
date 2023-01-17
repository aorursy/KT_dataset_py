import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sb

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.figure_factory as ff



from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold

from sklearn.preprocessing import StandardScaler  

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

df =df.drop(['RowNumber','Surname'],axis=1)

df.head()

# df_1 = df.select_dtypes(exclude='object')

# df_1.hist(figsize=(10,8))



df.select_dtypes(exclude='object').hist(figsize=(14,10),bins=20)

sb.countplot(df['Exited'])

df['Exited'].value_counts().unique()

corr = df.corr()

corr.style.background_gradient(cmap='Greens').set_precision(2)


# Seperate the target or response variable



y = df['Exited']



#Remove response from dataset



X = df.drop(['Exited'],axis =1)

X.head()



# apply train test split



train_X,valid_X,train_y,valid_y = train_test_split(X,y,test_size=0.20,random_state=1)



#LabelEncoding for categorical variables



from sklearn.preprocessing import LabelEncoder

lbenc = LabelEncoder()



cols = ['Geography','Gender']



for col in df[cols]:

    train_X[col] = lbenc.fit_transform(train_X[col])

    valid_X[col] = lbenc.transform(valid_X[col])

# import the class

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression(solver='liblinear',random_state=1,C=10.0)



# fit the model with data

logreg.fit(train_X,train_y)



#

y_pred=logreg.predict(valid_X)





# import the metrics class

from sklearn import metrics

from sklearn.metrics import classification_report

cnf_matrix = metrics.confusion_matrix(valid_y, y_pred)

cnf_matrix

print(classification_report(valid_y, y_pred,digits=5))
y_pred = logreg.predict_proba(valid_X)[::,1]

fpr, tpr, _ = metrics.roc_curve(valid_y,  y_pred)

auc = metrics.roc_auc_score(valid_y, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()



rs=1

models = [LogisticRegression(random_state=rs), LGBMClassifier(),KNeighborsClassifier(),

          BaggingClassifier(random_state=rs),DecisionTreeClassifier(random_state=rs),

          RandomForestClassifier(random_state=rs), GradientBoostingClassifier(random_state=rs),

          XGBClassifier(random_state=rs), MLPClassifier(random_state=rs),

          CatBoostClassifier(random_state=rs,verbose = False)]

alias = ["LogisticRegression","LGBM","KNN","Bagging",

             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]
print('Default model validation accuracies for the train data:', end = "\n\n")

for name, model in zip(alias, models):

    model.fit(train_X, train_y)

    y_pred = model.predict(valid_X) 

    print(name,':',"%.3f" % accuracy_score(y_pred, valid_y))
predictors=pd.concat([train_X,valid_X])

outcomes = []

print('10 fold Cross validation accuracy and std of the default models for the train data:', end = "\n\n")

for name, model in zip(alias, models):

    kfold = KFold(n_splits=10, random_state=2003)

    cv_results = cross_val_score(model, predictors, y, cv = kfold, scoring = "accuracy")

    outcomes.append(cv_results)

    print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))
cv_models = [LGBMClassifier(),LogisticRegression(random_state=rs),

          RandomForestClassifier(random_state=rs), GradientBoostingClassifier(random_state=rs),

          ]





params=[]

lgbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [1000,500,100],

             "max_depth": [3,5,10]}

log_params = {"penalty": ["l1","l2"],

               "C":[1, 3, 5],

            "solver":['lbfgs', 'liblinear', 'sag', 'saga'], "max_iter":[1000]}



rf_params = {"max_features": ["log2","auto","sqrt"],

                "min_samples_split":[2,3,5],

                "min_samples_leaf":[1,3,5],

                "bootstrap":[True,False],

                "n_estimators":[50,100,150],

                "criterion":["gini","entropy"]}

gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}



params = [lgbm_params,log_params,rf_params,gbm_params]

new_alias = ["LGBM","LogisticRegression","RandomForest","GBM"]

cv_result = {}

best_estimators = {}

for name, model,param in zip(new_alias,cv_models,params):

        clf = GridSearchCV(model, param_grid=param, cv =10, scoring = "accuracy", n_jobs = -1,verbose = False)

        clf.fit(train_X,train_y)

        cv_result[name]=clf.best_score_

        best_estimators[name]=clf.best_estimator_

        print(clf.best_estimator_)

        print(name,'cross validation accuracy : %.3f'%cv_result[name])

        print(clf.best_score_)
accuracies={}

print('Validation accuracies of the tuned models for the train data:', end = "\n\n")

for name, model_tuned in zip(best_estimators.keys(),best_estimators.values()):

    y_pred =  model_tuned.fit(train_X,train_y).predict(valid_X)

    accuracy=accuracy_score(y_pred, valid_y)

    print(name,':', "%.3f" %accuracy)

    accuracies[name]=accuracy
n=3

accu=sorted(accuracies, reverse=True, key= lambda k:accuracies[k])[:n]

firstn=[[k,v] for k,v in best_estimators.items() if k in accu]

print(firstn)



votingC = VotingClassifier(estimators = firstn, voting = "soft", n_jobs = -1)

model= votingC.fit(train_X, train_y)

print("\nAccuracy_score is:",accuracy_score(model.predict(valid_X),valid_y))