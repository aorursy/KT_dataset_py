# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%pylab inline

import seaborn as sns

import xgboost

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_dataset = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
heart_dataset.head()
sns.countplot(x = "target", data = heart_dataset)
splitBySex = pd.crosstab(heart_dataset["target"], heart_dataset["sex"])

splitBySex[1] /= sum(splitBySex[1])

splitBySex[0] /= sum(splitBySex[0])

splitBySex.plot(kind = "bar")
sexSplitByDisease = pd.crosstab(heart_dataset["sex"], heart_dataset["target"])

sexSplitByDisease[0] /= sum(sexSplitByDisease[0])

sexSplitByDisease[1] /= sum(sexSplitByDisease[1])
sexSplitByDisease.plot(kind = "bar")
positives = heart_dataset.loc[heart_dataset["target"] == 1]

negatives = heart_dataset.loc[heart_dataset["target"] == 0]
sns.distplot(positives["age"],kde = False)
sns.distplot(negatives["age"],kde = False)
pd.crosstab(heart_dataset["target"], heart_dataset["cp"]).plot(kind = "bar")
sns.distplot(positives["trestbps"])
sns.distplot(negatives["trestbps"])
sns.jointplot(x = positives["trestbps"], y = positives["age"],kind = "hex")
sns.jointplot(x = negatives["trestbps"], y = negatives["age"], kind = "hex")
(positives["ca"].value_counts()).plot(kind = "bar")
negatives["ca"].value_counts().plot(kind = "bar")
pd.crosstab(heart_dataset["exang"],heart_dataset["target"]).plot(kind = "bar")
sns.swarmplot(y = heart_dataset["oldpeak"], x = heart_dataset["slope"], hue = heart_dataset["target"])
sns.distplot(positives["chol"], kde = False)
sns.distplot(negatives["chol"], kde = False)
from  sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
plt.plot(heart_dataset["target"])
X = heart_dataset.sample(frac=1) #shuffle shuffle shuffle

X = X.drop(["thal"], axis = 1)
#Insert the ca boolean variable in place of the ca variable

X["cabool"] = X["ca"].apply(lambda x : 1 if x > 0 else 0)

X = X.drop(["ca"], axis = 1)

X["ca"] = X["cabool"]

X = X.drop(["cabool"], axis = 1)
#copy the targets to y

y = X["target"].copy()

X = X.drop(["target"], axis = 1)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
#Using pipelines to do everything together - makes life easy

onehot_encoder = OneHotEncoder(handle_unknown = "ignore", sparse = False)

numerical_scaler = StandardScaler()

preprocessor = ColumnTransformer(transformers = [("numerical_scaler",numerical_scaler,["trestbps","chol","thalach","oldpeak"]),("onehot_encoder",onehot_encoder,["cp","restecg","slope"])], remainder = "passthrough")

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size = 0.8)
from sklearn.linear_model import LogisticRegression

fit_model = LogisticRegression(max_iter = 1000, C = 0.5, class_weight = "balanced")

lr_pipeline = Pipeline(steps = [("preprocessor",preprocessor),("model",fit_model)])
from sklearn.model_selection import GridSearchCV

param_grid = {"model__C" : np.logspace(-5,0,10)}

search = GridSearchCV(lr_pipeline, param_grid, n_jobs = -1)

search.fit(X_train,y_train)

search.best_score_
search.best_params_
#Redefine the pipeline with the best C

fit_model = LogisticRegression(max_iter = 1000, C = search.best_params_["model__C"], class_weight = "balanced")

lr_pipeline = Pipeline(steps = [("preprocessor",preprocessor),("model",fit_model)])
lr_pipeline.fit(X_train,y_train)

lr_preds = lr_pipeline.predict(X_valid)

print("Accuracy = ",sum(lr_preds == y_valid)/len(y_valid))
from sklearn.metrics import confusion_matrix,f1_score

lr_CM = confusion_matrix(y_valid, lr_preds)

lr_TP = lr_CM[1,1]

lr_TN = lr_CM[0,0]

lr_FP = lr_CM[0,1]

lr_FN = lr_CM[1,0]

print(lr_CM)
lr_recall = lr_TP/(lr_TP + lr_FN)

lr_specificity = lr_TN/(lr_TN + lr_FP)

lr_precision = lr_TP/(lr_TP + lr_FP)



print(lr_recall,lr_specificity)
lr_f1_score = f1_score(y_valid, lr_preds)

lr_f1_score
#Prep for ROC score curves

from sklearn.metrics import roc_curve, roc_auc_score

lr_y_proba = lr_pipeline.predict_proba(X_valid)[:,1]

lr_fpr,lr_tpr,lr_threshold = roc_curve(y_valid,lr_y_proba)
fit_model.coef_, fit_model.intercept_
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 1000)

rf_pipeline = Pipeline([("preprocessor",preprocessor),("model",rf_model)])
rf_search = GridSearchCV(rf_pipeline,param_grid = {"model__n_estimators":np.arange(100,1100,100)},n_jobs = -1)

rf_search.fit(X_train,y_train)
rf_search.best_params_
rf_search.best_score_
rf_model = RandomForestClassifier(n_estimators = rf_search.best_params_["model__n_estimators"])

rf_pipeline = Pipeline([("preprocessor",preprocessor),("model",rf_model)])

rf_pipeline.fit(X_train,y_train)
rf_preds = rf_pipeline.predict(X_valid)

print(sum(rf_preds == y_valid)/len(rf_preds))
all(rf_preds == lr_preds) #Confirmation in case the accuracies turn out to be equal
rf_CM = confusion_matrix(y_valid, rf_preds)

rf_TP = rf_CM[1,1]

rf_TN = rf_CM[0,0]

rf_FP = rf_CM[0,1]

rf_FN = rf_CM[1,0]

print(rf_CM)
rf_recall = rf_TP/(rf_TP + rf_FN)

rf_specificity = rf_TN/(rf_TN + rf_FP)

lr_precision = rf_TP/(rf_TP + rf_FP)



print(rf_recall,rf_specificity)
print(lr_recall, lr_specificity)
rf_y_proba = rf_pipeline.predict_proba(X_valid)[:,1]

rf_fpr,rf_tpr,rf_threshold = roc_curve(y_valid,rf_y_proba)
plt.plot(lr_fpr,lr_tpr,"b", label = "Logistic Regression")

plt.plot(rf_fpr, rf_tpr, "r", label = "Random Forest")

plt.plot([0,1],ls = "--")

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
print("logistic regression AUC = ",roc_auc_score(y_valid,lr_preds))

print("Random forests AUC = ",roc_auc_score(y_valid,rf_preds))