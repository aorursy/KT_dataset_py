from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport

#Data Understanding & Preparing
diabetes = pd.read_csv("../input/diabetes-data-set/diabetes.csv")

df = diabetes.copy()
df.info()
df.describe().T
df["Outcome"].value_counts().plot.pie(autopct = "%.1f")
df = df.drop(["Outcome"], axis = 1)
df = df.replace({0 : np.nan})
df[df["BMI"] > 50]
df.isnull().sum()
df = pd.concat([df, diabetes["Outcome"]], axis = 1)
df.head()
df.isnull().sum()
#Filling missing values with median based on Outcome(y)
#pregnancies
df.groupby("Outcome")["Pregnancies"].median()
df.loc[(df["Outcome"] == 0) & (df["Pregnancies"].isnull()), "Pregnancies"] = 3
df.loc[(df["Outcome"] == 1) & (df["Pregnancies"].isnull()), "Pregnancies"] = 5
#glucose
df.groupby("Outcome")["Glucose"].median()
df.loc[(df["Outcome"] == 0) & (df["Glucose"].isnull()), "Glucose"] = 107
df.loc[(df["Outcome"] == 1) & (df["Glucose"].isnull()), "Glucose"] = 140
#BloodPressure
df.groupby("Outcome")["BloodPressure"].median()
df.loc[(df["Outcome"] == 0) & (df["BloodPressure"].isnull()), "BloodPressure"] = 70
df.loc[(df["Outcome"] == 1) & (df["BloodPressure"].isnull()), "BloodPressure"] = 74.5
#SkinThickness 
df.groupby("Outcome")["SkinThickness"].median()
df.loc[(df["Outcome"] == 0) & (df["SkinThickness"].isnull()), "SkinThickness"] = 27
df.loc[(df["Outcome"] == 1) & (df["SkinThickness"].isnull()), "SkinThickness"] = 32 
#Insulin
df.groupby("Outcome")["Insulin"].median()
df.loc[(df["Outcome"] == 0) & (df["Insulin"].isnull()), "Insulin"] = 102.5
df.loc[(df["Outcome"] == 1) & (df["Insulin"].isnull()), "Insulin"] = 169.5
#BMI
df.groupby("Outcome")["BMI"].median()
df.loc[(df["Outcome"] == 0) & (df["BMI"].isnull()), "BMI"] = 30.1
df.loc[(df["Outcome"] == 1) & (df["BMI"].isnull()), "BMI"] = 34.3
df.isnull().sum()
df.loc[(df["Glucose"] < 160) & (df["Outcome"] == 0)]
df.loc[(df["Glucose"] > 160) & (df["Outcome"] == 1)]
df.groupby("Outcome")["Pregnancies"].median()
df.isnull().sum()
df.shape
#Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer
df.head()
scaler = RobustScaler()
y = df["Outcome"]
X = df.drop('Outcome', axis=1)
y.shape
cols = X.columns
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns= cols)
X.head()
X.shape
y.head()
y.shape
#PREDICTION
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
def before_tuned():
    models = []
    models.append (("LightGBM", LGBMClassifier()))
    models.append (("CART",  DecisionTreeClassifier()))
    models.append (("XGB", GradientBoostingClassifier()))
# evaluate each model in turn
    results = []
    names = [] 
    for name, model in models:
            kfold = KFold(n_splits = 10, random_state = 123456)
            cv_results = cross_val_score(model, X, y, cv = 10, scoring = "accuracy")
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

#scores before tuning.
before_tuned()

#LGBM Tuned
lgbm = LGBMClassifier(random_state = 12345)

lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}
gs_cv = GridSearchCV(lgbm, 
                     lgbm_params, 
                     cv = 5, 
                     n_jobs = -1, 
                     verbose = 2).fit(X, y)
lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)
lgbm_sc = cross_val_score(lgbm_tuned, X, y, cv = 10).mean()
lgbm_sc
Importance = pd.DataFrame({'Importance':lgbm_tuned.feature_importances_}, 
                          index = X.columns)
Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True)
#XGB Tuned
xgb_model = GradientBoostingClassifier(random_state = 12345).fit(X,y)
xgb_params = {"max_depth": [2,3,4,5,8],
              "n_estimators": [100,200,500,1000]}
xgb_cv_model = GridSearchCV(xgb_model,
                           xgb_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2).fit(X, y)
xgb_cv_model.best_params_
xgb_tuned = RandomForestClassifier(**xgb_cv_model.best_params_).fit(X,y)
xgb_sc= cross_val_score(xgb_tuned, X, y, cv = 10).mean()
xgb_sc
Importance = pd.DataFrame({'Importance':xgb_tuned.feature_importances_}, 
                          index = X.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True)
#CART Tuned
cart_model = DecisionTreeClassifier(random_state=12345).fit(X,y)
cart_params = {"max_depth": [2,3,4,5,10,20, 100, 1000],
              "min_samples_split": [2,10,5,30,50,10]}
cart_cv_model = GridSearchCV(cart_model, 
                    cart_params,
                    cv = 10,
                    n_jobs = -1,
                    verbose = 2).fit(X, y)
cart_tuned = DecisionTreeClassifier(**cart_cv_model.best_params_).fit(X,y)
cart_sc = cross_val_score(cart_tuned, X, y, cv = 10).mean()
cart_sc
Importance = pd.DataFrame({'Importance':cart_tuned.feature_importances_}, 
                          index = X.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True)
#Reporting
models = pd.DataFrame({"Model" : ["LGBM","XGBOOST", "CART"],
                     "Score" : [lgbm_sc,xgb_sc,cart_sc]})
models.sort_values("Score")
