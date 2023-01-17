# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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

from xgboost import XGBClassifier



from warnings import filterwarnings

filterwarnings('ignore')
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
models = []



models.append(("LR", LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVR', SVC()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('RandomForests', RandomForestClassifier()))

models.append(('GradientBoosting', GradientBoostingClassifier()))

models.append(('XGBoost', XGBClassifier()))

models.append(('Light GBM', LGBMClassifier()))
X = df.drop("Outcome",axis=1)

y = df["Outcome"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=46)



for name,model in models:

    mod = model.fit(X_train,y_train) #trainleri modele fit etmek

    y_pred = mod.predict(X_test) # tahmin

    acc = accuracy_score(y_test, y_pred) #rmse hesabı

    cvscore = cross_val_score(model, X,y, cv = 10).mean()

    print("Holdout Method:",end=" ")

    print(name,acc) #yazdırılacak kısım

    print("Cross Val Score",end=" ")

    print(name,cvscore)

    print("------------------------------------")
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.describe().T
df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)
df.isnull().sum()
naValues = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]



for i in naValues:

    df[i][(df[i].isnull()) & (df["Outcome"] == 0)] = df[i][(df[i].isnull()) & (df["Outcome"] == 0)].fillna(df[i][df["Outcome"] == 0].mean())

    df[i][(df[i].isnull()) & (df["Outcome"] == 1)] = df[i][(df[i].isnull()) & (df["Outcome"] == 1)].fillna(df[i][df["Outcome"] == 1].mean())
df.isnull().sum()
df.head()
df.info()
df.shape
df["Outcome"].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
df.Outcome.value_counts().plot.barh()
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.80,0.90,0.95,0.99]).T
plt.scatter(df.Glucose,df.Insulin)
sns.catplot(x = "Outcome", y = "Glucose", data = df);
sns.catplot(x = "Outcome", y = "Insulin", data = df);
sns.catplot(x = "Outcome", y = "BloodPressure", data = df);
sns.catplot(x = "Outcome", y = "BMI", data = df);#the 0 value for outcome feature is more dense around 20 BMI. Healthy people are more generally 0.
sns.boxplot(df)
df.corr()
f, ax = plt.subplots(figsize= [20,15])

sns.heatmap(df.corr(),annot=True,ax=ax)
df.describe().T
for feature in df:



    Q1 = df[feature].quantile(0.05)

    Q3 = df[feature].quantile(0.95)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):

        print(feature,"yes")

        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])

        print("lower",lower,"\nupper",upper)

        df.loc[df[feature] > upper,feature] = upper

    else:

        print(feature, "no")

        
df.describe().T
df.head()
df['BMIRanges'] = pd.cut(x=df['BMI'], bins=[0,18.5,25,30,100],labels = ["Underweight","Healthy","Overweight","Obese"])
df.head()
df.groupby(["Outcome","BMIRanges"]).describe()
df.head()
df["Insulin"].describe().T
def set_insulin(row):

    if row["Insulin"] >= 16 and row["Insulin"] <= 166:

        return "Normal"

    else:

        return "Abnormal"
df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))

df.head()
df['NewGlucose'] = pd.cut(x=df['Glucose'], bins=[0,70,99,126,200],labels = ["Low","Normal","Secret","High"])
df.head()
df = pd.get_dummies(df,drop_first=True)
df.head()
from sklearn.preprocessing import RobustScaler

r_scaler = RobustScaler()

df_r = r_scaler.fit_transform(df.drop(["Outcome","BMIRanges_Healthy","BMIRanges_Overweight","BMIRanges_Obese","INSULIN_DESC_Normal","NewGlucose_Normal","NewGlucose_Secret","NewGlucose_High"],axis=1))



df_r = pd.DataFrame(df_r, columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
df_r
df = pd.concat([df_r,df[["Outcome","BMIRanges_Healthy","BMIRanges_Overweight","BMIRanges_Obese","INSULIN_DESC_Normal","NewGlucose_Normal","NewGlucose_Secret","NewGlucose_High"]]],axis=1)
df
df.info()
X = df.drop("Outcome",axis=1)

y = df["Outcome"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=46)



for name,model in models:

    mod = model.fit(X_train,y_train) #trainleri modele fit etmek

    y_pred = mod.predict(X_test) # tahmin

    acc = accuracy_score(y_test, y_pred) #rmse hesabı

    cvscore = cross_val_score(model, X,y, cv = 10).mean()

    print("Holdout Method:",end=" ")

    print(name,acc) #yazdırılacak kısım

    print("Cross Val Score",end=" ")

    print(name,cvscore)

    print("------------------------------------")
knn_params = {"n_neighbors": np.arange(2,30,1)}



knn_model = KNeighborsClassifier()



knn_cv_model = GridSearchCV(knn_model, knn_params, cv = 10).fit(X,y)
knn_cv_model.best_params_
knn_tuned = KNeighborsClassifier(**knn_cv_model.best_params_).fit(X,y)

cross_val_score(knn_tuned, X,y, cv = 10).mean()
cart_model = DecisionTreeClassifier()
cart_params = {"max_depth": [2,3,4,5,10,20,100, 1000],

              "min_samples_split": [2,10,5,30,50,10]}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10, n_jobs = -1, verbose =  2).fit(X,y)
cart_cv_model.best_params_
cart_tuned = DecisionTreeClassifier(**cart_cv_model.best_params_).fit(X,y)
cross_val_score(cart_tuned, X,y, cv = 10).mean()
rf_params = {"max_depth": [5,10,None],

            "max_features": [2,5,10],

            "n_estimators": [100, 500, 900],

            "min_samples_split": [2,10,30]}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X,y)
rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X,y)
cross_val_score(rf_tuned, X,y, cv = 10).mean()
svc_model = SVC()
svc_params = {"C": [0.01,0.001, 0.2, 0.1,0.5,0.8,0.9,1, 10, 100, 500,1000]}



svc_cv_model = GridSearchCV(svc_model, svc_params, cv = 10, n_jobs = -1, verbose =  2).fit(X,y)
svc_cv_model.best_params_
svc_tuned = SVC(**svc_cv_model.best_params_).fit(X,y)
cross_val_score(svc_tuned, X,y, cv = 10).mean()
gb_model = GradientBoostingClassifier()
gbm_params = {"learning_rate": [0.001,0.1,0.01],

             "max_depth": [3,5,8,10],

             "n_estimators": [200,500,1000],

             "subsample": [1,0.5,0.8]}
gbm_cv_model = GridSearchCV(gb_model, 

                            gbm_params, 

                            cv = 10, 

                            n_jobs=-1, 

                            verbose = 2).fit(X,y)
gbm_cv_model.best_params_

gbm_tuned = GradientBoostingClassifier(**gbm_cv_model.best_params_).fit(X,y)
cross_val_score(gbm_tuned, X,y, cv = 10).mean()
lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],

              "n_estimators": [200, 500, 1000],

              "max_depth":[5,8,10],

              "colsample_bytree": [1,0.5,0.3]}

lgbm_model = LGBMClassifier()
lgbm_cv_model = GridSearchCV(lgbm_model, 

                     lgbm_params, 

                     cv = 10, 

                     n_jobs = -1, 

                     verbose = 2).fit(X,y)
lgbm_cv_model.best_params_

lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X,y)

cross_val_score(lgbm_tuned, X,y, cv = 10).mean()
xgb_model = XGBClassifier()
xgb_params = {"learning_rate": [0.1,0.01,1],

             "max_depth": [2,5,8],

             "n_estimators": [100,500,1000],

             "colsample_bytree": [0.3,0.6,1]}
xgb_cv_model  = GridSearchCV(xgb_model,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X,y)
xgb_cv_model.best_params_

xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_).fit(X,y)

cross_val_score(xgb_tuned, X,y, cv = 10).mean()