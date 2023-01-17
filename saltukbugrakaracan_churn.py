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
churn = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv")
churn.head()
churn.info()
churn.shape
churn["Geography"].value_counts()
churn["NumOfProducts"].value_counts()
churn.drop(["RowNumber","CustomerId","Surname"],axis=1,inplace=True)#We've dropped these features because they don't effect my target.
churn.head()
churn = pd.get_dummies(churn,drop_first=True)#one hot encoding,LabelEncoder
churn.head()
churn.describe([0.01,0.05,0.10,0.20,0.25,0.40,0.50,0.60,0.75,0.90,0.995]).T
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

from catboost import CatBoostClassifier

from xgboost import XGBClassifier





from warnings import filterwarnings

filterwarnings('ignore')
models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('SVC', SVC(gamma='auto')))

models.append(('GradientBoosting', GradientBoostingClassifier()))

models.append(("LightGBM", LGBMClassifier()))

models.append(("XGBoost", XGBClassifier()))
models
churn.drop("Exited",axis=1)
churn[["Exited"]]
X = churn.drop("Exited",axis=1)

y = churn["Exited"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12345)



#



lr = LogisticRegression()

model = lr.fit(X_train,y_train) # 8000 satır.

y_pred = model.predict(X_test) # 2000 satır.

accuracy_score(y_pred,y_test)
X = churn.drop("Exited",axis=1)

y = churn["Exited"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12345)





for name, model in models:

    

        mod = model.fit(X_train,y_train)

        y_pred = mod.predict(X_test)

        res = accuracy_score(y_test,y_pred)

        print(name+": "+str(res))
churn.head()
df = churn.copy()
df.head()
df["Exited"].value_counts()
import matplotlib.pyplot as plt





# Plot

plt.scatter(df["Tenure"], df["Age"])

plt.title('Scatter plot pythonspot.com')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
df.head()
df[["CreditScore"]].describe()
df["Age"].max()
df.head()
df[["CreditScore"]][df["IsActiveMember"] == 0].describe().T
df["Tenure"].value_counts()
df.describe().T
df[["EstimatedSalary"]][df["Balance"] == 0 ].describe([0.01,0.025,0.05,0.10,0.15,0.25,0.50,0.75,0.90,0.99]).T
df[["Balance","EstimatedSalary"]].head(30)
#df["New"] = df["Balance"] / df[]
#df["Using"] = df["Tenure"] / df["Age"]



df['CreditSegment'] = pd.qcut(df['CreditScore'], 4 ,labels = [1,2,3,4])
df[["CreditSegment"]].dtypes
df[["CreditSegment"]] = df[["CreditSegment"]].astype("int64")
df.head()
df.describe()
for feature in df:



    Q1 = df[feature].quantile(0.25)

    Q3 = df[feature].quantile(0.75)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):

        print(feature,"yes")

        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])

        print("lower",lower,"\nupper",upper)

        #df.loc[df[feature] > upper,feature] = upper

    else:

        print(feature, "no")
df["Balance"][df["Balance"] > 200000] = 200000
df.max()
df["Balance"].describe()
df.isnull().sum()
df.head()
# LOF yöntemi ile tüm değişkenler arasındaki uç değerleri belirleriz

from sklearn.neighbors import LocalOutlierFactor

lof =LocalOutlierFactor(n_neighbors= 10)

lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_

np.sort(df_scores)[0:30]
#Lof skorlarına göre eşik değerini seçiyoruz

threshold = np.sort(df_scores)[7]

threshold
#Eşikten daha yüksek olanları siliyoruz

outlier = df_scores > threshold#-1.62

df = df[outlier]
df.head()
df["CreditScore"].describe()
cols = []



for i in df.columns:

    cols.append(i)
cols
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(df)

df = scaler.transform(df)

#fit_transform

#min max 0-1 arasına yayma işlemi







df = pd.DataFrame(df,columns=cols) #numpy arrayden dataframe a çevirme işlemi

#df[:10]
df.head()
X = df.drop("Exited",axis=1)

y = df["Exited"]
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled
X_resampled
y_resampled.value_counts()
y_resampled
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12345)




X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=12345)
df["Exited"].value_counts()
for name, model in models:

    

        mod = model.fit(X_train,y_train)

        y_pred = mod.predict(X_test)

        res = accuracy_score(y_test,y_pred)

        print(name+": "+str(res))
df.shape
df.head()
knn_params = {"n_neighbors": np.arange(2,30,1)}



knn_model = KNeighborsClassifier()



knn_cv_model = GridSearchCV(knn_model, knn_params, cv = 10).fit(X_train, y_train)
knn_cv_model.best_params_
knn_tuned = KNeighborsClassifier(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
cart_model = DecisionTreeClassifier()

cart_params = {"max_depth": [2,3,4,5,10,20,100, 1000],

              "min_samples_split": [2,10,5,30,50,10]}

cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10, n_jobs = -1, verbose =  2).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeClassifier(**cart_cv_model.best_params_).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
rf_params = {"max_depth": [5,10,None],

            "max_features": [2,5,10],

            "n_estimators": [100, 500, 900],

            "min_samples_split": [2,10,30]}

rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train , y_train)
rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
svc_model = SVC()



svc_params = {"C": [0.01,0.001, 0.2, 0.1,0.5,0.8,0.9,1, 10, 100, 500,1000]}



svc_cv_model = GridSearchCV(svc_model, svc_params, cv = 10, n_jobs = -1, verbose =  2).fit(X_train, y_train)
svc_cv_model.best_params_
svc_tuned = SVC(**svc_cv_model.best_params_).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
gb_model = GradientBoostingClassifier()

gbm_params = {"learning_rate": [0.001,0.1],

             "max_depth": [3,5,8,10],

             "n_estimators": [200,500,1000],

             "subsample": [1,0.5]}

gbm_cv_model = GridSearchCV(gb_model, 

                            gbm_params, 

                            cv = 10, 

                            n_jobs=-1, 

                            verbose = 2).fit(X_train, y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingClassifier(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],

              "n_estimators": [200, 500, 1000, 1500],

              "max_depth":[3,5,8,10],

              "colsample_bytree": [1,0.5,0.3]}

lgbm_model = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm_model, 

                     lgbm_params, 

                     cv = 10, 

                     n_jobs = -1, 

                     verbose = 2).fit(X_train, y_train)
lgbm_cv_model.best_params_

lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)



accuracy_score(y_test,y_pred)
xgb_model = XGBClassifier()

xgb_params = {"learning_rate": [0.1,0.01,1],

             "max_depth": [2,5,8],

             "n_estimators": [100,500,1000],

             "colsample_bytree": [0.3,0.6,1]}

xgb_cv_model  = GridSearchCV(xgb_model,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)



accuracy_score(y_test,y_pred)