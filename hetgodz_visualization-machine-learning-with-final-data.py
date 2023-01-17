# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt#visualization

%matplotlib inline

import seaborn as sns#visualization

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import pickle
df = pd.read_csv('../input/weatherson.csv')

df.head()
df.drop("date",axis=1,inplace=True)
df.head()
df.dtypes
cleanup_nums = {"events": {"Rain": 1, "Snow": 2, "Rain-Snow": 3, "Fog-Rain": 4,"Fog-Snow": 5, "Fog": 6, "Rain-Thunderstorm":7,"Fog-Rain-Snow":8 ,

                            "Fog-Rain-Hail-Thunderstorm":9, "Fog-Rain-Thunderstorm":10 , "Thunderstorm":11 }}
df.replace(cleanup_nums, inplace=True)

df.head()
(df["events"].value_counts()

.plot.barh()

.set_title("bar grafiği"));
sns.catplot(x="events", y="cloudcover",data=df);
sns.catplot(x="events", y="max_humidity", hue="cloudcover",data=df);
(df["winddirdegrees"]

.plot

.hist(bins=100)

.set_title("histogram grafiği")); 
sns.kdeplot(df.winddirdegrees, shade= True); 
sns.catplot(x="events",y="max_sea_level_pressurein",hue="cloudcover", kind="point", data=df);
sns.boxplot(x="events",y="cloudcover", data=df); #events&cloudiness
sns.boxplot(x="cloudcover",y="precipitationin", data=df); #cloudiness&precipitation
sns.boxplot(x="events",y="precipitationin", data=df);
sns.boxplot(x="mean_visibilitymiles",y="mean_humidity", data=df); # humidity up visibility downb
sns.boxplot(x="events",y="mean_temperaturef", data=df);
from sklearn import linear_model

import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier



from warnings import filterwarnings

filterwarnings('ignore')
y=df["events"]

X= df.drop(["events"], axis=1)
loj= LogisticRegression(solver="liblinear")

loj_model=loj.fit(X,y)

loj_model
loj_model.intercept_
loj_model.coef_ 
y_pred= loj_model.predict(X)
confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)
loj_model.predict(X) [0:10]
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.30, 

                                                    random_state = 42)
loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X_train,y_train)

loj_model
cv_loj_score=cross_val_score(loj_model, X_test, y_test, cv = 10).mean()

cv_loj_score
knn= KNeighborsClassifier()
knn_model= knn.fit(X_train,y_train)

knn_model
cv_knn_score=cross_val_score(knn_model, X_test, y_test, cv = 10).mean()

cv_knn_score
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)
svc=linear_svc.score(X_test, y_test)

svc
from sklearn.preprocessing import StandardScaler  

scaler= StandardScaler()
scaler.fit(X_train)

X_train_scaled= scaler.transform(X_train)

X_test_scaled= scaler.transform(X_test)
X_train_scaled[0:5]
from sklearn.neural_network import MLPClassifier
mlpc= MLPClassifier().fit(X_train_scaled,y_train)
y_pred = mlpc.predict(X_test) 
mlpc=mlpc.score(X_test, y_test)

mlpc
rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model
y_pred = rf_model.predict(X_test) 
rf=rf_model.score(X_test, y_test)

rf
xgb_model = XGBClassifier().fit(X_train, y_train)
xgb_model
y_pred = xgb_model.predict(X_test) 
xgb=xgb_model.score(X_test, y_test)

xgb
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)

lgbm=lgbm_model.score(X_test, y_test)

lgbm
cv_light_score=cross_val_score(lgbm_model, X_test, y_test, cv = 10).mean()

cv_light_score
nb=GaussianNB()

nb_model=nb.fit (X_train,y_train)

nb_model
nb_model.predict(X_test)[0:10]
y_pred=nb_model.predict(X_test)
gnb=nb_model.score(X_test, y_test)

gnb
from sklearn.linear_model import Perceptron

perceptron = Perceptron(max_iter=5)

p_model=perceptron.fit(X_train, y_train)

p_model
y_pred=perceptron.predict(X_test)
per=p_model.score(X_test, y_test)

per
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train) 



y_pred = decision_tree.predict(X_test)  



dec=decision_tree.score(X_test, y_test)

dec
results = pd.DataFrame({

    'Model': ['SVC', "YSA", 'KNN', 'Logistic Regression', 

              'Random Forest', 'Decision Tree', "Light GBM", "GNB", "XGB","Perceptron"],

    'Score': [svc, mlpc, cv_knn_score, 

              cv_loj_score, rf, dec, 

              lgbm, gnb,xgb,per]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(10)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(xgb_model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(25)
importances.plot.bar();
df  = df.drop("max_visibilitymiles", axis=1)

df  = df.drop("mean_wind_speedmph", axis=1)
lgbm_model
lgbm_params = {

        'n_estimators': [100,500, 1000],

        'subsample': [0.6, 0.8, 1.0],

        'max_depth': [3, 5,8],

        'learning_rate': [0.1,0.01,0.05],

        "min_child_samples": [5,10,20]}
lgbm = LGBMClassifier()



lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 

                             cv = 10, 

                             n_jobs = -1, 

                             verbose = 2)
#lgbm_cv_model.fit(X_train, y_train)
#lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.05, 

                       max_depth = 10,

                       subsample = 0.8,

                       n_estimators = 500,

                       min_child_samples = 5)



lgbm_tuned = lgbm.fit(X_train,y_train)
lgbm_tuned.score(X_test, y_test)
