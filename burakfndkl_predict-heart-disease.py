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
df = pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")
df.head()
df.isnull().sum()
# Since the average and median values are close in most variables, I filled the nan values according to the median.



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

imputer.fit(df.iloc[:,:-1].values)

df.iloc[:,:-1] = imputer.transform(df.iloc[:,:-1].values)
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
#Corr matrix

plt.figure(figsize=(20,20))

cor = df.corr()

sns_heat=sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

fig=sns_heat.get_figure

plt.show()
df.hist(figsize=(20, 20))

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(data=df,palette='RdBu',orient='h')
X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

stdc = StandardScaler()

X_train = stdc.fit_transform(X_train)

X_test = stdc.transform(X_test)
from sklearn.metrics import roc_curve,roc_auc_score,classification_report,accuracy_score



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)



preds = lr.predict(X_test)

print(classification_report(y_test, preds))
import numpy as np





def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()

    

probs = lr.predict_proba(X_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)
cross_val_score(lr,X_test,y_test,cv=10).mean()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier().fit(X_train,y_train)

result = knn.predict(X_test)

print(classification_report(y_test, result))

#ROC

probs = knn.predict_proba(X_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)
#Knn model tuning

knn_params ={"n_neighbors": np.arange(1,10), "metric": ["euclidean","minkowski","manhattan"]}

knn_cv_model = GridSearchCV(knn,knn_params, cv=10,n_jobs=-1, verbose=2).fit(X_train,y_train)

knn_cv_model.score(X_test,y_test)
knn_cv_model.best_params_
#Knn tuned model

knn_tuned = KNeighborsClassifier(metric='euclidean',n_neighbors=8).fit(X_train,y_train)



result = knn_tuned.predict(X_test)



print(classification_report(y_test, result))

#ROC

probs = knn_tuned.predict_proba(X_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)
from sklearn.svm import SVC

svm_model = SVC().fit(X_train,y_train)

result = svm_model.predict(X_test)

print(classification_report(y_test, result))
#svm model tuning

svm = SVC() 

svm_params ={"C": [0.1,1,2,3],"kernel": ["rbf","poly"],"degree" : [0, 1, 2,3,4]}

svm_cv_model = GridSearchCV(svm,svm_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_
#svm tuned model

svm_tuned = SVC(C=2, kernel= "poly",degree=3).fit(X_train,y_train)

result = svm_tuned.predict(X_test)

print(classification_report(y_test, result))
from sklearn.neural_network import MLPClassifier



mlpc_model = MLPClassifier(activation="logistic").fit(X_train,y_train)

result = mlpc_model.predict(X_test)

accuracy_score(y_test,result)
mlpc_params={"alpha": [0.1,0.01,0.03],"hidden_layer_sizes": [(10,10),(100,100,100),(3,5)],"solver": ["lbfgs","adam"]}
#mlpc tuning

mlpc=MLPClassifier()

mlpc_cv_model = GridSearchCV(mlpc,mlpc_params ,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
mlpc_cv_model.best_score_
mlpc_cv_model.best_params_
#mlpc tuned

mlpc_tuned = MLPClassifier(activation="logistic",alpha=0.1,hidden_layer_sizes=(3,5),solver="adam").fit(X_train,y_train)

result = mlpc_tuned.predict(X_test)

accuracy_score(y_test,result)
from sklearn.tree import DecisionTreeClassifier

cart_model = DecisionTreeClassifier().fit(X_train,y_train)

result = cart_model.predict(X_test)

accuracy_score(y_test,result)
cart = DecisionTreeClassifier()

cart_params={"max_depth": [1,3,5,7],"min_samples_split": [2,3,6,10,15,20]}
cart_cv_model = GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeClassifier(max_depth=1,min_samples_split=2).fit(X_train,y_train)
result = cart_tuned.predict(X_test)

accuracy_score(y_test,result)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf_model = RandomForestClassifier().fit(X_train,y_train)
result = rf_model.predict(X_test)

accuracy_score(y_test,result)
#rf model tuning

rf = RandomForestClassifier()

rf_params = {"n_estimators": [100,200,500,1000],

            "max_features": [1,3,5,7],

            "min_samples_split": [2,5,7]}

rf_cv_model = GridSearchCV(rf,rf_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)

rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(max_features= 1,min_samples_split= 7,n_estimators=200).fit(X_train,y_train)

result = rf_tuned.predict(X_test)

accuracy_score(y_test,result)
from sklearn.ensemble import GradientBoostingClassifier

gbm_model = GradientBoostingClassifier().fit(X_train,y_train)
result = gbm_model.predict(X_test)

accuracy_score(y_test,result)
#GBM model tuning

gbm = GradientBoostingClassifier()

gbm_params = {"learning_rate": [0.1,0.001,0.05],

              "n_estimators": [100,500,1000],

              "max_depth": [2,3,5,8]}

gbm_cv_model= GridSearchCV(gbm,gbm_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingClassifier(learning_rate=0.001,max_depth=2,n_estimators=100).fit(X_train,y_train)

result = gbm_tuned.predict(X_test)

accuracy_score(y_test,result)
from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(X_train,y_train)
result = xgb_model.predict(X_test)

accuracy_score(y_test,result)
xgb = XGBClassifier()

xgb_params = {"learning_rate": [0.1,0.001,0.05],

              "n_estimators": [100,500,1000,2000],

              "max_depth": [2,3,5,8]}
xgb_cv_model = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBClassifier(learning_rate= 0.1,max_depth= 2,n_estimators= 100).fit(X_train,y_train)
result = xgb_tuned.predict(X_test)

accuracy_score(y_test,result)
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier().fit(X_train,y_train)

lgbm_model
result = lgbm_model.predict(X_test)

accuracy_score(y_test,result)
lgbm = LGBMClassifier()

lgbm_params= {"learning_rate": [0.1,0.001,0.05],

              "n_estimators": [100,500,1000],

              "max_depth": [2,3,5,8]}

lgbm_cv_model = GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
result = lgbm_model.predict(X_test)

accuracy_score(y_test,result)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMClassifier(learning_rate=0.1,max_depth=3,n_estimators=100).fit(X_train,y_train)

result = lgbm_model.predict(X_test)

accuracy_score(y_test,result)