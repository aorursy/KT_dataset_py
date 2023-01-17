import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')
diabetes = pd.read_csv("../input/docspot/datasets_228_482_diabetes.csv")
data = diabetes.copy()
data = data.dropna()
data.head()
data.info()
data.describe().T
data["Outcome"].value_counts()
data["Outcome"].value_counts().plot.barh();
y = data["Outcome"]
X = data.drop(["Outcome"], axis=1)
log = sm.Logit(y, X)
log_model = log.fit()
log_model.summary()
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(solver="liblinear")
log_model = log.fit(X,y)
log_model
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)
confusion_matrix(y, y_pred)
accuracy_score(y,y_pred)
print(classification_report(y,y_pred))
log_model.predict(X)[:5]
log_model.predict_proba(X)[:5]
y[:5]
y_probs = log_model.predict_proba(X)
y_probs = y_probs[:,1]
y_probs[:5]
y_pred = [1 if i > 0.5 else 0 for i in y_probs]
y_pred[:5]
confusion_matrix(y, y_pred)
accuracy_score(y,y_pred)
print(classification_report(y,y_pred))
logit_roc_auc = roc_auc_score(y, log_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, log_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.title('ROC')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
log = LogisticRegression(solver = "liblinear")
log_model = log.fit(X_train,y_train)
log_model
accuracy_score(y_test,log_model.predict(X_test))
cross_val_score(log_model,X_test, y_test,cv=10).mean()
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
nb = GaussianNB()
nb_model = nb.fit(X_train,y_train)
nb_model
nb_model.predict(X_test)[:10]
nb_model.predict_proba(X_test)[:10]
y_pred = nb_model.predict(X_test)
accuracy_score(y_test,y_pred)
cross_val_score(nb_model,X_test, y_test,cv=10).mean()
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
cross_val_score(knn_model,X_test, y_test,cv=10).mean()
knn_params = {"n_neighbors": np.arange(50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,knn_params, cv=10)
knn_cv.fit(X_train,y_train)
print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_["n_neighbors"]))
knn = KNeighborsClassifier(11)
knn_tuned =knn.fit(X_train,y_train)
knn_tuned.score(X_test,y_test)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
svc_model = SVC(kernel="linear").fit(X_train,y_train)
svc_model
y_pred = svc_model.predict(X_test)
accuracy_score(y_test,y_pred)
svc_params = {"C": np.arange(1,10)}

svc = SVC(kernel="linear")
svc_cv_model = GridSearchCV(svc, svc_params, cv=10, n_jobs=-1,verbose=2)

svc_cv_model.fit(X_train,y_train)
print("Best Parameters: " + str(svc_cv_model.best_params_["C"]))
svc_tuned = SVC(kernel="linear", C=5).fit(X_train,y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
svc_model = SVC(kernel="rbf").fit(X_train,y_train)
svc_model
y_pred =svc_model.predict(X_test)
accuracy_score(y_test,y_pred)
svc_params = {
    "C": [0.0001,0.001,0.01,0.1,5,10,50,100,200],
    "gamma": [0.0001,0.001,0.01,0.1,5,10,50,100,200]
}
svc = SVC()
svc_cv_model = GridSearchCV(svc,svc_params,cv = 10,n_jobs = -1, verbose=2)
svc_cv_model.fit(X_train,y_train)
print("Best Parameters: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(C=10, gamma =0.0001).fit(X_train,y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlpc = MLPClassifier().fit(X_train_scaled,y_train)
mlpc.coefs_
y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
mlpc_params = {
    "alpha": [0.1,0.01,0.02,0.005,0.001,0.0001,0.00001],
    "hidden_layer_sizes": [(10,10,10),
                           (100,100,100),
                           (100,100),
                           (3,5),
                           (5,3)],
    "solver": ["lbfgs","adam","sgd"],
    "activation": ["relu","logistic"]
}
mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs = -1, verbose=2)
mlpc_cv_model.fit(X_train_scaled,y_train)
print("Best Parameters: " + str(mlpc_cv_model.best_params_))
mlpc_tuned = MLPClassifier(alpha = 0.001 , hidden_layer_sizes = (100, 100, 100), solver = "sgd", activation = "relu")
mlpc_tuned.fit(X_train_scaled,y_train)
y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train,y_train)
cart_model
x = [3]
y_pred = cart_model.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
cart_grid = {
    "max_depth": list(range(10)),
    "min_samples_split": range(2,50),
}
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart,cart_grid, cv=10, n_jobs=-1,verbose=2)
cart_cv_model = cart_cv.fit(X_train,y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
cart = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 19)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
rf_params = {
    "max_depth": [2,5,8,10],
    "max_features": [5,8,10],
    "n_estimators": [10,500,1000,2000],
    "min_samples_split": [2,5,10]
}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model, rf_params,cv=10, n_jobs=-1, verbose=2)
rf_cv_model.fit(X_train,y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 10,max_features = 8,n_estimators = 1000,min_samples_split = 8)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable Significance Levels");
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
gbc_model = GradientBoostingClassifier().fit(X_train,y_train)
y_pred = gbc_model.predict(X_test)
accuracy_score(y_test, y_pred)
gbm_params = {
    "learning_rate":[0.001,0.01,0.1,0.05],
    "max_depth": [3,5,10],
    "n_estimators": [100,500,1000],
    "min_samples_split": [2,5,10]
    }
gbm = GradientBoostingClassifier()
gbm_cv = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)
gbm_cv.fit(X_train,y_train)
print("Best Parameters: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate=  0.1,max_depth = 3,n_estimators = 100,min_samples_split = 5)
gbm_tuned = gbm.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
xgbm_model = XGBClassifier().fit(X_train,y_train)
y_pred = xgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
xgbm = XGBClassifier()
xgbm_params = {
        'n_estimators': [100, 500, 1000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01,0.02,0.05],
        "min_samples_split": [2,5,10]}

xgbm_cv_model = GridSearchCV(xgbm,xgbm_params, cv=10,n_jobs=-1,verbose=2)
xgbm_cv_model.fit(X_train,y_train)
print("Best Parameters: " + str(xgbm_cv_model.best_params_))
xgb = XGBClassifier(learning_rate = 0.02, 
                    max_depth = 3,
                    min_samples_split = 2,
                    n_estimators = 100,
                    subsample = 0.6)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
lgbm_model = LGBMClassifier().fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}
lgbm = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose = 2)

lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.01, 
                       max_depth = 3,
                       subsample = 0.6,
                       n_estimators = 500,
                       min_child_samples = 20)
lgbm_tuned = lgbm.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
catb_model = CatBoostClassifier().fit(X_train,y_train)
y_pred = catb_model.predict(X_test)
accuracy_score(y_test, y_pred)
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.05, 0.1],
    'depth': [3,5,8] }
catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb = CatBoostClassifier(iterations = 200, 
                          learning_rate = 0.01, 
                          depth = 8)

catb_tuned = catb.fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
models = [
    knn_tuned,
    log_model,
    svc_tuned,
    nb_model,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    catb_tuned,
    lgbm_tuned,
    xgb_tuned
    
]


for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
    
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratio of Models');    

