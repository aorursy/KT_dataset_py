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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df_train = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
df_train.index
df_test = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
print(df_train.shape)
df_train.head()
df_train.columns
df_train.price_range.value_counts()
print(df_train.isnull().sum())
df_train.drop(axis=1,columns = ["price_range"]).corrwith(df_train["price_range"]).plot(kind="barh",grid=True,xlim = (-1,1),xticks = np.linspace(start=-1,stop=1,num=21),figsize=(10,8),fontsize=12)
plt.figure(figsize=(10,10))
sns.boxplot(y = df_train["ram"],x = df_train['price_range'])
plt.figure(figsize=(10,10))
sns.swarmplot(y = df_train["ram"],x = df_train['price_range'])
plt.figure(figsize=(20,15))
sns.heatmap(df_train.corr(),cmap="cividis",annot=True,center=0.5,linewidths=.5,cbar_kws={"orientation": "horizontal"})

df_train.dtypes.unique()
def column_type(df):
    categorical_column = []
    continuous_column = []
    for col in df.columns:
        if len(df[col].unique())<=10:
            categorical_column.append(col)
        else:
            continuous_column.append(col)
    return [categorical_column,continuous_column]
            
            
        
cat_col, cont_col = column_type(df_train.drop(columns=["price_range"],axis=1))
cat_col
cont_col
sns.pairplot(df_train, x_vars = cat_col,y_vars = ["price_range"],hue ="price_range",kind="scatter")
#fig,ax = plt.subplots(figsize=(10,6))
#plt.figure(figsize=(15,8))

sns.pairplot(df_train, x_vars = cont_col[:6],y_vars = ["price_range"],height = 5,aspect=0.5,hue = "price_range")
sns.pairplot(df_train, x_vars = cont_col[6:],y_vars = ["price_range"],height = 5,aspect=0.5,hue = "price_range")
# g.fig.set_figwidth(25)
# g.fig.set_figheight(5)
X = df_train.drop(axis=1,columns = ["price_range"])
y = df_train.price_range
X_ = pd.get_dummies(X,columns=cat_col)
print(X_.shape)
X_.head()
x_train,x_test,y_train,y_test = train_test_split(X_,y,train_size=0.8, random_state = 1)
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression(solver="liblinear",random_state=42,C=1)
model_1.fit(x_train,y_train)
print(model_1.score(x_train,y_train)*100)
print(model_1.score(x_test, y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
model_2 = KNeighborsClassifier(n_neighbors = 5,weights='distance')
model_2.fit(x_train,y_train)
print(model_2.score(x_train,y_train)*100)
print(model_2.score(x_test, y_test)*100)
from sklearn.naive_bayes import GaussianNB
model_3 = GaussianNB()
model_3.fit(x_train,y_train)
print(model_3.score(x_train,y_train)*100)
print(model_3.score(x_test, y_test)*100)
from sklearn.svm import SVC
model_4 = SVC(kernel = "rbf",C=0.9)
model_4.fit(x_train,y_train)
print(model_4.score(x_train,y_train)*100)
print(model_4.score(x_test, y_test)*100)
from sklearn.tree import DecisionTreeClassifier
model_5 = DecisionTreeClassifier(random_state=42,max_depth=7,criterion="gini",splitter = "best")
model_5.fit(x_train,y_train)
print(model_5.score(x_train,y_train)*100)
print(model_5.score(x_test, y_test)*100)
model_5.get_params()
from sklearn.ensemble import RandomForestClassifier
model_6 = RandomForestClassifier(max_depth = 10,n_estimators=250,criterion='gini',random_state=42)
model_6.fit(x_train,y_train)
print(model_6.score(x_train,y_train)*100)
print(model_6.score(x_test, y_test)*100)
from sklearn.ensemble import GradientBoostingClassifier
model_7 = GradientBoostingClassifier(random_state=42)
model_7.fit(x_train,y_train)
print(model_7.score(x_train,y_train)*100)
print(model_7.score(x_test, y_test)*100)
from xgboost import XGBClassifier
model_8 = XGBClassifier()
model_8.fit(x_train,y_train)
print(model_8.score(x_train,y_train)*100)
print(model_8.score(x_test, y_test)*100)
from sklearn.model_selection import GridSearchCV
params = {"C":np.linspace(0,10,11), "solver":["liblinear","lbfgs","newton-cg"],"max_iter":[200]}

#scoring= ['accuracy', 'precision','recall']
log_reg = LogisticRegression()
gridsearch = GridSearchCV(estimator=log_reg,param_grid = params,scoring = "accuracy",n_jobs = -1,cv = 5)
gridsearch.fit(x_train,y_train)
gridsearch.best_estimator_
          
gridsearch.best_params_
gridsearch.best_score_
tuned_log_reg = LogisticRegression(C= 1.0, max_iter= 200, solver= 'newton-cg')
tuned_log_reg.fit(x_train,y_train)
print(tuned_log_reg.score(x_train,y_train)*100)
print(tuned_log_reg.score(x_test, y_test)*100)

params = {"C":[0.01,0.1,1,3,5,8,10,18,25],"gamma":[0.05,0.1,0.5,0.75,1],"kernel":['linear','rbf','sigmoid']}#,"degree":[1,2,3,4,5],"coef0":[0.001,0.05,0.1,1,2,3,4,5]}
svc = SVC()
gridsearch = GridSearchCV(estimator=svc,param_grid = params,scoring = "accuracy",n_jobs = -1,cv = 5,iid = True)
gridsearch.fit(x_train,y_train)
# print(gridsearch.best_estimator_)
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)

tuned_svc_1 = SVC(C=8, gamma=0.05, kernel='linear')
tuned_svc_1.fit(x_train,y_train)

print(tuned_svc_1.score(x_train,y_train)*100)

print(tuned_svc_1.score(x_test, y_test)*100)
params = {"gamma":[0.05,0.1,0.5,0.75,1],"kernel":['poly'],"degree":[1,2,3],"coef0":[0.001,0.05,0.1,1,2,4,5]}
svc = SVC()
gridsearch = GridSearchCV(estimator=svc,param_grid = params,scoring = "accuracy",n_jobs = -1,cv = 5,iid = True)
gridsearch.fit(x_train,y_train)
# print(gridsearch.best_estimator_)
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)
tuned_svc = SVC(coef0=0.001, degree=1, gamma=1, kernel='poly')
tuned_svc.fit(x_train,y_train)

print(tuned_svc.score(x_train,y_train)*100)

print(tuned_svc.score(x_test, y_test)*100)
params = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(1, 20))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(1, 20))
          }

tree = DecisionTreeClassifier(random_state=42)
gridsearch = GridSearchCV(tree, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3, iid=True)
gridsearch.fit(x_train,y_train)
# print(gridsearch.best_estimator_)
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)
tuned_dtc = DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, random_state=42)
tuned_dtc.fit(x_train,y_train)

print(tuned_dtc.score(x_train,y_train)*100)

print(tuned_dtc.score(x_test, y_test)*100)
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rand_forest = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(estimator=rand_forest, param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(x_train,y_train)
# # print(rf_random.best_estimator_)
# # print(rf_random.best_params_)
# # print(rf_random.best_score_)
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
learning_rate = [0.001,0.1,0.2,0.5,0.75,1]

params = {'learning_rate': learning_rate,'n_estimators': n_estimators, 'max_features': max_features}
               #'max_depth': max_depth}# 'min_samples_split': min_samples_split,
              # 'min_samples_leaf': min_samples_leaf}
gbc = GradientBoostingClassifier()

gridsearch = GridSearchCV(gbc, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3, iid=True)
gridsearch.fit(x_train,y_train)
# print(gridsearch.best_estimator_)
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)


tuned_gbc = GradientBoostingClassifier(learning_rate=1, max_features='auto',
                           n_estimators=200,validation_fraction=0.15)
tuned_gbc.fit(x_train,y_train)

print(tuned_gbc.score(x_train,y_train)*100)

print(tuned_gbc.score(x_test, y_test)*100)

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 0.99]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight = [1, 2, 3, 4]

hyperparameter_grid = {'n_estimators': [200,500], 'max_depth': [5,10,15],
                       'learning_rate' : learning_rate}#, 'min_child_weight' : min_child_weight, 
                       #'booster' : booster, 'base_score' : base_score
                      #}

xgb_model = XGBClassifier()

xgb_cv = RandomizedSearchCV(estimator=xgb_model, param_distributions=hyperparameter_grid,
                               cv=5, n_iter=650, scoring = 'accuracy',n_jobs =-1, iid=True,
                               verbose=1, return_train_score = True, random_state=42)
xgb_cv.fit(x_train, y_train)
print(xgb_cv.best_estimator_)
print(xgb_cv.best_params_)
print(xgb_cv.best_score_)

tuned_xgb = XGBClassifier(n_estimators= 500, max_depth = 5, learning_rate= 0.2)
tuned_xgb.fit(x_train,y_train)

print(tuned_xgb.score(x_train,y_train)*100)

print(tuned_xgb.score(x_test, y_test)*100)

