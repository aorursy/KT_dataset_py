import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from scipy import stats

from scipy.stats import norm,skew



from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone



import xgboost as xgb



import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

data.head()
data.drop("car name",inplace=True,axis=1)
data = data.rename(columns = {"mpg":"target"})
data.shape
data.info()
data['horsepower']=data['horsepower'].replace('?','150')

data['horsepower']=data['horsepower'].astype('int')

'?' in data['horsepower']
data.describe().T
data.isnull().sum()
sns.distplot(data.horsepower);
data_corr = data.corr()
sns.clustermap(data_corr,annot=True);

plt.title("Correlation Between Features");

plt.show();
threshold = 0.75

filter_ = np.abs(data_corr["target"]) > threshold

corr_features = data_corr.columns[filter_].tolist()



sns.heatmap(data[corr_features].corr(),annot=True,fmt=".2f");

plt.title("Correlation Between Features");

plt.show();
# result: multicollinearity
sns.pairplot(data,diag_kind="kde",markers="+");

plt.show();
#cylinders and origin can be categorical
plt.figure();

sns.countplot(data["cylinders"]);
print(data["cylinders"].value_counts());
plt.figure();

sns.countplot(data["origin"]);
print(data["origin"].value_counts())
for i in data.columns:

    plt.figure();

    sns.boxplot(x=i,data=data,orient="v")
th = 2



Q1_hp = data.horsepower.quantile(0.25)

Q3_hp = data.horsepower.quantile(0.75)

IQR_hp = Q3_hp - Q1_hp

print(IQR_hp)



top_limit_hp = Q3_hp + th * IQR_hp

top_limit_hp



bottom_limit_hp = Q1_hp - th * IQR_hp

bottom_limit_hp



filter_hp_bottom = bottom_limit_hp < data.horsepower

filter_hp_top = data.horsepower < top_limit_hp

filter_hp = filter_hp_bottom & filter_hp_top



data = data[filter_hp]

data.shape
th = 2



Q1_ac = data.acceleration.quantile(0.25)

Q3_ac = data.acceleration.quantile(0.75)

IQR_ac = Q3_ac - Q1_ac

print(IQR_ac)



top_limit_ac = Q3_ac + th * IQR_ac

top_limit_ac

bottom_limit_ac = Q1_ac - th * IQR_ac

bottom_limit_ac



filter_ac_bottom = bottom_limit_ac < data.acceleration

filter_ac_top = data.acceleration < top_limit_ac

filter_ac = filter_ac_bottom & filter_ac_top



data = data[filter_ac]

data.shape
# target - dependent variable
sns.distplot(data.target,fit=norm);
(mu,sigma) = norm.fit(data["target"])

print("mu: {}, sigma: {}".format(mu,sigma))
# qq plot

plt.figure();

stats.probplot(data["target"],plot = plt);

plt.show();
data["target"] = np.log1p(data["target"])
plt.figure();

sns.distplot(data.target,fit=norm);
# qq plot

plt.figure();

stats.probplot(data["target"],plot = plt);

plt.show();
# feature - independent variable

skewed_features = data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame(skewed_features,columns=["skewed"])

skewness
# cylinders & origin
data["cylinders"] = data["cylinders"].astype(str) 

data["origin"] = data["origin"].astype(str) 
data = pd.get_dummies(data)
data.head()
x = data.drop(["target"],axis=1)

y = data.target
test_size = 0.9

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=42)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
lr = LinearRegression()

lr.fit(x_train,y_train)

print("LR Coef: ", lr.coef_)

y_pred = lr.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("LR MSE: ",mse)
ridge = Ridge(random_state=42,max_iter=10000)

alphas = np.logspace(-4,-0.5,30)

tuned_params = [{'alpha':alphas}]

n_folds = 5



clf = GridSearchCV(ridge,tuned_params, cv=n_folds,scoring="neg_mean_squared_error",refit=True)

clf.fit(x_train,y_train)

scores = clf.cv_results_["mean_test_score"]

scores_std = clf.cv_results_["std_test_score"]



print("Ridge Coef: ", clf.best_estimator_.coef_)

ridge = clf.best_estimator_



print("Ridge Best Estimator: ",ridge)

y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("Ridge MSE: ",mse)
plt.figure();

plt.semilogx(alphas,scores);

plt.xlabel("alpha");

plt.ylabel("score");

plt.title("Ridge(L2)");
lasso = Lasso(random_state=42,max_iter=10000)

alphas = np.logspace(-4,-0.5,30)

tuned_params = [{'alpha':alphas}]

n_folds = 5



clf = GridSearchCV(lasso,tuned_params, cv=n_folds,scoring="neg_mean_squared_error",refit=True)

clf.fit(x_train,y_train)

scores = clf.cv_results_["mean_test_score"]

scores_std = clf.cv_results_["std_test_score"]



print("Lasso Coef: ", clf.best_estimator_.coef_)

lasso = clf.best_estimator_



print("Lasso Best Estimator: ",lasso)

y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("Lasso MSE: ",mse)
plt.figure();

plt.semilogx(alphas,scores);

plt.xlabel("alpha");

plt.ylabel("score");

plt.title("Lasso(L1)");
parametersGrid = {"alpha": alphas,

                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}



eNet = ElasticNet(random_state=42, max_iter=10000)

clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)

clf.fit(x_train, y_train)





print("ElasticNet Coef: ",clf.best_estimator_.coef_)

print("ElasticNet Best Estimator: ",clf.best_estimator_)





y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("ElasticNet MSE: ",mse)
parametersGrid = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'learning_rate': [.03, 0.05, .07], 

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500,1000]}



model_xgb = xgb.XGBRegressor()



clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)



clf.fit(x_train, y_train)

model_xgb = clf.best_estimator_



y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("XGBRegressor MSE: ",mse)
class AveragingModels():

    def __init__(self,models):

        self.models = models

    

    def fit(self,x,y):

        self.models_ = [clone(x) for x in self.models]

        

        for model in self.models_:

            model.fit(x,y)   

        return self

    

    def predict(self,x):

        predictions = np.column_stack([model.predict(x) for model in self.models_])

        return np.mean(predictions,axis=1)
averaged_models = AveragingModels(models = (model_xgb, lasso))

averaged_models.fit(x_train, y_train)
y_pred = averaged_models.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print("Averaged Models MSE: ",mse)