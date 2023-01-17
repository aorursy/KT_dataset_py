import pandas as pd
data = pd.read_csv('/kaggle/input/dataset/Life_Expectancy_Data.csv')
data.head()
import matplotlib.pyplot as plt
plt.hist(data['Life_expectancy'])
plt.show()
print('Shape of Data {}'.format(data.shape))
import seaborn as sns
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

for i in range(len(corrmat.columns)):
  for j in range(len(corrmat.index)):
    if corrmat.iloc[i,j]>0.80 and corrmat.iloc[i,j] != 1.0:
      print('Multi-Collinearity Feature {} and Feature {} --> Correlation Score {}'.format(corrmat.columns[i],corrmat.columns[j],corrmat.iloc[i,j]))
data = data.drop(['thinness_5-9 years','GDP','infant_deaths'],axis=1)
corrmat['Life_expectancy']


fig, ax = plt.subplots()
ax.scatter(x = data['Adult_Mortality'], y = data['Life_expectancy'])
plt.ylabel('Life_expectancy', fontsize=13)
plt.xlabel('Adult_Mortality', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = data['Schooling'], y = data['Life_expectancy'])
plt.ylabel('Life_expectancy', fontsize=13)
plt.xlabel('Schooling', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = data['Income_composition_of_resources'], y = data['Life_expectancy'])
plt.ylabel('Life_expectancy', fontsize=13)
plt.xlabel('Income_composition_of_resources', fontsize=13)
plt.show()
data = data.drop(data[data['Income_composition_of_resources']<0.2].index)
data = data.drop(data[data['Schooling']<2].index)
data = data.drop(data[data['Adult_Mortality']<80].index)
null_cols=[]
for col in data.columns:
  if data[col].isnull().sum() !=0:
    print('{} ---- null values : {} ---- data type : {}'.format(col, data[col].isnull().sum(), type(data[col][0])))
    null_cols.append(col)
data = data.fillna(data.mean())
data.head()
data = pd.get_dummies(data,columns=['Status','Country'])
y = data['Life_expectancy']
x = data.drop(['Life_expectancy'],axis=1)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
#)
!pip install catboost
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV,Lasso,LassoCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings

lr=0.1
n = 200
kf = KFold(n_splits=20, random_state=42, shuffle=True)
svr_model = SVR(C=1)
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor(n_estimators=n)
adaboost_model = AdaBoostRegressor(n_estimators=n,learning_rate=lr)
gradientboost_model = GradientBoostingRegressor(learning_rate=lr,n_estimators = n)
catboost_model = CatBoostRegressor(learning_rate = lr,iterations = n, depth=3,loss_function='RMSE',verbose=0)
ridge_model = Ridge(alpha=0.1)
ridge_cv_model = RidgeCV(alphas=[0.1,0.01,0.001,1],cv=10)
lasso_model = Lasso(alpha=0.1)
lasso_cv_model = LassoCV(alphas=[0.1,0.01,0.001,1],cv=10)






#Params for each model are adjusted using GridSearchCV hyperparameter tuning
#illustration

# random_forest_regressor = AdaBoostRegressor()
# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_estimators':[100,200,300,400,500,800,1000],'learning_rate':[1,0.1,0.001,0.001]}
# grid_search = GridSearchCV(estimator = random_forest_regressor, param_grid = param_grid, 
#                         cv = 5, n_jobs = -1, verbose = 2)
# grid_search.fit(xtrain,ytrain)
# grid_search.best_params_
# best_grid = grid_search.best_estimator_
# ypred = best_grid.predict(xtest)
# metrics(ytest,ypred)


def metrics(ytest,ypred):
  return np.sqrt(mean_squared_error(ytest,ypred))
result = pd.DataFrame([],columns=['Model','CV_rmse','Prediction_rmse'])
def compute(model,i):
  cv_rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf))
  model.fit(xtrain,ytrain)
  ypred = model.predict(xtest)
  result.loc[i] = [str(model)[:str(model).index('(')] ,cv_rmse.mean(),metrics(ytest,ypred) ]


models = [svr_model,decision_tree_model,random_forest_model,adaboost_model,gradientboost_model,ridge_model,ridge_cv_model,lasso_model,lasso_cv_model]
for model in range(len(models)):
  compute(models[model],model)
result = result.sort_values('CV_rmse')
warnings.filterwarnings("ignore")
result
catboost_model.fit(xtrain, ytrain)
ypred = catboost_model.predict(xtest)
warnings.filterwarnings("ignore")
print('CatBoost Regressor RMSE {}'.format(metrics(ytest,ypred)))
from mlxtend.regressor import StackingCVRegressor
stack_gen = StackingCVRegressor(regressors=(decision_tree_model,random_forest_model,catboost_model,adaboost_model,gradientboost_model,ridge_model,ridge_cv_model,lasso_model,lasso_cv_model),
                                meta_regressor=ridge_model,
                                use_features_in_secondary=True,cv=30)
stack_gen.fit(np.array(xtrain),np.array(ytrain))
ypred = stack_gen.predict(np.array(xtest))
warnings.filterwarnings("ignore")
print('StackingCV Regressor RMSE {}'.format(metrics(ytest,ypred)))
