import pandas as pd

import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])
dataset.info()
dataset.drop(['id','date'],axis=1,inplace=True)
dataset.isnull().any()
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x = 'price', data = dataset, orient = 'h',  
                 fliersize = 3, showmeans=True, ax = ax)
plt.show()
import numpy as np

corrmat = dataset.corr()
cols = corrmat.nlargest(30, 'price').index
cm = np.corrcoef(dataset[cols].values.T)
plt.subplots(figsize=(16,12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.jointplot(x=dataset['sqft_living'], y = dataset['price'], kind='reg');
dataset = dataset.drop(dataset[dataset['sqft_living']>12500].index).reset_index(drop=True)
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=dataset['grade'], y=dataset['price'])
sns.jointplot(x=dataset['sqft_above'], y = dataset['price'], kind='reg');
f, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x=dataset['bathrooms'], y=dataset['price'])
locs, labels = plt.xticks()
plt.xticks(rotation=90);
dataset = dataset.drop(dataset[dataset['bathrooms']==7.5].index).reset_index(drop=True)
f, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x=dataset['view'], y=dataset['price']);
sns.jointplot(x=dataset['sqft_basement'], y = dataset['price'], kind='reg');
f, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x=dataset['bedrooms'], y=dataset['price'])
dataset = dataset.drop(dataset[dataset['bedrooms']==33].index).reset_index(drop=True)
f, ax = plt.subplots(figsize=(5, 5))
sns.boxplot(x=dataset['waterfront'], y=dataset['price']);
f, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x=dataset['floors'], y=dataset['price']);
dataset.info()
sns.jointplot(x=dataset['sqft_lot'], y=dataset['price'], kind='reg');
plt.subplots(figsize=(16, 8))
sns.boxplot(x=dataset['yr_built'], y=dataset['price'])
locs, labels = plt.xticks()
plt.xticks(locs[0:115:3],labels[0:115:3],rotation=90);
plt.subplots(figsize=(16, 8))
sns.boxplot(x=dataset['yr_built'], y=dataset['price'])
locs, labels = plt.xticks()
plt.xticks(locs[0:115:3],labels[0:115:3],rotation=90);
plt.subplots(figsize=(6, 6))
sns.boxplot(x=dataset['condition'], y=dataset['price']);
data = []
for x,y in zip(dataset['yr_built'],dataset['yr_renovated']):
    if y != 0:
        data.append(y)
    else:
        data.append(x)
data = pd.Series(data)
dataset['age'] = -(2015-data)
dataset['basement_existence'] = dataset['sqft_basement'].apply(lambda x: 1 if x>1 else 0)
for i in ('waterfront','view','condition','grade','basement_existence','zipcode'):
    dataset[i] = dataset[i].astype(str)

dataset = pd.get_dummies(dataset)
dataset.info()
corrmat = dataset.corr()
cols = corrmat.nlargest(30, 'price').index
dataset = dataset[cols]
dataset.info()
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
input_features = dataset.columns.tolist()
input_features.remove('price')

X = dataset[input_features]
y = dataset['price']
X_train, X_test,y_train,y_test = train_test_split(dataset[input_features],dataset['price'], train_size = 0.75, random_state = 20)
print(X_train.shape, X_test.shape)
def param_selector(estimator,params,scaler):
    if scaler:
        estimator = Pipeline([('scaler',StandardScaler()),('estimator',estimator)])
    cv = GridSearchCV(estimator,param,scoring="neg_mean_squared_error")
    cv.fit(X_train,y_train)
    score = np.sqrt(mean_squared_error(y_test,cv.predict(X_test)))
    print("RMSE error: {:.4f} ".format(score))
    print("Best parameters {}".format(cv.best_params_))
lasso = Lasso()
param = {'alpha': [0.05,0.1,0.5,1,5,10],'normalize': [True,False]}
param_selector(lasso,param,False)

lasso = Lasso(alpha=10,normalize=False)
param = {'alpha': [0.001,0.05,0.1,0.5,1,5,10],'normalize':[True,False]}
ridge = Ridge()
param_selector(ridge,param,False)
ridge = Ridge(alpha=1,normalize=False)
param = {'estimator__alpha': [0.005,0.05,0.1,1,0.01,10],'estimator__l1_ratio':[.1, .2, .8,.9]}
Enet = ElasticNet()
param_selector(Enet,param,True)
Enet = make_pipeline(StandardScaler(),ElasticNet(alpha=0.1,l1_ratio=0.9))
param = {'estimator__n_neighbors': list(range(14,16)),'estimator__weights':['distance']}
knn = KNeighborsRegressor()
param_selector(knn,param,True)
knn = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=14,weights='distance'))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

GBoost.fit(X_train,y_train)
score = np.sqrt(mean_squared_error(y_test,GBoost.predict(X_test)))
print("score: {:.4f} \n".format(score))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
import itertools

for j in range(1,5):
    for i in itertools.combinations([lasso,ridge,knn,Enet],j):
        averaged_models = AveragingModels(i)
        averaged_models.fit(X_train,y_train)
        score = np.sqrt(mean_squared_error(y_test,averaged_models.predict(X_test)))
        print("score: {:.4f} ".format(score))
averaged_models = AveragingModels(models = (lasso,knn))
averaged_models.fit(X_train,y_train)
score = np.sqrt(mean_squared_error(y_test,averaged_models.predict(X_test)))
print("RMSE error: {:.4f}".format(score))
Enet.fit(X_train,y_train)
score = np.sqrt(mean_squared_error(y_test,averaged_models.predict(X_test)*0.3+GBoost.predict(X_test)*0.7))
print("RMSE error: {:.4f}".format(score))