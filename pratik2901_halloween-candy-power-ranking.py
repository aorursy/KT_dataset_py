%load_ext autoreload
%autoreload 2

%matplotlib inline

import pandas as pd
from pandas_summary import DataFrameSummary
from IPython.display import display
import matplotlib.pyplot as plt

from fastai.imports import *
from fastai.structured import *

from scipy.stats import spearmanr

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
data = pd.read_csv('../input/candy-data.csv')
display(data.T)
display_all(data.describe(include='all').T)
data.sugarpercent = round(data.sugarpercent,3)
data.pricepercent = round(data.pricepercent,3)
data.winpercent = round(data.winpercent,3)
data.info()
plt.figure(figsize=(20,10))
sns.heatmap(data.corr().abs(),annot=True)
def corrank(X):
    import itertools
    df = pd.DataFrame([[i,j,X.corr().abs().loc[i,j]] for i,j in list(itertools.combinations(X.corr().abs(), 2))],columns=['Feature1','Feature2','corr'])    
    return df.sort_values(by='corr',ascending=False).reset_index(drop=True)

# prints a descending list of correlation pair (Max on top)
display_all(corrank(data))
data.isnull().sum()/len(data)
winners = data[data.winpercent>data.winpercent.quantile(.6)]
from mlxtend.frequent_patterns import apriori
df =  winners[data.columns[1:-3]]
association = apriori(df, min_support=0.3,use_colnames=True).sort_values(by='support')


association.plot(kind='barh',x='itemsets',y='support',title=f'Most Frequently Used Composition',sort_columns=True,figsize = (10,5),legend=False)
sns.boxplot(x="chocolate", y="winpercent", data=winners).set_title('Relation of Chocolate and Win Percent');
sns.boxplot(x="fruity", y="winpercent", data=winners).set_title('Relation of Fruity and Win Percent');
sns.boxplot(x="caramel", y="winpercent", data=winners).set_title('Relation of Caramel and Win Percent');
sns.boxplot(x="peanutyalmondy", y="winpercent", data=winners).set_title('Relation of Peanut/Almond and Win Percent');
sns.boxplot(x="nougat", y="winpercent", data=winners).set_title('Relation of Nougat and Win Percent');
sns.boxplot(x="crispedricewafer", y="winpercent", data=winners).set_title('Relation of Wafer and Win Percent');
sns.boxplot(x="hard", y="winpercent", data=winners).set_title('Relation of Hardness of Candy and Win Percent');
sns.boxplot(x="bar", y="winpercent", data=winners).set_title('Relation of Bar and Win Percent');
sns.boxplot(x="pluribus", y="winpercent", data=winners).set_title('Relation of Pluribus and Win Percent');
sns.jointplot(x="sugarpercent", y="winpercent", data=winners,kind="kde",stat_func=spearmanr)
sns.jointplot(x="pricepercent", y="winpercent", data=winners,kind="kde",stat_func=spearmanr)
popularity = data[['competitorname','winpercent']].sort_values(by='winpercent')
pd.concat([popularity.head(5),popularity.tail(5)],axis=0).plot(x='competitorname',y='winpercent',kind='barh',title='Popularity of various candies',sort_columns=True,figsize = (10,5),legend=False)
association.plot(kind='barh',x='itemsets',y='support',title=f'Most Frequently Used Composition',sort_columns=True,figsize = (10,5),legend=False)
from sklearn import tree
reg = tree.DecisionTreeRegressor(max_depth=3).fit(data[data.columns[1:-1]],data[data.columns[-1]])
imp = pd.DataFrame.from_dict({'Name':data.columns[1:-1],'Importance':reg.feature_importances_})
imp_plt = imp.sort_values(by='Importance',ascending=True).reset_index(drop=True)
imp_plt[imp_plt.Importance>0].plot(kind='barh',x='Name',y='Importance',title='Feature Importance',sort_columns=True,figsize = (10,5),legend=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[1:-1]],data[data.columns[-1]], test_size=0.33, random_state=42)
rmse_err = []
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=5).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))
rmse_err[-1]
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=200).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,rf_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import LinearRegression
lr_reg = LinearRegression().fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,lr_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,sgd_reg.predict(X_test))))
rmse_err[-1]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)

lg = LinearRegression().fit(X_, y_train)

rmse_err.append(math.sqrt(mean_squared_error(y_test,lg.predict(X_test_))))
rmse_err[-1]
from sklearn.linear_model import Ridge
r_reg = Ridge(alpha = .5).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,r_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import Lasso
l_reg = Lasso(alpha = 0.1).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,l_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import LassoLars
ll_reg = LassoLars(alpha = 0.1).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,ll_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import BayesianRidge
b_reg = BayesianRidge().fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,b_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import PassiveAggressiveRegressor
par_reg = PassiveAggressiveRegressor(max_iter=1000, random_state=0).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,par_reg.predict(X_test))))
rmse_err[-1]
from sklearn.linear_model import ElasticNet
en_reg = ElasticNet(max_iter=1000, random_state=0).fit(X_train,y_train)
rmse_err.append(math.sqrt(mean_squared_error(y_test,en_reg.predict(X_test))))
rmse_err[-1]
models = ['Decision Tree','RandomForest','Linear','SGD','Polynomial','Ridge','Lasso','LassoLars','Bayesian Ridge','Passive Aggressive','ElasticNet']
pd.DataFrame.from_dict({'Name':models,'RMSE':rmse_err}).sort_values(by='RMSE',ascending=False).plot(x='Name',y='RMSE',kind='barh',sort_columns=True,figsize = (10,5),legend=False,title='Performance of various Regression based algos')