%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.svm import SVC
import warnings 
warnings.filterwarnings("ignore")
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
dt=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.head(5)
df.info()
df.describe()
categorical_features = df.select_dtypes(include=['object']).columns
print('Categorical')
print(categorical_features)
print('Numerical')
numerical_features = df.select_dtypes(exclude = ["object"]).columns
print(numerical_features)
df_num=df[numerical_features]
df_cat=df[categorical_features]
sns.distplot(df.skew(),color='red',axlabel ='Skewness')
plt.figure(figsize = (12,8))
sns.distplot(df.kurt(),color='blue',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
plt.show()
%matplotlib inline
import scipy.stats as st
y=df['SalePrice']
plt.style.use('fivethirtyeight')
plt.figure(1);
plt.title('johnson su')
sns.distplot(y,kde=False,fit=st.johnsonsu)
plt.figure(2);
plt.title('normal distrbutation')
sns.distplot(y,kde=False,fit=st.norm)
plt.figure(3);
plt.title('log normal')
sns.distplot(y,kde=False,fit=st.lognorm)
import seaborn as sns
plt.figure(figsize=(12,12), dpi= 80)
sns.heatmap(df.corr(), cmap='RdYlGn', center=0)

# Decorations
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
df_corr=df.corr()['SalePrice'][:-1]
feature_list=df_corr[abs(df_corr)>0.5].sort_values(ascending=False)
feature_list
year_feature = [feature for feature in df_num if 'Yr' in feature or 'Year' in feature]
discrete_feature=[feature for feature in df_num if len(df[feature].unique())<25 and feature not in year_feature+['Id']]
for feature in discrete_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_feature=[feature for feature in df_num if feature not in discrete_feature+year_feature+['Id']]
for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
dy=pd.DataFrame(df.groupby('YrSold')['SalePrice'].mean().reset_index().values,
                    columns=["YrSold","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice over the Year Sold",'xaxis':{'title':"Year Sold"}
                        ,'yaxis':{'title':"Average SalePrice"}})
# Add traces
fig.add_trace(go.Bar(x=dy.YrSold, y=dy.SalePrice,marker=dict(color="blue")))
fig.show()
dy=pd.DataFrame(df.groupby('MoSold')['SalePrice'].mean().reset_index().values,
                    columns=["MoSold","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice over the Month Sold",'xaxis':{'title':"Month Sold"}
                        ,'yaxis':{'title':"Average SalePrice"}})
# Add traces
fig.add_trace(go.Scatter(x=dy.MoSold, y=dy.SalePrice))
fig.show()
dy=pd.DataFrame(df.groupby('SaleType')['SalePrice'].mean().reset_index().values,
                    columns=["SaleType","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice with Respect to SaleType",'xaxis':{'title':"Sale Type"}
                        ,'yaxis':{'title':"Average SalePrice"}})

# Add traces
fig.add_trace(go.Bar(x=dy.SaleType, y=dy.SalePrice,marker=dict(color="brown")))
fig.show()
dy=pd.DataFrame(df.groupby('YearBuilt')['SalePrice'].mean().reset_index().values,
                    columns=["YearBuilt","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice with Respect to Year Built",'xaxis':{'title':"Year Built"}
                        ,'yaxis':{'title':"Average SalePrice"}})

# Add traces
fig.add_trace(go.Scatter(x=dy.YearBuilt, y=dy.SalePrice,mode='lines+markers',marker=dict(color="red")))
fig.show()
dy=pd.DataFrame(df.groupby('Neighborhood')['SalePrice'].mean().reset_index().values,
                    columns=["Neighborhood","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice over the Neighborhood",'xaxis':{'title':"Neighborhood"}
                        ,'yaxis':{'title':"Average SalePrice"},'xaxis_tickangle':-45})
# Add traces
fig.add_trace(go.Scatter(x=dy.Neighborhood, y=dy.SalePrice,mode='lines+markers'))
fig.show()
dy=pd.DataFrame(df.groupby('HouseStyle')['SalePrice'].mean().reset_index().values,
                    columns=["HouseStyle","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice with Respect to House Style",'xaxis':{'title':"House Style"}
                        ,'yaxis':{'title':"Average SalePrice"}})

# Add traces
fig.add_trace(go.Bar(x=dy.HouseStyle, y=dy.SalePrice,marker=dict(color="green")))
fig.show()
dy=pd.DataFrame(df.groupby('SaleCondition')['SalePrice'].mean().reset_index().values,
                    columns=["SaleCondition","SalePrice"])
fig = go.Figure(layout={'title':"Average SalePrice with Respect to SaleType",'xaxis':{'title':"Sale Condition"}
                        ,'yaxis':{'title':"Average SalePrice"}})

# Add traces
fig.add_trace(go.Bar(x=dy.SaleCondition, y=dy.SalePrice,marker=dict(color="black")))
fig.show()
df['saletype'] = 'Low Range'
df.loc[(df['SalePrice'] >= 143000) & (df['SalePrice'] <= 254000), 'saletype'] = 'Medium Range'
df.loc[(df['SalePrice'] > 254000), 'saletype'] = 'High Range'

df_flight = pd.DataFrame(df['saletype'].value_counts().reset_index().values, columns=["saletype", "AggregateType"])
labels = ["Low Range","Medium Range","High Range"]
value = [df_flight['AggregateType'][0],df_flight['AggregateType'][1],df_flight['AggregateType'][2]]
# colors=['lightcyan','cyan','royalblue']
figs = go.Figure(data=[go.Pie(labels=labels, values=value, pull=[0, 0, 0.2],textinfo = 'label+percent', hole = 0.35, 
                              hoverinfo="label+percent")],layout={'title':"SalePrice by Range",
                                                'annotations':[dict(text='<b>Saleprice<b>', x=0.5, y=0.5, font_size=11, showarrow=False)]})
figs.update_traces( textinfo='label + percent', textfont_size=10)
figs.show()
from scipy import stats
from scipy.stats import norm, skew
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()
x=df
x['SalePrice']=np.log1p(x['SalePrice'])
fig = plt.figure()
res = stats.probplot(x['SalePrice'], plot=plt)
plt.show()
%matplotlib inline
target=np.log1p(df["SalePrice"])
plt.figure(4);
plt.scatter(x=df['GarageArea'],y=target)
df=df[df['GarageArea']<1200]
target=np.log1p(df["SalePrice"])
plt.figure(9);
plt.scatter(x=df['GarageArea'],y=target)
plt.xlim(-200,1600)
target=np.log1p(df["SalePrice"])
plt.figure(5);
plt.scatter(x=df['GrLivArea'],y=target)
plt.figure(6);
plt.scatter(x=df['YearBuilt'],y=target)
plt.figure(7);
plt.scatter(x=df['TotalBsmtSF'],y=target)
plt.figure(8);
plt.scatter(x=df['1stFlrSF'],y=target)

import missingno as msno
msno.heatmap(df)
total=df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df.shape
data=pd.concat((df.loc[:,'MSSubClass':'SaleCondition'],
                dt.loc[:,'MSSubClass':'SaleCondition']))
var1=['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']
for v in var1:
  data[v]=data[v].fillna(data[v].mode()[0])
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

for col in ['GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',"PoolQC",'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
  data[col]=data[col].fillna('None')

for col in ['GarageYrBlt','GarageArea','GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF']:
  data[col]=data[col].fillna(0)

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data['Functional']=data['Functional'].fillna('Typ')

data=pd.get_dummies(data)
data.shape
print(data.isnull().sum().sum())
data.head(3)

for c in df_cat:
    df[c] = df[c].astype('category')
    if df[c].isnull().any():
        df[c] = df[c].cat.add_categories(['MISSING'])
        df[c] = df[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols('SalePrice ~ MSSubClass+LotArea+OverallQual+OverallCond+YearBuilt+YearRemodAdd+BsmtFinSF1+BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+LowQualFinSF+GrLivArea+BsmtFullBath+BsmtHalfBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+TotRmsAbvGrd+Fireplaces+GarageCars+GarageArea+WoodDeckSF+OpenPorchSF+EnclosedPorch+ScreenPorch+PoolArea+MiscVal+MoSold+YrSold', data=df_num).fit()
print(model.summary()) 
corr=df.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]
#creating matrices for sklearn:
X = data[:df.shape[0]]
y = df.SalePrice
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
import numpy as np


from yellowbrick.model_selection import ValidationCurve

from sklearn.tree import DecisionTreeRegressor

# Load a regression dataset

viz = ValidationCurve(
    DecisionTreeRegressor(), param_name="max_depth",
    param_range=np.arange(1, 11), cv=10, scoring="r2"
)

# Fit and show the visualizer
viz.fit(X, y)
viz.show()
import xgboost as xgb
# model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) 
model_xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.1, 
                             verbosity=1, silent=None, objective='reg:linear', booster='gbtree', 
                             n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                             subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.2, reg_lambda=1.2, 
                             scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain') 

model_xgb.fit(X_train, y_train)
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model_xgb)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model_xgb)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.regressor import PredictionError
rdf_r=RandomForestRegressor(n_estimators=600,random_state=0, n_jobs= -1)
visualizer = PredictionError(rdf_r)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(rdf_r)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
#import scikitplot as skplt
#skplt.estimators.plot_learning_curve(rdf_r, X, y)
#plt.show()
from sklearn.linear_model import Ridge
from yellowbrick.regressor import PredictionError
model_rdg=Ridge(alpha=3.181)
visualizer = PredictionError(model_rdg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model_rdg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
import scikitplot as skplt
skplt.estimators.plot_learning_curve(model_rdg, X, y)
plt.show()
from yellowbrick.regressor import PredictionError
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0)
visualizer = PredictionError(dtree)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(dtree)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
import scikitplot as skplt
skplt.estimators.plot_learning_curve(dtree, X, y)
plt.show()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model_lasso)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model_lasso)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
import scikitplot as skplt
skplt.estimators.plot_learning_curve(model_lasso, X, y)
plt.show()
X_test = X[:50]
y_test = y[:50]
pred1 = model_xgb.predict(X_test)
pred2 = rdf_r.predict(X_test)
pred3 = model_lasso.predict(X_test)
pred4 = dtree.predict(X_test)
pred5=model_rdg.predict(X_test)
plt.style.use('ggplot')
plt.figure(figsize=(8,6))
plt.plot(y_test, 'b*', label='ActualValue' )
plt.plot(pred1, 'gd', label='XGBoost')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='Lasso')
plt.plot(pred4, 'p', label='DecisionTree')
plt.plot(pred5, 'r*', ms=10, label='Ridge')

print('XGBoost :',model_xgb.score(X_test, y_test))
print('RandomForestRegressor :',rdf_r.score(X_test, y_test))
print('Lasso :',model_lasso.score(X_test, y_test))
print('DecisionTree :',dtree.score(X_test, y_test))
print('Ridge :',model_rdg.score(X_test, y_test))
# plt.tick_params(axis='x', which='both', bottom=False, top=False,
#                 labelbottom=False)
plt.tick_params(axis='x', which='both', bottom=True, top=True,
                labelbottom=True)
plt.ylabel('predicted Value')
plt.xlabel('Test samples')
plt.legend(loc="best")
plt.title('Model predictions and their average')
plt.show()