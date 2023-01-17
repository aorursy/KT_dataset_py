import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import graphviz

from scipy.stats import boxcox

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.stats.api as sms

import statsmodels.formula.api as smf

import statsmodels.graphics.tsaplots as smgt

from statsmodels.stats import stattools

from sklearn.preprocessing import StandardScaler,Normalizer,LabelEncoder,PolynomialFeatures,MinMaxScaler

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,StratifiedKFold,learning_curve,KFold,RandomizedSearchCV

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,SGDRegressor

from sklearn.tree import DecisionTreeRegressor,export_graphviz

from sklearn.svm import LinearSVR

from sklearn.feature_selection import SelectFromModel, RFE,SelectPercentile,f_regression,VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import umap

from scipy import stats

import warnings

warnings.filterwarnings("ignore")
def calculate_vif(X):

    X['intercept'] = 1

    vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif["features"] = X.columns

    vif['VIF Factor'] = round(vif['VIF Factor'],2)

    vif = vif[vif['features'] != 'intercept']

    vif.sort_values('VIF Factor',ascending=False,inplace=True)

    return vif
def scale_data(X):

    return pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns.values)
def make_data(n_points, err = 1.0, random_state = 42):

    rnd_gen = np.random.RandomState(random_state)

    X = rnd_gen.rand(n_points, 1) ** 2

    y = 10 - 1. / (X.ravel() + 0.1)

    if err > 0:

        y += err * rnd_gen.randn(n_points)

    return X, np.abs(y)
def set_poly_feature(X,y,degree,plot_type = 'reg'):

    poly = PolynomialFeatures(degree = degree)

    X_poly = poly.fit_transform(X)

    lr = LinearRegression()

    lr.fit(X_poly,y)

    y_pred = lr.predict(X_poly)

    if plot_type == 'reg':

        sns.regplot(x='X',y='y',data = dftemp,fit_reg=False)

        sns.lineplot(dftemp['X'],y_pred,c = 'r',lw = 1,label = 'degree ' + str(degree))

        plt.legend(loc='upper left')

    else:

        residuals = y_pred - y

        sns.scatterplot(x = y_pred,y = residuals)

        plt.axhline(y = 0, c = 'r', lw = 1)

        plt.title('Residual for degree ' + str(degree))
def plot_residuals(y_pred,residuals):

    print("Durbin-Watson test statistics is " + str(round(stattools.durbin_watson(residuals),2)))

    fig,ax = plt.subplots(1,3,figsize=(18,4))

    sns.scatterplot(x = y_pred,y = residuals,ax = ax[0])

    ax[0].axhline(y=0, c='r', lw=1)

    stats.probplot(residuals, plot = ax[1])

    smgt.plot_acf(residuals,ax = ax[2])

    plt.show()
def print_reg_result(estimator,X,y,y_pred):

    mse = round(mean_squared_error(y, y_pred),2)

    print("Mean Square Error (MSE): ", mse)



    rmse = round(np.sqrt(mse),2)

    print("Root Mean Square Error (RMSE):", rmse)



    mae = round(mean_absolute_error(y,y_pred),2)

    print("Mean Absolute Error :", mae)



    r_sq = round(estimator.score(X, y),2)

    print('R-square         :', r_sq)

    

    adj_r_sq = round(1 - (1 - r_sq) * (len(y) - 1) / (len(y_pred) - X.shape[1] - 1),2)

    print('R-square adj.    :',adj_r_sq)
dfdata = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")

dfdata.columns = dfdata.columns.str.lower()

dfdata.head()
dfdata.describe().T
print("Data : ",dfdata.shape)

print("Duplicate rows : ",dfdata[dfdata.duplicated()].shape)

dfdata.drop_duplicates(inplace = True)

print("After drop duplicates : ",dfdata.shape)

dfdata.info()
dfdata.isnull().sum()
dfdata[dfdata['distance'].isnull() == True]
dfdata.drop(dfdata[dfdata['distance'].isnull() == True].index,axis=0,inplace=True)

dfdata.shape
dfdata[dfdata['bedroom2'].isnull() == True][['rooms','price','landsize','car']].describe().T
pd.crosstab(dfdata['bedroom2'],dfdata['rooms'])

sns.countplot(dfdata[dfdata['bedroom2'].isnull() == True]['rooms'])

plt.show()
dfdata['bedroom2'].fillna(dfdata['rooms'],inplace=True)

dfdata.isnull().sum()
pd.crosstab(dfdata['bathroom'],dfdata['rooms'])
dftemp = dfdata.groupby(['rooms'],as_index=False)['bathroom'].median()

indices = dfdata[dfdata['bathroom'].isnull() == True].index

dfdata.loc[indices,'bathroom'] = dfdata.loc[indices,'rooms'].apply(lambda x : dftemp[dftemp['rooms']==x]['bathroom'].values[0])

dfdata.isnull().sum()
dfdata[dfdata['bathroom'] == 'Nan']
def fill_via_suburb(colname):

    indices = dfdata[dfdata[colname].isnull() == True].index

    dfdata.loc[indices,colname] = dfdata.loc[indices,'suburb'].map(lambda x: dfdata[dfdata['suburb'] == x][colname].mode()[0])



fill_via_suburb('councilarea')

fill_via_suburb('regionname')

fill_via_suburb('propertycount')

dfdata.isnull().sum()
dfdata.drop(['car','landsize','buildingarea','yearbuilt','lattitude','longtitude'],axis=1,inplace=True)

dfdata.isnull().sum()
dfdata.dropna(subset = {'price'},inplace = True)

dfdata.describe()
dfdata['price_log'] = np.log1p(dfdata['price'])

fig,ax = plt.subplots(2,3,figsize = (24,8))

sns.distplot(dfdata['price'], ax = ax[0][0])

sns.boxplot(dfdata['price'],ax = ax[0][1])

stats.probplot(dfdata['price'],plot = ax[0][2])

sns.distplot(dfdata['price_log'], ax = ax[1][0])

sns.boxplot(dfdata['price_log'],ax = ax[1][1])

stats.probplot(dfdata['price_log'],plot = ax[1][2])

plt.show()
dfdata[dfdata['price'] < 150000].head()
dfdata[dfdata['price'] > 7000000]
dfdata.columns
dfdata['rooms'] = dfdata['rooms'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='rooms',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='rooms',ax=ax[1])

plt.show()
dftemp = dfdata['rooms'].value_counts().reset_index()

indices = dfdata[dfdata['rooms'].isin(dftemp[dftemp['rooms'] < 1000]['index'].values)].index

dfdata.loc[indices,'rooms'] = 'other'

print("Number of unique rooms : ",dfdata['rooms'].nunique())
dfdata['rooms'] = dfdata['rooms'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='rooms',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='rooms',ax=ax[1])

plt.show()
dfdata['bedroom2'] = dfdata['bedroom2'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='bedroom2',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='bedroom2',ax=ax[1])

plt.show()
dftemp = dfdata['bedroom2'].value_counts().reset_index()

indices = dfdata[dfdata['bedroom2'].isin(dftemp[dftemp['bedroom2'] < 1000]['index'].values)].index

dfdata.loc[indices,'bedroom2'] = 'other'

print("Number of unique bedroom2 : ",dfdata['bedroom2'].nunique())
dfdata['bedroom2'] = dfdata['bedroom2'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='bedroom2',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='bedroom2',ax=ax[1])

plt.show()
dfdata['bathroom'] = dfdata['bathroom'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='bathroom',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='bathroom',ax=ax[1])

plt.show()
dftemp = dfdata['bathroom'].value_counts().reset_index()

indices = dfdata[dfdata['bathroom'].isin(dftemp[dftemp['bathroom'] < 1000]['index'].values)].index

dfdata.loc[indices,'bathroom'] = 'other'

print("Number of unique bathroom : ",dfdata['bathroom'].nunique())
dfdata['bathroom'] = dfdata['bathroom'].astype('str')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.boxplot(data = dfdata,x='bathroom',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='bathroom',ax=ax[1])

plt.show()
dfdata['street'] = dfdata['address'].apply(lambda x: str(x).split(' ')[1])

dfdata['suburb_street'] = dfdata['suburb'] + '_' + dfdata['street']

print("Number of unique suburb : ",dfdata['suburb'].nunique())

print("Number of unique address : ",dfdata['address'].nunique())

print("Number of unique street : ",dfdata['street'].nunique())

print("Number of unique suburb_street : ",dfdata['suburb_street'].nunique())

print("Number of unique postcode : ",dfdata['postcode'].nunique())

print("Number of unique type : ",dfdata['type'].nunique())

print("Number of unique method : ",dfdata['method'].nunique())

print("Number of unique sellerg : ",dfdata['sellerg'].nunique())

print("Number of unique councilarea : ",dfdata['councilarea'].nunique())

print("Number of unique regionname : ",dfdata['regionname'].nunique())
dfdata['suburb_street_mean'] = dfdata.groupby('suburb_street')['price'].transform('mean')

dfdata['suburb_mean'] = dfdata.groupby('suburb')['price'].transform('mean')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.distplot(dfdata['suburb_street_mean'],ax = ax[0])

sns.distplot(dfdata['suburb_mean'],ax = ax[1])

plt.show()
dftemp = dfdata.groupby('suburb')['postcode'].nunique()

dftemp[dftemp != 1]
fig,ax = plt.subplots(1,4,figsize=(18,4))

sns.boxplot(data = dfdata,x='type',y='price',ax=ax[0])

sns.countplot(data = dfdata,x='type',ax=ax[1])

sns.boxplot(data = dfdata,x='method',y='price',ax=ax[2])

sns.countplot(data = dfdata,x='method',ax=ax[3])

plt.show()
dftemp = dfdata['sellerg'].value_counts().reset_index()

print(dftemp[['sellerg']].describe().T)

plt.figure(figsize=(18,4))

sns.countplot(dftemp['sellerg'])

plt.xticks(rotation=90)

plt.show()

indices = dfdata[dfdata['sellerg'].isin(dftemp[dftemp['sellerg'] < 10]['index'].values)].index

dfdata.loc[indices,'sellerg'] = 'other'

print("Number of unique sellerg : ",dfdata['sellerg'].nunique())

dfdata['sellerg_mean'] = dfdata.groupby('sellerg')['price'].transform('mean')
fig,ax=plt.subplots(1,2,figsize=(18,4))

sns.countplot(x='regionname',data = dfdata,ax=ax[0])

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=45)

sns.boxplot(x='regionname',y='price', data = dfdata, ax = ax[1])

ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=45)

plt.show()
dfdata['regionname_mean'] = dfdata.groupby('regionname')['price'].transform('mean')
fig,ax=plt.subplots(2,1,figsize=(18,8))

sns.countplot(x='councilarea',data = dfdata,ax=ax[0])

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=45)

sns.boxplot(x='councilarea',y='price', data = dfdata, ax = ax[1])

ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=45)

plt.show()
dfdata['councilarea_mean'] = dfdata.groupby('councilarea')['price'].transform('mean')
dfdata['date'] = pd.to_datetime(dfdata['date'])

dfdata['year'] = dfdata['date'].dt.year

dfdata['month'] = dfdata['date'].dt.month

dfdata['dayofweek'] = dfdata['date'].dt.dayofweek

dfdata['week'] = dfdata['date'].dt.week

dfdata['day'] = dfdata['date'].dt.day
plt.figure(figsize=(18,5))

sns.lineplot(x='date',y='price',data = dfdata)

plt.show()
fig,ax = plt.subplots(1,4,figsize=(18,5))

sns.countplot(x='month',data = dfdata,ax=ax[0])

sns.boxplot(x='month',y='price',data = dfdata,ax=ax[1])

sns.countplot(x='dayofweek',data = dfdata,ax=ax[2])

sns.boxplot(x='dayofweek',y='price',data = dfdata,ax=ax[3])

plt.show()

dfdata['issaturday'] = np.where(dfdata['dayofweek'] == 5,'1','0')
fig,ax = plt.subplots(1,2,figsize=(18,5))

sns.countplot(x='week',data = dfdata,ax=ax[0])

sns.boxplot(x='week',y='price',data = dfdata,ax=ax[1])

plt.show()
fig,ax = plt.subplots(1,2,figsize=(18,5))

sns.countplot(x='year',data = dfdata,ax=ax[0])

sns.boxplot(x='year',y='price',data = dfdata,ax=ax[1])

plt.show()
fig,ax = plt.subplots(1,2,figsize=(18,5))

sns.countplot(x='day',data = dfdata,ax=ax[0])

sns.boxplot(x='day',y='price',data = dfdata,ax=ax[1])

plt.show()
drop_list = ['suburb','address','postcode','street','year','day','dayofweek','date','week']

dfdata.drop(drop_list,inplace=True,axis=1)

dfdata['month'] = dfdata['month'].astype('str')
dfdata.columns
fig,ax = plt.subplots(1,4,figsize=(18,4))

sns.distplot(dfdata['distance'],ax=ax[0])

sns.distplot(dfdata['propertycount'],ax=ax[1])

sns.distplot(dfdata['suburb_street_mean'],ax=ax[2])

sns.distplot(dfdata['suburb_mean'],ax=ax[3])

plt.show()
fig,ax = plt.subplots(1,3,figsize=(18,4))

sns.distplot(dfdata['sellerg_mean'],ax=ax[0])

sns.distplot(dfdata['regionname_mean'],ax=ax[1])

sns.distplot(dfdata['councilarea_mean'],ax=ax[2])

plt.show()
numeric_features = list(dfdata.columns[dfdata.dtypes != 'object'].values)

print("Numeric features :" ,numeric_features)

numeric_features.remove('price')

numeric_features.remove('price_log')

categoric_features = list(dfdata.columns[dfdata.dtypes == 'object'].values)

print("Categoric features :" ,categoric_features)
calculate_vif(dfdata[numeric_features])
le = LabelEncoder()

for feature in categoric_features:

    colname = 'le_' + feature

    dfdata[colname] = le.fit_transform(dfdata[feature])



encoded_features = ['le_' + x for x in categoric_features]
calculate_vif(dfdata[encoded_features])
corr_matrix = np.corrcoef(dfdata[numeric_features],rowvar=False)

corr_matrix = dfdata[numeric_features].corr()

sns.heatmap(corr_matrix,annot=True,fmt='.1g')

plt.show()
numeric_features.remove('suburb_street_mean')

numeric_features.remove('regionname_mean')

numeric_features.remove('suburb_mean')

corr_matrix = np.corrcoef(dfdata[numeric_features],rowvar=False)

corr_matrix = dfdata[numeric_features].corr()

sns.heatmap(corr_matrix,annot=True,fmt='.1g')

plt.show()
corr_matrix = np.corrcoef(dfdata[encoded_features],rowvar=False)

corr_matrix = dfdata[encoded_features].corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr_matrix,annot=True,fmt='.1g')

plt.show()
categoric_features.remove('bedroom2')

encoded_features = ['le_' + x for x in categoric_features]

corr_matrix = np.corrcoef(dfdata[encoded_features],rowvar=False)

corr_matrix = dfdata[encoded_features].corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr_matrix,annot=True,fmt='.1g')

plt.show()
calculate_vif(dfdata[numeric_features + encoded_features])
corr_matrix = np.corrcoef(dfdata[numeric_features + encoded_features],rowvar=False)

corr_matrix = dfdata[numeric_features + encoded_features].corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr_matrix,annot=True,fmt='.1g')

plt.show()
dfdata.drop(['bedroom2','suburb_street_mean','regionname_mean','suburb_mean'],axis=1,inplace=True)

dfdata.columns
print("Numeric features :" ,numeric_features)

print("Categoric features :" ,categoric_features)
col_list = ['price'] + numeric_features

sns.pairplot(dfdata[col_list],kind='scatter',diag_kind='kde')

plt.show()
col_list = ['price'] + encoded_features

sns.pairplot(dfdata[col_list],kind='scatter',diag_kind='hist')

plt.show()
X = dfdata[numeric_features]

y = dfdata['price_log']
lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

plot_residuals(y_pred,residuals)
X = scale_data(dfdata[numeric_features])

lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

plot_residuals(y_pred,residuals)
model = smf.ols('price_log ~ distance + propertycount + sellerg_mean + councilarea_mean ', data = dfdata)

fitted = model.fit()

print(fitted.summary())
N = 200 #number of points

X, y = make_data(200)

dftemp = pd.DataFrame({'X':X.ravel(), 'y':y})

print(dftemp.shape)

dftemp.head()
lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

residuals = y - y_pred

fig,ax=plt.subplots(1,2,figsize=(18,4))

sns.scatterplot(x = y_pred,y = residuals,ax = ax[0])

ax[0].axhline(y = 0, c = 'r', lw = 1)

ax[0].set(title='Residual Plot')

sns.regplot(x='X',y='y',data = dftemp,fit_reg=False, ax = ax[1])

sns.lineplot(dftemp['X'],y_pred,c = 'r',lw = 1,ax = ax[1])

ax[1].set(title='Regression Plot')

plt.show()
fig,ax = plt.subplots(1,4,figsize=(18,4))

plt.subplot(141)

set_poly_feature(X,y,2)

plt.subplot(142)

set_poly_feature(X,y,10)

plt.subplot(143)

set_poly_feature(X,y,30)

plt.subplot(144)

set_poly_feature(X,y,50)

plt.show()
fig,ax = plt.subplots(1,4,figsize=(18,4))

plt.subplot(141)

set_poly_feature(X,y,2,'res')

plt.subplot(142)

set_poly_feature(X,y,10,'res')

plt.subplot(143)

set_poly_feature(X,y,30,'res')

plt.subplot(144)

set_poly_feature(X,y,50,'res')

plt.show()
score = []

rmse = []

for degree in range(1,20,2):

    X_poly = PolynomialFeatures(degree = degree).fit_transform(X)

    lr = LinearRegression()

    lr.fit(X_poly,y)

    y_pred = lr.predict(X_poly)

    score.append(lr.score(X_poly,y))

    rmse.append(np.sqrt(mean_squared_error(y,y_pred)))

plt.plot(score,c = 'b')

plt.plot(rmse,c='r')

plt.xticks(range(len(rmse)),range(1,20,2))

plt.show()
X_poly = PolynomialFeatures(degree = 8).fit_transform(X)

lr = LinearRegression()

lr.fit(X_poly,y)

y_pred = lr.predict(X_poly)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X_poly,y,y_pred)

plot_residuals(y_pred,residuals)
X = scale_data(dfdata[numeric_features + encoded_features])

y = dfdata['price_log']
lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

plot_residuals(y_pred,residuals)
n_zero_coefs = []

score = []

alpha_list = [10,1,0.5,0.1,0.05,0.01,0.001]

for alpha in alpha_list:

    lasso = Lasso(alpha = alpha)

    lasso.fit(X,y)

    coef = np.round(lasso.coef_,4)

    n_zero_coefs.append(len(coef[coef == 0]))

    score.append(round(lasso.score(X,y),2))

    

pd.DataFrame(zip(alpha_list,n_zero_coefs,score),columns = ['alpha','zero_coef','score'])
lasso = Lasso(alpha = 0.01)

lasso.fit(X,y)

y_pred = lasso.predict(X)

residuals = y - y_pred

print(round(lasso.intercept_,2),np.round(lasso.coef_,2))

print_reg_result(lasso,X,y,y_pred)

plot_residuals(y_pred,residuals)
n_zero_coefs = []

score = []

alpha_list = [100,10,1,0.1,0.01,0.001]

for alpha in alpha_list:

    ridge = Ridge(alpha = alpha)

    ridge.fit(X,y)

    coef = np.round(ridge.coef_,4)

    n_zero_coefs.append(coef.max())

    score.append(round(ridge.score(X,y),2))

    

pd.DataFrame(zip(alpha_list,n_zero_coefs,score),columns = ['alpha','zero_coef','score'])
ridge = Ridge(alpha = 1000)

ridge.fit(X,y)

y_pred = ridge.predict(X)

print(round(ridge.intercept_,2),np.round(ridge.coef_,2))

print_reg_result(ridge,X,y,y_pred)

plot_residuals(y_pred,residuals)
elastic = ElasticNet(alpha = 0.1, l1_ratio=0.1)

elastic.fit(X,y)

y_pred = elastic.predict(X)

print(round(elastic.intercept_,2),np.round(elastic.coef_,2))

print_reg_result(elastic,X,y,y_pred)

plot_residuals(y_pred,residuals)
dftemp = pd.DataFrame(zip(X.columns.values,np.abs(lr.coef_),np.abs(lasso.coef_),np.abs(ridge.coef_),np.abs(elastic.coef_)),columns = ['feature','lr_coef','lasso_coef','ridge_coef','elastic_coef'])

plt.figure(figsize=(18,5))

plt.plot(dftemp['lr_coef'],label = 'lr',c = 'r')

plt.plot(dftemp['lasso_coef'],label='lasso',c='g')

plt.plot(dftemp['ridge_coef'],label='ridge',c='b')

plt.plot(dftemp['elastic_coef'],label='elastic',c='y')

plt.legend(loc='best')

plt.xticks(range(X.shape[1]), X.columns.values,rotation = 45)

plt.show()

dftemp
p_values = np.round(f_regression(X, y)[1],4)

dftemp = pd.DataFrame(zip(X.columns.values,p_values),columns = ['feature','pval'])

dftemp.sort_values('pval',ascending=False)
vt_filter = VarianceThreshold(threshold=0.1)

vt_filter.fit(X)

drop_list = [column for column in X.columns if column not in X.columns[vt_filter.get_support()]]

print(drop_list)
model_dict = {'lr':lr,'lasso':lasso,'elastic':elastic}

for key,value in model_dict.items():

    rfe = RFE(value).fit(X,y)

    dftemp[key] = rfe.ranking_

dftemp
features = numeric_features + encoded_features

features.remove('le_sellerg')

features.remove('le_councilarea')

features.remove('le_month')

features.remove('propertycount')

features.remove('le_suburb_street')

print(features)

numeric_features = ['distance', 'sellerg_mean', 'councilarea_mean']

categoric_features = ['rooms', 'type', 'method', 'bathroom', 'regionname', 'issaturday']
X = dfdata[features]

lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

plot_residuals(y_pred,residuals)
n_unique = []

for feature in categoric_features:

    n_unique.append(dfdata[feature].nunique())

dftemp = pd.DataFrame(zip(categoric_features,n_unique),columns = ['feature','n_unique'])

dftemp.sort_values('n_unique',ascending=False)
X = dfdata[numeric_features]

X_encoded = pd.get_dummies(dfdata[categoric_features],drop_first = True)

X = pd.concat([X,X_encoded],axis=1)

print(X.shape)
X = scale_data(X)

lr = LinearRegression()

lr.fit(X,y)

y_pred = lr.predict(X)

residuals = y - y_pred

print(round(lr.intercept_,2),np.round(lr.coef_,2))

print_reg_result(lr,X,y,y_pred)

plot_residuals(y_pred,residuals)
X = scale_data(X)

dt = DecisionTreeRegressor(max_depth = 8,min_samples_split=10).fit(X,y)

y_pred = dt.predict(X)

residuals = y - y_pred

print_reg_result(dt,X,y,y_pred)

plot_residuals(y_pred,residuals)
X = scale_data(X)

sgd = SGDRegressor(alpha = 0.01)

sgd.fit(X,y)

y_pred = sgd.predict(X)

residuals = y - y_pred

print(np.round(sgd.intercept_,2),np.round(sgd.coef_,2))

print_reg_result(sgd,X,y,y_pred)

plot_residuals(y_pred,residuals)
X = dfdata[numeric_features + encoded_features]

y = dfdata['price_log']
X_train,X_test,y_train,y_test = train_test_split(X,y)

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

cv_results = np.round(100 * cross_val_score(sgd,X_train_scaled,y_train,cv = 5),1)

print("Cross validation score : ",cv_results)

print("Mean cross val score : ", round(np.mean(cv_results),1))



kfold = KFold(n_splits = 5,shuffle = True)

cv_results = np.round(100 * cross_val_score(sgd,X_train_scaled,y_train,cv = kfold),1)

print("Cross validation score : ",cv_results)

print("Mean cross val score : ", round(np.mean(cv_results),1))
X_train,X_test,y_train,y_test = train_test_split(X,y)

X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train)

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)
values = [100,10,1,0.1,0.01,0.001]

best_score = 0

for alpha in values:

    for ratio in values:

        elastic = ElasticNet(alpha=alpha,l1_ratio=ratio).fit(X_train_scaled,y_train)

        score = elastic.score(X_validation_scaled,y_validation)

        if best_score < score:

            best_score = score

            best_params = {'alpha':alpha,'l1_ratio': ratio}

print("Best Score : ",round(best_score,2))

print("Best Parameters : ",best_params)
elastic = ElasticNet(**best_params)

elastic.fit(X_train_scaled, y_train)

y_pred = elastic.predict(X_test_scaled)

print(round(elastic.intercept_,2),np.round(elastic.coef_,2))

print_reg_result(elastic,X_test_scaled,y_test,y_pred)
dt_overfit = DecisionTreeRegressor().fit(X_train_scaled,y_train)

print("Overfit prediction : ",round(100 * dt_overfit.score(X_train_scaled,y_train),1))

dt = DecisionTreeRegressor(max_depth = 8,min_samples_split=10).fit(X_train_scaled,y_train)

print("Normal prediction : ",round(100 * dt.score(X_train_scaled,y_train),1))

train_size,train_scores,test_scores = learning_curve(dt,X_train_scaled,y_train,train_sizes = np.linspace(0.05,1,20))

train_scores_mean = np.mean(train_scores,axis=1)

test_scores_mean = np.mean(test_scores,axis=1)



train_size_overfit,train_scores_overfit,test_scores_overfit = learning_curve(dt_overfit,X_train_scaled,y_train,train_sizes = np.linspace(0.05,1,20))

train_scores_overfit_mean = np.mean(train_scores_overfit,axis=1)

test_scores_overfit_mean = np.mean(test_scores_overfit,axis=1)



fig,ax = plt.subplots(1,2,figsize=(18,5))

sns.lineplot(x=train_size,y=train_scores_mean,c='r',label='train',ax=ax[0])

sns.lineplot(x=train_size,y=test_scores_mean,c='b',label='test',ax=ax[0])

ax[0].legend(loc='best')

sns.lineplot(x=train_size_overfit,y=train_scores_overfit_mean,c='r',label='train',ax=ax[1])

sns.lineplot(x=train_size_overfit,y=test_scores_overfit_mean,c='b',label='test',ax=ax[1])

ax[1].legend(loc='best')

plt.show()
alpha_list = [100, 10, 1, 0.1, 0.01]

l1_ratio_list =  [0.1, 0.01, 0.05, 0.001]

param_grid = {'alpha': alpha_list, 'l1_ratio' : l1_ratio_list}

estimator = ElasticNet(random_state = 12)
grid_search = GridSearchCV(estimator,param_grid,cv = 5)

grid_search.fit(X_train_scaled,y_train)



print("Grid Search Result : ", round(100 * grid_search.score(X_validation_scaled,y_validation),1))

print("Grid Search Best Parameters : ", grid_search.best_params_)

print("Grid Search Best Score : ", round(100 * grid_search.best_score_,1))

print("Grid Search Best Estimator : ", grid_search.best_estimator_)
df_result = pd.DataFrame(grid_search.cv_results_)

print(df_result.shape)

df_result[['param_alpha','param_l1_ratio','mean_test_score']].head(10)
mean_scores = np.round(100 * np.array(df_result['mean_test_score']).reshape(len(alpha_list),len(l1_ratio_list)),2)

sns.heatmap(mean_scores,annot = True,fmt = '2.1f')

plt.xticks(range(len(l1_ratio_list)),l1_ratio_list,rotation = 45)

plt.yticks(range(len(alpha_list)),alpha_list,rotation = 90)

plt.show()
randomized_search = RandomizedSearchCV(estimator,param_grid,cv = 5,n_iter = 10)

randomized_search.fit(X_train_scaled,y_train)



print("Randomized Search Result : ", round(100 * randomized_search.score(X_validation_scaled,y_validation),1))

print("Randomized Search Best Parameters : ", randomized_search.best_params_)

print("Randomized Search Best Score : ", round(100 * randomized_search.best_score_,1))

print("Randomized Search Best Estimator : ", randomized_search.best_estimator_)
df_result = pd.DataFrame(randomized_search.cv_results_)

print(df_result.shape)

df_result[['param_alpha','param_l1_ratio','mean_test_score']].head(10)