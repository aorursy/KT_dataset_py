import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



import collections

import itertools



import scipy.stats as stats

from scipy.stats import norm

from scipy.special import boxcox1p



import statsmodels

import statsmodels.api as sm

#print(statsmodels.__version__)



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

from plotly.subplots import make_subplots



init_notebook_mode(connected=True)



from tqdm import tqdm_notebook



from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score

from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.utils import resample



from xgboost import XGBRegressor



#Model interpretation modules

import lime

import lime.lime_tabular

import shap

shap.initjs()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
Combined_data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

Combined_data.head()
print('Number of features: {}'.format(Combined_data.shape[1]))

print('Number of examples: {}'.format(Combined_data.shape[0]))
#for c in df.columns:

#    print(c, dtype(df_train[c]))

Combined_data.dtypes
Combined_data['last_review'] = pd.to_datetime(Combined_data['last_review'],infer_datetime_format=True) 
total = Combined_data.isnull().sum().sort_values(ascending=False)

percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
Combined_data.drop(['host_name','name'], axis=1, inplace=True)
Combined_data[Combined_data['number_of_reviews']== 0.0].shape
Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)
earliest = min(Combined_data['last_review'])

Combined_data['last_review'] = Combined_data['last_review'].fillna(earliest)

Combined_data['last_review'] = Combined_data['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())
total = Combined_data.isnull().sum().sort_values(ascending=False)

percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
fig, axes = plt.subplots(1,3, figsize=(21,6))

sns.distplot(Combined_data['price'], ax=axes[0])

sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])

axes[1].set_xlabel('log(1+price)')

sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2]);
Combined_data = Combined_data[np.log1p(Combined_data['price']) < 8]

Combined_data = Combined_data[np.log1p(Combined_data['price']) > 3]
fig, axes = plt.subplots(1,3, figsize=(21,6))

sns.distplot(Combined_data['price'], ax=axes[0])

sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])

axes[1].set_xlabel('log(1+price)')

sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2]);
Combined_data['price'] = np.log1p(Combined_data['price'])
print(Combined_data.columns)
print('In this dataset there are {} unique hosts renting out  a total number of {} properties.'.format(len(Combined_data['host_id'].unique()), Combined_data.shape[0]))
Combined_data = Combined_data.drop(['host_id', 'id'], axis=1)
sns.catplot(x='neighbourhood_group', kind='count' ,data=Combined_data)

fig = plt.gcf()

fig.set_size_inches(12, 6)
fig, axes = plt.subplots(1,3, figsize=(21,6))

sns.distplot(Combined_data['latitude'], ax=axes[0])

sns.distplot(Combined_data['longitude'], ax=axes[1])

sns.scatterplot(x= Combined_data['latitude'], y=Combined_data['longitude'])
sns.catplot(x='room_type', kind='count' ,data=Combined_data)

fig = plt.gcf()

fig.set_size_inches(8, 6)
fig, axes = plt.subplots(1,2, figsize=(21, 6))



sns.distplot(Combined_data['minimum_nights'], rug=False, kde=False, color="green", ax = axes[0])

axes[0].set_yscale('log')

axes[0].set_xlabel('minimum stay [nights]')

axes[0].set_ylabel('count')



sns.distplot(np.log1p(Combined_data['minimum_nights']), rug=False, kde=False, color="green", ax = axes[1])

axes[1].set_yscale('log')

axes[1].set_xlabel('minimum stay [nights]')

axes[1].set_ylabel('count')
Combined_data['minimum_nights'] = np.log1p(Combined_data['minimum_nights'])
fig, axes = plt.subplots(1,2,figsize=(18.5, 6))

sns.distplot(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month'], rug=True, kde=False, color="green", ax=axes[0])

sns.distplot(np.sqrt(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month']), rug=True, kde=False, color="green", ax=axes[1])

axes[1].set_xlabel('ln(reviews_per_month)')
fig, axes = plt.subplots(1,1, figsize=(21,6))

sns.scatterplot(x= Combined_data['availability_365'], y=Combined_data['reviews_per_month'])
Combined_data['reviews_per_month'] = Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month']
fig, axes = plt.subplots(1,1,figsize=(18.5, 6))

sns.distplot(Combined_data['availability_365'], rug=False, kde=False, color="blue", ax=axes)

axes.set_xlabel('availability_365')

axes.set_xlim(0, 365)
corrmatrix = Combined_data.corr()

f, ax = plt.subplots(figsize=(15,12))

sns.heatmap(corrmatrix, vmax=0.8, square=True)
#sns.pairplot(Combined_data.select_dtypes(exclude=['object']))#
categorical_features = Combined_data.select_dtypes(include=['object'])

print('Categorical features: {}'.format(categorical_features.shape))
categorical_features_one_hot = pd.get_dummies(categorical_features)

categorical_features_one_hot.head()
Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)
numerical_features =  Combined_data.select_dtypes(exclude=['object'])

y = numerical_features.price

numerical_features = numerical_features.drop(['price'], axis=1)

print('Numerical features: {}'.format(numerical_features.shape))
X = np.concatenate((numerical_features, categorical_features_one_hot), axis=1)

X_df = pd.concat([numerical_features, categorical_features_one_hot], axis=1)

#print('Dimensions of the design matrix: {}'.format(X.shape))

#print('Dimension of the target vector: {}'.format(y.shape))
Processed_data = pd.concat([X_df, y], axis = 1)

#Processed_data.to_csv('NYC_Airbnb_Processed.dat')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Dimensions of the training feature matrix: {}'.format(X_train.shape))

print('Dimensions of the training target vector: {}'.format(y_train.shape))

print('Dimensions of the test feature matrix: {}'.format(X_test.shape))

print('Dimensions of the test target vector: {}'.format(y_test.shape))
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
Xy_train = pd.concat([pd.DataFrame(X_train, columns = X_df.columns), pd.DataFrame(y_train, columns=['price'])], axis=1)
Xy_train.shape
n_folds = 5



# squared_loss

def rmse_cv(model, X = X_train, y=y_train):

    kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)

    return cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
best_alpha = 0.0001

lasso_CV_best = -rmse_cv(Ridge(alpha = best_alpha))

lasso = Lasso(alpha = best_alpha) 

lasso.fit(X_train, y_train) 

y_train_lasso = lasso.predict(X_train)

y_test_lasso = lasso.predict(X_test)

lasso_results = pd.DataFrame({'sample':['original'],

            'CV error': lasso_CV_best.mean(), 

            'CV std': lasso_CV_best.std(),

            'training error': [mean_squared_error(y_train_lasso, y_train)],

            'test error': [mean_squared_error(y_test_lasso, y_test)],

            'training_r2_score': [r2_score(y_train, y_train_lasso)],

            'test_r2_score': [r2_score(y_test, y_test_lasso)]})

lasso_results
lasso_results
lasso_coefs = pd.DataFrame(lasso.coef_, columns=['original'], index = X_df.columns)

lasso_coefs


lasso_all = lasso_results

lasso_coefs_all = lasso_coefs



for i in range(0, 100):

    print('Bootstrap sample no. '+str(i))

    X_bs, y_bs = resample(X_train, y_train, n_samples = X_df.shape[0], random_state=i) 

    best_alpha = 0.0001

    lasso_CV_best = -rmse_cv(Lasso(alpha = best_alpha))

    lasso = Lasso(alpha = best_alpha) 

    lasso.fit(X_bs, y_bs) 

    y_bs_lasso = lasso.predict(X_bs)

    y_test_lasso = lasso.predict(X_test)

    lasso_bs_results = pd.DataFrame({'sample':['Bootstrap '+str(i)],

            'CV error': lasso_CV_best.mean(), 

            'CV std': lasso_CV_best.std(),

            'training error': [mean_squared_error(y_bs, y_bs_lasso)],

            'test error': [mean_squared_error(y_test_lasso, y_test)],

            'training_r2_score': [r2_score(y_bs, y_bs_lasso)],

            'test_r2_score': [r2_score(y_test, y_test_lasso)]})

    lasso_all = pd.concat([lasso_all, lasso_bs_results], ignore_index=True)

    lasso_coefs_all = pd.concat([lasso_coefs_all, pd.DataFrame(lasso.coef_, columns=['Bootstrap' + str(i)], index = X_df.columns)], axis=1)
lasso_all
lasso_all.describe()
lasso_coefs_all.head(10)
lasso_coefs_all['mean'] = lasso_coefs.mean(axis=1)

lasso_coefs_all.sort_values(by='mean', inplace=True)

lasso_coefs_all
lasso_coefs_all['nonzero_vals'] = lasso_coefs_all.astype(bool).sum(axis=1)

lasso_coefs_all.sort_values(by='nonzero_vals', inplace=True)

lasso_coefs_all
sns.distplot(lasso_coefs_all['nonzero_vals'], kde=False)
trace = go.Histogram(

    x=lasso_coefs_all['nonzero_vals'],

    marker=dict(

        color='blue'

    ),

    opacity=0.75

)



layout = go.Layout(

    title='LASSO Variable Importance on boot',

    height=450,

    width=1200,

    xaxis=dict(

        title='No. nonzero occurrences'

    ),

    yaxis=dict(

        title='No. features'

    ),

    bargap=0.2,

)



data= [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
#RR_coefs_all[RR_coefs_all.index=='longitude']

lasso_coefs_all.loc['longitude']
def getACoefficientSlice(start=0, end=1):

    df_lasso = lasso_coefs_all.iloc[start:end]

    df_lasso['std'] = df_lasso.std(axis=1)

    df_lasso = df_lasso.sort_values(by='std', ascending=True)

    df_lasso.drop(columns=['mean','std'], inplace=True)

    return df_lasso



getACoefficientSlice(0,10)
fig = go.Figure()



df = getACoefficientSlice(0,10)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(10,20)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(20,30)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(30,40)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(40,50)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(50,60)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(60,70)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(70,80)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(80,90)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(90,100)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(100,110)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(110,120)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(120,130)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(130,140)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(140,150)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(160,170)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(170,180)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(180,190)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(190,200)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(200,210)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(210,220)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(220,230)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
fig = go.Figure()



df = getACoefficientSlice(230,243)



for predictor in df.index:

    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))

    

fig.update_layout(showlegend=False)

fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))



fig.show()
lasso_coefs_all.drop(columns=['mean'], inplace=True)

lasso_coefs_all.to_csv('Lasso_regr')