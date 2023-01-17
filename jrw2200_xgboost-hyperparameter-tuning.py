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



from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.metrics import mean_squared_error, balanced_accuracy_score

from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.utils import resample



from xgboost import XGBRegressor

import lightgbm as lgb



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
#sns.pairplot(Combined_data.select_dtypes(exclude=['object']))
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

Processed_data.to_csv('NYC_Airbnb_Processed.dat')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Dimensions of the training feature matrix: {}'.format(X_train.shape))

print('Dimensions of the training target vector: {}'.format(y_train.shape))

print('Dimensions of the test feature matrix: {}'.format(X_test.shape))

print('Dimensions of the test target vector: {}'.format(y_test.shape))
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
n_folds = 5



# squared_loss

def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)

    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)



def rmse_lv_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)

    return cross_val_score(model, Xlv_train, y_train, scoring='neg_mean_squared_error', cv=kf)
xgb_baseline = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping=5)

kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)

cv_res = cross_val_score(xgb_baseline, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

xgb_baseline.fit(X_train, y_train)

y_train_xgb_base = xgb_baseline.predict(X_train)

y_test_xgb_base = xgb_baseline.predict(X_test)

xgb_baseline_results = pd.DataFrame({'algorithm':['XGBRegressor[baseline]'],

            'CV error': cv_res.mean(), 

            'CV std': cv_res.std(),

            'training error': [mean_squared_error(y_train_xgb_base, y_train)]})
d = {'Learning Rate':[],

            'Mean CV Error': [],

            'CV Error Std': [],

            'Training Error': []}

for lr in [0.01, 0.05, 0.1, 0.5]:

    continue

    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=lr, early_stopping=5)

    cv_res = -cross_val_score(xgb_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

    xgb_model.fit(X_train, y_train)

    y_train_xgb = xgb_model.predict(X_train)

    d['Learning Rate'].append(lr)

    d['Mean CV Error'].append(cv_res.mean())

    d['CV Error Std'].append(cv_res.std())

    # makes no sense to look at max/min when we only have 3 CV folds

    #d['Max CV Error'].append(max(cv_res)

    #d['Min CV Error'].append(max(cv_res)

    d['Training Error'].append(mean_squared_error(y_train_xgb, y_train))



#without early stopping

d = {'Learning Rate':[0.01, 0.05, 0.1, 0.5],

        'Mean CV Error': [0.184223, 0.177748, 0.175002, 0.188239],

        'CV Error Std': [0.00626211, 0.00575213, 0.00544426, 0.00525595],

        'Training Error': [0.179093, 0.164874, 0.154238, 0.109885]}



xgb_tuning_1 = pd.DataFrame(d)

xgb_tuning_1
fig, ax = plt.subplots(1, 1, figsize=(20,6))



ax.plot(xgb_tuning_1['Learning Rate'], xgb_tuning_1['Mean CV Error'], color='red')

ax.plot(xgb_tuning_1['Learning Rate'], xgb_tuning_1['Mean CV Error'], 'o', color='black')

ax.fill_between(xgb_tuning_1['Learning Rate'], xgb_tuning_1['Mean CV Error'] - xgb_tuning_1['CV Error Std'], xgb_tuning_1['Mean CV Error'] + xgb_tuning_1['CV Error Std'], color='r', alpha=.1)

ax.plot(xgb_tuning_1['Learning Rate'], xgb_tuning_1['Training Error'], color='blue')

ax.plot(xgb_tuning_1['Learning Rate'], xgb_tuning_1['Training Error'], 'o', color='black')

ax.legend(fontsize=12, loc = 'center right');

ax.set_ylim(0.1, 0.2)

ax.set_xlabel('Learning Rate')

ax.set_ylabel('Mean Squared Error')

#ax.set_title('')
d = {'Max_depth':[],

             'Min_child_weight': [],

            'Mean CV Error': [],

            'CV Error Std': [],

            'Training Error': []}

xgbreg = XGBRegressor(n_estimators=2, learning_rate=0.05, early_stopping=5)

params2 = {'max_depth': list(range(3,10,2)), 'min_child_weight': list(range(1,6,2))}

#print(params2)

#xgb_random.fit(X_train, y_train)

kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(X_train)

for md in params2['max_depth']:

    for mcw in params2['min_child_weight']:

        continue

        xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, early_stopping=5, max_depth=md, min_child_weight=mcw )

        cv_res = -cross_val_score(xgb_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

        xgb_model.fit(X_train, y_train)

        y_train_xgb = xgb_model.predict(X_train)

        d['Max_depth'].append(md)

        d['Min_child_weight'].append(mcw)

        d['Mean CV Error'].append(cv_res.mean())

        d['CV Error Std'].append(cv_res.std())

        # makes no sense to look at max/min when we only have 3 CV folds

        #d['Max CV Error'].append(max(cv_res)

        #d['Min CV Error'].append(max(cv_res)

        d['Training Error'].append(mean_squared_error(y_train_xgb, y_train))



#print(d)



d = {'Max_depth': [3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9], 'Min_child_weight': [1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5], 

 'Mean CV Error': [0.1750024956601357, 0.17483011840929769, 0.17493846554576997, 0.17309889297300166, 0.17316622731288867, 

        0.17351576928079232, 0.17662213266155447, 0.17623539711716868, 0.17586167155362295, 0.18027062402369495, 0.1795815552171006, 0.1794402792605232], 

 'CV Error Std': [0.0054442612607845196, 0.005346726848155686, 0.005781224325978589, 0.0047992091315554805, 0.005078460548746871, 0.0055470435006580825, 

                  0.004522282538112627, 0.005521088520254507, 0.005182127039391581, 0.00548502303198156, 0.0056636180606624885, 0.005837983614899652],

 'Training Error': [0.15423828100740364, 0.1548338435116449, 0.15489721899341147, 0.1174713383813709, 0.11768836644071619, 0.11962286723882598, 

                    0.07157996439924702, 0.07249081997317249, 0.0809473890478948, 0.03364907441870936, 0.03787025803370217, 0.045449523400453724]}

        

xgb_tuning_2 = pd.DataFrame(d)

xgb_tuning_2
fig, ax = plt.subplots(1, 1, figsize=(20,6))



colors = ['green','blue','red']

for i, mcw in enumerate(params2['min_child_weight']):

    color = colors[i]

    xgb_tuning_3 = xgb_tuning_2[xgb_tuning_2['Min_child_weight']==mcw]

    ax.plot(xgb_tuning_3['Max_depth'], xgb_tuning_3['Mean CV Error'], color=color, label= 'mean_child_weight='+str(mcw)+', CV')

    ax.plot(xgb_tuning_3['Max_depth'], xgb_tuning_3['Mean CV Error'], 'o', color='black', label='_nolegend_')

    #ax.fill_between(xgb_tuning_3['Max_depth'], xgb_tuning_3['Mean CV Error'] - xgb_tuning_3['CV Error Std'], 

                    #xgb_tuning_3['Mean CV Error'] + xgb_tuning_3['CV Error Std'], color='r', alpha=.1, label='_nolegend_')

    ax.plot(xgb_tuning_3['Max_depth'], xgb_tuning_3['Training Error'], color=color, label='mean_child_weight='+str(mcw)+', Training')

    ax.plot(xgb_tuning_3['Max_depth'], xgb_tuning_3['Training Error'], 'o', color='black', label='_nolegend_')



ax.legend(fontsize=12, loc = 'center right');

ax.set_ylim(0, 0.25)

ax.set_xlabel('Max_depth')

ax.set_ylabel('Mean Squared Error')

#ax.set_title('')
print('Optimal values of max_depth and mean_child_weight are: \n')

print(xgb_tuning_2.iloc[xgb_tuning_2.idxmin()['Mean CV Error']])
d = {'gamma':[],

            'Mean CV Error': [],

            'CV Error Std': [],

            'Training Error': []}

params3 = {'gamma': [i/10.0 for i in range(0,12, 2)]}

#print(params2)

#xgb_random.fit(X_train, y_train)

kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(X_train)

for g in params3['gamma']:

        #print(g)

        #continue

        xgb_model = XGBRegressor(n_estimators=1000, learning_rate=lr, early_stopping=5, max_depth=5, min_child_weight=1, gamma = g)

        cv_res = -cross_val_score(xgb_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

        xgb_model.fit(X_train, y_train)

        y_train_xgb = xgb_model.predict(X_train)

        d['gamma'].append(g)

        d['Mean CV Error'].append(cv_res.mean())

        d['CV Error Std'].append(cv_res.std())

        # makes no sense to look at max/min when we only have 3 CV folds

        #d['Max CV Error'].append(max(cv_res)

        #d['Min CV Error'].append(max(cv_res)

        d['Training Error'].append(mean_squared_error(y_train_xgb, y_train))

        

print(d)



xgb_tuning_4 = pd.DataFrame(d)

xgb_tuning_4