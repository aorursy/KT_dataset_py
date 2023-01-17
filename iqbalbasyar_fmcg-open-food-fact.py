import numpy as np 

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_palette(sns.cubehelix_palette(10))



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
food = pd.read_csv('../input/world-food-facts/en.openfoodfacts.org.products.tsv', delimiter='\t', low_memory=False)
fig, ax = plt.subplots(1,1,figsize=(25,10))

fig= sns.barplot(x='index',y="main_category_en", data=food['main_category_en'].value_counts().reset_index().head(10))

ax.set(ylabel = 'Frequency', xlabel='Category', title='Categories Frequencies')

plt.show(fig)
selected_categories = ['Plant-based foods and beverages', 'Beverages']

selected_categories = ['Meats']

target_var = 'nutrition-score-uk_100g'



pbf = food[food['main_category_en'].isin(selected_categories)] 



threshold = 0.5 #atleast 50% data should not na

df = pbf.select_dtypes('number').dropna(axis=1, thresh = int(len(pbf)*threshold)).dropna(how='all').dropna(subset=['nutrition-score-uk_100g']).drop(columns=['nutrition-score-fr_100g'])

df.loc[:,df.columns != 'nutrition-score-uk_100g'] = df.loc[:,df.columns != 'nutrition-score-uk_100g'].fillna(0)

df
corrmat = df.corr()

sns.set(context="paper", font_scale = 1.2)

f, ax = plt.subplots(figsize=(11, 11))

cols = corrmat.nlargest(25, 'nutrition-score-uk_100g')['nutrition-score-uk_100g'].index

cm = corrmat.loc[cols, cols] 

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 9}, linewidth = 0.1,

                 yticklabels=cols.values, xticklabels=cols.values, )

f.text(0.5, 0.93, "Correlation with nutrition score", ha='center', fontsize = 24)

plt.show()
df = df.drop(columns=['salt_100g'])

df
from scipy import stats

from scipy.stats import norm, skew 



fig, ax = plt.subplots(figsize=(20, 10))

sns.distplot(df[target_var], fit=norm)

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df[target_var])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')



#Get also the QQ-plot

f, ax = plt.subplots(figsize=(20, 10))

res = stats.probplot(df[target_var], plot=plt)

plt.show()
def classify(x):

    if x <= 9:

        return ('A')

    else:

        return ('B')



df_plot = df.copy()

df_plot['class'] = df_plot[target_var].apply(classify)

df_plot['class'].value_counts()
fig, ax = plt.subplots(1,2, figsize = (20,5))

sns.distplot(df_plot[df_plot['class'] == 'A'][target_var], color = "blue", ax = ax[0], bins=10)

sns.distplot(df_plot[df_plot['class'] == 'B'][target_var], color = "orange", ax = ax[1], bins=10)

ax[0].set_title("Class A")

ax[1].set_title("Class B")
from sklearn.manifold import TSNE

predictor = df.columns[df.columns != target_var]

X_embedded = TSNE(n_components=2, random_state = 127001).fit_transform(df[predictor])

df_embedded = pd.DataFrame(X_embedded)
df_combined = df_embedded.copy()

df_combined['class'] = df_plot['class'].values
fig, ax = plt.subplots(figsize=(15,15))

ax = sns.scatterplot(x=0, y=1, data=df_combined, hue='class')

ax.set(ylabel = 'Embedded Feature 2', xlabel='Embedded Feature 1')

fig.text(0.5, 0.93, "Meat Data Distribution (After Embedded)", ha='center', fontsize = 24)

plt.show()
fig, ax = plt.subplots(10,1, figsize = (20,30))

for i in range(10):

    sns.violinplot(df[predictor[i]], ax=ax[i])

fig.tight_layout(pad=5)

plt.show()
c1 = additives_outlier = df['additives_n'] > 10

c2 = ing_oil_outlier = df['ingredients_from_palm_oil_n'] > 0.2

c3 = ing_may_oil_outlier = df['ingredients_that_may_be_from_palm_oil_n'] > 1

c4 = energy_outlier = df['energy_100g'] > 2500

c5 = fat_outlier = df['fat_100g'] > 60

c6 = sat_fat_outlier = df['saturated-fat_100g'] > 25

c7 = carbo_outlier = df['carbohydrates_100g'] > 20

c8 = sugar_outlier = df['sugars_100g'] > 5

c9 = protein_outlier = df['proteins_100g'] > 40

c10 =sodium_outlier = df['sodium_100g'] > 3



# drop all outlier

df_clean = df.drop(df[c1|c2|c3|c4|c5|c6|c7|c8|c9|c10].index)

df_clean
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



X_train, X_test, y_train, y_test = train_test_split(df_clean[predictor], df_clean[target_var], test_size=0.1, random_state=127001)



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=127001).get_n_splits(df_clean.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def rmsle(model):

    rmse = np.sqrt(mean_squared_error(model.predict(X_test), y_test))

    return rmse
# Base models 

linear = LinearRegression()

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=127001))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=127001))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =127001)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =127001, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# base models score

rmse_linear = rmsle_cv(linear)

rmse_lasso = rmsle_cv(lasso)

rmse_enet = rmsle_cv(ENet)

rmse_krr = rmsle_cv(KRR)

rmse_gboost = rmsle_cv(GBoost)

rmse_xgb = rmsle_cv(model_xgb)

rmse_lgb = rmsle_cv(model_lgb)
print(f"""Model RMSE on {n_folds} Folds of Training Data

linear model:\t mean= {rmse_linear.mean()},\t std= {rmse_linear.std()}

lasso model:\t mean= {rmse_lasso.mean()},\t std= {rmse_lasso.std()}

enet model:\t mean= {rmse_enet.mean()},\t std= {rmse_enet.std()}

krr model:\t mean= {rmse_krr.mean()},\t std= {rmse_krr.std()}

gboost model:\t mean= {rmse_gboost.mean()},\t std= {rmse_gboost.std()}

xgboost model:\t mean= {rmse_xgb.mean()},\t std= {rmse_xgb.std()}

lgboost model:\t mean= {rmse_lgb.mean()},\t std= {rmse_lgb.std()}

""")
linear.fit(X_train, y_train)

lasso.fit(X_train, y_train)

ENet.fit(X_train, y_train)

KRR.fit(X_train, y_train)

GBoost.fit(X_train, y_train)

model_xgb.fit(X_train, y_train)

model_lgb.fit(X_train, y_train)
rmse_t_linear = rmsle(linear)

rmse_t_lasso = rmsle(lasso)

rmse_t_enet = rmsle(ENet)

rmse_t_krr = rmsle(KRR)

rmse_t_gboost = rmsle(GBoost)

rmse_t_xgb = rmsle(model_xgb)

rmse_t_lgb = rmsle(model_lgb)
print(f"""Model RMSE on Data Test

linear model:\t mean= {rmse_t_linear}

lasso model:\t mean= {rmse_t_lasso}

enet model:\t mean= {rmse_t_enet}

krr model:\t mean= {rmse_t_krr}

gboost model:\t mean= {rmse_t_gboost}

xgboost model:\t mean= {rmse_t_xgb}

lgboost model:\t mean= {rmse_t_lgb}

""")