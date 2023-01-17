# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head()
#descriptive statistics summary

df_train['price'].describe()
#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
#skewness and kurtosis

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']), plot=plt)
df_train['price'] = np.log1p(df_train['price'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
#saleprice correlation matrix

k = 10 #number of variables for heatmap

corrmat = abs(df_train.corr(method='spearman')) # correlation 전체 변수에 대해서 계산

cols = corrmat.nlargest(k, 'price').index # nlargest : Return this many descending sorted values

cm = np.corrcoef(df_train[cols].values.T) # correlation 특정 컬럼에 대해서

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(18, 8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living15']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living15', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_above']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_above', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bathrooms', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import plotly.graph_objs as go



import time

import random



#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach

def mis_value_graph(data):  

    data = [

    go.Bar(

        x = data.columns,

        y = data.isnull().sum(),

        name = 'Counts of Missing value',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

    layout= go.Layout(

        title= '"Total Missing Value By Column"',

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='skin')

    

def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color



df_all = pd.concat([df_train,df_test])

del df_all['price']

mis_value_graph(df_all)
### 유니크 갯수 계산

train_unique = []

columns = ['bedrooms','bathrooms','floors','waterfront','view','condition','grade']



for i in columns:

    train_unique.append(len(df_train[i].unique()))

unique_train = pd.DataFrame()

unique_train['Columns'] = columns

unique_train['Unique_value'] = train_unique



data = [

    go.Bar(

        x = unique_train['Columns'],

        y = unique_train['Unique_value'],

        name = 'Unique value in features',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

        title= "Unique Value By Column",

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
df_train.loc[df_train['sqft_living'] > 13000]
df_train = df_train.loc[df_train['id']!=8990]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train = df_train.loc[df_train['id']!=456]

df_train = df_train.loc[df_train['id']!=7259]

df_train = df_train.loc[df_train['id']!=2777]
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
df_train.loc[df_train['bedrooms']>=10]
df_test.loc[df_test['bedrooms']>=10]
df_train = df_train.loc[df_train['bedrooms']<10]
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
df_train.head()
for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')
df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')

del df_train['per_price']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import RidgeCV



y_reg = df_train['price']

del df_train['price']

del df_train['id']

test_id = df_test['id']

del df_test['id']



kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



def cv_rmse(model):

    rmse = np.sqrt(-cross_val_score(model, df_train, y_reg, 

                                   scoring="neg_mean_squared_error", 

                                   cv = kfolds))

    return(rmse)



def ridge_selector(k):

    ridge_model = make_pipeline(RobustScaler(),

                                RidgeCV(alphas = [k],

                                        cv=kfolds)).fit(df_train, y_reg)

    

    ridge_rmse = cv_rmse(ridge_model).mean()

    return(ridge_rmse)



r_alphas = [.0001, .0003, .0005, .0007, .0009, 

          .01, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10]



ridge_scores = []

for alpha in r_alphas:

    score = ridge_selector(alpha)

    ridge_scores.append(score)

    

plt.plot(r_alphas, ridge_scores, label='Ridge')

plt.legend('center')

plt.xlabel('alpha')

plt.ylabel('score')
alphas_alt = [5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7]



ridge_model2 = make_pipeline(RobustScaler(),

                            RidgeCV(alphas = alphas_alt,

                                    cv=kfolds)).fit(df_train, y_reg)



print("Ridge rmse : ",cv_rmse(ridge_model2).mean())
print("Best of alpha in ridge model :" ,ridge_model2.steps[1][1].alpha_)
ridge_coef = pd.DataFrame(np.round_(ridge_model2.steps[1][1].coef_, decimals=3), 

df_test.columns, columns = ["penalized_regression_coefficients"])

# remove the non-zero coefficients

ridge_coef = ridge_coef[ridge_coef['penalized_regression_coefficients'] != 0]

# sort the values from high to low

ridge_coef = ridge_coef.sort_values(by = 'penalized_regression_coefficients', 

ascending = False)



# plot the sorted dataframe

fig = plt.figure(figsize = (25,25))

ax = sns.barplot(x = 'penalized_regression_coefficients', y= ridge_coef.index , 

data=ridge_coef)

ax.set(xlabel='Penalized Regression Coefficients')
train_columns = [c for c in df_train.columns if c not in ['id']]
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}

         

#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
##plot the feature importance

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
test_ridge_preds = np.expm1(ridge_model2.predict(df_test))

test_lgb_preds = np.expm1(predictions)

test_ensemble_preds = 0.7*test_lgb_preds + 0.3*test_ridge_preds
submission0 = pd.DataFrame({'id': test_id, 'price': test_ridge_preds})

submission0.to_csv('ridge.csv', index=False)
submission = pd.DataFrame({'id': test_id, 'price': test_lgb_preds})

submission.to_csv('lightgbm.csv', index=False)
submission1 = pd.DataFrame({'id': test_id, 'price': test_ensemble_preds})

submission1.to_csv('WeightAvg1.csv', index=False)