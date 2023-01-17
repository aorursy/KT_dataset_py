import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd 



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split



from sklearn.preprocessing import LabelEncoder, RobustScaler



from sklearn.linear_model import Lasso, Ridge

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor





from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score
# read data

red = pd.read_csv('../input/wine-rating-and-price/Red.csv')

white = pd.read_csv('../input/wine-rating-and-price/White.csv')

sparkling = pd.read_csv('../input/wine-rating-and-price/Sparkling.csv')

rose = pd.read_csv('../input/wine-rating-and-price/Rose.csv')
red.head()
red['WineStyle'] = 'red'

white['WineStyle'] = 'white'

sparkling['WineStyle'] = 'sparkling'

rose['WineStyle'] = 'rose'

wines =  pd.concat([red, white, sparkling, rose], ignore_index=True)
wines.info()
# N.V. wines is a nonvintage wine, which is usually a blend from the produce of two or more years

# we can choose any free number to encode it



wines['Year'] = wines['Year'].replace('N.V.', 2030) # it's important, that there were no 2030 year wines in list before

wines['Year'] = wines['Year'].astype('int')
wines.sample(frac=1).head()
wines.shape
wines.info()
wines.describe()
wines.Country.nunique()
wines.Country.value_counts()
LEV_countries = wines.Country.value_counts()[:12] #Countries with the largest export volume
plt.figure(figsize=(10,4))



country = wines.Country.value_counts()[:12]



graph = sns.countplot(x='Country', 

                  data=wines[wines.Country.isin(LEV_countries.index.values)],

                 color='olive')

graph.set_title("Countries with the largest export volume", fontsize=20)

graph.set_xlabel("Country", fontsize=15)

graph.set_ylabel("Volume", fontsize=15)

graph.set_xticklabels(graph.get_xticklabels(),rotation=45)



plt.show()

plt.figure(figsize=(10, 4))

graph = sns.countplot(x='Rating', data=wines, color='mediumpurple')

graph.set_title("Rating Count distribuition ", fontsize=20)

graph.set_xlabel("Rating", fontsize=15) 

graph.set_ylabel("Count", fontsize=15)

plt.show()
plt.figure(figsize=(16,6))



graph = sns.boxplot(x='Country', y='Rating',

                 data=wines[wines.Country.isin(LEV_countries.index.values)],

                 color='mediumpurple')

graph.set_title("Rating by Country", fontsize=20)

graph.set_xlabel("Country", fontsize=15)

graph.set_ylabel("Rating", fontsize=15)

graph.set_xticklabels(graph.get_xticklabels())



plt.show()
MP_regions = wines['Region'].value_counts()[:100].index #most productive regions

print(wines[wines['Region'].isin(MP_regions)].groupby('Region').Rating.mean().sort_values(ascending=False)[:20])

#Regions with the best rating from most productive onece
MP_wineries = wines['Winery'].value_counts()[:100].index #most productive wineries

print(wines[wines['Winery'].isin(MP_wineries)].groupby('Winery').Rating.mean().sort_values(ascending=False)[:20])

#wineries with the best rating from most productive onece
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

graph = sns.distplot(wines['Price'], color='coral')

graph.set_title("Price distribuition", fontsize=20) # seting title and size of font

graph.set_xlabel("Price (EUR)", fontsize=15) # seting xlabel and size of font

graph.set_ylabel("Frequency", fontsize=15) # seting ylabel and size of font



plt.subplot(2,1,2)

graph1 = sns.distplot(np.log(wines['Price']) , color='coral')

graph1.set_title("Price Log distribuition", fontsize=20) # seting title and size of font

graph1.set_xlabel("Price(EUR)", fontsize=15) # seting xlabel and size of font

graph1.set_ylabel("Frequency", fontsize=15) # seting ylabel and size of font

graph1.set_xticklabels(np.exp(graph1.get_xticks()).astype(int))



plt.subplots_adjust(hspace = 0.3,top = 0.9)

plt.show()
plt.figure(figsize=(16,18))



plt.subplot(3,1,1)

graph = sns.boxplot(x='Year', y=np.log(wines['Price']),

                    data=wines,

                    color='coral')

graph.set_title("Price by Year", fontsize=20)

graph.set_xlabel("Year", fontsize=15)

graph.set_ylabel("Price(EUR)", fontsize=15)

graph.set_xticklabels(graph.get_xticklabels(),rotation=45)

graph.set_yticklabels(np.exp(graph.get_yticks()).astype(int))



plt.subplot(3,1,2)

graph1 = sns.boxplot(x='WineStyle', y=np.log(wines['Price']),

                 data=wines,

                 color='coral')

graph1.set_title("Price by WineStyle", fontsize=20)

graph1.set_xlabel("WineStyle", fontsize=15)

graph1.set_ylabel("Price(EUR)", fontsize=15)

graph1.set_xticklabels(graph1.get_xticklabels())

graph1.set_yticklabels(np.exp(graph1.get_yticks()).astype(int))



plt.subplot(3,1,3)

graph2 = sns.boxplot(x='Country', y=np.log(wines['Price']),

                 data=wines[wines.Country.isin(LEV_countries.index.values)],

                 color='coral')

graph2.set_title("Price by Country", fontsize=20)

graph2.set_xlabel("Country", fontsize=15)

graph2.set_ylabel("Price(EUR)", fontsize=15)

graph2.set_yticklabels(np.exp(graph2.get_yticks()).astype(int))



plt.subplots_adjust(hspace = 0.3, top = 0.9)



plt.show()
plt.figure(figsize=(16,6))

graph = sns.boxplot(x='Country', y=wines['Rating']/wines['Price'],

                 data=wines[wines.Country.isin(LEV_countries.index.values)],

                 color='olive')

graph.set_title("Rating/Price by Countries", fontsize=20)

graph.set_xlabel("Country", fontsize=15)

graph.set_ylabel("Rating/Price", fontsize=15)

graph.set_xticklabels(graph.get_xticklabels())



plt.show()
plt.figure(figsize=(13,5))



graph = sns.regplot(x=np.log(wines['Price']), y='Rating', 

                    data=wines, fit_reg=False, color='olive')

graph.set_title("Rating x Price Distribuition", fontsize=20)

graph.set_xlabel("Price(EUR)", fontsize= 15)

graph.set_ylabel("Rating", fontsize= 15)

graph.set_xticklabels(np.exp(graph.get_xticks()).astype(int))



plt.show()
corrs = wines[['Rating','NumberOfRatings','Price','Year']].corr() #Heatmap for numetrical columns

fig, ax = plt.subplots(figsize=(7,5))        



sns.heatmap(corrs,annot = True,ax=ax,linewidths=.6, cmap = 'YlGnBu');
plt.figure(figsize=(10,15))



plt.subplot(3,1,1)

graph = sns.distplot(wines['NumberOfRatings'], color='olive')

graph.set_title("Number Of Ratings distribuition", fontsize=20) 

graph.set_xlabel("Number Of Ratings", fontsize=15)

graph.set_ylabel("Frequency", fontsize=15) 



plt.subplot(3,1,2)

graph1 = sns.distplot(np.log(wines['NumberOfRatings']), color='olive')

graph1.set_title("Number Of Ratings Log distribuition", fontsize=20) 

graph1.set_xlabel("Number Of Ratings", fontsize=15) 

graph1.set_ylabel("Frequency", fontsize=15)

graph1.set_xticklabels(np.exp(graph1.get_xticks()).astype(int))



plt.subplot(3,1,3)

graph = sns.distplot(wines[wines['NumberOfRatings']<1000]['NumberOfRatings'], color='olive')

graph.set_title("Number Of Ratings <1000 distribuition", fontsize=20)

graph.set_xlabel("Number Of Ratings", fontsize=15) 

graph.set_ylabel("Frequency", fontsize=15) 



plt.subplots_adjust(hspace = 0.3,top = 0.9)

plt.show()
varieties = pd.read_csv('../input/wine-rating-and-price/Varieties.csv')
wines['Variety'] = np.nan

for index in wines.index:

    for variety in varieties['Variety']:    

        if variety in wines.loc[index, 'Name']:

            wines.loc[index, 'Variety'] = variety

            break
print('Now we have variety for', wines.Variety.notna().sum(),'wines,',

      '%s%%' % int(wines.Variety.notna().sum()/len(wines)*100), 'of all')
# replace NaN's

wines.Variety = wines.Variety.fillna('unknown')
wines.Variety.value_counts().head(20)
wines_enc = wines.copy().drop(columns = ['Name'])
#One-hot encoder for winestyle

wines_enc = pd.get_dummies(wines_enc, columns = ['WineStyle'])
wines_enc.head()
categorical_cols = [col for col in wines_enc.columns if wines_enc[col].dtype == "object"]
# Apply label encoder

label_encoder = LabelEncoder()

for col in categorical_cols:

    wines_enc[col] = label_encoder.fit_transform(wines_enc[col])
wines_enc.head()
y = wines_enc['Rating']

X = wines_enc.drop(['Rating'], axis = 1)
kfolds = KFold(n_splits=6, shuffle=True,

               random_state=0)
def cv_mae(model, X=X, y=y):

    mae = -cross_val_score(model, X, y,

                          scoring="neg_mean_absolute_error",

                          cv=kfolds)

    return mae
lightgbm = LGBMRegressor(objective='regression',

                         metric='mean_absolute_error',

                         num_leaves=10,

                         learning_rate=0.05,

                         n_estimators=3000,

                         max_depth=5,

                         max_bin=400,

                         bagging_fraction=0.75,                         

                         bagging_freq=5,

                         bagging_seed=7,

                         reg_alpha=0.7,

                         reg_lambda=1.2,

                         feature_fraction=0.6,

                         feature_fraction_seed=7,

                         verbose=-1,

                         min_data_in_leaf=3,

                         min_sum_hessian_in_leaf=11

                         )

xgboost = XGBRegressor(n_estimators=3000,

                       learning_rate=0.02,

                       max_depth=5, 

                       min_child_weight=2,

                       subsample=0.8,

                       colsample_bytree=0.7,

                       nthread=-1,

                       gamma=0,

                       reg_alpha=0.1,

                       reg_lambda=1.8

                       )

catboost = CatBoostRegressor(iterations=3000,

                             learning_rate=0.03,

                             depth=6,

                             l2_leaf_reg = 2,

                             verbose=0

                            )
maes_lgbm = cv_mae(lightgbm)
print('Average lightgbm mae:', np.average(maes_lgbm), ' Standard deviation: ', np.std(maes_lgbm))
maes_xgb = cv_mae(xgboost)
print('Average xgboost mae:', np.average(maes_xgb), ' Standard deviation: ', np.std(maes_xgb))
maes_catboost = cv_mae(catboost)
print('Average catboost mae:', np.average(maes_catboost), ' Standard deviation: ', np.std(maes_catboost))
lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100,150,200,250,300]}

lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_absolute_error',cv=kfolds)

lasso_regressor.fit(X,y)

print('Best lasso mae:', -lasso_regressor.best_score_,'with',lasso_regressor.best_params_)
ridge=Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100,150,200,250,300]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_absolute_error',cv=kfolds)

ridge_regressor.fit(X,y)

print('Best ridge mae:', -ridge_regressor.best_score_,'with',ridge_regressor.best_params_)
wines_enc['NumberOfRatings'].sort_values()[int(0.1*(len(wines_enc))):int(0.9*(len(wines_enc)))]
wines_low_NumberOfRatings = wines_enc[wines_enc['NumberOfRatings']<35]

wines_high_NumberOfRatings = wines_enc[wines_enc['NumberOfRatings']>838]

wines_mid_NumberOfRatings = wines_enc[wines_enc['NumberOfRatings']>35][wines_enc['NumberOfRatings']<838]
X_low_NumberOfRatings_test = wines_low_NumberOfRatings.drop(['NumberOfRatings','Rating'], axis = 1)

y_low_NumberOfRatings_test = wines_low_NumberOfRatings['Rating']



X_high_NumberOfRatings_test = wines_high_NumberOfRatings.drop(['NumberOfRatings','Rating'], axis = 1)

y_high_NumberOfRatings_test = wines_high_NumberOfRatings['Rating']



X_mid = wines_mid_NumberOfRatings.drop(['NumberOfRatings','Rating'], axis = 1)

y_mid = wines_mid_NumberOfRatings['Rating']

X_train, X_random_test, y_train, y_random_test = train_test_split(X_mid, y_mid, test_size=len(X_low_NumberOfRatings_test))
print('Train data size:', len(X_train))

print('Test data sizes:', len(X_low_NumberOfRatings_test), len(X_high_NumberOfRatings_test), len(X_random_test))
lgbm = lightgbm.fit(X_train, y_train)
res_low_NumberOfRatings = lgbm.predict(X_low_NumberOfRatings_test)

res_high_NumberOfRatings = lgbm.predict(X_high_NumberOfRatings_test)

res_random_NumberOfRatings = lgbm.predict(X_random_test)
print('MAE of predictions with low NumberOfRatings:   ', mean_absolute_error(y_low_NumberOfRatings_test, res_low_NumberOfRatings))

print('MAE of predictions with high NumberOfRatings:  ', mean_absolute_error(y_high_NumberOfRatings_test, res_high_NumberOfRatings))

print('MAE of predictions with middle NumberOfRatings:', mean_absolute_error(y_random_test, res_random_NumberOfRatings))