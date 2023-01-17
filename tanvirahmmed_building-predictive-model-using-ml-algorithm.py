import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
df = pd.read_csv('/kaggle/input/house-prices-data/train.csv')
dt = pd.read_csv('/kaggle/input/house-prices-data/test.csv')
test_dataY = pd.read_csv('/kaggle/input/submission/sample_submission.csv')
sns.boxplot(x=df['SalePrice'])
df = df[df.SalePrice < 350000]
df.reset_index(drop=True, inplace=True)
df.shape
sns.boxplot(x=df['SalePrice'])
df.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
dt.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
y = df['SalePrice']
df.drop(['SalePrice'], axis = 1, inplace = True)
data = pd.concat([df,dt], axis = 0)
data.shape
(((data.isnull().sum())*100)/len(data)).sort_values(
            ascending = False, kind = 'mergesort').head(30)
year_all = ['YearBuilt', 'YearRemodAdd','YrSold','MoSold','GarageYrBlt']
for i in year_all:
    data[i] = data[i].astype(object)
qual_listt = ['HeatingQC','OverallQual','ExterQual','BsmtQual','KitchenQual','FireplaceQu','GarageQual']
cond_listt = ['OverallCond','ExterCond','BsmtCond','GarageCond']

data['BsmtQual'] = data['BsmtQual'].fillna('NA')
dic = {'NA':.5,'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 3.5, 'Ex': 5}
for i in (qual_listt+cond_listt):
  data[i] = data[i].fillna(data[i].mode()[0])
  if data[i].dtype == object:
    data[i] = data[i].map(dic)
data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')
data['GarageType'] = data['GarageType'].fillna('NA')
data['GarageFinish'] = data['GarageFinish'].fillna('NA')
for j in data:
    if data[j].dtype == object:
        data[j] = data[j].fillna(data[j].mode()[0])
    else:
        data[i] = data[i].astype('float64')
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mean())
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean())
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean())
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean())
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mode()[0])
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0])
data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mode()[0])
house_style = {'1.5Unf':1,'SFoyer':2, '1.5Fin': 3, '2.5Unf': 4, 'SLvl': 5, '1Story': 6, '2Story': 7, '2.5Fin': 8}
utilities = {'NoSeWa':1,'AllPub':2}
roof_matl = {'Roll':1,'ClyTile':2, 'CompShg': 3, 'Metal': 4, 'Tar&Grv': 5, 'WdShake': 6, 'Membran': 7, 'WdShngl': 8}
heating = {'Floor':1,'Grav':2, 'Wall': 3, 'OthW': 4, 'GasW': 5, 'GasA': 6}
electrical = {'Mix':1,'FuseP':2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}

data['Utilities'] = data['Utilities'].map(utilities)
data['HouseStyle'] = data['HouseStyle'].map(house_style)
data['RoofMatl'] = data['RoofMatl'].map(roof_matl)
data['Heating'] = data['Heating'].map(heating)
data['Electrical'] = data['Electrical'].map(electrical)
data['RemodAdd'] = data['YearBuilt']

for i in range(len(data)):
    if data['YearBuilt'].iloc[i] == data['YearRemodAdd'].iloc[i]:
        data['RemodAdd'].iloc[i] = 0
    else:
        data['RemodAdd'].iloc[i] = abs(data['YearBuilt'].iloc[i]- data['YearRemodAdd'].iloc[i])

data['DiffEx'] = data['ExterCond']

for i in range(len(data)):
    if data['ExterQual'].iloc[i] == data['ExterCond'].iloc[i]:
        data['DiffEx'].iloc[i] = 0
    else:
        data['DiffEx'].iloc[i] = abs(data['ExterQual'].iloc[i]- data['ExterCond'].iloc[i])


data['CompletedBstmSf'] = data['TotalBsmtSF'] - data['BsmtUnfSF']
data['CompletedFloorSF'] = data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = data['BsmtFullBath'] + data['BsmtHalfBath'] + data['FullBath'] + data['HalfBath']
data['GarageAreaPerCar'] = (data['GarageArea']+1) / (data['GarageCars'] +1)
data['TotalExtraArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch']+ data['PoolArea']
data['AgeOfHouse'] = abs(data['YrSold'] - data['YearBuilt'])

data.select_dtypes(exclude=object).shape
tx = data.iloc[:len(y), :]
tx = np.log1p(tx.select_dtypes(exclude=object).copy())
#tx.drop(['SalePrice'],axis=1,inplace=True)
ty = np.log1p(y)
X_train, X_test, y_train, y_test = train_test_split(tx, ty, test_size=0.2, random_state=13)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#print(pos)
#print(np.array(tx.columns)[sorted_idx])
fig = plt.figure(figsize=(40, 30))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(tx.columns)[sorted_idx])
aa = (pos, np.array(tx.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')
aa[1]
xgb_selection = ['GarageAreaPerCar',
       'HeatingQC', 'TotalExtraArea', 'LotFrontage', 'BsmtFinSF1',
       'Fireplaces', 'YearRemodAdd', 'AgeOfHouse', 'BsmtQual', '1stFlrSF',
       'YearBuilt', 'CompletedBstmSf', 'GarageCars', 'OverallCond',
       'LotArea', 'GarageArea', 'GrLivArea', 'ExterQual', 'KitchenQual',
       'TotalBath', 'TotalBsmtSF', 'CompletedFloorSF', 'OverallQual']
tx = data.iloc[:len(y), :]
tx = np.log1p(tx.select_dtypes(exclude=object).copy())
filter = []
for i in tx:
  if tx[i].dtypes != object:
    r = tx[i].corr(np.log1p(y))
    if r >= .3 or r<=-.3:
      print(i,' ',r)
      filter.append(str(i))
# feature selection
def select_features(X, Y, func):
  bestfeatures = SelectKBest(score_func=func, k='all')
  fit = bestfeatures.fit(X,Y)
  return fit,bestfeatures
fit,fs = select_features(tx, np.log1p(y), mutual_info_regression)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(tx.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 

mutual_info = featureScores.nlargest(30,'Score')
mutual_info = list(mutual_info['Specs'])
print(len(mutual_info),'\n',mutual_info)
print(featureScores.nlargest(32,'Score'))

skewed = []
c = 0
for i in tx:
  if tx[i].skew() <=.5 and tx[i].skew() >=-.5:
    skewed.append(i)
    c+=1
    #print('fairly symmetrical: ',i,'\n')
  elif tx[i].skew() <=-.5 and tx[i].skew() >=-1:
    skewed.append(i)
    c+=1
    #print('negatively skewed: ',i,'\n')
  elif tx[i].skew() >=.5 and tx[i].skew() <=1:
    skewed.append(i)
    c+=1
    #print('positively skewed: ',i,'\n')
  #elif tx[i].skew() <-1:
    #skewed.append(i)
    #c+=1
    #print('negatively highly skewed: ',i,'\n')
  #elif tx[i].skew() >1:
    #skewed.append(i)
    #c+=1
    #print('positively highly skewed: ',i,'\n')
print(c)
len(skewed),len(filter),len(xgb_selection), len(mutual_info)
c = 0
c1 = 0
for i in skewed:
  if i in filter:
    c+=1
  if i in xgb_selection:
    c1+=1
print(c,' ',c1)
data.select_dtypes(include=object).shape
features = data.copy()
features_skewed = features.filter(skewed,axis = 1)
features_filter = features.filter(filter,axis = 1)
features_xgb_selection = features.filter(xgb_selection,axis = 1)
features_mutual_info = features.filter(mutual_info,axis = 1)


features_skewed = pd.concat([features_skewed,features.select_dtypes(include=object)], axis = 1)
features_mutual_info = pd.concat([features_mutual_info,features.select_dtypes(include=object)], axis = 1)
features_filter = pd.concat([features_filter,features.select_dtypes(include=object)], axis = 1)
features_xgb_selection = pd.concat([features_xgb_selection,features.select_dtypes(include=object)], axis = 1)

features_skewed.shape,features_filter.shape,features_xgb_selection.shape,features_mutual_info.shape
features_skewed = pd.get_dummies(features_skewed, drop_first=True)
features_filter = pd.get_dummies(features_filter, drop_first=True)
features_mutual_info = pd.get_dummies(features_mutual_info, drop_first=True)
features_xgb_selection = pd.get_dummies(features_xgb_selection, drop_first=True)
all_features = pd.get_dummies(features, drop_first=True)

all_features.shape, features_skewed.shape,features_filter.shape,features_xgb_selection.shape,features_mutual_info.shape
'''
HistGradientBoostingRegressor(l2_regularization=0, learning_rate=0.1,
                          loss='least_absolute_deviation', max_bins=255,
                          max_depth=15, max_iter=500, max_leaf_nodes=15,
                          min_samples_leaf=20, n_iter_no_change=None,
                          random_state=None, scoring=None, tol=1e-07,
                          validation_fraction=0.1, verbose=0,
                          warm_start=False),'''
classifiers = [
    LinearRegression(),
    Ridge(alpha=.7),
    PassiveAggressiveRegressor(max_iter=100000, random_state=5000,tol=1e-3),
    AdaBoostRegressor(random_state=3500, n_estimators=1000,loss='square'),
    GradientBoostingRegressor(n_estimators=5000, learning_rate=0.009,
                                max_depth=25, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=300),
    XGBRegressor(learning_rate=0.001, n_estimators=3500,
                       max_depth=5, min_child_weight=0,
                       gamma=0.01, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=30,
                       reg_alpha=0.00005),
    
    SVR(C=20, epsilon=0.009, gamma=0.0004, )

]
all_data = [all_features,features_skewed, features_filter, features_xgb_selection, features_mutual_info]
data_name = ['all_features','features_skewed', 'features_filter', 'features_xgb_selection', 'features_mutual_info']
Y = np.log1p(y)
for data,nm in zip(all_data,data_name):
    print(nm)
    train = data.iloc[:len(y), :]
    test = data.iloc[len(train):, :]
    X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.1, random_state=1)
    for clf in classifiers:
        try:
            clf.fit(X_train,y_train)
            print(clf.__class__.__name__,' ', round(clf.score(X_test, y_test) * 100, 2))
            print(mean_squared_error(y_test, clf.predict(X_test)))
            print()
        except:
            continue
            #print('hello') 
    print('___END___')
    
l = 0
log_data = data.copy()
sqrt_data = data.copy()
box_data = data.copy()
for i in box_data:
    if box_data[i].dtypes != object:
        if box_data[i].skew() <=.5 and box_data[i].skew() >=-.5:
            box_data[i],lam = stats.boxcox(box_data[i]+1)
            log_data[i] = np.log1p(log_data[i])
        sqrt_data[i] = np.sqrt(sqrt_data[i])
log_data = pd.get_dummies(log_data, drop_first=True)
sqrt_data = pd.get_dummies(sqrt_data, drop_first=True)
box_data = pd.get_dummies(box_data, drop_first=True)
y_log = np.log1p(y)
y_box,l = stats.boxcox(y+1)
all_data = [log_data,sqrt_data, box_data]
all_y = [y_log, y_box]
data_name = ['log_data','sqrt_data', 'box_data']
Y = np.log1p(y)
for data,nm in zip(all_data,data_name):
    print(nm)
    for dy in all_y:
        train = data.iloc[:len(y), :]
        #test = data.iloc[len(train):, :]
        X_train, X_test, y_train, y_test = train_test_split(train, dy, test_size=0.1, random_state=1)
        for clf in classifiers:
            try:
                clf.fit(X_train,y_train)
                print(clf.__class__.__name__,' ', round(clf.score(X_test, y_test) * 100, 2))
                print(mean_squared_error(y_test, clf.predict(X_test)))
                print()
            except:
                continue
                #print('hello') 
        print('__end__')
    print('___END___')
    
scale = [
    StandardScaler(),
    MinMaxScaler(),
    RobustScaler(),
    PowerTransformer(method='yeo-johnson'),
    QuantileTransformer(output_distribution='normal'),
    QuantileTransformer(output_distribution='uniform'),
    Normalizer()
]
train = features_skewed.iloc[:len(y), :]
test = features_skewed.iloc[len(train):, :]

for scl in scale:
    print(scl)
    train = data.iloc[:len(y), :]
    #test = data.iloc[len(train):, :]
    try:
        train = scl.fit_transform(train)
        X_train, X_test, y_train, y_test = train_test_split(train, y_log, test_size=0.1, random_state=1)
        for clf in classifiers:
            clf.fit(X_train,y_train)
            print(clf.__class__.__name__,' ', round(clf.score(X_test, y_test) * 100, 2))
            print(mean_squared_error(y_test, clf.predict(X_test)))
            print()
    except:
        continue
        #print('hello') 
    print('___END___')
train = log_data.iloc[:len(y), :]
test = log_data.iloc[len(train):, :]
X_train, X_test, y_train, y_test = train_test_split(train, y_log, test_size=0.1, random_state=1)
'''
parameters = {'n_estimators':[5000,4000], 'learning_rate':[.005,.009,.0001],
              'max_depth':[30,45],'max_features':['sqrt'],'min_samples_leaf':[20,25],
              'min_samples_split':[10,20],'loss':['ls','huber'],'random_state':[500,1000]}
'''
clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,
                                max_depth=50, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=20,
                                loss='huber', random_state=500)
scores = []
cv = KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in cv.split(train):
    X_train, X_test, y_train, y_test = train.iloc[train_index], train.iloc[test_index], y_log[train_index], y_log[test_index]
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
    lin_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    print('Mean squared error: %.2f' % mean_squared_error(y_test, lin_pred))
    #The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'% r2_score(y_test, lin_pred))

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print('Mean squared error: %.2f' % mean_squared_error(y_test, clf.predict(X_test)))
y_pred = clf.predict(test)
y_pred = np.expm1(y_pred)

pred_y = y_pred.reshape(-1)
all_id = np.array(test_dataY['Id'])
y_pred = pd.DataFrame(list(zip(all_id, pred_y)),columns =['Id', 'SalePrice'])
y_pred.to_csv("svr.csv", index=False)
