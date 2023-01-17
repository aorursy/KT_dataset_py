#import libs
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
# see the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv');
sns.distplot(train['SalePrice'])
# outlier first
# pos = np.where(train['SalePrice'] < 40000)
# train.drop(labels=pos[0],axis=0,inplace=True)
# train.drop(labels=[825,88],axis=0,inplace=True)
train_len = len(train)
test_len = len(test)
print(train_len)
print(test_len)
ids = test['Id']
if 'SalePrice' in train:
    y = train['SalePrice']

if 'Id' in test:
    ids = test.pop('Id')
    train.pop('Id')
dataset = pd.concat([train, test],axis=0,ignore_index=True)
dataset.reset_index(drop=True)
print(dataset.shape)
# dataset.isnull().sum()
dataset = dataset.reindex(index=range(dataset.shape[0]))
dataset['Alley'].unique()
null_alley = dataset.loc[dataset['Alley'].isnull(),'Alley']
null_num = len(null_alley)
L = [ 'Grvl', 'Pave']
np.random.seed(125)
random_list = np.random.random_integers(1,null_num,null_num)
random_val = [ L[ x % len(L)] for x in random_list]
dataset.loc[dataset['Alley'].isnull(),'Alley'] = random_val
dataset['Alley'].unique()
sns.factorplot(x='Alley',y='SalePrice',data=dataset,kind='strip',jitter=True)
Bsmt_fl = ['BsmtCond','BsmtExposure','BsmtFinSF1','BsmtFinSF2','BsmtFinType1','BsmtFinType2','BsmtFullBath','BsmtHalfBath','BsmtQual','BsmtUnfSF']
pd.Series([ True, False]) & pd.Series ([False , True] )
all_null_basement = dataset['BsmtCond'].isnull() &\
dataset['BsmtExposure'].isnull()  &\
dataset['BsmtFinType1'].isnull()  &\
dataset['BsmtFinType2'].isnull()  &\
dataset['BsmtQual'].isnull() &\
(dataset['BsmtFinSF1'] ==0.0)  &\
(dataset['BsmtFinSF2'] ==0.0) &\
(dataset['BsmtUnfSF'] == 0) &\
(dataset['BsmtFullBath'] == 0) &\
(dataset['BsmtHalfBath'] == 0)

print('all null and o sum is' ,sum (all_null_basement.where(all_null_basement==True,0) ) )
all_notnull_basement = (all_null_basement == False)
dataset['FullnullBsmt']=all_null_basement
sns.factorplot(x='FullnullBsmt',y='SalePrice',data=dataset,kind='strip')
Bsmt_catigorail_fl = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']
conds =  (train['SalePrice'] < 200000) & all_notnull_basement[:train_len]
childset = train.loc[conds,Bsmt_catigorail_fl]
print('describe dtype: ',type(childset.describe()))
childset.describe()
for x in Bsmt_catigorail_fl :
    dataset.loc[all_null_basement,x] =childset.describe().loc['top'][x] 
dataset.loc[dataset['BsmtCond'].isnull() , Bsmt_fl]
dataset.loc[2040,['BsmtCond']]='Ex'
dataset.loc[[2120,2185],['BsmtCond']]='TA'
for x in Bsmt_catigorail_fl :
    dataset.loc[[2188,2524,2120],x] =childset.describe().loc['top'][x] 
dataset[Bsmt_fl].isnull().sum()
dataset.loc[dataset['BsmtExposure'].isnull() ,Bsmt_fl]
dataset.loc[[948,1487,2348],['BsmtExposure']]='Gd'
dataset[Bsmt_catigorail_fl].isnull().sum()
dataset.loc[dataset['BsmtFinType2'].isnull() ,Bsmt_fl]
dataset.loc[[332],['BsmtFinType2']]='GLQ'
dataset.loc[dataset['BsmtQual'].isnull() ,Bsmt_fl]
dataset.loc[[2217],['BsmtQual']]='Fa'
dataset.loc[[2218],['BsmtQual']]='TA'
dataset[Bsmt_catigorail_fl].isnull().sum()
dataset[Bsmt_fl].isnull().sum()
dataset.loc[dataset['BsmtFinSF1'].isnull() ,Bsmt_fl]
dataset.loc[dataset['BsmtFinSF1'].isnull() ,'BsmtFinSF1']=0
dataset.loc[dataset['BsmtFinSF2'].isnull() ,'BsmtFinSF2']=0
dataset.loc[dataset['BsmtFullBath'].isnull() ,'BsmtFullBath']=0
dataset.loc[dataset['BsmtHalfBath'].isnull() ,'BsmtHalfBath']=0
dataset.loc[dataset['BsmtUnfSF'].isnull() ,'BsmtUnfSF']=0
dataset[Bsmt_fl].isnull().sum()
dataset.loc[dataset['Electrical'].isnull(),'Electrical']='FuseA'
ex=dataset.loc[dataset['Exterior1st'].isnull()]
sns.factorplot(y='Exterior1st',x='SalePrice',data=train,kind='strip',size=5,orient='h')

sns.countplot(y='Exterior2nd',data=dataset,orient='h')
dataset.loc[dataset['Exterior1st'].isnull(),'Exterior1st']='VinylSd'
dataset.loc[dataset['Exterior2nd'].isnull(),'Exterior2nd']='VinylSd'
dataset['Fence'].unique()
# train.loc[train['Fence'].isnull()]
sns.countplot(x='Fence',data=dataset)
sns.factorplot(x='Fence',y='SalePrice',data=train,kind='strip',size=5) 
dataset['Fence']=dataset['Fence'].astype('category')
dataset['Fence'].unique()
# make cat numerical to apply factorplot
dataset['Fence_num']=dataset['Fence'].cat.codes
dataset['Fence_num'] =  dataset['Fence_num'].astype(int)
sns.factorplot(y='LandContour',x='Fence_num',data=dataset.loc[dataset['Fence'].isnull()==False],kind='strip',jitter=True,size=5) 
sns.factorplot(y='LandSlope',x='Fence_num',data=dataset.loc[dataset['Fence'].isnull()==False],kind='strip',jitter=True,size=5) 
#Condition1
sns.factorplot(y='Condition1',x='Fence_num',data=dataset.loc[dataset['Fence'].isnull()==False],kind='strip',jitter=True,size=5) 
#Condition2
sns.factorplot(y='Condition2',x='Fence_num',data=dataset.loc[dataset['Fence'].isnull()==False],kind='strip',jitter=True,size=5) 
#MSZoning
sns.factorplot(y='MSZoning',x='Fence_num',data=dataset.loc[dataset['Fence'].isnull()==False],kind='strip',jitter=True,size=5)
dataset['Fence'].unique()
fence_notnull_set = dataset.loc[dataset['Fence'].isnull()==False]
lens = len(fence_notnull_set['Fence'] )
print(lens)
probs= (fence_notnull_set['Fence'].value_counts() / lens ).tolist()
print(probs)
fence_t = dataset['Fence'].cat.categories.tolist()
fence_null_nums  = len(dataset.loc[dataset['Fence'].isnull()])
fence_null = np.random.choice(fence_t,fence_null_nums,p=probs)

dataset.loc[dataset['Fence'].isnull(),'Fence'] = fence_null

dataset['Fence'].unique()

np.random.randint(10,1000,1)[0]
dataset.loc[dataset['LandContour']=='HLS','Fence' ] ='MnPrv'
dataset.loc[dataset['LotConfig']=='RH','Fence' ] ='MnPrv'
dataset.loc[dataset['LandSlope']=='Sev','Fence' ] ='GdWo'
dataset.loc[ (dataset['Condition1']=='RRNn') |  (dataset['Condition1']=='PosA')  |  (dataset['Condition1']=='RRne') ,'Fence' ] ='GdPrv'
dataset.loc[ (dataset['Condition2']=='PosN') ,'Fence' ] ='GdPrv'
dataset.loc[ (dataset['MSZoning']=='RH') ,'Fence' ] ='GdPrv'

tmp = dataset.loc[dataset['MSZoning']==r'C (all)','Fence'] 
d = ['GdWo','MnPrv']
lens = len(tmp)
dataset.loc[dataset['MSZoning']==r'C (all)','Fence']  = [ d[ x % len(d)] for x in np.random.randint(10,1000,lens)]

tmp = dataset.loc[dataset['MSZoning']==r'FV','Fence'] 
d = ['GdWo','GdPrv']
lens = len(tmp)
dataset.loc[dataset['MSZoning']==r'FV','Fence']  = [ d[ x % len(d)] for x in np.random.randint(10,1000,lens)]


dataset.drop(labels=['Fence_num'],axis=1,inplace=True)
sns.countplot(x='FireplaceQu',data=dataset)
sns.factorplot(x='FireplaceQu',y='Fireplaces',data=dataset,kind='strip',jitter=True)
sns.factorplot(x='FireplaceQu',y='SalePrice',data=dataset,kind='strip',jitter=True)
tmp = dataset.loc[dataset['FireplaceQu'].isnull()] 
d = ['TA','Gd'] 
# not include Po , thus fits condition :  if Fireplaces == 1  FireplaceQu != Po
lens = len(tmp)
dataset.loc[dataset['FireplaceQu'].isnull(),'FireplaceQu']  = [ d[ x % len(d)] for x in np.random.randint(10,1000,lens)]

sns.countplot(x='MSZoning',data=dataset)
dataset.loc[dataset['MSZoning'].isnull() ,'MSZoning'] = 'RL';
dataset.loc[dataset['MasVnrArea'].isnull() ,'MasVnrArea'] = dataset['MasVnrArea'].median()
dataset.loc[dataset['MasVnrType'].isnull()]
sns.countplot(x='MasVnrType',data=dataset)
sns.factorplot(x='MasVnrType',y='SalePrice',data=dataset,kind='strip',jitter=True)
# tbd here make it simple
dataset.loc[dataset['MasVnrType'].isnull(),'MasVnrType'] = 'None'

MiscFeature_notnull_set = dataset.loc[dataset['MiscFeature'].isnull()==False]
sns.countplot('MiscFeature',data=MiscFeature_notnull_set)
sns.factorplot(x='MiscFeature',y='SalePrice',data=MiscFeature_notnull_set,kind='strip',jitter=True)
dataset.loc[ dataset['MiscFeature'].isnull(),'MiscFeature']='Shed' 
PoolQC_notnull_set =  dataset.loc[dataset['PoolQC'].isnull() == False]
len(PoolQC_notnull_set)
sns.countplot(x='PoolQC',data=PoolQC_notnull_set)
if 'PoolQC' in dataset.columns:
     dataset.drop( labels='PoolQC',axis=1,inplace=True);
dataset.loc[dataset['TotalBsmtSF'].isnull(),'TotalBsmtSF'] = dataset['TotalBsmtSF'].median()
dataset.loc[dataset['Utilities'].isnull(),'Utilities'] = 'AllPub'
dataset.loc[dataset['SaleType'].isnull(),'SaleType'] = 'Other'
dataset.loc[dataset['LotFrontage'].isnull(),'LotFrontage'] = dataset['LotFrontage'].median()
dataset.loc[dataset['GarageArea'].isnull(),'GarageArea'] = dataset['GarageArea'].median()
dataset.loc[dataset['GarageCars'].isnull(),'GarageCars'] = dataset['GarageCars'].median()
dataset.loc[dataset['Functional'].isnull(),'Functional']='Typ'
dataset.loc[dataset['KitchenQual'].isnull(),'KitchenQual']='TA'
Garage_fl =  ['GarageCond','GarageQual','GarageArea','GarageCars','GarageFinish','GarageType']
Garage_cat_fl = ['GarageCond','GarageQual','GarageFinish','GarageType']
null_cond = (dataset['GarageCond'].isnull() ) |\
                                               (dataset['GarageQual'].isnull() ) |\
                                               (dataset['GarageFinish'].isnull() ) |\
                                                (dataset['GarageType'].isnull() )
Garage_null_set=dataset.loc[null_cond]
Garage_notnull_set=dataset.loc[ (dataset['GarageCond'].isnull()==False) | (dataset['MiscFeature'] == 'Gar2' )]
sns.factorplot(x='GarageCond',y='SalePrice',data=Garage_notnull_set,kind='strip',jitter=True)
Garage_notnull_set[Garage_cat_fl].describe()
not_null_cond = (Garage_null_set['GarageCond'].isnull() == False) |\
                                   (Garage_null_set['GarageQual'].isnull() == False ) |\
                                  (Garage_null_set['GarageArea'] != 0 ) |\
                                  (Garage_null_set['GarageCars'] != 0 ) |\
                                  (Garage_null_set['GarageFinish'].isnull() == False ) |\
                                  (Garage_null_set['GarageType'].isnull() == False )
Garage_null_set.loc[ not_null_cond,Garage_fl ]
for x in Garage_cat_fl :
    dataset.loc[null_cond,x] =Garage_notnull_set[Garage_cat_fl].describe().loc['top'][x] 
dataset.loc[dataset['GarageYrBlt'].isnull(),'GarageYrBlt'] = dataset.loc[dataset['GarageYrBlt'].isnull(),'YearBuilt'] 
if 'SalePrice' in dataset:
    dataset.drop(labels='SalePrice',axis=1,inplace=True)
sum (dataset.isnull().sum() )
dataset.dtypes
dataset.columns
year_fl = ['YearBuilt','YearRemodAdd','YrSold','GarageYrBlt']
dataset.loc[ (dataset['YearBuilt'] >dataset['YearRemodAdd'])  | (dataset['YearRemodAdd'] >dataset['YrSold']) ,year_fl]
dataset['YearBuiltTillNow'] = 2018-dataset['YearBuilt']
dataset['YearRemodTillNow'] = 2018-dataset['YearRemodAdd']
dataset['YearSoldTillNow'] = 2018-dataset['YrSold']
dataset['YearGaBuiltNow'] = 2018-dataset['GarageYrBlt']
dataset['RemodAfterBuilt'] = dataset['YearBuiltTillNow']  -  dataset['YearRemodTillNow'] 
dataset['SoldAfterBuilt'] = dataset['YearBuiltTillNow']  -  dataset['YearSoldTillNow'] 

dataset.loc[dataset['YearGaBuiltNow'] < 0,'YearGaBuiltNow'] = dataset.loc[dataset['YearGaBuiltNow'] < 0,'YearBuilt'] 

for x in year_fl:
    if x in dataset:
        dataset.drop(labels=x,axis=1,inplace= True)
dataset.dtypes.unique()
cat_dataset = dataset.select_dtypes(include=['object','bool'])
for col in cat_dataset:
    cat_dataset[col] = cat_dataset[col].astype('category')

cat_dataset.dtypes.unique()
num_dataset = dataset.select_dtypes(exclude=['object','category']).astype('float64')
pos = np.where( num_dataset < 0 )
pos[0]
num_dataset.loc[ num_dataset['RemodAfterBuilt'] < 0,'RemodAfterBuilt'] = 0
num_dataset.loc[ num_dataset['SoldAfterBuilt'] < 0,'SoldAfterBuilt'] = 0
pos = np.where( num_dataset < 0 )
pos[0]
from scipy.stats import skew
skew_feats = num_dataset.apply(skew).sort_values(ascending=False)
skew_feats.head(10)
sns.distplot(num_dataset[num_dataset['MiscVal'] != 0]['MiscVal'])
skew_feats = skew_feats[abs(skew_feats) > 1]

print(skew_feats)

for feat in skew_feats.index:
    num_dataset[feat] = np.log1p(num_dataset[feat])
sns.distplot(num_dataset[num_dataset['MiscVal'] != 0]['MiscVal'])
cat_dataset = pd.get_dummies(cat_dataset, columns = cat_dataset.columns);
np.where(num_dataset < 0)
for x in num_dataset:
     num_dataset[x] = (num_dataset[x] - num_dataset[x].mean()) / (num_dataset[x].std())
#         num_dataset[x] = (num_dataset[x] - num_dataset[x].mean()) / (num_dataset[x].max() - num_dataset[x].min())
sum (np.isnan(num_dataset).any())
np.where(np.isnan(num_dataset))
num_dataset = num_dataset.fillna(num_dataset.median())
sum (np.isnan(num_dataset).any())
dataset = pd.concat([cat_dataset,num_dataset],axis=1)

X = dataset[:train_len]
X_test   = dataset[train_len:]
sum(X.isnull().sum())

# y = train.pop('SalePrice')
sum(X_test.isnull().sum() )
X.shape
X_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import Ridge,RidgeCV
# K折线 学习曲线 

from sklearn.model_selection import GridSearchCV, \
                                                            cross_val_score, \
                                                            StratifiedKFold, \
                                                            learning_curve,\
                                                            KFold,\
                                                            cross_val_predict;
from sklearn.model_selection import train_test_split;
sns.set(style='white', context='notebook', palette='deep');
from sklearn.metrics import mean_squared_error,mean_squared_log_error,median_absolute_error
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def train_model(estimator=None, X_train=None,y_train=None,X_cv=None,y_cv=None,scoring=None):
    m = estimator
    if(scoring == 'neg_mean_squared_error'):
        m.fit(X_train,np.log1p(y_train))
    if(scoring == 'neg_mean_squared_log_error'):
        m.fit(X_train,y_train)
    pred=estimator.predict(X_cv)
#     print(np.where(np.isnan(pred)))
    if(scoring == 'neg_mean_squared_error'):
         score = np.sqrt (mean_squared_error(np.log1p(y_cv),pred))
    if(scoring == 'neg_mean_squared_log_error'):
        score = np.sqrt (mean_squared_log_error(y_cv,pred))
    #What's inside mean_squared_log_error...
#     sq_diff = np.square(np.log(y_cv)-np.log(pred))
#     score= np.sqrt(np.sum(sq_diff)/y_cv.shape[0])
#     print('s2 ',score)
    return score,m    
def gen_submission(csvname=None,X_test=None,models=None,scoring=None):
    if(scoring == 'neg_mean_squared_error'):
        pred=np.expm1(models.predict(X_test))
    if(scoring == 'neg_mean_squared_log_error'):
        pred=models.predict(X_test)
    result  = pd.concat([ids,pd.Series(pred).astype('float64')],axis=1);
    result.columns = ['Id','SalePrice']
    result.to_csv(csvname+r'.csv',index=False)
    pass
X.dtypes.unique()
X = X.astype('float64')
X.shape
sum(np.isnan(X).all())
sum (np.isinf(X).any() )
# 1170
# outlier=[1170,825,88]
# X.drop(labels=outlier,axis=0,inplace=True)
# y.drop(labels=outlier,axis=0,inplace=True)

# 916
# X.drop(labels=[916],axis=0,inplace=True)
# y.drop(labels=[916],axis=0,inplace=True)
scores_cv = [] 
scores_test=[]
# scoring= 'neg_mean_squared_log_error'
scoring= 'neg_mean_squared_error'

models = []
LR_ridge = Ridge(alpha=20)
if scoring == 'neg_mean_squared_error':
    LR_lasso = Lasso(alpha=0.0005)
if scoring == 'neg_mean_squared_log_error':
    LR_lasso = Lasso(alpha=80)

models.append( ['ridge',LR_ridge ])
models.append(['lasso',LR_lasso])

times = 10;

score_mat = np.empty( (times,len(models),2))

print('test vs cv score:')


for i in range(times):

    X_train, X_vld, y_train,y_vld = train_test_split(X,y,test_size=0.2,shuffle=True)
    j=0;
    for n,m in models:
        s,fin = train_model(m,X,y,X,y,scoring=scoring)
#         print('model ',n, ' test score: ',s)
        score_mat[i,j,0] = s
        s,fin = train_model(m,X_train,y_train,X_vld,y_vld,scoring=scoring)
#          print('model ',n, ' cv score: ',s)
        score_mat[i,j,1] = s
        j = j+1

# print(score_mat)
print('For models', list(map(lambda x: x[0] ,models)))
print('Ave test score: ' ,np.mean(score_mat,axis=0)[:,0])
print('Ave cv score: ' ,np.mean(score_mat,axis=0)[:,1])
scoring
scores = -1 * cross_val_score(LR_lasso,X,np.log1p(y),scoring=scoring,cv=5)
np.sqrt(scores).mean()
LR = Lasso()
kf = StratifiedKFold(n_splits=10,shuffle=True)
lasso_param_grid = {
                "alpha":[0.001,0.0005,0.0007] ,
#                 "alpha":[80,90,100,120,140] ,
                 "max_iter":[1000,800,500],
                 "tol":[0.001,0.002,0.005,0.01,0.02,0.04,],
#                 "fit_intercept":[True,False],
}
#                   "tol":[0.005,0.01,0.02],
#                   "fit_intercept":[True,False]}

rcv_param_grid = {}

gsLR = GridSearchCV(LR,param_grid = lasso_param_grid, cv=5,n_jobs= -1, verbose = 1)

gsLR.scoring = scoring


gsLR.fit(X,np.log1p(y))

bestLR= gsLR.best_estimator_
                 
print(bestLR)
# Best score
print(np.sqrt(-gsLR.best_score_))
def plot_learning_curve(estimator, title, X, y, scoring = None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    
    if(scoring == 'neg_mean_squared_log_error'):
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)
    if(scoring == 'neg_mean_squared_error'):
        train_scores = np.log1p(train_scores)
        test_scores =  np.log1p(test_scores)  
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# g = plot_learning_curve(bestLR,"LR mearning curves",X,np.log1p(y),scoring='neg_mean_squared_error',cv=5,)
g = plot_learning_curve(bestLR,"LR mearning curves",X,np.log1p(y),scoring=scoring,cv=5)
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, np.log1p(y), scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
alphas = [0.00001,0.00005,0.0001,0.0005,0.001,]
#confirm 0.0005

#confirm 20
alphas = [0.0001,0.0003,0.0005,0.0007,0.001]

cv_ridge = [rmse_cv(Lasso(alpha = alpha)).mean() 
            for alpha in alphas]


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
print(cv_ridge)
plt.xlabel("alpha")
plt.ylabel("rmse")
print(X.shape)
LR_lasso_cv = LassoCV(alphas=[0.0004,0.0005,0.0006]);
LR_lasso_cv.fit(X,np.log1p(y))
g = plot_learning_curve(LR_lasso_cv,"Lasso cv mearning curves",X,np.log1p(y),scoring='neg_mean_squared_error',cv=5)
# got  0.11992 
gen_submission('Lasso_cv',X_test,LR_lasso_cv,'neg_mean_squared_error')
