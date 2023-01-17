import datetime
import numpy as np
from scipy import stats, special
import statsmodels.formula.api as smf
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline

# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

from xgboost.sklearn import XGBRegressor

%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.dpi']=200
houses     = pd.read_csv('../input/train.csv', index_col='Id')
housesTest = pd.read_csv('../input/test.csv',  index_col='Id')

trainSet = []
trainSet.append(houses.shape[0])
trainSet.append(houses.shape[1])
testSet  = housesTest.shape

houses = pd.concat([houses,housesTest])
def renameColumns(df):
    # Rename columns which names start with a number to workaround problems with statsmodels library

    toRename={}
    for c in df.columns:
        if c[0].isdigit():
            toRename[c]='n'+c

    df.rename(columns=toRename, inplace=True)
renameColumns(houses)
# All _UPPERCASE categories will be used for imputed NaNs
# _UNAVAILABLE: item is unavailable in the house, e.g. no pool, no garage in the house
# _UNKNOWN: a serious data leak

categoricalFeatures=[
     'BldgType',
     'Condition1',
     'Condition2',
     'Electrical',
     'Exterior1st',
     'Exterior2nd',
     'Foundation',
     'Heating',
     'MSZoning',
     'MasVnrType',
     'MiscFeature',
     'Neighborhood',
     'RoofMatl',
     'RoofStyle',
     'SaleCondition',
     'SaleType'
]

# According to data_description.txt, most NaN in following features have a meaning
categoricalOrderedFeatures={
    'Alley':        ['_UNAVAILABLE','Grvl', 'Pave'],
    'BsmtCond':     ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'BsmtExposure': ['_UNAVAILABLE','No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['_UNAVAILABLE','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'BsmtFinType2': ['_UNAVAILABLE','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'BsmtQual':     ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'CentralAir':   ['N','Y'],
    'ExterCond':    ['Po','Fa','TA','Gd','Ex'],
    'ExterQual':    ['Po','Fa','TA','Gd','Ex'],
    'Fence':        ['_UNAVAILABLE','MnWw','GdWo','MnPrv','GdPrv'],
    'FireplaceQu':  ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'Functional':   ['_UNKNOWN','Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
    'GarageCond':   ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'GarageFinish': ['_UNAVAILABLE','Unf','RFn','Fin'],
    'GarageQual':   ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'GarageType':   ['_UNAVAILABLE','Detchd','CarPort','BuiltIn','Basment','Attchd','2Types'],
    'HeatingQC':    ['Po','Fa','TA','Gd','Ex'],
    'HouseStyle':   ['SLvl', 'SFoyer', '2.5Unf', '2.5Fin','2Story', '1.5Unf', '1.5Fin', '1Story'],
    'KitchenQual':  ['_UNKNOWN','Po','Fa','TA','Gd','Ex'],
    'LandContour':  ['Low','HLS','Bnk','Lvl'],
    'LandSlope':    ['Sev','Mod','Gtl'],
    'LotConfig':    ['Inside','Corner','CulDSac','FR2','FR3'],
    'LotShape':     ['IR3','IR2','IR1','Reg'],
    'PavedDrive':   ['N','P','Y'],
    'PoolQC':       ['_UNAVAILABLE','Po','Fa','TA','Gd','Ex'],
    'Street':       ['Grvl','Pave'],
    'Utilities':    ['_UNKNOWN','NoSeWa','AllPub']
}


# Features which values might be unknown before house sale
futureFeatures = [
    'MoSold',
    'SaleType',
    'SaleCondition',
    'SalePrice',
    'YrSold'
]
def convertCategorical(df):
    df['Functional'].fillna('Typ', inplace=True) # according to data_description.txt

    # NaN in following features are trully _UNKNOWN
    df['Utilities'].fillna('_UNKNOWN', inplace=True)
    df['MSZoning'].fillna('_UNKNOWN', inplace=True)


    df['Electrical'].fillna('_UNKNOWN', inplace=True)
    df['Exterior1st'].fillna('_UNKNOWN', inplace=True)
    df['Exterior2nd'].fillna('_UNKNOWN', inplace=True)
    df['KitchenQual'].fillna('_UNKNOWN', inplace=True)
    df['SaleType'].fillna('_UNKNOWN', inplace=True)
    
    # NaN in following features are trully _UNAVAILABLE
    df['MiscFeature'].fillna('_UNAVAILABLE', inplace=True)
    df['MasVnrType'].fillna('_UNAVAILABLE', inplace=True)


    # Convert some columns to Pandas category data type, ordered and unordered
    for col in categoricalFeatures:
        df[col] = df[col].astype(CategoricalDtype())
        df[col].cat.add_categories(['_AGGREGATED_MINORITIES'],inplace=True)

    for col in categoricalOrderedFeatures.keys():
        # All NaNs have a meaning here
        # We can make this imputation after inspecting NaN a few cells below
        df[col].fillna('_UNAVAILABLE', inplace=True)
        df[col] = df[col].astype(CategoricalDtype(categories=categoricalOrderedFeatures[col],ordered=True))
        df[col].cat.add_categories(['_AGGREGATED_MINORITIES'],inplace=True)
convertCategorical(houses)
houses.info()
# Inspect order of each category.
# Find NaNs and figure out what to do with them...
for c in houses.dtypes[houses.dtypes=='category'].index.sort_values():
    if np.nan in houses[c].unique().tolist():
        # must print nothing because we handled NaNs a few cells above
        print(c)
        print(houses[c].unique())
        print(houses[c].unique().tolist())
        print()
houses.plot.scatter(x='GrLivArea',y='SalePrice');
houses[(houses.GrLivArea>4000) & (houses.SalePrice<300000)][['GrLivArea','SalePrice']]
def dropOutliers(df):
    if 'SalePrice' in df.columns:
        df.drop(df[(df.GrLivArea>4000) & (df.SalePrice<300000)].index,inplace=True)
dropOutliers(houses)
trainSet[0]=trainSet[0]-2
def skewAnalysis(df):
    """
    Return a DataFrame with skeweness of each numeric feature, most skewed first.
    High ranked skew variables are candidates to be log()ed or boxcox()ed before any regression.
    
    Written by Avi Alkalay
    avi at unix dot sh
    http://Avi.Alkalay.net
    2018-09-27
    """
    skewList=df.select_dtypes('number').apply(lambda x: stats.skew(x, nan_policy='omit')).sort_values(ascending=False)
    nTypes=skewList.axes[0]
    
    sk=pd.DataFrame({'skew' : skewList})
    sk['rank'] = range(1, len(sk) + 1)    
    
    for i in nTypes:
        shift = 0.0
        
        if df[i].min() <= 0.0:
            shift=-df[i].min()+1
            
        sk.loc[i,'BoxCox λ'] = stats.boxcox_normmax(df[i]+shift, method='mle')
        sk.loc[i,'BoxCox shift'] = shift
        sk.loc[i,'Has NaN'] = df[i].isnull().values.any()
        
    return sk




def analyzeNormalization(df,variables,typ='log',bins=None):
    """
    Plots graphics for each variable passed in variables array used to visualy spot if it
    requires a log or boxcox transformation. A value distribution before and
    after (typ=)log/boxcox transformation against a perfect Gaussian curve and probability
    plot before and after the (typ=)log/boxcox transformation, summing 4 plots per variable.
    
    Written by Avi Alkalay
    avi at unix dot sh
    http://Avi.Alkalay.net
    2018-09-27
    """
    # typ=boxcox still unstable, waiting for scikit 1.1 for better boxcox implementation
    
    for v in variables:
        shift=0.0
        
        if df[v].min() <= 0.:
            shift=-df[v].min() + 1
            
        fig=plt.figure(num=f'Gaussian fit analisys for «{v}»')
        fig.suptitle(f'Gaussian fit analisys for «{v}»',fontweight='bold')

        axs=fig.subplots(nrows=2,ncols=2)

        # Set common style for all 4 subplots...
        for row in axs:
            for axes in row:
                # add grid:
                axes.grid(linewidth=0.1)
                
                # thinner line width and smaller scatter size:
                axes.set_prop_cycle(linewidth=[.5],markersize=[1])
                
                # remove ticks:
                axes.tick_params(which='both',
                                 bottom=False, top=False,
                                 left=False, right=False,
                                 labelbottom=False, labeltop=False,
                                 labelleft=False, labelright=False)

        
        #(μ,σ) = stats.norm.fit(df[v])
        sns.distplot(df[v],fit=stats.norm,ax=axs[0][0],bins=bins)
        axs[0][0].set_title(f'Dist × perfect Gauss curve', fontsize='small')
        axs[0][0].set_ylabel('Frequency', fontsize='small')
        axs[0][0].set_xlabel(f'{v}', fontsize='small')
        #axs[0][0].legend(['Gauss curve (μ={:.2f} • σ={:.2f})'.format(μ,σ),f'«{v}»'], loc='best')
        axs[0][0].legend(['Normal'], loc='best')


        if typ == 'boxcox':
            v_trans, λ = stats.boxcox(df[v] + shift)
        else:
            v_trans = np.log(df[v] + shift)

        #(μ,σ) = stats.norm.fit(v_log)
        sns.distplot(v_trans,fit=stats.norm,ax=axs[1][0],bins=bins)
        axs[1][0].set_ylabel('Frequency', fontsize='small')

        if typ == 'boxcox':
            axs[1][0].set_xlabel('BoxCox({}), λ={:.3f}'.format(v,λ), fontsize='small')
        else:
            axs[1][0].set_xlabel(f'logₑ({v})', fontsize='small')

        a=stats.probplot(df[v], fit=True, plot=axs[0][1])
        a=stats.probplot(v_trans, fit=True, plot=axs[1][1])
        axs[1][1].set_title('')
        axs[0][1].set_xlabel('')
        
        # cleanup
        del v_trans
        del a
        del axs
        del axes
        del fig


pd.set_option('display.max_rows', 100)
skewAnalysis(houses)
pd.reset_option('display.max_rows')
houses['KitchenQual'].value_counts().plot(kind='bar')
houses['KitchenQual'].value_counts()
houses[(houses.KitchenQual=='_UNKNOWN')][['KitchenQual']]
houses['KitchenQual'].mode().head(1)
def modeToFeatureValue(df,feature,value):
    df.loc[df[feature]==value, feature] = df[feature].mode().head(1).values[0]
modeToFeatureValue(houses,'KitchenQual','_UNKNOWN')
houses['KitchenQual'].value_counts()
inspect=['Utilities','MSZoning','Electrical','Exterior1st','Exterior2nd','SaleType']

fig=plt.figure()

i=1
for f in inspect:
    ax=fig.add_subplot((len(inspect)//2)+(len(inspect)%2),2,i)
    houses[f].value_counts().plot(kind='bar',title=f,ax=ax)
    i=i+1

plt.rcParams['figure.dpi']=200
plt.show()
for f in inspect:
    print()
    print(f'Distribution of «{f}» values:')
    print(houses[f].value_counts(normalize=True))

modeToFeatureValue(houses,'Utilities','_UNKNOWN')
modeToFeatureValue(houses,'MSZoning','_UNKNOWN')
modeToFeatureValue(houses,'Electrical','_UNKNOWN')
modeToFeatureValue(houses,'Exterior1st','_UNKNOWN')
modeToFeatureValue(houses,'Exterior2nd','_UNKNOWN')
modeToFeatureValue(houses,'SaleType','_UNKNOWN')
# Figure out what to do with NaNs on MasVnrArea
houses[houses.MasVnrType=='_UNAVAILABLE'][['MasVnrType','MasVnrArea']]
def zeroToFeature(df,feature):
    df[feature].fillna(0, inplace=True)
def meanToFeature(df,feature):
    df[feature]=Imputer().fit_transform(df[[feature]])
zeroToFeature(houses,'MasVnrArea')
houses[houses.MasVnrType=='_UNAVAILABLE'][['MasVnrType','MasVnrArea']]
# Figure out what to do with NaNs on GarageYrBlt

# Compare this NaNed feature with other Garage-related features
houses[(pd.isnull(houses.GarageYrBlt)) & (houses.GarageCars!=0.0 )][['GarageYrBlt','GarageCars','GarageArea']]
def copyToFeature(df,source,target):
    """Fill NaNs on target feature with what is on source feature"""
    df.loc[pd.isnull(df[target]),target] = df[pd.isnull(df[target])][source]
copyToFeature(houses,'YearBuilt','GarageYrBlt')
houses[pd.isnull(houses['GarageYrBlt'])][['GarageYrBlt']]
#meanToFeature(houses,'GarageYrBlt')
houses[houses.GarageCars==0][['GarageCars','GarageYrBlt','GarageArea']].head(80)
houses[pd.isnull(houses.GarageCars)][['GarageCars','GarageArea']]
zeroToFeature(houses,'GarageCars')
zeroToFeature(houses,'GarageArea')

houses.loc[2577][['GarageCars']]
houses[(pd.isnull(houses.BsmtFinSF1))|(pd.isnull(houses.BsmtFinSF2))][['BsmtFinSF1','BsmtFinType1','BsmtFinSF2','BsmtFinType2']]
houses[(houses.BsmtFinType1=='_UNAVAILABLE') | (houses.BsmtFinType2=='_UNAVAILABLE')][['BsmtFinSF1','BsmtFinType1','BsmtFinSF2','BsmtFinType2']]
zeroToFeature(houses,'BsmtFinSF1')
zeroToFeature(houses,'BsmtFinSF2')
houses[(pd.isnull(houses.BsmtHalfBath)) | (pd.isnull(houses.BsmtFullBath))][['BsmtHalfBath','BsmtFullBath']]
zeroToFeature(houses,'BsmtHalfBath')
zeroToFeature(houses,'BsmtFullBath')
houses[(pd.isnull(houses.TotalBsmtSF))][['TotalBsmtSF','BsmtQual']]
zeroToFeature(houses,'TotalBsmtSF')
houses[(pd.isnull(houses.BsmtUnfSF))][['BsmtUnfSF','BsmtQual']]
zeroToFeature(houses,'BsmtUnfSF')
def aggregateMinorCategories(df,rate=0.05,targetCategory='_AGGREGATED_MINORITIES'):
    for feature in list(categoricalOrderedFeatures.keys())+categoricalFeatures:
        for c, r in df[feature].value_counts(normalize=True).items():
            if r<rate:
                df.loc[df[feature]==c, feature] = targetCategory
# Figure out what to do with NaNs on LotFrontage

# Lets see if we can estimate LotFrontage from LotArea...
studylotfrontage=houses[['LotArea','LotFrontage']].copy()
studylotfrontage['LogLotArea']=np.log(studylotfrontage['LotArea'])
studylotfrontage['LogLotFrontage']=np.log(studylotfrontage['LotFrontage'])

plt.rcParams['figure.dpi']=200
pd.plotting.scatter_matrix(studylotfrontage);
plt.scatter(x=studylotfrontage['LogLotArea'],y=studylotfrontage['LogLotFrontage'],s=1)
plt.xlabel("LogLotArea")
plt.ylabel("LogLotFrontage")
plt.show()
# Use original data
lotFrontageEstimator = LinearRegression().fit(
    X=np.log(houses[pd.notnull(houses.LotFrontage)][['LotArea']]),
    y=np.log(houses[pd.notnull(houses.LotFrontage)][['LotFrontage']])
)
print('R²={}'.format(lotFrontageEstimator.score(
    X=np.log(houses[pd.notnull(houses.LotFrontage)][['LotArea']]),
    y=np.log(houses[pd.notnull(houses.LotFrontage)][['LotFrontage']])
)))
predictedLotFrontage=lotFrontageEstimator.predict(np.log(houses[pd.isnull(houses.LotFrontage)][['LotArea']]))

plt.scatter(x=studylotfrontage['LogLotArea'],y=studylotfrontage['LogLotFrontage'],s=1)
plt.xlabel("LogLotArea")
plt.ylabel("LogLotFrontage")

plt.scatter(x=np.log(houses[pd.isnull(houses.LotFrontage)]['LotArea']),y=predictedLotFrontage,c='red', marker='^', s=3)

plt.show()
def estimateLotFrontageByLotAreaRegression(df):
#     lotFrontageEstimator = LinearRegression().fit(
#         X=np.log(df[pd.notnull(df.LotFrontage)][['LotArea']]),
#         y=np.log(df[pd.notnull(df.LotFrontage)][['LotFrontage']])
#     )
    
    predictedLotFrontage=lotFrontageEstimator.predict(np.log(df[pd.isnull(df.LotFrontage)][['LotArea']]))

    nanindex=df[pd.isnull(df.LotFrontage)].index

    # Precision fill...
    df.loc[pd.isnull(df.LotFrontage),'LotFrontage']=np.exp(predictedLotFrontage)
print("{} predicted Lot Frontages for {} NaN Lot Frontages in dataset. Follows their IDs:".format(predictedLotFrontage.shape[0],houses[pd.isnull(houses.LotFrontage)].shape[0]))
nanindex=houses[pd.isnull(houses.LotFrontage)].index
print(nanindex)
def estimateLotFrontageByNeighborhoodMedian(df):
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#estimateLotFrontageByLotAreaRegression(houses)
estimateLotFrontageByNeighborhoodMedian(houses)
# Inspect
houses[['LotFrontage']]
def addTotalSF(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['n1stFlrSF'] + df['n2ndFlrSF']
addTotalSF(houses)
plt.rcParams['figure.dpi']=200
analyzeNormalization(houses,['LotArea','PoolArea','LowQualFinSF','LotFrontage','TotalBsmtSF','n1stFlrSF','GrLivArea','n2ndFlrSF','n3SsnPorch','TotalSF'],bins=200)
analyzeNormalization(houses.head(trainSet[0]),['SalePrice'],typ='boxcox',bins=200)
#analyzeNormalization(houses,['LotArea','LotFrontage'],typ='boxcox', bins=200)
#analyzeNormalization(houses.head(trainSet[0]),['TotalSF','n1stFlrSF','LotArea','BsmtFinSF2','EnclosedPorch','ScreenPorch','KitchenAbvGr'],typ='boxcox',bins=200)
analyzeNormalization(houses,['TotalSF','n1stFlrSF','LotArea','LotFrontage','BsmtFinSF2','EnclosedPorch','ScreenPorch','KitchenAbvGr','GrLivArea','n1stFlrSF'],typ='boxcox',bins=200)
def boxcoxToFeature(df,feature):
    df[feature] = special.boxcox1p(df[feature],0.15)

def logToFeature(df,feature):
    df[feature] = np.log1p(df[feature])
logToFeature(houses,'TotalSF')
logToFeature(houses,'LotArea')
logToFeature(houses,'n1stFlrSF')
logToFeature(houses,'GrLivArea')
function = None

for feature in houses.columns.sort_values():
    if feature in futureFeatures:
        # skip features unknown in house pre-sales time, more likely to be Y and not X
        continue
        
    col=feature.strip() # clean strange chars
    if function:
        function = '{} + {}'.format(function,col)
    else:
        function = col

function=f'np.log(SalePrice) ~ {function}'

# model = smf.ols(function, houses).fit()
# print(model.summary())
# unfinished...
def encodeOneHotFeatures(df):
    for feature in list(categoricalOrderedFeatures.keys())+categoricalFeatures:
        df2=pd.DataFrame(index=df.index)
        df2["numerical_" + feature] = LabelEncoder().fit_transform(df[feature])
        x=OneHotEncoder().fit_transform(df2[["numerical_" + feature]])
        print(x)
        df3=pd.DataFrame(data=x)
        df.join(df3,how='left',rsuffix='_' + feature)
def encodeUnorderedFeatures(df):
    # Incrementally add encoded unordered categorical features
    for feature in categoricalFeatures:
        df["numerical_" + feature] = LabelEncoder().fit_transform(df[feature])
encodeUnorderedFeatures(houses)

print(houses.shape)
houses.head()
def encodeOrderedFeatures(df):
    # Incrementally add encoded ordered categorical features
    for feature in categoricalOrderedFeatures.keys():
        df["numerical_" + feature]=-1 # initialize target feature
        i=0
        for category in df[feature].unique().categories:

            # Get row indexes from this category from source DataFrame
            indexes = df.index[df[feature] == category]

            # Imputation
            df.loc[indexes,"numerical_" + feature] = i

            # move along
            i=i+1
encodeOrderedFeatures(houses)

print(houses.shape)
houses.head()
def dropNonNumeric(df):
    numerical=df.select_dtypes(['number']).columns
    al=df.columns
    toDrop=[]
    for c in al:
        if c not in numerical:
            toDrop.append(c)
    
    df.drop(columns=toDrop, inplace=True)
dropNonNumeric(houses)
houses.head()
def encodeAndDrop(df):
    df=pd.get_dummies(df)
    return df
def featureEngineering(dataset):
    renameColumns(dataset)
    convertCategorical(dataset)
    
    dropOutliers(dataset)
    
    zeroToFeature(dataset,'MasVnrArea')
    zeroToFeature(dataset,'BsmtFinSF1')
    zeroToFeature(dataset,'BsmtFinSF2')
    zeroToFeature(dataset,'BsmtHalfBath')
    zeroToFeature(dataset,'BsmtFullBath')
    zeroToFeature(dataset,'TotalBsmtSF')
    zeroToFeature(dataset,'BsmtUnfSF')
    zeroToFeature(dataset,'GarageCars')
    zeroToFeature(dataset,'GarageArea')

    copyToFeature(dataset,'YearBuilt','GarageYrBlt')
#     meanToFeature(dataset,'GarageYrBlt')

    modeToFeatureValue(dataset,'KitchenQual','_UNKNOWN')
    modeToFeatureValue(dataset,'Utilities','_UNKNOWN')
    modeToFeatureValue(dataset,'MSZoning','_UNKNOWN')
    modeToFeatureValue(dataset,'Electrical','_UNKNOWN')
    modeToFeatureValue(dataset,'Exterior1st','_UNKNOWN')
    modeToFeatureValue(dataset,'Exterior2nd','_UNKNOWN')
    modeToFeatureValue(dataset,'SaleType','_UNKNOWN')

#     aggregateMinorCategories(dataset,0.001)
    
    addTotalSF(dataset)
    
#     boxcoxToFeature(dataset,'TotalSF')
#     boxcoxToFeature(dataset,'LotArea')
#     boxcoxToFeature(dataset,'n1stFlrSF')
#     boxcoxToFeature(dataset,'GrLivArea')


    logToFeature(dataset,'TotalSF')
    logToFeature(dataset,'LotArea')
    logToFeature(dataset,'n1stFlrSF')
    logToFeature(dataset,'GrLivArea')


    estimateLotFrontageByLotAreaRegression(dataset)
    #estimateLotFrontageByNeighborhoodMedian(dataset)
    
    encodeUnorderedFeatures(dataset)
    encodeOrderedFeatures(dataset)
#     encodeOneHotFeatures(dataset)

    dropNonNumeric(dataset)

    #dataset=pd.get_dummies(dataset)
del houses
houses=pd.read_csv('../input/train.csv', index_col='Id')
featureEngineering(houses)
def paramSeeker(estimator,param_grid,df):
    #kf = StratifiedKFold(n_splits=5, shuffle=True)
    hparamsearch = GridSearchCV(estimator = estimator, param_grid = param_grid,
                                cv=5, scoring='neg_mean_squared_error')

    # hparamsearch = BayesSearchCV(estimator = priceEstimator, search_spaces = searchSpace,
    #                              cv=5, n_iter=200, n_points=5, scoring='neg_mean_squared_error')

    hparamsearch.fit(df.drop(columns='SalePrice'),np.log(df['SalePrice']))

    results = dict(
        best_score = np.sqrt(-hparamsearch.best_score_),
        best_params = hparamsearch.best_params_
    )
    
    return results
xgbPriceEstimator=Pipeline([
    ('scale',RobustScaler()),
    ('xgb',XGBRegressor(n_jobs=4))
])

xgbParamGrid={
    'xgb__n_estimators': [1800],
    'xgb__max_depth': [3, 4],
    'xgb__reg_alpha': [0.2, 0.25],
    'xgb__reg_lambda': [1.1, 1.2],

    'xgb__colsample_bytree': [0.4603],
    'xgb__gamma': [0.0468],
    'xgb__learning_rate': [0.05],
    'xgb__min_child_weight': [1.7817],
    'xgb__subsample': [0.5213],
    'xgb__random_state': [42]
}
enetPriceEstimator=Pipeline([
    ('scale',RobustScaler()),
    ('enet',ElasticNet())
])

enetParamGrid={
    'enet__alpha': [0.0002, 0.0005, 0.001, 0.1],
    'enet__l1_ratio': [0.7, 0.9, 1.1],
    'enet__random_state': [42]
}
xgbResults=paramSeeker(xgbPriceEstimator,xgbParamGrid,houses)
enetResults=paramSeeker(enetPriceEstimator,enetParamGrid,houses)

print(xgbResults)
print()
print(enetResults)
xgbPriceEstimator.set_params(**xgbResults['best_params'])
enetPriceEstimator.set_params(**enetResults['best_params'])
xgbPriceEstimator.fit(houses.drop(columns='SalePrice'),np.log(houses['SalePrice']))
enetPriceEstimator.fit(houses.drop(columns='SalePrice'),np.log(houses['SalePrice']))
housesTest=pd.read_csv('../input/test.csv', index_col='Id')
featureEngineering(housesTest)
results=pd.DataFrame(index=housesTest.index)
results['xgbSalePrice']=np.exp(xgbPriceEstimator.predict(housesTest))
results['enetSalePrice']=np.exp(enetPriceEstimator.predict(housesTest))
housesTest['SalePrice']=(results['xgbSalePrice']+results['enetSalePrice'])/2
housesTest[['SalePrice']].to_csv(path_or_buf='submission.csv.gz',compression='gzip')
# RandomizedSearchCV 2018-09-21 16:51:16.586857
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 196, 'xgb__reg_alpha': 0.9, 'xgb__reg_lambda': 0.1}
# hparamsearch.best_score_ = 0.8985498074355338

# GridSearchCV 2018-09-22 09:40:45.967409
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.6, 'xgb__reg_lambda': 0.1}
# hparamsearch.best_score_ = 0.9002134583272592

# GridSearchCV 2018-09-22 10:00:53.267628
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.9, 'xgb__reg_lambda': 0.1}
# hparamsearch.best_score_ = 0.898309723179664

# GridSearchCV 2018-09-22 10:25:11.086841
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.125}
# hparamsearch.best_score_ = 0.8985959477587327

# GridSearchCV 2018-09-22 16:27:26.066470
# hparamsearch.best_params_ = {'xgb__max_depth': 5, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.18}
# hparamsearch.best_score_ = 0.8988287397759668

# GridSearchCV(RobustScaler) 2018-09-22 18:46:30.349091
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.96, 'xgb__reg_lambda': 0.2}
# hparamsearch.best_score_ = 0.9003688412258621

# GridSearchCV(RobustScaler+TotalSF) 2018-09-22 19:00:47.473969
# hparamsearch.best_params_ = {'xgb__max_depth': 5, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.98, 'xgb__reg_lambda': 0.2}
# hparamsearch.best_score_ = 0.9014648252764706

# GridSearchCV(RobustScaler+TotalSF+BoxCox) 2018-09-22 19:21:14.675157
# hparamsearch.best_params_ = {'xgb__max_depth': 5, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.98, 'xgb__reg_lambda': 0.2}
# hparamsearch.best_score_ = 0.9012942441479855

# GridSearchCV(RobustScaler+TotalSF+BoxCox) 2018-09-22 20:06:12.663002
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 235, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.24}
# hparamsearch.best_score_ = 9.403486391711356e-05

# GridSearchCV(RobustScaler+BoxCox+LotFrontageNeighborhood) 2018-09-22 20:36:12.203183
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.96, 'xgb__reg_lambda': 0.22}
# hparamsearch.best_score_ = 9.368462858961698e-05
# Kaggle score = 0.14483

# GridSearchCV(RobustScaler+LotFrontageNeighborhood) 2018-09-22 20:45:17.838028
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.96, 'xgb__reg_lambda': 0.22}
# hparamsearch.best_score_ = 9.368462858961698e-05
# Kaggle score = 0.12689

# GridSearchCV(RobustScaler+LotFrontageRegression) 2018-09-22 21:04:27.647685
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 235, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.24}
# hparamsearch.best_score_ = 9.406959048473793e-05
# Kaggle score = 0.12832

# GridSearchCV(RobustScaler+LotFrontageRegression+NoLog) 2018-09-22 21:10:51.990014
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.24}
# hparamsearch.best_score_ = 0.016329734882100273
# Kaggle score = 0.14262

# GridSearchCV(RobustScaler+LotFrontageGlobalRegression) 2018-09-22 21:19:57.323594
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 230, 'xgb__reg_alpha': 0.95, 'xgb__reg_lambda': 0.22}
# hparamsearch.best_score_ = 9.432671389584923e-05
# Kaggle score = 0.12680

# GridSearchCV(RobustScaler+LotFrontageGlobalRegression) 2018-09-22 21:29:11.874732
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 250}
# hparamsearch.best_score_ = 9.431363999930521e-05
# Kaggle score = 0.12667

# GridSearchCV(StandardScaler+LotFrontageGlobalRegression) 2018-09-22 21:34:12.214354
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 250}
# hparamsearch.best_score_ = 0.00010021662013134085
# Kaggle score = 0.13739

# GridSearchCV(RobustScaler+LotFrontageGlobalRegression+ElasticNet) 2018-09-22 22:27:19.940156
# hparamsearch.best_params_ = {'enet__alpha': 0.0005, 'enet__l1_ratio': 2.4}
# hparamsearch.best_score_ = 0.00013411002114302706
# Kaggle score = 

# GridSearchCV(RobustScaler+BoxCox+TotalSF+Mode) 2018-09-24 00:45:49.598859
# hparamsearch.best_params_ = {'xgb__max_depth': 4, 'xgb__n_estimators': 240, 'xgb__reg_alpha': 0.98, 'xgb__reg_lambda': 0.2}
# hparamsearch.best_score_ = 0.12542358749542298
# Kaggle score = 0.39074

# GridSearchCV(RobustScaler+Mode) 2018-09-24 00:45:49.598859
# hparamsearch.best_params_ = {'xgb__max_depth': 6, 'xgb__n_estimators': 240, 'xgb__reg_alpha': 1, 'xgb__reg_lambda': 0.2}
# hparamsearch.best_score_ = 0.12586607415571907
# Kaggle score = 0.12977

# GridSearchCV(RobustScaler+Mode+TotalSF+log1p+newhparam) 2018-09-24 09:24:15.272289
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__reg_alpha': 0.464, 'xgb__reg_lambda': 0.8571, 'xgb__subsample': 0.5213}
# hparamsearch.best_score_ = 0.12391159562822468
# Kaggle score = 0.12539

# GridSearchCV(RobustScaler+Mode+TotalSF+log1p+newhparam) 2018-09-24 09:35:05.235635
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 4, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.464, 'xgb__reg_lambda': 0.8571, 'xgb__subsample': 0.5213}
# hparamsearch.best_score_ = 0.12263193522001599
# Kaggle score = 0.12621

# GridSearchCV(RobustScaler+Mode+TotalSF+log1p+newhparam) 2018-09-24 09:48:44.979292
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.464, 'xgb__reg_lambda': 0.8571, 'xgb__subsample': 0.5213}
# hparamsearch.best_score_ = 0.12329124997588987
# Kaggle score = 0.12632

# GridSearchCV(RobustScaler+Mode+TotalSF+log1p+newhparam) 2018-09-24 10:03:43.146468
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 4, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.3, 'xgb__reg_lambda': 0.9, 'xgb__subsample': 0.5213}
# hparamsearch.best_score_ = 0.12187840824676832
# Kaggle score = 0.12618

# GridSearchCV(RobustScaler+Mode+TotalSF+log1p+newhparam) 2018-09-24 10:29:37.716113
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.2, 'xgb__reg_lambda': 0.9, 'xgb__subsample': 0.5213}
# hparamsearch.best_score_ = 0.12177648367932005
# Kaggle score = 0.12911

# GridSearchCV(RobustScaler+Mode+TotalSF+boxcox+newhparam) 2018-09-24 12:22:27.333097
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.2, 'xgb__reg_lambda': 0.9, 'xgb__subsample': 0.5213}
# hparamsearch.best_params_ = 0.12177192528415169
# Kaggle score = 

# GridSearchCV(RobustScaler+Mode+TotalSF+boxcox+newhparam+aggregatedMinorities5%) 2018-09-25 14:19:40.116817
# hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 4, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.2, 'xgb__reg_lambda': 1.2, 'xgb__subsample': 0.5213}
# hparamsearch.best_params_ = 0.1241218566722285
# Kaggle score = 0.12842

    # GridSearchCV(RobustScaler+enet+xgb+outliers+bagging) 2018-09-25 14:19:40.116817
    # hparamsearch.best_params_ = {'xgb__colsample_bytree': 0.4603, 'xgb__gamma': 0.0468, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 4, 'xgb__min_child_weight': 1.7817, 'xgb__n_estimators': 2200, 'xgb__random_state': 42, 'xgb__reg_alpha': 0.2, 'xgb__reg_lambda': 1.2, 'xgb__subsample': 0.5213}
    # hparamsearch.best_params_ = 0.114443634127553
    # Kaggle score = 0.11924