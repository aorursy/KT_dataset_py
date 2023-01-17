# Data Pre-Processing
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
import seaborn as sns
import matplotlib.mlab as mlab
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import cross_val_score
# Read data
train_data = pd.read_csv('../input/train.csv')
train_data.drop(train_data[(train_data["GrLivArea"]>4000)&(train_data["SalePrice"]<300000)].index,inplace=True)
# Missing Value Count Function
def show_missing():
    missing = train_data.columns[train_data.isnull().any()].tolist()
    return missing

# Missing data counts and percentage
print('Missing Data Count')
print(train_data[show_missing()].isnull().sum().sort_values(ascending = False))
print('--'*40)
print('Missing Data Percentage')
print(round(train_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(train_data)*100,2))
# Functions to address missing data

# Explore features
def feat_explore(column):
    return train_data[column].value_counts()

# Function to impute missing values
def feat_impute(column, value):
    train_data.loc[train_data[column].isnull(),column] = value
# Features with over 50% of its observations missings will be removed
#train_data.Fence.value_counts()
#train_data.Alley.value_counts()
#train_data.PoolQC.value_counts()
#train_data.MiscFeature.value_counts()
#train_data = train_data.drop(['PoolQC','MiscFeature','Alley','Fence'],axis = 1)
train_data['PoolQC'] = train_data['PoolQC'].fillna('None')
train_data['MiscFeature'] = train_data['MiscFeature'].fillna('None')
train_data['Alley'] = train_data['Alley'].fillna('None')
train_data['Fence'] = train_data['Fence'].fillna('None')
# FireplaceQu missing data
# Impute the nulls with None 
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')

# LotFrontage will impute with the mean
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
null_garage = ['GarageYrBlt','GarageType','GarageQual','GarageCond','GarageFinish']
# Impute null garage features
for cols in null_garage:
   if train_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)
# Basement Features
# BsmtFinType2 and BsmtExposure are both missing 38 observations
# Check that data is missing in the same rows
# Confirm if all nulls correspond to homes without basements
null_basement = ['BsmtFinType2','BsmtExposure']

# Impute the only null BsmtFinType2 with a basement at index 332 with most frequent value
train_data.iloc[332, train_data.columns.get_loc('BsmtFinType2')] = train_data['BsmtFinType2'].mode()[0]
#train_data.set_value(332,'BsmtFinType2',train_data['BsmtFinType2'].mode()[0])

# Impute the only null BsmtExposure with a basement at index 948 with most frequent value
train_data.iloc[948, train_data.columns.get_loc('BsmtExposure')] = train_data['BsmtExposure'].mode()[0]

# Impute the remaining nulls as None
for cols in null_basement:
   if train_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)

# Basement Features Part 2
null_basement2 = ['BsmtFinType1', 'BsmtCond','BsmtQual']

# Impute nulls to None or 0
for cols in null_basement2:
    if train_data[cols].dtype ==np.object:
        cols = feat_impute(cols, 'None')
    else:
        cols = feat_impute(cols, 0)    

# MasVnrArea and MasVnrType are each missing 8 observations

# Impute MasVnrArea with the most frequent values
# feat_explore('MasVnrArea')
# feat_impute('MasVnrArea','None')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])

# Impute MasVnrType with the most frequent values
# feat_explore('MasVnrType')
# feat_impute('MasVnrType',0.0)
train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])

# Electrical is only missing one value
# Impute Electrical with the most frequent value, 'SBrkr'
# feat_explore('Electrical')
# feat_impute('Electrical','SBrkr')
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
# Confirm all changes
print('Missing Data Count')
print(train_data[show_missing()].isnull().sum().sort_values(ascending = False))
print('No Missing Values')
train_data.info()
# Data Types
# Categorical Features
print('Categorical Features:\n ', train_data.select_dtypes(include=['object']).columns)
print('--'*40)

# Numeric Features
print('Numeric Features:\n ', train_data.select_dtypes(exclude=['object']).columns)
catcols = train_data.select_dtypes(['object'])
for cat in catcols:
    print('--'*40)
    print(cat)
    print(train_data[cat].value_counts())
# Encode ordinal data
train_data['LotShape'] = train_data['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})
train_data['LandContour'] = train_data['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})
train_data['Utilities'] = train_data['Utilities'].map({'NoSeWa':0,'NoSeWa':1,'AllPub':2})
train_data['BldgType'] = train_data['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})
train_data['HouseStyle'] = train_data['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
train_data['LandSlope'] = train_data['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})
train_data['Street'] = train_data['Street'].map({'Grvl':0,'Pave':1})
train_data['MasVnrType'] = train_data['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})
train_data['CentralAir'] = train_data['CentralAir'].map({'N':0,'Y':1})
train_data['GarageFinish'] = train_data['GarageFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3})
train_data['PavedDrive'] = train_data['PavedDrive'].map({'N':0,'P':1,'Y':2})
train_data['BsmtExposure'] = train_data['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
train_data['ExterQual'] = train_data['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['ExterCond'] = train_data['ExterCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['BsmtCond'] = train_data['BsmtCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['BsmtQual'] = train_data['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['HeatingQC'] = train_data['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['KitchenQual'] = train_data['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['FireplaceQu'] = train_data['FireplaceQu'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['GarageQual'] = train_data['GarageQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train_data['GarageCond'] = train_data['GarageCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

# Encode Categorical Variables
train_data['Foundation'] = train_data['Foundation'].map({'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5})
train_data['Heating'] = train_data['Heating'].map({'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5})
train_data['Electrical'] = train_data['Electrical'].map({'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4})
train_data['Functional'] = train_data['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})
train_data['GarageType'] = train_data['GarageType'].map({'None':0,'Detchd':1,'CarPort':2,'BuiltIn':3,'Basment':4,'Attchd':5,'2Types':6})
train_data['SaleType'] = train_data['SaleType'].map({'Oth':0,'ConLD':1,'ConLI':2,'ConLw':3,'Con':4,'COD':5,'New':6,'VWD':7,'CWD':8,'WD':9})
train_data['SaleCondition'] = train_data['SaleCondition'].map({'Partial':0,'Family':1,'Alloca':2,'AdjLand':3,'Abnorml':4,'Normal':5})
train_data['MSZoning'] = train_data['MSZoning'].map({'A':0,'FV':1,'RL':2,'RP':3,'RM':4,'RH':5,'C (all)':6,'I':7})
train_data['LotConfig'] = train_data['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})
train_data['Neighborhood'] = train_data['Neighborhood'].map({'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3, 'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,
                                                             'IDOTRR':9,'MeadowV':10,'Mitchel':11, 'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15, 'NWAmes':16,
                                                             'OldTown':17,'SWISU':18,'Sawyer':19, 'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24})
train_data['Condition1'] = train_data['Condition1'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})
train_data['Condition2'] = train_data['Condition2'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})
train_data['RoofStyle'] = train_data['RoofStyle'].map({'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5})
train_data['RoofMatl'] = train_data['RoofMatl'].map({'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7})
train_data['Exterior1st'] = train_data['Exterior1st'].map({'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,
                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'WdShing':16})
train_data['Exterior2nd'] = train_data['Exterior2nd'].map({'AsbShng':0,'AsphShn':1,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,
                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'Wd Shng':16})  
train_data['Alley'] = train_data['Alley'].map({'None':0,'Grvl':1,'Pave':2})
train_data['MiscFeature'] = train_data['MiscFeature'].map({'None':0,'Shed':1,'Gar2':2,'Othr':3,'TenC':4})
train_data['PoolQC'] = train_data['PoolQC'].map({'None':0,'Gd':1,'Fa':2,'Ex':3})
train_data['Fence'] = train_data['Fence'].map({'None':0,'MnPrv':1,'GdPrv':2,'GdWo':3,'MnWw':4})
# Confirm encoding
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress = False)
train_data.describe().transpose()
# Statistical Summary
print("SalePrice Statistical Summary:\n")
print(train_data['SalePrice'].describe())
print("Median Sale Price:", train_data['SalePrice'].median(axis = 0))
print('Skewness:',train_data['SalePrice'].skew())
skew = train_data['SalePrice'].skew()

# mean distribution
mu = train_data['SalePrice'].mean()

# std distribution
sigma = train_data['SalePrice'].std()
num_bins = 40

# Histogram of SalesPrice
plt.figure(figsize=(11, 6))
n, bins, patches = plt.hist(train_data['SalePrice'], num_bins, normed=1,edgecolor = 'black', lw = 1, alpha = .40)

# Normal Distribution
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('Sale Price')
plt.ylabel('Probability density')

plt.title(r'$\mathrm{Histogram\ of\ SalePrice:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma))
plt.grid(True)
#fig.tight_layout()
plt.show()
train_data.info()
def getObjectFeature(df, col, datalength=1458):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    elif len([i for i in df[col].T.notnull() if i == True])!=datalength: # if there's missing data..
        print('feature',col,'is missing data.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
        df1[col] = [counts.index.tolist().index(i) for i in df1[col]] # do the conversion
        return df1 # make the new (integer) column from the conversion
import sklearn.feature_selection as fs # feature selection library in scikit-learn
included_features = [col for col in list(train_data)
                    if len([i for i in train_data[col].T.notnull() if i == True])==1458
                    and col!='SalePrice' and col!='id']
# define the training data X...
X = train_data[included_features] # the feature data
Y = train_data[['SalePrice']] # the target
yt = [i for i in Y['SalePrice']] # the target list 
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
X.head()
# and the data for the competition submission...
#X_test = [included_features]
mir_result = fs.mutual_info_regression(X, yt) # mutual information regression feature ordering
feature_scores = []
for i in np.arange(len(included_features)):
    feature_scores.append([included_features[i],mir_result[i]])
sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True) 
print(np.array(sorted_scores))
# define a function to do the necessary model building....
def getModel(sorted_scores,train,numFeatures):
    included_features = np.array(sorted_scores)[:,0][:numFeatures] # ordered list of important features
    # define the training data X...
    X = train[included_features]
    Y = train[['SalePrice']]
    # transform categorical data if included in X...
    for col in list(X):
        if X[col].dtype=='object':
            X = getObjectFeature(X, col)
    # define the number of estimators to consider
    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    mean_rfrs = []
    std_rfrs_upper = []
    std_rfrs_lower = []
    yt = [i for i in Y['SalePrice']]
    np.random.seed(999)
    # for each number of estimators, fit the model and find the results for 8-fold cross validation
    for i in estimators:
        model = rfr(n_estimators=i,max_depth=None)
        scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
        mean_rfrs.append(scores_rfr.mean())
        std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
        std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
    return mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators

# define a function to plot the model expected variance results...
def plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators,numFeatures):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(estimators,mean_rfrs,marker='o',
           linewidth=4,markersize=12)
    ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                    facecolor='green',alpha=0.3,interpolate=True)
    ax.set_ylim([-.2,1])
    ax.set_xlim([0,80])
    plt.title('Expected Variance of Random Forest Regressor: Top %d Features'%numFeatures)
    plt.ylabel('Expected Variance')
    plt.xlabel('Trees in Forest')
    plt.grid()
    plt.show()
    return
# top 20 features...
mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators = getModel(sorted_scores,train_data,20)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators,20)
# top 30 features...
mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators = getModel(sorted_scores,train_data,30)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators,30)
# top 40 features...
mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators = getModel(sorted_scores,train_data,40)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,estimators,40)
# build the model with the desired parameters...
numFeatures = 60 # the number of features to inlcude
trees = 40 # trees in the forest
included_features = np.array(sorted_scores)[:,0][:numFeatures]
# define the training data X...
X = train_data[included_features]
Y = train_data[['SalePrice']]
#bonf_outlier = [523,691,803,898,1046,1169,1182,1298]
#X = X.drop(bonf_outlier)
#Y = Y.drop(bonf_outlier)
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
yt = [i for i in Y['SalePrice']]
np.random.seed(11111)

model = rfr(n_estimators=trees,max_depth=None)
scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
print('explained variance scores for k=10 fold validation:',scores_rfr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
# fit the model
model.fit(X,yt)
# Load test data
test_data = pd.read_csv('../input/test.csv')
# Missing Value Count Function
def show_missing():
    missing = test_data.columns[test_data.isnull().any()].tolist()
    return missing

# Missing data counts and percentage
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print('Missing Data Count')
print(test_data[show_missing()].isnull().sum().sort_values(ascending = False))
print('--'*40)
print('Missing Data Percentage')
print(round(test_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(test_data)*100,2))
# Function to impute missing values
def feat_impute(column, value):
    test_data.loc[test_data[column].isnull(),column] = value
# Features with over 50% of its observations missings will be removed
test_data['PoolQC'] = test_data['PoolQC'].fillna('None')
test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')
test_data['Alley'] = test_data['Alley'].fillna('None')
test_data['Fence'] = test_data['Fence'].fillna('None')
# FireplaceQu missing data
# Impute the nulls with None 
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')
# LotFrontage nulls
# Impute with mean
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())
# Garage features null
# The null values may be homes that do not have Garages at all.
# Most of the nulls are associated with homes without a garage.  
# However, there are exceptions that must be addressed before we can inpute the remaining nulls with 'None'
# Inpute nulls at index 666 that have a garage with most common value in each column for categorical variables 
test_data.iloc[666, test_data.columns.get_loc('GarageYrBlt')] = test_data['GarageYrBlt'].mode()[0]
test_data.iloc[666, test_data.columns.get_loc('GarageCond')] = test_data['GarageCond'].mode()[0]
test_data.iloc[666, test_data.columns.get_loc('GarageFinish')] = test_data['GarageFinish'].mode()[0]
test_data.iloc[666, test_data.columns.get_loc('GarageQual')] = test_data['GarageQual'].mode()[0]
test_data.iloc[666, test_data.columns.get_loc('GarageType')] = test_data['GarageType'].mode()[0]

# Inpute nulls at index 1116 that have a garage with most common value in each column for categorical variables 
test_data.iloc[1116, test_data.columns.get_loc('GarageYrBlt')] = test_data['GarageYrBlt'].mode()[0]
test_data.iloc[1116, test_data.columns.get_loc('GarageCond')] = test_data['GarageCond'].mode()[0]
test_data.iloc[1116, test_data.columns.get_loc('GarageFinish')] = test_data['GarageFinish'].mode()[0]
test_data.iloc[1116, test_data.columns.get_loc('GarageQual')] = test_data['GarageQual'].mode()[0]
test_data.iloc[1116, test_data.columns.get_loc('GarageType')] = test_data['GarageType'].mode()[0]

# Inpute nulls at index 1116 that have a garage with median value in each column for continuous variables 
test_data.iloc[1116, test_data.columns.get_loc('GarageCars')] = test_data['GarageCars'].median()
test_data.iloc[1116, test_data.columns.get_loc('GarageArea')] = test_data['GarageArea'].median()

# Impute the remaining nulls as None
null_garage = ['GarageYrBlt','GarageCond','GarageFinish','GarageQual', 
                 'GarageType','GarageCars','GarageArea']

for cols in null_garage:
   if test_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)
null_basement1 = ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']
 
# The null values may be homes that do not have Garages at all.
# Most of the nulls are associated with homes without a basement.  
# However, there are exceptions that must be addressed before we can inpute the remaining nulls with 'None'
# Inpute nulls of BasmtExposure that have a basement with most common value 
test_data.iloc[27, test_data.columns.get_loc('BsmtExposure')] = test_data['BsmtExposure'].mode()[0]
test_data.iloc[888, test_data.columns.get_loc('BsmtExposure')] = test_data['BsmtExposure'].mode()[0]

# Inpute nulls of BsmtCond that have a basement with most common value 
test_data.iloc[540, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]
test_data.iloc[580, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]
test_data.iloc[725, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]
test_data.iloc[1064, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]
test_data.iloc[1064, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]

# Inpute nulls in BsmetQualthat have a basement with most common value
test_data.iloc[757, test_data.columns.get_loc('BsmtQual')] = test_data['BsmtQual'].mode()[0]
test_data.iloc[758, test_data.columns.get_loc('BsmtQual')] = test_data['BsmtQual'].mode()[0]

# Inpute nulls in basement features with 'None' for categorical variables or zero for numeric variables
for cols in null_basement1:
   if test_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)
null_basement2= ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']  

for cols in null_basement2:
   if test_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)
null_masonry = ['MasVnrType','MasVnrArea']

# Impute exceptions to assumption that nulls correspond to homes with no exposure
test_data.iloc[1150, test_data.columns.get_loc('MasVnrType')] = test_data['MasVnrType'].mode()[0]

# Impute the remaining nulls with 'None' or zero
for cols in null_masonry:
   if test_data[cols].dtype ==np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)
# Impute other categorical features with most frequent value
null_others = ['MSZoning', 'Utilities','Functional','Exterior2nd','Exterior1st','SaleType','KitchenQual'] 
# Impute with most common value
for cols in null_others:
    test_data[cols] = test_data[cols].mode()[0]
# LotFrontage nulls
# Impute with mean
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())
test_data.info()
# Encode ordinal data
test_data['LotShape'] = test_data['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})
test_data['LandContour'] = test_data['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})
test_data['Utilities'] = test_data['Utilities'].map({'NoSeWa':0,'NoSeWa':1,'AllPub':2})
test_data['BldgType'] = test_data['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})
test_data['HouseStyle'] = test_data['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
test_data['LandSlope'] = test_data['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})
test_data['Street'] = test_data['Street'].map({'Grvl':0,'Pave':1})
test_data['MasVnrType'] = test_data['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})
test_data['CentralAir'] = test_data['CentralAir'].map({'N':0,'Y':1})
test_data['GarageFinish'] = test_data['GarageFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3})
test_data['PavedDrive'] = test_data['PavedDrive'].map({'N':0,'P':1,'Y':2})
test_data['BsmtExposure'] = test_data['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
test_data['ExterQual'] = test_data['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['ExterCond'] = test_data['ExterCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['BsmtCond'] = test_data['BsmtCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['BsmtQual'] = test_data['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['HeatingQC'] = test_data['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['KitchenQual'] = test_data['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['FireplaceQu'] = test_data['FireplaceQu'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['GarageQual'] = test_data['GarageQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test_data['GarageCond'] = test_data['GarageCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

# Encode Categorical Variables
test_data['Foundation'] = test_data['Foundation'].map({'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5})
test_data['Heating'] = test_data['Heating'].map({'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5})
test_data['Electrical'] = test_data['Electrical'].map({'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4})
test_data['Functional'] = test_data['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})
test_data['GarageType'] = test_data['GarageType'].map({'None':0,'Detchd':1,'CarPort':2,'BuiltIn':3,'Basment':4,'Attchd':5,'2Types':6})
test_data['SaleType'] = test_data['SaleType'].map({'Oth':0,'ConLD':1,'ConLI':2,'ConLw':3,'Con':4,'COD':5,'New':6,'VWD':7,'CWD':8,'WD':9})
test_data['SaleCondition'] = test_data['SaleCondition'].map({'Partial':0,'Family':1,'Alloca':2,'AdjLand':3,'Abnorml':4,'Normal':5})
test_data['MSZoning'] = test_data['MSZoning'].map({'A':0,'FV':1,'RL':2,'RP':3,'RM':4,'RH':5,'C (all)':6,'I':7})
test_data['LotConfig'] = test_data['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})
test_data['Neighborhood'] = test_data['Neighborhood'].map({'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3, 'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,
                                                             'IDOTRR':9,'MeadowV':10,'Mitchel':11, 'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15, 'NWAmes':16,
                                                             'OldTown':17,'SWISU':18,'Sawyer':19, 'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24})
test_data['Condition1'] = test_data['Condition1'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})
test_data['Condition2'] = test_data['Condition2'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})
test_data['RoofStyle'] = test_data['RoofStyle'].map({'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5})
test_data['RoofMatl'] = test_data['RoofMatl'].map({'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7})
test_data['Exterior1st'] = test_data['Exterior1st'].map({'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,
                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'WdShing':16})
test_data['Exterior2nd'] = test_data['Exterior2nd'].map({'AsbShng':0,'AsphShn':1,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,
                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'Wd Shng':16})  
test_data['Alley'] = test_data['Alley'].map({'None':0,'Grvl':1,'Pave':2})
test_data['MiscFeature'] = test_data['MiscFeature'].map({'None':0,'Shed':1,'Gar2':2,'Othr':3,'TenC':4})
test_data['PoolQC'] = test_data['PoolQC'].map({'None':0,'Gd':1,'Fa':2,'Ex':3})
test_data['Fence'] = test_data['Fence'].map({'None':0,'MnPrv':1,'GdPrv':2,'GdWo':3,'MnWw':4})
#test_data.Fence.value_counts()
test_data.describe().transpose()
# Missing data counts and percentage
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print('Missing Data Count')
print(test_data[show_missing()].isnull().sum().sort_values(ascending = False))
print('--'*40)
print('Missing Data Percentage')
print(round(test_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(test_data)*100,2))

# re-define a function to convert an object (categorical) feature into an int feature
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
#         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels
        df1[col] = [counts.index.tolist().index(i) 
                    if i in counts.index.tolist() 
                    else 0 
                    for i in df1[col] ] # do the conversion
        return df1 # make the new (integer) column from the conversion
# apply the model to the test data and get the output...
included_features = np.array(sorted_scores)[:,0][:numFeatures]
X_test = test_data[included_features]
for col in list(X_test):
    if X_test[col].dtype=='object':
        X_test = getObjectFeature(X_test, col, datalength=1459)
# print(X_test.head(20))
y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0
print(y_output)
# Create submission
# define the data frame for the results
saleprice = pd.DataFrame(y_output, columns=['SalePrice'])
results = pd.concat([test_data['Id'],saleprice['SalePrice']],axis=1)
results.head()
# and write to output
#results.to_csv('housepricing_submission1.csv', index = False)
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
y = train_data.SalePrice
X_train = train_data[included_features]
X = X_train
model = sm.OLS(y,X)
results = model.fit()
bonf_test = results.outlier_test()['bonf(p)']
bonf_outlier = list(bonf_test[bonf_test<1e-3].index) 
print(bonf_test[bonf_test<1e-3])
#bonf_outlier = [523,691,803,898,1046,1169,1182,1298]
X_train1 = X_train.drop(bonf_outlier)
y=y.drop(bonf_outlier)
X = X_train1
#X_train.info()
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
from xgboost import XGBRegressor
import xgboost as xgb
#my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
#my_model.fit(train_X, train_y, verbose=False)
my_model = XGBRegressor(subsample = 0.4,colsample_bytree = 0.4,n_estimators=1200, learning_rate=0.02)
my_model.fit(train_X, train_y, early_stopping_rounds=30, 
             eval_set=[(test_X, test_y)], verbose=False)
cv_score = cross_val_score(my_model, train_X, train_y, cv = 10)
print('CV Score is: '+ str(np.mean(cv_score)))
#train_X.shape
#train_X.columns
test_ID = test_data['Id']
X_test = test_data[included_features]
test = X_test
#test.info()
#test_data.describe()
predictions = my_model.predict(test.values)
saleprice1 = pd.DataFrame(predictions, columns=['SalePrice'])
results1 = pd.concat([test_ID,saleprice1['SalePrice']],axis=1)
results1.head()
results1.to_csv('housepricing_submission4.csv', index = False)
