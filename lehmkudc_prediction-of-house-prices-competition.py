import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print( df_train.shape )
print( df_test.shape )
id_train = df_train['Id']
X_train = df_train.drop(['Id','SalePrice'],axis=1)
y_train = df_train['SalePrice']
print( id_train.head() )
print( y_train.head() )
X_train.head()
id_test = df_test['Id']
X_test = df_test.drop(['Id'],axis=1)
print( id_test.head() )
X_test.head()
X = pd.concat([X_train,X_test])
def d_dummy( x, df_dum ):
    # Convert's a column of categorical data into columns of dummy variables
    dum = pd.get_dummies( x )
    for dcol in dum.columns:
        name = x.name + "_" + str(dcol)
        df_dum[name] = dum[dcol]
    return (df_dum)

def d_look( x , type):
    # Does a quick look at the status of a feature. Useful when cleaning individiual features
    print( 'Null Values = ', x.isnull().sum())
    if (type == 'cat'):
        print( x.value_counts() )
    if (type == 'num'):
        print( x.describe() )
        
def d_comb( x, df_dum, Options):
    # Makes dummy columns for features with multiple columns but same indicator options
    # Allows for multiples if more than one is present
    for o in Options:
        name = "Condition_" + o
        df_dum[name] = np.isin( x, o ).sum(axis=1) + 0
    return df_dum

def d_comb2(x, df_dum, Options):
    # As d_comb but does not allow for multiples
    for o in Options:
        name = "Condition_" + o
        df_dum[name] = (np.isin( x, o ).sum(axis=1) >= 1)+ 0
    return df_dum
X = pd.concat([X_train,X_test])
Xc = pd.DataFrame()
#======================================================
Xc = d_dummy( X['MSSubClass'], Xc)
Xc = d_dummy( X['MSZoning'].replace(np.NaN, "None"), Xc)
Xc['LotFrontage'] = X['LotFrontage'].replace( np.NaN, 0 )
Xc['LotArea'] = X.LotArea
Xc = d_dummy( X.Street, Xc)
Xc = d_dummy( X.Alley.replace(np.NaN, "None"), Xc)
Xc = d_dummy( X.LotShape, Xc)
Xc = d_dummy( X.LandContour, Xc)
#Xc = d_dummy( X.Utilities, Xc)  # Utilities Ignored due to lack of feature variation
Xc = d_dummy( X.LotConfig, Xc)
Xc = d_dummy( X.LandSlope, Xc)
Xc = d_dummy( X.Neighborhood, Xc)
conditions = ['Feedr','Artery','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
Xc = d_comb( X[['Condition1','Condition2']], Xc, conditions)
Xc = d_dummy( X.BldgType, Xc)
Xc = d_dummy( X.HouseStyle, Xc)
Xc['OverallQual'] = X.OverallQual
Xc['OverallCond'] = X.OverallCond
Xc['YearBuilt'] = X.YearBuilt
Xc['YearRemodAdd'] = X.YearRemodAdd
Xc = d_dummy( X.RoofStyle, Xc)
Xc = d_dummy( X.RoofMatl, Xc)
ex_rep = {np.NaN: "Other", "Brk Cmn":"BrkComm", "CmentBd":"CemntBd", "Wd Shng":"WdShing" }
Xc = d_comb2( X[['Exterior1st','Exterior2nd']].replace(ex_rep), Xc, 
             np.unique(X[['Exterior1st','Exterior2nd']].replace(ex_rep).values))
c = np.diag( X.MasVnrArea.replace(np.NaN, 0).values )
A = d_dummy( X.MasVnrType.replace(np.NaN, "None"), pd.DataFrame()).values
Xc['MasVnr_BrkCmn'] = np.dot( c, A)[:,0]
Xc['MasVnr_BrkFace'] = np.dot( c, A)[:,1]
Xc['MasVnr_Stone'] = np.dot( c, A)[:,3]
Xc = d_dummy( X.ExterQual, Xc)
Xc = d_dummy( X.ExterCond, Xc)
Xc = d_dummy( X.Foundation, Xc)
Xc = d_dummy( X.BsmtQual.replace(np.NaN, "None"), Xc)
Xc = d_dummy( X.BsmtCond.replace(np.NaN, "None"), Xc)
Xc = d_dummy( X.BsmtExposure.replace(np.NaN, "None"), Xc)
Options = ['GLQ','ALQ','BLQ','Rec','LwQ']
bsm1 = X.BsmtFinSF1.replace( np.NaN, 0)
bsm2 = X.BsmtFinSF2.replace( np.NaN, 0)
for o in Options:
    name = "BsmtFin_" + o
    Xc[name] =  (X.BsmtFinType1 == o)*bsm1 + (X.BsmtFinType2 == o)*bsm2
Xc['BsmtUnfSF'] = X.BsmtUnfSF.replace(np.NaN, 0)
Xc['TotalBsmtSF'] = X.TotalBsmtSF.replace(np.NaN, 0)
Xc = d_dummy( X.Heating, Xc)
Xc = d_dummy( X.HeatingQC, Xc)
Xc['CentralAir'] = (X.CentralAir == 'Y') + 0
Xc = d_dummy( X.Electrical.replace( np.NaN, 'Mix'), Xc)
Xc['1stFlrSF'] = X['1stFlrSF']
Xc['2ndFlrSF'] = X['2ndFlrSF']
Xc['LowQualFinSF'] = X['LowQualFinSF']
Xc['GrLivArea'] = X['GrLivArea']
Xc['BsmtFullBath'] = X.BsmtFullBath.replace(np.NaN, 0)
Xc['BsmtHalfBath'] = X.BsmtHalfBath.replace(np.NaN, 0)
Xc['FullBath'] = X['FullBath']
Xc['HalfBath'] = X['HalfBath']
Xc['BedroomAbvGr'] = X['BedroomAbvGr']
Xc['KitchenAbvGr'] = X['KitchenAbvGr']
Xc = d_dummy(X['KitchenQual'].replace(np.NaN, 'TA'), Xc)
Xc['TotRmsAbvGrd'] = X['TotRmsAbvGrd']
Xc = d_dummy(X['Functional'].replace(np.NaN, 'Typ'), Xc)
Xc['Fireplaces'] = X['Fireplaces']
Xc = d_dummy( X['FireplaceQu'].replace(np.NaN, "None"),Xc)
Xc = d_dummy( X['GarageType'].replace(np.NaN, "None"),Xc)
Xc['GarageYrBlt'] = X['GarageYrBlt'].replace(np.NaN, 0)
Xc = d_dummy(X['GarageFinish'].replace(np.NaN,"no_garage"), Xc)
Xc['GarageCars'] = X['GarageCars'].replace(np.NaN, 0)
Xc['GarageArea'] = X['GarageArea'].replace(np.NaN, 0)
Xc = d_dummy(X['GarageQual'].replace(np.NaN,"no_garage"), Xc)
Xc = d_dummy(X['GarageCond'].replace(np.NaN,"no_garage"), Xc)
Xc['PavedDrive'] = X['PavedDrive'] == 'Y'
Xc['WoodDeckSF'] = X['WoodDeckSF']
Xc['OpenPorchSF'] = X['OpenPorchSF']
Xc['EnclosedPorch'] = X['EnclosedPorch']
Xc['3SsnPorch'] = X['3SsnPorch']
Xc['ScreenPorch'] = X['ScreenPorch']
Xc['PoolArea'] = X['PoolArea']
#PoolQC omitted for lack of feature variation
Xc = d_dummy(X['Fence'].replace(np.NaN, "None"), Xc)
c = np.diag( X.MiscVal.replace(np.NaN, 0).values )
A = d_dummy( X.MiscFeature.replace(np.NaN, "None"), pd.DataFrame()).values
Xc['MiscFeature_Gar2'] = np.dot(c,A)[:,0]
Xc['MiscFeature_Othr'] = np.dot(c,A)[:,2]
Xc['MiscFeature_Shed'] = np.dot(c,A)[:,3]
Xc['MiscFeature_TenC'] = np.dot(c,A)[:,4]
Xc['YrSold'] = X.MoSold/12 + X.YrSold
Xc = d_dummy( X.SaleType.replace(np.NaN, "Oth") ,Xc)
Xc = d_dummy( X.SaleCondition , Xc)
print( Xc.isnull().sum().sum())
Xc.head()
scaler = StandardScaler()
Xs = scaler.fit_transform(Xc)
Xs_train = Xc.iloc[:1460,:]
Xs_test = Xc.iloc[1460:,]
print(Xs_train.shape)
print(Xs_test.shape)
def d_method( dX, dy, model, random_state = 0, k = 5 ):
    # Fits a categorical model and outputs a cross-validation result of:
    # Accuracy, Recall, Precision, and the model thats fit last.
    # The data is train/test split and shuffled systematically
    kf = ShuffleSplit( n_splits = k )
    mse = np.zeros(k)
    i = 0
    for train_index, test_index in kf.split(dX):
        t_model = clone(model)
        dX_train, dX_test = dX.iloc[train_index], dX.iloc[test_index]
        dy_train, dy_test = dy[train_index], dy[test_index]
        t_model.fit( dX_train, dy_train )
        dy_pred = t_model.predict(dX_test)
        mse[i] = mean_squared_error( dy_pred, dy_test)
        i = i+1
    return( mse, t_model )
mse, model = d_method( Xs_train, y_train , LinearRegression() )
print( mse )
mse, model = d_method( Xs_train, y_train , S() )
print( mse )
