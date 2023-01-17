import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, preprocessing
from sklearn.linear_model import LinearRegression
import pickle as pickle



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_rows', 100)
#Import CSV Files
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
#Remove NA here before combining files, since SalesPrice will be removed otherwise. 
#Since "Test" doesn't have SalePrice, it will be removed when checking for columns with high NA. 
#So we will remove "Train" columns with High NA,then remove the repesctive "Test" columns
print(train.shape)
print(test.shape)

#Dropped variables with 20% of NA or more. 
train = train[train.columns[train.isnull().mean() < 0.2]]
print(train.shape)
test.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], inplace=True)
print(test.shape)
#Combibe both files to allow for proper one hot encoding
data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)

data.drop(columns='Id',inplace=True)

#Drop empty columns
data.dropna(axis=1, how='all', inplace=True)

#Convert to Float to allow for NA to be removed.
#for c in data.select_dtypes(np.number).columns:
#    try:
#        data[c] = data[c].astype('Float64')
#    except:
#        print('could not cast {} to Float64'.format(c))

print(data.dtypes)
print(data.shape)

#One Hot Encoding. We apply prefixes here due to duplicate column names.
data=pd.get_dummies(data, prefix=
             {'MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
            'Functional','GarageType', 'GarageFinish', 'GarageQual',
            'GarageCond', 'PavedDrive','SaleType', 'SaleCondition'}, 
        prefix_sep='', columns=
            ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
            'Functional', 'GarageType', 'GarageFinish', 'GarageQual',
            'GarageCond', 'PavedDrive','SaleType', 'SaleCondition'])



print(data.shape)
data.to_csv('Data After Data Preprocessing.csv')

#Split Data into origional files after Data Preprocessing
train=data[0:1460]
test=data[1460:2919]
print(train.shape)
print(test.shape)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
#impute missing values then convert numpy array to pandas df.
X_train=train.columns
my_imputer = SimpleImputer()
train_with_imputed_values = my_imputer.fit_transform(train)
train = pd.DataFrame(train_with_imputed_values, columns = X_train)
train.to_csv('train after imputation.csv')
corr_new_train=train.corr()
corr_dict2=corr_new_train['SalePrice'].sort_values(ascending=False).to_dict()
corr_dict2
#Removes Multicollinearity
best_columns=[]
for key,value in corr_dict2.items():
    if ((value>=0.3175) & (value<0.9)) | (value<=-0.315):
        best_columns.append(key)
best_columns
len(best_columns)
#Positive Skewness and Kurtosis
plt.figure(figsize=(10,8))
sb.set(font_scale=1.2)
sb.distplot(train['SalePrice'],color='violet')
plt.xlabel('SalePrice',fontsize=20)
print('Skew Dist:',train['SalePrice'].skew())
print('Kurtosis Dist:',train['SalePrice'].kurt())
train['SalePrice_Log1p'] = np.log1p(train.SalePrice)
plt.figure(figsize=(10,8))
sb.set(font_scale=1.2)
sb.distplot(train['SalePrice_Log1p'],color='indigo')
plt.xlabel('SalePrice_Log1p',fontsize=20)
print('Skew Dist:',train['SalePrice_Log1p'].skew())
print('Kurtosis Dist:',train['SalePrice_Log1p'].kurt())
#Since Each variables is a feature variable of the SalesPrice, we cannot utilize the Z score method to remove outliers.
#As such, we must utilize visualizations or a better method, to take into account SalesPrice and each feature variable.

plt.style.use('ggplot')
fig, axes = plt.subplots(20, 2,figsize=(20,60))
fig.subplots_adjust(hspace=0.8)
sb.set(font_scale=1.2)
colors=[plt.cm.prism_r(each) for each in np.linspace(0, 1, len(best_columns))]
for i,ax,color in zip(best_columns,axes.flatten(),colors):
    sb.regplot(x=train[i], y=train["SalePrice_Log1p"], fit_reg=True,marker='o',scatter_kws={'s':50,'alpha':0.7},color=color,ax=ax)
    plt.xlabel(i,fontsize=12)
    plt.ylabel('SalePrice_Log1p',fontsize=12)
    ax.set_title('SalePrice_Log1p'+' - '+str(i),color=color,fontweight='bold',size=20)
plt.style.use('dark_background')
fig, axes = plt.subplots(20, 2,figsize=(20,80))
fig.subplots_adjust(hspace=0.6)
colors=[plt.cm.prism_r(each) for each in np.linspace(0, 1, len(best_columns))]
for i,ax,color in zip(best_columns,axes.flatten(),colors):
    sb.regplot(x=train[i], y=train["SalePrice"], fit_reg=True,marker='o',scatter_kws={'s':50,'alpha':0.8},color=color,ax=ax)
    plt.xlabel(i,fontsize=12)
    plt.ylabel('SalePrice',fontsize=12)
    ax.set_yticks(np.arange(0,900001,100000))
    ax.set_title('SalePrice'+' - '+str(i),color=color,fontweight='bold',size=20)
"Looking at the SalePrice and SalePrice_Log1p visualizations"

train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice>100000)].index)
train = train.drop(train[(train['1stFlrSF']>4000) & (train.SalePrice>100000)].index)
train = train.drop(train[(train.MasVnrArea>1400) & (train.SalePrice>100000)].index)
train = train.drop(train[(train.BsmtFinSF1>5000) & (train.SalePrice>100000)].index)
train = train.drop(train[(train.LotFrontage>300) & (train.SalePrice>100000)].index)
train = train.drop(train[(train.WoodDeckSF>800) & (train.SalePrice>100000)].index)
#Drop all variables except those in best_columns df for both train and test.
train = train.filter(['OverallQual',
 'GrLivArea',
 'GarageCars',
 'GarageArea',
 'TotalBsmtSF',
 '1stFlrSF',
 'FullBath',
 'GarageQualEx',
 'TotRmsAbvGrd',
 'YearBuilt',
 'YearRemodAdd',
 'StreetEx',
 'SaleTypePConc',
 'MasVnrArea',
 'GarageYrBlt',
 'Fireplaces',
 'ExterCondGd',
 'ExterCondEx',
 'Exterior1stGLQ',
 'ElectricalEx',
 'GarageTypeFin',
 'HouseStyleNridgHt',
 'BsmtFinSF1',
 'RoofStyleNew',
 'BsmtFinType2Partial',
 'HeatingAttchd',
 'LotFrontage',
 'GarageFinishStone',
 'HouseStyleNoRidge',
 'WoodDeckSF',
 'StreetGd',
 '2ndFlrSF',
 'SaleTypeCBlock',
 'HeatingDetchd',
 'GarageFinishNone',
 'GarageTypeUnf',
 'GarageQualTA',
 'StreetTA',
 'ExterCondTA','SalePrice','SalePrice_Log1p'])
train.to_csv('train after column adjustment.csv')
#Drop all variables except those in best_columns df for both train and test.
test = test.filter(['OverallQual',
 'GrLivArea',
 'GarageCars',
 'GarageArea',
 'TotalBsmtSF',
 '1stFlrSF',
 'FullBath',
 'GarageQualEx',
 'TotRmsAbvGrd',
 'YearBuilt',
 'YearRemodAdd',
 'StreetEx',
 'SaleTypePConc',
 'MasVnrArea',
 'GarageYrBlt',
 'Fireplaces',
 'ExterCondGd',
 'ExterCondEx',
 'Exterior1stGLQ',
 'ElectricalEx',
 'GarageTypeFin',
 'HouseStyleNridgHt',
 'BsmtFinSF1',
 'RoofStyleNew',
 'BsmtFinType2Partial',
 'HeatingAttchd',
 'LotFrontage',
 'GarageFinishStone',
 'HouseStyleNoRidge',
 'WoodDeckSF',
 'StreetGd',
 '2ndFlrSF',
 'SaleTypeCBlock',
 'HeatingDetchd',
 'GarageFinishNone',
 'GarageTypeUnf',
 'GarageQualTA',
 'StreetTA',
 'ExterCondTA','SalePrice'])
test.to_csv('train after column adjustment.csv')
X = train.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = train.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
x = test.iloc[:, 0].values.reshape(-1, 1)
y = test.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
Y_pred = pd.DataFrame(Y_pred, columns=['SalePrice'])
Y_pred.index.name= 'Id'
Y_pred.to_csv('Precition2.csv')
print(Y_pred)
X = train.drop(columns=['SalePrice','SalePrice_Log1p']) 
y = train['SalePrice_Log1p']
lab_enc = preprocessing.LabelEncoder()
y = lab_enc.fit_transform(train['SalePrice_Log1p'])
test.shape
#SciKit-Learn Multiple Linear Regression Model

#Preprocess
X=preprocessing.StandardScaler().fit(X).transform(X)

#Train/Test Split. #Random State=0 ensure same output everytime

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=0)
print('X_train Shape :',X_train.shape)
print('X_test Shape :',X_test.shape)
print('y_train Shape :',y_train.shape)
print('y_test Shape :',y_test.shape)

#Algorithim Setup
clf=svm.SVC(kernel='linear',gamma=0.001,C=100.0)

#Model fitting
clf.fit(X_train,y_train)
np.set_printoptions(precision=2)#Limits float to 2 decimal places, default is 8.

#Prediction
house_price_prediction = clf.predict(X_test)
house_price_prediction = pd.DataFrame(house_price_prediction, columns=['SalePrice'])
house_price_prediction.index.name= 'Id'
house_price_prediction.to_csv('Precition.csv')
print(house_price_prediction)
from catboost import CatBoostRegressor
cols=train.select_dtypes(include=['object']).columns
model=CatBoostRegressor()
y=train.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(train.iloc[:,:-1],y)
model.fit(x_train,y_train,cat_features=cols)
print(model.score(x_train,y_train))
predicted=model.predict(test)
ids=pd.Series(test.all)
Saleprice=pd.Series(predicted)
ss=pd.concat([ids,Saleprice],axis=1)
ss['SalePrice']=ss[0]
ss.drop(0,inplace=True,axis=1)
print(ss)
ss.to_csv('house1.csv')