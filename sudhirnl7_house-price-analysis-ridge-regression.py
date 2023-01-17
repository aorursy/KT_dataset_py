import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import lightgbm as  lgb
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
seed = 4432
path = '../input/'
#path = 'dataset/'
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path + 'test.csv')
print('Number of rows and columns in train dataset:', train.shape)
print('Number of rows and columns in test dataset:', test.shape)
train.head()
test.head()
train.describe()
no_missing_col = [c for c in train.columns if train[c].isnull().sum() ==0]
missing_col = [c for c in train.columns if train[c].isnull().sum() >0]
print(f'Missing value in {len(missing_col)} columns and no missing value in {len(no_missing_col)} columns')

missing = train[missing_col].isnull().sum()
plt.figure(figsize=(14,6))
sns.barplot(x = missing.index, y = missing.values)
plt.xticks(rotation=90);
no_missing_col = [c for c in test.columns if test[c].isnull().sum() ==0]
missing_col = [c for c in test.columns if test[c].isnull().sum() >0]
print(f'Missing value in {len(missing_col)} columns and no missing value in {len(no_missing_col)} columns')

missing = test[missing_col].isnull().sum()
plt.figure(figsize=(14,6))
sns.barplot(x = missing.index, y = missing.values)
plt.xticks(rotation=90);
missing = train[missing_col].isnull()
plt.figure(figsize =(14,6))
sns.heatmap(missing, cbar=False, cmap='viridis')
def Numeric_plot(df,column = '', title='',ncols=2,trans_func = None):
    """ Histogram plot Box plot of Numeric variable"""
    
    # Box plot
    trace1 = go.Box(y = df[column],name='Box')
    
    # Histogram
    trace2 = go.Histogram(x = df[column], name = 'x')
    
    fig = tools.make_subplots(rows=1, cols=ncols)
    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    fig['layout'].update(height=300, title=title)
    fig['layout']['yaxis1'].update(title= column)

    # Histogram after transformation
    if trans_func != None:
        tmp = df[column].apply(trans_func)
        trace3 = go.Histogram(x = tmp, name = trans_func+'(x)')
        fig.append_trace(trace3, 1,3)
    
    py.iplot(fig)

def Categorical_plot(df, column ='', title = '',limit=10):
    """ Barplot: of categorical variable
        Boxplot: of categoriucal and taraget variable"""
    # Barplot
    bar = df[column].value_counts()[:limit]/df.shape[0]
    bar_round = [round(w,2) for w in bar.values *100]
    trace1 = go.Bar(x = bar.index, y = bar_round, name='% Count' )    
    # Boxplot
    box = df[column].isin(bar.index[:limit])
    box =df.loc[box][[column,'SalePrice']]
    trace2 = go.Box(x = box[column], y= box['SalePrice'],name='Sale Price')

    # Figure legend
    fig = tools.make_subplots(rows=1, cols=2,)#subplot_titles= ('',''))
    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    
    fig['layout']['yaxis1'].update(title='% Count')
    fig['layout']['yaxis2'].update(title='Sale Price')
    fig['layout'].update(height=400, title=title,showlegend=False)
    py.iplot(fig)
def Regression_plot(df,column=''):
    """Regression plot: with pearsonr correlation value """
    cor = round(df[['SalePrice',column]].corr().iloc[0,1], 3)
    sns.jointplot(x= df[column], y = df['SalePrice'], kind= 'reg',
                  label = 'r: '+str(cor),color='blue')
    plt.legend()
    #plt.title('Regression plot ')
drop_col = []
categorical_col = []
numeric_col = []
Numeric_plot(train, column='SalePrice',title='Sale Price',ncols=3,trans_func='log1p')
# Run this only once
map_value = {20: '1-STORY 1946 & NEWER ALL STYLES',
            30: '1-STORY 1945 & OLDER',
            40: '1-STORY W/FINISHED ATTIC ALL AGES',
            45: '1-1/2 STORY - UNFINISHED ALL AGES',
            50: '1-1/2 STORY FINISHED ALL AGES',
            60: '2-STORY 1946 & NEWER',
            70: '2-STORY 1945 & OLDER',
            75: '2-1/2 STORY ALL AGES',
            80: 'PLIT OR MULTI-LEVEL',
            85: 'SPLIT FOYER',
            90: 'DUPLEX - ALL STYLES AND AGES',
            120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
            150: '1-1/2 STORY PUD - ALL AGES',
            160: '2-STORY PUD - 1946 & NEWER',
            180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
            190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'}

train['MSSubClass'] = train['MSSubClass'].map(map_value)
test['MSSubClass'] = test['MSSubClass'].map(map_value)
Categorical_plot(train, column='MSSubClass', title='MSSubClass: The building class',limit=None)
# Add to list of categorical column
categorical_col.append('MSSubClass')
map_value = { 
            'A': 'Agriculture',
            'C': 'Commercial',
            'FV': 'Floating Village Residential',
            'I': 'Industrial',
            'RH': 'Residential High Density',
            'RL': 'Residential Low Density',
            'RP': 'Residential Low Density Park',
            'RM': 'Residential Medium Density',
            }
train['MSZoning'] = train['MSZoning'].map(map_value)
test['MSZoning'] = test['MSZoning'].map(map_value)
Categorical_plot(train, column= 'MSZoning', title ='MSZoning: Identifies the general zoning classification of the sale')
# Add to list of categorical column
categorical_col.append('MSZoning')
Numeric_plot(train, column= 'LotFrontage', ncols=3, trans_func='log', title='Linear feet of street connected to property')
Regression_plot(train, column='LotFrontage')
# Add to list of Numeric column list
numeric_col.append('LotFrontage')
Numeric_plot(train, column = 'LotArea',ncols=3, trans_func='log1p', title='Lot size in square feet')
Regression_plot(train, column='LotArea')
# Add to list of Numeric column list
numeric_col.append('LotArea')
Categorical_plot(train, column='Street', title= 'Street: Type of road access to property')
# Add to list of Drop column list
drop_col.append('Street')
Categorical_plot(train, column='Alley', title= 'Type of alley access to property')
# Add to list of categorical column list
drop_col.append('Alley')
Categorical_plot(train, column='LotShape', title= 'General shape of property')
# Add to list of categorical column list
categorical_col.append('LotShape')
Categorical_plot(train, column='LandContour', title= 'Flatness of the property')
# Add to list of categorical column list
categorical_col.append('LandContour')
Categorical_plot(train, column='Utilities', title= 'Type of utilities available')
# Add to list of Drop column list
drop_col.append('Utilities')
Categorical_plot(train, column='LotConfig', title= 'Lot configuration')
# Add to list of categorical column list
categorical_col.append('LotConfig')
Categorical_plot(train, column='LandSlope', title= 'Lot configuration')
# Add to list of categorical column list
categorical_col.append('LandSlope')
Categorical_plot(train, column='Neighborhood', title= 'Top 10 Lot configuration',limit=10)
# Add to list of categorical column list
categorical_col.append('Neighborhood')
Categorical_plot(train, column='Condition1', title= 'Proximity to various conditions',limit=None)
# Add to list of categorical column list
categorical_col.append('Condition1')
Categorical_plot(train, column='Condition2', title= 'Proximity to various conditions',limit=None)
# Add to list of categorical column list
categorical_col.append('Condition2')
Categorical_plot(train, column='BldgType', title= 'Type of dwelling',limit=None)
# Add to list of categorical column list
categorical_col.append('BldgType')
Categorical_plot(train, column='HouseStyle', title= 'Style of dwelling',limit=None)
# Add to list of categorical column list
categorical_col.append('HouseStyle')
map_values = {10: 'Very Excellent', 
             9: 'Excellent', 
             8: 'Very Good',
             7: 'Good',
             6: 'Above Average',
             5: 'Average',
             4: 'Below Average',
             3: 'Fair',
             2: 'Poor',
             1: 'Very Poor'
            }
train['OverallQual'] = train['OverallQual'].map(map_values)
test['OverallQual'] = test['OverallQual'].map(map_values)
Categorical_plot(train, column='OverallQual', title= 'Rates the overall material and finish of the house',limit=None)
# Add to list of categorical column list
categorical_col.append('OverallQual')
map_values = {10: 'Very Excellent', 
             9: 'Excellent', 
             8: 'Very Good',
             7: 'Good',
             6: 'Above Average',
             5: 'Average',
             4: 'Below Average',
             3: 'Fair',
             2: 'Poor',
             1: 'Very Poor'
            }
train['OverallCond'] = train['OverallCond'].map(map_values)
test['OverallCond'] = test['OverallCond'].map(map_values)
Categorical_plot(train, column='OverallCond', title= 'Rates the overall condition of the house',limit=None)
# Add to list of categorical column list
categorical_col.append('OverallCond')
Numeric_plot(train, column='YearBuilt', title= 'Original construction date', ncols=2,)# trans_func='sqrt')
Regression_plot(train, column='YearBuilt')
# Add to numeric column list
numeric_col.append('YearBuilt')
Numeric_plot(train, column='YearRemodAdd', title= 'Remodel date', ncols=2,)# trans_func='log')
Regression_plot(train, column='YearRemodAdd')
# Add to numeric column list
numeric_col.append('YearRemodAdd')
Categorical_plot(train, column='RoofStyle', title= 'Type of roof',limit=None)
# Add to list of categorical column list
categorical_col.append('RoofStyle')
Categorical_plot(train, column='RoofMatl', title= 'Roof material',limit=None)
# Add to list of drop column list
drop_col.append('RoofMatl')
Categorical_plot(train, column='Exterior1st', title= 'Exterior covering on house',limit=None)
# Add to list of categorical column list
categorical_col.append('Exterior1st')
Categorical_plot(train, column='Exterior2nd', title= 'Exterior covering on house',limit=None)
# Add to list of categorical column list
categorical_col.append('Exterior2nd')
Categorical_plot(train, column='MasVnrType', title= 'Masonry veneer type',limit=None)
# Add to list of categorical column list
categorical_col.append('MasVnrType')
Numeric_plot(train, column= 'MasVnrArea', title= 'Masonry veneer area in square feet',) #ncols=3, trans_func='sqrt')
Regression_plot(train, column='MasVnrArea')
# Add to list of numeric column list
numeric_col.append('MasVnrArea')
map_values = { 
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'Po': 'Poor'
            }
train['ExterQual'] = train['ExterQual'].map(map_values)
test['ExterQual'] = test['ExterQual'].map(map_values)
Categorical_plot(train, column='ExterQual', title= 'Evaluates the quality of the material on the exterior',limit=None)
# Add to list of categorical column list
categorical_col.append('ExterQual')
map_values = { 
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'Po': 'Poor'
            }
train['ExterCond'] = train['ExterCond'].map(map_values)
test['ExterCond'] = test['ExterCond'].map(map_values)
Categorical_plot(train, column='ExterCond', title= 'Evaluates the present condition of the material on the exterior',limit=None)
# Add to list of categorical column list
categorical_col.append('ExterCond')
Categorical_plot(train, column='Foundation', title= 'Type of foundation',limit=None)
# Add to list of categorical column list
categorical_col.append('Foundation')
Categorical_plot(train, column='BsmtQual', title= 'Evaluates the height of the basement',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtQual')
Categorical_plot(train, column='BsmtCond', title= 'Evaluates the general condition of the basement',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtCond')
Categorical_plot(train, column='BsmtExposure', title= 'Refers to walkout or garden level walls',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtExposure')
Categorical_plot(train, column='BsmtFinType1', title= 'Rating of basement finished area',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtFinType1')
Numeric_plot(train, column='BsmtFinSF1', title='Type 1 finished square feet')#,ncols=3, trans_func='log1p')
Regression_plot(train, column= 'BsmtFinSF1')
# Add to list of numeric column list
numeric_col.append('BsmtFinSF1')
Categorical_plot(train, column='BsmtFinType2', title= 'Rating of basement finished area',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtFinType2')
Numeric_plot(train, column='BsmtFinSF2', title='Type 2 finished square feet')#,ncols=3, trans_func='log1p')
Regression_plot(train, column= 'BsmtFinSF2')
# Add to list of numeric column list
numeric_col.append('BsmtFinSF2')
Numeric_plot(train, column='BsmtUnfSF', title='Unfinished square feet of basement area')#,ncols=3, trans_func='log1p')
Regression_plot(train, column= 'BsmtUnfSF')
# Add to list of numeric column list
numeric_col.append('BsmtUnfSF')
Numeric_plot(train, column='TotalBsmtSF', title='Total square feet of basement area')#,ncols=3, trans_func='log1p')
Regression_plot(train, column= 'TotalBsmtSF')
# Add to list of numeric column list
numeric_col.append('TotalBsmtSF')
Categorical_plot(train, column='Heating', title= 'Type of heating',limit=None)
# Add to list of dro column list
drop_col.append('Heating')
Categorical_plot(train, column='HeatingQC', title= 'Heating quality and condition',limit=None)
# Add to list of categorical column list
categorical_col.append('HeatingQC')
Categorical_plot(train, column='CentralAir', title= 'Central air conditioning',limit=None)
# Add to list of categorical column list
categorical_col.append('CentralAir')
Categorical_plot(train, column='Electrical', title= 'Electrical system',limit=None)
# Add to list of categorical column list
categorical_col.append('Electrical')
g = sns.pairplot(train, vars=['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'],
                palette = 'viridis', kind= 'reg', aspect=1.5)
# Add to
numeric_col.extend(['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'])
Categorical_plot(train, column='BsmtFullBath', title= 'Basement full bathrooms',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtFullBath')
Categorical_plot(train, column='BsmtHalfBath', title= 'Basement half bathrooms',limit=None)
# Add to list of categorical column list
categorical_col.append('BsmtHalfBath')
Categorical_plot(train, column='FullBath', title= ' Full bathrooms above grade',limit=None)
# Add to list of categorical column list
categorical_col.append('FullBath')
Categorical_plot(train, column='HalfBath', title= 'Half baths above grade',limit=None)
# Add to list of categorical column list
categorical_col.append('HalfBath')
Categorical_plot(train, column='BedroomAbvGr', title= 'Bedrooms above grade',limit=None)
# Add to list of categorical column list
categorical_col.append('BedroomAbvGr')
Categorical_plot(train, column='KitchenAbvGr', title= 'Kitchens above grade',limit=None)
# Add to list of categorical column list
categorical_col.append('KitchenAbvGr')
Categorical_plot(train, column='KitchenQual', title= 'Kitchen quality',limit=None)
# Add to list of categorical column list
categorical_col.append('KitchenQual')
Categorical_plot(train, column='TotRmsAbvGrd', title= 'Total rooms above grade',limit=None)
# Add to list of categorical column list
categorical_col.append('KitchenQual')
Categorical_plot(train, column='Functional', title= 'Home functionality',limit=None)
# Add to list of categorical column list
categorical_col.append('Functional')
Categorical_plot(train, column='Fireplaces', title= 'Number of fireplaces',limit=None)
# Add to list of categorical column list
categorical_col.append('Fireplaces')
Categorical_plot(train, column='FireplaceQu', title= 'Fireplace quality',limit=None)
# Add to list of categorical column list
categorical_col.append('FireplaceQu')
Categorical_plot(train, column='GarageType', title= 'Garage location',limit=None)
# Add to list of categorical column list
categorical_col.append('GarageType')
Numeric_plot(train, column='GarageYrBlt', title= 'Year garage was built')
# Add to list of Numeric column list
numeric_col.append('GarageYrBlt')
Categorical_plot(train, column='GarageFinish', title= 'Interior finish of the garage')
# Add to list of calegtory column list
categorical_col.append('GarageFinish')
Categorical_plot(train, column='GarageCars', title= 'Size of garage in car capacity')
# Add to list of calegtory column list
categorical_col.append('GarageCars')
Numeric_plot(train, column='GarageArea', title= 'Size of garage in square feet')
# Add to list of numeric column list
numeric_col.append('GarageArea')
Categorical_plot(train, column='GarageQual', title= 'Garage quality')
# Add to list of calegtory column list
categorical_col.append('GarageQual')
Categorical_plot(train, column='GarageCond', title= 'Garage condition')
# Add to list of calegtory column list
categorical_col.append('GarageCond')
Categorical_plot(train, column='PavedDrive', title= 'Paved driveway')
# Add to list of calegtory column list
categorical_col.append('PavedDrive')
g = sns.pairplot(data= train, kind= 'reg',
                 vars= ['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea'],)
# Add to
numeric_col.extend(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea'])
Categorical_plot(train, column='PoolQC', title= 'Pool quality')
# Add to list of calegtory column list
drop_col.append('PoolQC')
Categorical_plot(train, column='Fence', title= 'Fence quality')
# Add to list of calegtory column list
categorical_col.append('Fence')
Categorical_plot(train, column='MiscFeature', title= 'Miscellaneous feature not covered in other categories')
# Add to list of drop column list
drop_col.append('MiscFeature')
Numeric_plot(train, column='MiscVal', title='$Value of miscellaneous feature',)# ncols=3, trans_func='log1p')
# Add to numeric column list
numeric_col.append('MiscVal')
Categorical_plot(train, column='MoSold', title='Month Sold (MM)',)
# Add to categorical column list
categorical_col.append('MoSold')
Categorical_plot(train, column='YrSold', title='Year Sold',)
# Add to categorical column list
categorical_col.append('YrSold')
Categorical_plot(train, column='SaleType', title='Type of sale',)
# Add to categorical column list
categorical_col.append('SaleType')
Categorical_plot(train, column='SaleCondition', title='Condition of sale',)
# Add to categorical column list
categorical_col.append('SaleCondition')
# Check column 
print('Check number of column',train.shape, len(categorical_col)+len(drop_col)+len(numeric_col))
train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)
test['SalePrice'] = np.nan
train_test = pd.concat([train, test], axis =0)
train_test.shape
def Binary_encoding(df,columns):
    """Binary encoding"""
    print('*'*5,'Binary encoding','*'*5)
    lb = LabelBinarizer()
    print('Original shape:',df.shape)
    original_col = df.columns
    #columns = [i for i in columns if df[columns].nunique()>2]
    for i in columns:
        if df[i].nunique() >2:
            result = lb.fit_transform(df[i].fillna(df[i].mode()[0],axis=0))
            col = ['BIN_'+ str(i)+'_'+str(c) for c in lb.classes_]
            result1 = pd.DataFrame(result, columns=col)
            df = df.join(result1)
    print('After:',df.shape)
    #new_col = [c for c in df.columns if c not in original_col]
    return df
#train_test = Binary_encoding(train_test, categorical_col[1])
def OneHotEncoding(df, columns, nan_as_category=True, drop_first=True):
    """One Hot Encoding: of categorical variable"""
    print(10*'*'+'One Hot Encoding:',df.shape,10*'*')
    lenght = df.shape[0]
    # Concatenate dataframe
    #df = pd.concat([df1,df2], axis=0)
    
    # OHE
    df = pd.get_dummies(data = df, columns= columns, drop_first=drop_first, 
                        dummy_na=nan_as_category)
    
    print(10*'*','After One Hot Encoding:',df.shape,10*'*')
    return df
train_test = OneHotEncoding(train_test, columns=categorical_col)
def Fill_missing_value(df,column):
    """Fill missing value with Mean"""
    for c in column:
        if df[c].isnull().sum() >0:
            df[c] = df[c].fillna(df[c].mean())
    print('Check Missing value:',df.isnull().sum().sum())
    return df
train_test = Fill_missing_value(train_test,numeric_col)
def Descriptive_stat_feat(df,columns):
    """ Descriptive statistics feature
    genarating function: Mean,Median,Q1,Q3"""
    print('*'*5,'Descriptive statistics feature','*'*5)
    print('Before',df.shape)
    mean = df[columns].mean()
    median = df[columns].median()
    Q1 = np.percentile(df[columns], 25, axis=0)
    Q3 = np.percentile(df[columns], 75, axis=0)
    for i,j in enumerate(columns):
        df['mean_'+j] = (df[j] < mean[i]).astype('int8')
        df['median_'+j] = (df[j] > median[i]).astype('int8')
        df['Q1'+j] = (df[j] < Q1[i]).astype('int8')
        df['Q3'+j] = (df[j] > Q3[i]).astype('int8')
    print('After ',df.shape)
    return df
train_test = Descriptive_stat_feat(train_test, columns = numeric_col)
train_test.isnull().sum().sum()
length = train.shape[0]
test_id = test['Id']
train1 = train_test[:length]
test1 = train_test[length:]
X = train1.drop(['Id','SalePrice'], axis=1)
y = np.log1p(train1['SalePrice'])
new_test = test1.drop(['Id','SalePrice'], axis=1)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=seed)
del train1, test1
from sklearn.model_selection import RandomizedSearchCV
reg = Ridge(alpha= 1.0)
rsCV = RandomizedSearchCV(reg,cv= 5,param_distributions={'alpha':np.linspace(0,20,100)},random_state= seed)
rsCV.fit(X,y)
rsCV.best_params_
kf = KFold(n_splits=5, random_state=seed,)

final_pred = 0
rmse = []
r_square = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Modelling {i+1} of {kf.n_splits} fold')
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    # L2 - Regression
    reg = Ridge(alpha = rsCV.best_params_['alpha'])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_valid)
    final_pred += reg.predict(new_test)
    r2 = reg.score(X_valid, y_valid)
    r_square.append(r2)
    print('*'*10,'R sqaure:',round(r2,3), '*'*10,'\n')
    rmse.append(mean_squared_error(y_valid, y_pred)**0.5)
print(rmse,'\nRMSE:',np.mean(rmse))
f = plt.figure(figsize= (14,6))

ax = f.add_subplot(121)
ax.scatter(y_valid, y_pred)
plt.title('Scatter plot of y_true vs y_pred')

residual = y_valid - y_pred
ax = f.add_subplot(122)
sns.distplot(residual, ax = ax)
plt.axvline(residual.mean())
plt.title('Residual plot');
#pred = reg.predict(new_test)
pred = np.expm1(final_pred/ kf.n_splits)
submit = pd.DataFrame({'Id':test_id,'SalePrice':pred})
submit.to_csv('houseprice.csv',index= False)
print('Shape: ',submit.shape)
submit.head()
from sklearn.model_selection import RandomizedSearchCV

param = {
    'n_estimators':[200, 500, 1000,2000],
    'learning_rate': np.linspace(0.001, 1, 10),
    'max_depth': [3,5,7,8,10],
    'num_leaves': [32, 64, 128],
    'feature_fraction': np.linspace(0.7,1,5),
    'bagging_fraction': np.linspace(0.6,1,5),
    'lambda_l1': np.linspace(0,1,20),
    'lambda_l2': np.linspace(0,1,20),
}

lgb_reg = lgb.LGBMRegressor(eval_metric ='mse',)
rsCV = RandomizedSearchCV(lgb_reg,cv= 5,param_distributions= param,random_state= seed)
rsCV.fit(X,y)
rsCV.best_params_
# Lightgbm
def model(X_train, X_valid, y_train, y_valid,test_new,random_seed, param):
    
    lg_param = {}
    lg_param['learning_rate'] = param['learning_rate']
    lg_param['n_estimators'] = param['n_estimators']
    lg_param['max_depth'] = param['max_depth']
    #lg_param['num_leaves'] = param['num_leaves']
    lg_param['boosting_type'] = 'gbdt'
    lg_param['feature_fraction'] = param['feature_fraction']
    lg_param['bagging_fraction'] = param['bagging_fraction']
    lg_param['lambda_l1'] = param['lambda_l1']
    lg_param['lambda_l2'] = param['lambda_l2']
    lg_param['silent'] = -1
    lg_param['verbose'] = -1
    lg_param['nthread'] = 4
    lg_param['seed'] = random_seed
    
    lgb_model = lgb.LGBMRegressor(**lg_param)
    print('-'*10,'*'*20,'-'*10)
    lgb_model.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_valid,y_valid)], 
                 eval_metric ='mse', verbose =100, early_stopping_rounds=50)
    y_pred = lgb_model.predict(X_valid)
    y_pred_new = lgb_model.predict(test_new)
    return y_pred,y_pred_new,lgb_model
kf = KFold(n_splits=5, random_state=seed,)

final_pred = 0
rmse = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Modelling {i+1} of {kf.n_splits} fold')
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    # GBM Regression
    print('\n{} fold of {} KFold'.format(i+1,kf.n_splits))
    y_pred,y_pred_new,lgb_model = model(X_train, X_valid, y_train, y_valid,new_test,random_seed = i,
                                    param = rsCV.best_params_)
    final_pred += y_pred_new
    rmse.append(mean_squared_error(y_valid, y_pred)**0.5)
    #print('*'*10,'Rmse:',round(r2,3), '*'*10,'\n')
print(rmse,'\nRMSE:',np.mean(rmse))
lgb.plot_importance(lgb_model,max_num_features=20)
f = plt.figure(figsize= (14,6))

ax = f.add_subplot(121)
ax.scatter(y_valid, y_pred)
plt.title('Scatter plot of y_true vs y_pred')

residual = y_valid - y_pred
ax = f.add_subplot(122)
sns.distplot(residual, ax = ax)
plt.axvline(residual.mean())
plt.title('Residual plot');
#pred = reg.predict(new_test)
pred = np.expm1(final_pred/ kf.n_splits)
submit = pd.DataFrame({'Id':test_id,'SalePrice':pred})
submit.to_csv('houseprice_lgb.csv',index= False)
print('Shape: ',submit.shape)
submit.head()