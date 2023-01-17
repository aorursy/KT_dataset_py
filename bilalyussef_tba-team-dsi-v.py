import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.style as style

import matplotlib.gridspec as gridspec

from scipy import stats

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import cross_val_score, train_test_split,KFold, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.impute import SimpleImputer

import warnings



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

style.use('fivethirtyeight')

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',300)

pd.set_option('display.max_columns',300)
hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

hp_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
hp.head()
hp_test.head()
hp.shape,hp_test.shape
hp.info()
hp_test.info()
def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    ## the two following line may seem complicated but its actually very simple. 

    if df.isnull().sum().sum() != 0:

        total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

        percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

        return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

    else:

        print (f'Congrats, No null values in your dataframe')
missing_percentage(hp)
missing_percentage(hp_test)
f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

f.set_figheight(7)

f.set_figwidth(16)

sns.heatmap(hp.isnull(),ax=ax1,cbar=False, yticklabels=False,cmap='viridis')

sns.heatmap(hp_test.isnull(),ax=ax2,cbar=False, yticklabels=False,cmap='viridis')

ax1.set_title('Train Data')

ax2.set_title('Test Data')
nnan_col = ["BsmtQual",'BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure','GarageQual','GarageFinish','GarageType',

            'GarageCond','FireplaceQu','Fence','Alley','MiscFeature','PoolQC']
def fill_fun(df,columns_list):

    for col in columns_list:

        df[col].fillna(value='NA',inplace = True)
fill_fun(hp,nnan_col)

fill_fun(hp_test,nnan_col)
def fill_missing(df):

    for col in df.columns:

        if df[col].dtypes == 'O':

            df[col].fillna(value=df[col].mode(dropna=True)[0],inplace=True)

        else:

            df[col].fillna(value=df[col].median(),inplace=True)

fill_missing(hp)

fill_missing(hp_test)
hp.GarageYrBlt = hp.GarageYrBlt.fillna(value=0.0)

hp_test.GarageYrBlt = hp_test.GarageYrBlt.fillna(value=0.0)
missing_percentage(hp)
missing_percentage(hp_test)
# c_columns = ['MSSubClass', 'OverallQual' , 'OverallCond' ]
# def change_columns(df):

#     column_wrong_type = c_columns

#     for col in column_wrong_type:

#         df[col]=df[col].astype(str)
# change_columns(hp)

# change_columns(hp_test)
def plotting_3_chart(df, feature):



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    
plotting_3_chart(hp, 'SalePrice')
stats.kurtosis(hp.SalePrice)
## Plot fig sizing. 

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(hp.drop(columns=['Id']).corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(hp.drop(columns=['Id']).corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, cbar=False);

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
feat_corr = abs(hp.corr().SalePrice).sort_values(ascending=False)[1:]

feat_corr
def outliers_nan(df):

    dff = df.drop(columns=['Id','SalePrice'])

    for col in dff.columns:

        if dff[col].dtypes != 'O':

            IQR = np.percentile(dff[col],75) - np.percentile(dff[col],25)

            upper_limit = np.percentile(dff[[col]],75)+(3*IQR)

            lower_limit = np.percentile(dff[[col]],25)-(3*IQR)

            df[col] = dff[col].apply(lambda x: np.nan if x > upper_limit or x < lower_limit else x) 

outliers_nan(hp)
hp.isnull().sum().sort_values()
hp_combined = pd.concat([hp,hp_test],join='inner')
y = hp.SalePrice
hp_combined = hp_combined[hp.columns[:-1]]
hp_combined.head()
hp_combined.info()
hp_combined_dum = pd.get_dummies(hp_combined, drop_first=True)

hp_combined_dum.shape
X_train = hp_combined_dum.iloc[0:hp.shape[0],:]

X_test =  hp_combined_dum.iloc[hp.shape[0]:,:]

y= hp.SalePrice

X_train['SalePrice']=y
L = []

for col in X_train.columns:

    try:

        if (abs(X_trian.corr().SalePrice[col])>0.5):

            L.append(col)

    except:

        L.append(col)

        

        

L.remove('Id')

c=L
X_train = X_train[L].dropna()

L.remove('SalePrice')
X_test = X_test[L]

y = X_train[['SalePrice']]

X_train.drop(columns='SalePrice', inplace=True)
X_train.shape,X_test.shape,y.shape
ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)

X_test_ss = ss.fit_transform(X_test)
lr_model = LinearRegression()

lr_model.fit(X_train_ss,y)
cross_val_score(lr_model,X_train_ss,y).mean()
ls_model = Lasso(alpha=5)

cross_val_score(ls_model.fit(X_train_ss,y),X_train_ss,y).mean()
lscv_model = LassoCV()
cross_val_score(lscv_model.fit(X_train_ss,y),X_train_ss,y).mean()
rg_model = Ridge(alpha=5)
cross_val_score(rg_model.fit(X_train_ss,y),X_train_ss,y).mean()
rgcv_model = RidgeCV(alphas=np.arange(0.1,10,0.1))
cross_val_score(rgcv_model.fit(X_train_ss,y),X_train_ss,y).mean()
encv_model = ElasticNetCV()
cross_val_score(encv_model.fit(X_train_ss,y),X_train_ss,y).mean()
rf_model = RandomForestRegressor(max_depth=15, random_state=101)
cross_val_score(rf_model.fit(X_train_ss,y),X_train_ss,y).mean()
grrf = GridSearchCV(rf_model, param_grid={'n_estimators':np.arange(1,50,1),'max_depth':np.arange(1,50,1)},

                   n_jobs=-1,verbose=1)
# grrf.fit(X_train_ss,y)


# cross_val_score(grrf,X_train_ss,y,cv=5).mean()
knn_model= KNeighborsRegressor(n_neighbors=5)
cross_val_score(knn_model.fit(X_train_ss,y),X_train_ss,y).mean()
from sklearn.svm import SVR
svm_model=SVR()
cross_val_score(svm_model.fit(X_train_ss,y),X_train_ss,y).mean()
submit = pd.DataFrame(columns=['Id','SalePrice'])
submit.Id = hp_test.Id

submit.SalePrice =( rgcv_model.predict(X_test_ss))

submit.head()
submit.to_csv('submission_rgcv_Final_t.csv',index=False)
pd.read_csv('submission_rgcv_Final_t.csv')