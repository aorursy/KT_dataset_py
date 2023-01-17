# Importing libraries we will use.

import warnings

import numpy as np

import pandas as pd



import scipy.stats as stat

from sklearn.linear_model import RidgeCV

from sklearn.model_selection import train_test_split



import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.stats.outliers_influence import variance_inflation_factor

import scipy.stats as sci



from statsmodels.graphics.mosaicplot import mosaic

from statsmodels.graphics.gofplots import ProbPlot

import matplotlib.pyplot as plt

import matplotlib.ticker as tck

import seaborn as sns



# Limiting floats output to 2 decimal points and adding the thousands operator that places a comma between all thousands

pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))



# Ignoring warnings

warnings.filterwarnings('ignore')



%matplotlib inline

# set style and color palette we will use for Seaborn visualization package

sns.set(style="ticks", palette="bright")

from IPython.display import Image



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
labelled =  pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #labelled data is the provided training data.

unlabelled = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #unlabelled data is the test data.
#Concate both sets for the sake of data exploration and wrangling

data = pd.concat([labelled, unlabelled], axis = 0, ignore_index = True)
data.shape
categorical = data.dtypes[data.dtypes == 'object'].index.tolist()

numerical = data.dtypes[data.dtypes != 'object'].index.drop(['Id','SalePrice']).tolist()
print('Categorical variables are:\n\n', categorical,'\n\nNumerical variables are:\n\n',numerical)
print('No. of categorical variables:', len(categorical), ' ,  No. of numerical variables:', len(numerical))
data[categorical].describe().transpose()
data[numerical].describe().transpose()
data_org = data.copy()
#Function for formating y-label to $(amount)k or $(amount)M

def currency(x,pos):

    if x >= 10**6 :

        return '${:1.2f}M'.format(x*1e-6)

    return '${:1.0f}k'.format(x*1e-3)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 

                                    figsize=(10,7), gridspec_kw={"height_ratios": (.15, .85)})

plt.suptitle('Distribution plot for "SalePrice"', fontsize = 20)

formater = tck.FuncFormatter(currency)

ax_hist.xaxis.set_major_formatter(formater)

sns.boxplot(data['SalePrice'].dropna(), ax= ax_box)

sns.distplot(data['SalePrice'].dropna(), ax= ax_hist)

ax_box.set(xlabel='')
print('"SalePrice" kurosis = ', np.round(data['SalePrice'].kurtosis(),2),

     '\n"SalePrice" skewness = ', np.round(data['SalePrice'].skew(),2)), 

data['SalePrice'].describe()
#List of varaibles having 'int64' type and we want to  make distplots for them rather than countplots.

int_as_float = ['LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 

        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

        'EnclosedPorch', '3SsnPorch', 'ScreenPorch','PoolArea', 'MiscVal',"YearBuilt", "YearRemodAdd"]
allcolumns = np.array(data.columns.drop(['Id','SalePrice']).tolist()+[None]).reshape(20, 4)

fig, ax = plt.subplots(figsize=(15, 70), ncols=4, nrows=20)

fig.suptitle('All Features Distribution plots', fontsize=20)

plt.subplots_adjust(

    left=0.125,

    right=0.9,

    bottom=0.1,

    top=0.97,

    wspace=0.275,

    hspace= 0.8)



prob_formater = tck.FuncFormatter(

    lambda x, p: np.format_float_scientific(x, precision=0, exp_digits=0))



for i in range(20):

    for j in range(4):

        if i == 19 and j == 3:

            continue

        ax[i][j].set_title(allcolumns[i][j], fontsize=14)

        for tick in ax[i][j].get_xticklabels():

            if data[allcolumns[i][j]].nunique() >= 5 and data[allcolumns[i][j]].dtype == 'object':

                tick.set_rotation(90)

        if (data[allcolumns[i][j]].dtypes == 'float64') | (allcolumns[i][j] in int_as_float):

            ax[i][j].yaxis.set_major_formatter(prob_formater)

            sns.distplot(data[allcolumns[i][j]].dropna(), ax=ax[i, j], kde=True)

        else:

            sns.countplot(data[allcolumns[i][j]].fillna('NA'), ax=ax[i, j])
fig, ax = plt.subplots(figsize=(15, 70), ncols=4, nrows=20)

fig.suptitle('All Features Distribution plots', fontsize=20)

plt.subplots_adjust(

    left=0.125,

    right=0.9,

    bottom=0.1,

    top=0.97,

    wspace=0.275,

    hspace= 0.8)



prob_formater = tck.FuncFormatter(

    lambda x, p: np.format_float_scientific(x, precision=0, exp_digits=0))



for i in range(20):

    for j in range(4):

        if i == 19 and j == 3:

            continue

        ax[i][j].set_title(allcolumns[i][j], fontsize=14)

        for tick in ax[i][j].get_xticklabels():

            if data[allcolumns[i][j]].nunique() >= 5 and data[allcolumns[i][j]].dtype == 'object':

                tick.set_rotation(90)

        if (data[allcolumns[i][j]].dtypes == 'float64') | (allcolumns[i][j] in int_as_float):

            ax[i][j].yaxis.set_major_formatter(prob_formater)

            sns.distplot(data.loc[(data[allcolumns[i][j]].isna() == False) & (data[allcolumns[i][j]] != 0),

                                      allcolumns[i][j]], ax=ax[i, j], kde=True)

        else:

            sns.countplot(data[allcolumns[i][j]].fillna('NA'), ax=ax[i, j])
data.loc[data.YearBuilt == data.YearRemodAdd, 'YearRemodAdd'].count()
data.loc[data.YearBuilt == data.YearRemodAdd, 'YearRemodAdd'] = np.nan

sns.distplot(data['YearRemodAdd'].dropna(), kde= True)
#Plotting the two distributions on the same axis.

sns.set(rc={'figure.figsize':(8,5)})

sns.distplot( data['YearBuilt'].dropna() , color="red", label="Year Built",  kde= False, )

sns.distplot( data['YearRemodAdd'].dropna() , color="blue", label="Year Remodelled", kde= False)

plt.legend()
#The houses remodeled before 1960

data.loc[data.YearRemodAdd < 1960]['YearRemodAdd'].value_counts()
#Calculating the difference between 1950 and 'YearBuilt'

data['difference'] = data.loc[data.YearRemodAdd == 1950, 'YearBuilt'].apply(lambda x: 1950 - x)
# Houses whose remodeling date will be changed

# We want to keep record of the original data in the sack of checking our work later

YearRem_bfr_change = data.loc[data.YearRemodAdd== 1950, [

    'Id', 'YearRemodAdd', 'YearBuilt']].rename(columns = {'YearRemodAdd':'YearRemodAdd_bfr'})



FrstOpt_YearRemodAdd = pd.DataFrame(data.YearRemodAdd)



# Assigning estimated 'YearRemodAdd' to houses with YearRemodAdd == 1950 and keep the new values in a separate df

FrstOpt_YearRemodAdd.loc[data.YearRemodAdd== 1950,'YearRemodAdd'] = data.loc[data.YearRemodAdd== 1950, [

    'YearBuilt', 'difference']].apply(lambda x: np.ceil(x[0] + 0.5*x[1]), axis=1)



#A comparison between the "YearBuilt" and modified "YearRemodAdd" to check if our constraints aren't violated

pd.concat([YearRem_bfr_change, FrstOpt_YearRemodAdd.loc[YearRem_bfr_change.index]],

          axis=1).rename(columns={'YearRemodAdd': 'YearRemodAdd_aftr'})
SecndOpt_YearRemodAdd = pd.DataFrame(data.YearRemodAdd)

SecndOpt_YearRemodAdd.loc[data.YearRemodAdd == 1950, 'YearRemodAdd'] = np.nan
fig, ax = plt.subplots(ncols=2, nrows=1, figsize= (15,5.5))

plt.subplots_adjust(top = 0.825)



fig.suptitle('Comparison between the two options', fontsize=20)

ax[0].set_title('1st Option: Estimate YearRemodelled', fontsize = 14)

ax[1].set_title('2nd Option: Replace YearRemodelled = 1950 by NA', fontsize = 14)



sns.distplot( data['YearBuilt'].dropna() , color="red", label="Year Built",  kde= False, ax= ax[0])

sns.distplot(FrstOpt_YearRemodAdd.YearRemodAdd.dropna(), color="blue", label="Year Remodelled",kde= False, ax=ax[0])

ax[0].legend()



sns.distplot( data['YearBuilt'].dropna() , color="red", label="Year Built",  kde= False, ax= ax[1] )

sns.distplot(SecndOpt_YearRemodAdd.YearRemodAdd.dropna(), color="blue", label="Year Remodelled", kde= False, ax= ax[1])

ax[1].legend()

data.loc[data.YearRemodAdd== 1950,'YearRemodAdd'] = FrstOpt_YearRemodAdd

#Drop the column 'difference' we created earlier

data.drop(columns=['difference'], inplace= True)
data.duplicated(subset= data.columns.drop('Id')).sum()
Image("../input/pic123/Filling missing values flowchart.jpg")
# Identify the number of NAs in each feature and select only those having NAs

total_NA = data.drop(columns='SalePrice').isnull().sum()[data.isnull().sum() != 0]



# Calculate the percentage of NA in each feature

percent_NA = data.drop(columns='SalePrice').isnull().sum()[data.isnull().sum() != 0]/data.shape[0]



# Summarize our findings in a dataframe

missing = pd.concat([total_NA, percent_NA], axis=1, keys=['Total NAs', 'Percentage']).sort_values('Total NAs', ascending=False)

missing
# Filling NAs for categorical features in group A.

data.loc[(data.Alley.isnull()) | (data.Fence.isnull()),

         ['Alley', 'Fence']] = 'Not exist'



data.loc[data.Electrical.isnull(), 'Electrical'] = data.loc[data.CentralAir ==

                                                            data.loc[data.Electrical.isnull(), 'CentralAir'].values[0], 'Electrical'].describe()['top']



data.loc[data.MSZoning.isnull(), 'MSZoning'] = data.loc[(data.Condition1 == data.loc[data.MSZoning.isnull(), 'Condition1'].values[0]) & (

    data.Condition2 == data.loc[data.MSZoning.isnull(), 'Condition2'].values[0]), 'MSZoning'].describe()['top']



data.loc[data.Utilities.isnull(), 'Utilities'] = data.loc[data.Functional ==

                                                          data.loc[data.Utilities.isnull(), 'Functional'].values[0], 'Utilities'].describe()['top']



data.loc[data.Functional.isnull(), 'Functional'] = data.loc[data.OverallCond ==

                                                            data.loc[data.Functional.isnull(), 'OverallCond'].values[0], 'Functional'].describe()['top']
sns.set(rc={'figure.figsize': (3, 9)})

sns.heatmap(data=data.corr()[['LotFrontage']].sort_values('LotFrontage', ascending=False)[

            1:], linewidths=0.025, annot= True).set_title('Correlation Coefficient\n vs. LotFrontage')
cnddt_vars = ['LotShape', 'GrLivArea', 'BldgType', 'LotArea', 'LotConfig', 'HouseStyle']

temp_vars = data[cnddt_vars].dropna(axis = 0)

temp_resp = data.iloc[temp_vars.index][['LotFrontage']]

subset = pd.concat([temp_vars, temp_resp], axis = 1)
fig, ax = plt.subplots(figsize=(12, 15), ncols=2, nrows=4)

fig.suptitle('Distribution plots for features and response variable', fontsize=20)

plt.subplots_adjust(

    left=0.125,

    right=0.9,

    bottom=0.1,

    top=0.93,

    wspace=0.275,

    hspace= 0.35)



allcolumns = np.array(subset.columns.tolist()+['']).reshape(4,2)

for i in range(4):

    for j in range(2):

        if i == 3 and j == 1:

            continue

        ax[i][j].set_title(allcolumns[i][j], fontsize=14)

        if (data[allcolumns[i][j]].dtypes == 'float64') | (allcolumns[i][j] in int_as_float):

            sns.distplot(data.loc[(data[allcolumns[i][j]].isna() == False) & (data[allcolumns[i][j]] != 0),

                                      allcolumns[i][j]], ax=ax[i, j], kde=True)

        else:

            sns.countplot(data[allcolumns[i][j]].fillna('NA'), ax=ax[i, j])
sns.pairplot(data= subset[['LotArea', 'GrLivArea', 'LotFrontage']])
subset.loc[((subset.LotFrontage > 300) | (subset.LotArea > 100000))]
sns.set(rc={'figure.figsize': (4,4)})

sns.heatmap(subset.corr(), annot= True)
fig, ax = plt.subplots(figsize=(15, 25), ncols=2, nrows=4)

fig.suptitle('Bivariate relationships: Categorical variable vs. Numeric variable', fontsize=20)

plt.subplots_adjust(

    left=0.125,

    right=0.9,

    bottom=0.1,

    top=0.95,

    wspace=0.2,

    hspace= 0.25)



sns.boxplot(data= subset, x= 'BldgType', y= 'LotFrontage', ax= ax[0][0])

sns.boxplot(data= subset, x= 'LotShape', y= 'LotFrontage', ax= ax[0][1])

sns.boxplot(data= subset, x= 'LotConfig', y= 'LotFrontage', ax= ax[1][0])

sns.boxplot(data= subset, x= 'HouseStyle', y= 'LotFrontage', ax= ax[1][1])



sns.boxplot(data= subset, x= 'BldgType', y= 'LotArea', ax= ax[2][0])

sns.boxplot(data= subset, x= 'LotShape', y= 'LotArea', ax= ax[2][1])

sns.boxplot(data= subset, x= 'LotConfig', y= 'LotArea', ax= ax[3][0])

sns.boxplot(data= subset, x= 'HouseStyle', y= 'LotArea', ax= ax[3][1])
fig, ax = plt.subplots(figsize=(10, 40), ncols=1, nrows=6)

fig.suptitle('Bivariate relationships: Categorical variable vs. Categorical variable', fontsize=20)

plt.subplots_adjust(

    left=0.125,

    right=0.9,

    bottom=0.1,

    top=0.96,

    wspace=0.275,

    hspace= 0.2)



#Find all possible combination of categorical features without repeation

ctg = subset.dtypes[subset.dtypes == 'object'].index.tolist()

x = []

for i in range(len(ctg)-1):

    j = i +1

    while j != len(ctg):

        x.append([ctg[i],ctg[j]])

        j += 1



for i in range(len(x)):

    title = '{0} vs. {1}'.format(x[i][0],x[i][1])

    mosaic(data=subset, index=[x[i][0], x[i][1]], axes_label=True,

       statistic=True, gap=0.005, title= title, ax= ax[i])

        

plt.show()
missing_LF = subset.loc[subset.LotFrontage.isnull()].drop(columns=['LotFrontage']) #Rows having LotFrontage to be imputed.

have_LF = subset.loc[~subset.LotFrontage.isnull()].reset_index()
lm1 = ols('LotFrontage ~ GrLivArea + LotArea + LotShape + BldgType +  LotConfig + HouseStyle',

         data=have_LF).fit()

lm1.summary()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,9))

fig.suptitle('Diagnostic Plots', fontsize=20)



ax[0][1].axhline(np.mean(lm1.resid), color='red', ls='--', lw = 0.75, label='Residuals Avg.')

ax[1][1].axhline(0, color='black', ls='--', lw = 0.75)



ax[0][0].set_title('Normal Probability Plot')

probplot = ProbPlot(lm1.resid, fit= True)

probplot.ppplot(line= 'r', ax= ax[0][0])



sns.scatterplot(x= lm1.fittedvalues, y= lm1.resid, ax= ax[0][1]).set_title('Residuals vs. Fitted values')

sns.distplot(lm1.resid, ax= ax[1][0])

sns.scatterplot(x= have_LF.index, y= lm1.resid, ax= ax[1][1]).set_title('Residuals vs. Observation order')

plt.show()
#The data point that could be influential

have_LF.iloc[[lm1.resid.sort_values().head(1).index[0]]]
k = have_LF.shape[1] - 1 #No. of predictors

n = have_LF.shape[0] #No. of observations

diffits_ref = 3*np.sqrt((k+2)/(n-k-2)) #Reference value of DIFFITS

influence = lm1.get_influence()

print(diffits_ref)
influence.summary_frame().loc[(influence.summary_frame(

).cooks_d > 0.5), influence.summary_frame().columns[-6:]]
have_LF.drop([260], inplace= True)
lm2 = ols('np.log(LotFrontage) ~ np.log(GrLivArea) + np.log(LotArea) + LotShape + BldgType +  LotConfig + HouseStyle',

         data=have_LF).fit()

lm2.summary()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

fig.suptitle('Diagnostic Plots after omitting influential data points', fontsize=20)



ax[0][1].axhline(0, color='black', ls='--', lw = 0.75)

ax[1][1].axhline(0, color='black', ls='--', lw = 0.75)



ax[0][0].set_title('Normal Probability Plot')

probplot = ProbPlot(lm2.resid, fit= True)

probplot.ppplot(line= 'r', ax= ax[0][0])



sns.scatterplot(x= lm2.fittedvalues, y= lm2.resid, ax= ax[0][1]).set_title('Residuals vs. Fitted values')

sns.distplot(lm2.resid, ax= ax[1][0])

sns.scatterplot(x= have_LF.index, y= lm2.resid, ax= ax[1][1]).set_title('Residuals vs. Observation order')

probplot = ProbPlot(lm2.resid, fit= True)
subset.loc[subset.LotFrontage.isnull(), 'LotFrontage'] = np.exp(lm2.predict(missing_LF))

data['LotFrontage'] = subset.LotFrontage
# Find rows with consistent NAs in all features describing the basement

data.loc[(data.BsmtQual.isnull() & data.BsmtCond.isnull() & data.BsmtExposure.isnull() & data.BsmtFinType1.isnull() & data.BsmtFinType2.isnull()) & (data.BsmtFinSF1 == 0) & (data.BsmtFinSF2 == 0) & (data.BsmtUnfSF == 0) & (data.TotalBsmtSF == 0) & (data.BsmtFullBath == 0) & (data.BsmtHalfBath == 0),

         ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].shape[0]
idx = data.loc[(data.BsmtQual.isnull() & data.BsmtCond.isnull() & data.BsmtExposure.isnull() & data.BsmtFinType1.isnull() & data.BsmtFinType2.isnull()) & (data.BsmtFinSF1 == 0) & (data.BsmtFinSF2 == 0) & (data.BsmtUnfSF == 0) & (data.TotalBsmtSF == 0) & (data.BsmtFullBath == 0) & (data.BsmtHalfBath == 0),

         ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].index
data.loc[idx, ['BsmtQual', 'BsmtCond', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinType2']] = 'Not exist'



data.loc[idx, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
# Find rows with non-consistent NAs across all features describing the basement.

data.loc[data.BsmtQual.isnull() | data.BsmtCond.isnull() | data.BsmtExposure.isnull() | data.BsmtFinType1.isnull() | data.BsmtFinType2.isnull() | (data.BsmtFinSF1.isnull()) | (data.BsmtFinSF2.isnull()) | (data.BsmtUnfSF.isnull()) | (data.TotalBsmtSF.isnull()) | (data.BsmtFullBath.isnull()) | (data.BsmtHalfBath.isnull()),

             ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].reset_index()
fig, ax = plt.subplots(ncols=2, nrows=1, figsize= (15,5))



ax[0].set_title('Basement Finish Type 2 by Basement Overall Condition', fontsize= 16)

ax[1].set_title('Basement Finish Type 2 by Basement Exposure', fontsize= 16)



sns.countplot(x='BsmtCond', hue='BsmtFinType2', data=data, ax= ax[0])

sns.countplot(x='BsmtExposure', hue='BsmtFinType2', data=data, ax= ax[1])



ax[0].legend(loc = [0.75,0.55])

ax[1].legend(loc = [0.75,0.55])
data.BsmtFinType2.iloc[[332]] = 'Rec'
data.iloc[[948, 1487, 2348]][['LandContour','LandSlope','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

              'BsmtFinType2','TotalBsmtSF']]
fig, ax = plt.subplots(ncols=2, nrows=1, figsize= (15,5))



ax[0].set_title('Basement Exposure by Land Contour', fontsize= 16)

ax[1].set_title('Basement Exposure by Land Slope', fontsize= 16)



sns.countplot(x='LandContour', hue='BsmtExposure', data=data.fillna('NA'), ax= ax[0])

sns.countplot(x='LandSlope', hue='BsmtExposure', data=data.fillna('NA'), ax= ax[1])



ax[0].legend(loc = 'best')

ax[1].legend(loc = (0.77, 0.625))
data.BsmtExposure.iloc[[948, 1487, 2348]] = 'No'
for i in [2040, 2185, 2524]:

    data.BsmtCond.iloc[i] = data.loc[(data.BsmtQual == data.BsmtQual.iloc[i]) & (

        data.BsmtFinType1 == data.BsmtFinType1.iloc[i]), 'BsmtCond'].describe()['top']
for i in [2217, 2218]:

    data.BsmtQual.iloc[i] = data.loc[(

        data.BsmtCond == data.BsmtCond.iloc[i]), 'BsmtQual'].describe()['top']
data.loc[2120 , ['BsmtQual', 'BsmtCond', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinType2']] = 'Not exist'



data.loc[2120 , ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
data.loc[2188 , ['BsmtQual', 'BsmtCond', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinType2']] = 'Not exist'



data.loc[2188 , ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
data.loc[data.FireplaceQu.isnull(), ['Fireplaces']].sum()
data.FireplaceQu.fillna('Not exist', inplace= True)
data.loc[data.GarageType.isnull() & data.GarageYrBlt.isnull() & data.GarageFinish.isnull() & data.GarageQual.isnull() & data.GarageCond.isnull() & (data.GarageArea == 0) & (data.GarageCars == 0),

         ['GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'GarageQual',

                                                                'GarageCond', 'GarageCars']].shape[0]
data.loc[data.GarageType.isnull() & data.GarageFinish.isnull() & data.GarageQual.isnull() & data.GarageCond.isnull() & (data.GarageArea == 0) & (data.GarageCars == 0),

         ['GarageType', 'GarageArea', 'GarageFinish', 'GarageQual',

                                                    'GarageCond']] = 'Not exist'
data.loc[data.GarageType.isnull() | data.GarageFinish.isnull() | data.GarageQual.isnull() | data.GarageCond.isnull() | data.GarageCars.isnull() | data.GarageArea.isnull(),

         ['GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'GarageQual',

                                                    'GarageCond', 'GarageCars']]
# Imputing GarageYrBlt, GarageFinish, GarageQual, and GarageCond for row no. 2126

data.at[2126,'GarageYrBlt'] = data.iloc[2126]['YearBuilt'] + data.loc[(data.GarageType == 'Detchd') & (data.Neighborhood == 'OldTown'), [

    'GarageYrBlt', 'YearBuilt']].apply(lambda x: x[0] - x[1], axis=1).mean()



data.at[2126, 'GarageFinish'] = data.loc[(data.GarageType == 'Detchd') & (

    data.GarageCars == 1)].GarageFinish.describe()['top']



data.at[2126, 'GarageQual'] = data.loc[(data.GarageType == 'Detchd') & (

    data.GarageCars == 1)].GarageQual.describe()['top']



data.at[2126, 'GarageCond'] = data.loc[(data.GarageType == 'Detchd') & (

    data.GarageCars == 1)].GarageCond.describe()['top']
data.loc[2576, ['GarageType','GarageFinish','GarageQual','GarageCond']] = 'Not exist'



data.loc[2576, ['GarageArea', 'GarageCars']] = 0
data.loc[data.PoolQC.isnull(), ['PoolArea']].sum()
data.loc[(data.PoolQC.isnull()) & (data.PoolArea != 0), ['PoolQC', 'PoolArea']]
data.loc[[2420,2503,2599],'PoolQC'] = data.PoolQC.describe()['top']
data.loc[(data.PoolQC.isnull()), 'PoolQC'] = 'Not exist'
data.loc[data.MiscFeature.isnull(), ['MiscVal']].sum()
data.loc[(data.MiscFeature.isnull()) & (data.MiscVal != 0), ['MiscFeature', 'MiscVal']]
data.loc[2549, 'MiscFeature'] = 'Othr'
data.MiscFeature.fillna('Not exist', inplace= True)
indx = data.loc[data.MasVnrArea.isnull() & data.MasVnrType.isnull()].index

data.loc[indx, 'MasVnrType'] = 'None'

data.loc[indx, 'MasVnrArea'] = 0
data.loc[data.MasVnrType.isnull(), 'MasVnrType'] = data.MasVnrType.value_counts().index[1]
data.loc[data.Exterior1st.isnull(), ['Exterior1st', 'Exterior2nd']] = 'Other'
data.loc[data.KitchenQual.isnull(), 'KitchenQual'] = data.loc[data.OverallQual == data.loc[data.KitchenQual.isnull(), 'OverallQual'].values[0],

         'KitchenQual'].describe()['top']
data.loc[data.SaleType.isnull(), 'SaleType'] = data.loc[data.SaleCondition == data.loc[data.SaleType.isnull(), 'SaleCondition'].values[0],

         'SaleType'].describe()['top']
data.drop(columns='SalePrice').isnull().sum()[data.isnull().sum() != 0]
data.groupby(['YrSold'])[['SalePrice']].describe()
sns.distplot(data.SalePrice.dropna())
sns.scatterplot(data= data.dropna(), x= 'GrLivArea', y='SalePrice')
data.drop(index= data.loc[(data.GrLivArea > 4000)&(data.SalePrice.notnull())].index, inplace= True)
data.to_csv('modified_data.csv', index= True)