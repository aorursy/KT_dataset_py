import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (9, 6)

infolder = '../input/'

df_train_raw = pd.read_csv(infolder + 'train.csv')
df_test_raw = pd.read_csv(infolder +  'test.csv')

print('Training data: {s[0]} rows, {s[1]} columns'.format(s=df_train_raw.shape))
print('Testing data: {s[0]} rows, {s[1]} columns'.format(s=df_test_raw.shape))
df_train_raw.head()
plt.scatter(df_train_raw['GrLivArea'], df_train_raw['SalePrice'],
            alpha=0.3, edgecolor='none');
plt.axvline(4000, ls='--', c='r');
plt.title('Outlier check: training data contains {} outlier(s)\n'
          'test data contains {} outlier(s)'
          .format(sum(df_train_raw['GrLivArea'] > 4000),
                  sum(df_test_raw['GrLivArea'] > 4000)));
plt.xlabel('GrLivArea (ft^2)');
plt.ylabel('SalePrice ($)');

# The big moment. Don't forget to reset the index.
df_train_raw.drop(df_train_raw[(df_train_raw['GrLivArea'] > 4000)].index, 
                  inplace=True)
df_train_raw.reset_index(inplace=True, drop=True)
y = df_train_raw['SalePrice'] # Dependent variable
df_train_raw.drop(['SalePrice'], axis=1, inplace=True)

ntrain = df_train_raw.shape[0] # so we can split the data back up later

df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
def condition_housing_data(df):
    """General data-conditioning function to prepare the housing DataFrame for
    analysis. Mostly NaN filling
    """

    fillnone = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
                'MasVnrType']

    fillzero = ['GarageArea', 'TotalBsmtSF', 'LotFrontage', 'MasVnrArea',
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

    fillmode = ['Electrical', 'KitchenQual', 'SaleType', 'Exterior1st',
                'Exterior2nd', 'Functional', 'MasVnrType', 'MSZoning']

    # has some NaNs. Value is highly correlated with YearBuilt
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    # There seems to be an erroneous value for GarageYrBlt of 2207
    # Based on the YearBuilt being 2006, I assume it should be 2007
    df.loc[df.GarageYrBlt == 2207.0, 'GarageYrBlt'] = 2007.0

    # Convert column to strings. It's categorical data stored as int64
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Really only one value present
    df.drop(['Utilities'], axis=1, inplace=True)

    # Apparently this can't be done without looping.
    for colname in fillnone:
        df[colname].fillna('none', inplace=True)

    for colname in fillzero:
        df[colname].fillna(0, inplace=True)

    for colname in fillmode:
        df[colname].fillna(df[colname].mode()[0], inplace=True)

    return df

nullcols = df_all_raw.isnull().sum(axis=0)
nullcols = nullcols[nullcols > 0].sort_values(ascending=False)

plt.figure();
nullcols.plot(kind='bar');
plt.title('NaN counts in train + test data')
plt.ylim(0, 3250)
for xpos, ypos in enumerate(nullcols.values):
    plt.text(xpos + .06, ypos + 30, str(ypos), 
             ha='center', va='bottom', rotation=90, color='black');
plt.xticks(range(len(nullcols)), nullcols.index, rotation=90);
print(df_train_raw['Utilities'].groupby(df_train_raw['Utilities']).count() \
                               .sort_values(ascending=False))
plt.scatter(df_all_raw['GarageYrBlt'], df_all_raw['YearBuilt'],
            alpha=0.4, edgecolor='none');
plt.title('GarageYrBlt and YearBuilt exploration');
plt.xlabel('GarageYrBlt (Year A.D.)');
plt.ylabel('YearBuilt (Year A.D.)');
df_all = condition_housing_data(df_all_raw)
print('There are now {} null values in the data.'
      .format(df_all.isnull().sum().sum()))

# split the DataFrames back up for analysis
df_train = df_all[:ntrain]
df_test = df_all[ntrain:]
def explore_categorical(df):

    plt.rc('figure', figsize=(10.0, 5.0))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # discard non-object data
    df_ob = df.loc[:, df.dtypes == 'object'].fillna('none')

    for column in df_ob.columns:

        values = df_ob[column].groupby(
            df_ob[column]).count().sort_values(ascending=False)

        fig1, (ax11, ax12) = plt.subplots(1, 2)
        plt.suptitle(column)
        ax11.bar(values.index, values.values, color=colors)
        plt.title('Feature value counts')
        ax11.set_xlabel('Feature value')
        ax11.set_ylabel('Count')

        for label in values.index:

            data = df.loc[df_ob[column] == label, 'SalePrice'].values
            if len(data) > 1:
                sns.distplot(data, hist=False, ax=ax12)
                plt.title('PDF per feature-value')
                ax12.set_xlabel('SalePrice ($)')
                ax12.set_ylabel('Relative requency of occurance\n'
                                'Units are not that useful')
                ax12.set_yticks([])
                # Maybe consider CDF as an alternative
                # data.hist(bins=len(data), cumulative=True,
                #          density=True, histtype='step')

        plt.show()
        
        
def explore_numerical(df):

    df_num = df.loc[:, df.dtypes != 'object']
    df_num.fillna(0, inplace=True)
    #df_num.drop(labels='Id', axis=1, inplace=True)

    for column in df_num.columns:
        plt.scatter(df_num[column].values, df_num['SalePrice'].values,
                    alpha=0.4, edgecolors='none')
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('SalePrice ($)')
        plt.show()
sns.heatmap(pd.concat([df_train, y], axis=1).corr().sort_values('SalePrice', 
                                                                axis=1,
                                                                ascending=False,),
            vmin=-1.0,
            vmax=1.0,
            cmap='bwr');
plt.scatter(df_train['GarageArea'], y, alpha=0.3,
            edgecolor='none');
plt.title('Investigating zero-fill effect');
plt.xlabel('GarageArea ( ft^2)');
plt.ylabel('SalePrice ($)');
print('Pearson\'s moment coefficient of skewness for SalePrice: {}'
      .format(stats.skew(y)))

# And let's plot the SalePrice distribution so we can see what that looks like.
sns.distplot(y);
plt.title('Distribution of dependent variable \'SalePrice\'');

# And a Q-Q plot for good measure
plt.figure();
stats.probplot(y, dist='norm', plot=plt);
plt.title('SalePrice Q-Q plot');
y_log = np.log1p(y)

print('Pearson\'s moment coefficient of skewness for transformed SalePrice: {}'
      .format(stats.skew(y_log)))

# And let's plot the SalePrice distribution so we can see what that looks like.
sns.distplot(y_log);
plt.title('Distribution of transformed dependent variable \'SalePrice\'');

# And a Q-Q plot for good measure
plt.figure();
stats.probplot(y_log, dist='norm', plot=plt);
plt.title('Transformed SalePrice Q-Q plot');
plt.figure();
plt.subplots_adjust(wspace=.4, hspace=.4)
plt.subplot(221);
plt.scatter(df_train['OverallQual'], y);
plt.title('Before transformation');
plt.xlabel('OverallQual (rating 1-10)');
plt.ylabel('SalePrice ($)');

plt.subplot(222);
plt.scatter(df_train['OverallQual'], y_log);
plt.title('After transformation');
plt.xlabel('OverallQual (rating 1-10)');
plt.ylabel('SalePrice (log$)');

plt.subplot(223);
plt.scatter(df_train['TotalBsmtSF'], y);
plt.title('Before transformation');
plt.xlabel('TotalBsmtSF (ft^2)');
plt.ylabel('SalePrice ($)');

plt.subplot(224);
plt.scatter(df_train['TotalBsmtSF'], y_log);
plt.title('After transformation');
plt.xlabel('TotalBsmtSF (ft^2)');
plt.ylabel('SalePrice (log$)');
def dummify_data(df):

    categoricals = df.columns[df.dtypes == 'object']
    df_new = df.drop(categoricals, axis=1)
    df_new = pd.concat([df_new, pd.get_dummies(df[categoricals])],
                       axis=1)

    return df_new


df_all_dummified = dummify_data(df_all)
df_all_dummified.head()
df_all_dummified.shape

# split the data back up
Xtrain = df_all_dummified[:ntrain]
Xtest = df_all_dummified[ntrain:]
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge

def RidgeCV_custom(model_ridge, alphas, X, y):
    """We are assuming y (SalePrice) has already been log transformed"""
    scores = pd.DataFrame()
    for alpha in alphas:
        model_ridge.alpha = alpha
        scores_temp = cross_val_score(model_ridge,
                                      X,
                                      y,
                                      cv=5,
                                      scoring='neg_mean_squared_error')
        scores['{}'.format(alpha)] = np.sqrt(-scores_temp)
        
    return scores

# Randomize the data since cross_val_score doesn't
randstate = 4
Xtrain_rand = Xtrain.sample(frac=1, random_state=randstate)
y_log_rand = y_log.sample(frac=1, random_state=randstate)

model_ridge = Ridge(alpha=.1,
                    normalize=True,
                    max_iter=1e5)
alphas = [.0001, .001, .01, .1, 1, 10]
cv_test_1 = RidgeCV_custom(model_ridge, alphas, Xtrain_rand, y_log_rand)
cv_test_1
randstate = 10
Xtrain_rand = Xtrain.sample(frac=1, random_state=randstate)
y_log_rand = y_log.sample(frac=1, random_state=randstate)

alphas = [.01, .05, .1, .5, 1]
cv_test_2 = RidgeCV_custom(model_ridge, alphas, Xtrain_rand, y_log_rand)
cv_test_2
residuals = y_log_rand - cross_val_predict(model_ridge,
                                           Xtrain_rand,
                                           y_log_rand,
                                           cv=5)

print('Normality of residuals a la Shapiro-Wilk test (W = 1.0 -> perfectly normal) \n'
      'W statistic: {vals[0]}\n'
      'p-value: {vals[1]}'
       .format(vals=stats.shapiro(residuals)))

sns.distplot(residuals);
plt.title('Distribution of residuals from cross-validation');
plt.xlabel('SalePrice (log$)');
plt.ylabel('Relative Frequency (arbitrary units)');
