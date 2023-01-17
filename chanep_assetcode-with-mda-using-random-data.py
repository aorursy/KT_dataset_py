import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn import preprocessing
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7
def make_test_train(df, split=0.80):
    # Label encode the assetCode feature
    X = df[df.universe==1]
    le = preprocessing.LabelEncoder()
    X = X.assign(assetCode = le.fit_transform(X.assetCode))
    
    # split test and train
    train_ct = int(X.shape[0]*split)
    y_train, y_test = X['returnsOpenNextMktres10'][:train_ct], X['returnsOpenNextMktres10'][train_ct:]
    X = X.drop(['time', 'returnsOpenNextMktres10'], axis=1)
    X_train, X_test = X.iloc[:train_ct,], X.iloc[train_ct:,]
    return X, X_train, X_test, y_train, y_test
def make_lgb(X_train, X_test, y_train, y_test, categorical_cols = ['assetCode']):
    # Set up LightGBM data structures
    train_cols = X_train.columns.tolist()
    dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(X_test.values, y_test, feature_name=train_cols, categorical_feature=categorical_cols)
    return dtrain, dvalid
# Set up the LightGBM params
lgb_params = dict(
    objective='regression_l1', learning_rate=0.1, num_leaves=127, max_depth=-1, bagging_fraction=0.75,
    bagging_freq=2, feature_fraction=0.5, lambda_l1=1.0, seed=1015
)
# @marketneutral: Create some random assetCodes (this is a nice snippet from McKinney, Wes. Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython (p. 340). O'Reilly Media. Kindle Edition.)
import random; random.seed(0)
import string
num_stocks = 1250
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in range(n)])
assetCodes = np.array([rands(5) for _ in range(num_stocks)])
# @marketneutral:  Spoof intraday and overnight returns
days_in_year = 260
total_days = days_in_year*7
on_vol_frac = 0.2  # overnight volatility fraction

annualized_vol = 0.20
open_to_close_returns = np.random.normal(0.0, scale=annualized_vol*(1-on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
close_to_open_returns = np.random.normal(0, scale=annualized_vol*(on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
open_to_open_returns = close_to_open_returns + open_to_close_returns 
close_to_close_returns = close_to_open_returns + np.roll(open_to_close_returns, -1)

# Make price series
prices_close = 100*np.cumprod(1+close_to_close_returns, axis=0)
prices_open = prices_close*(1+close_to_open_returns)
import itertools

# Make into a DataFrame
dates = pd.date_range(end=pd.Timestamp('2017-12-31'), periods=total_days)
spoofed_df = pd.DataFrame(
    data={'close': prices_close.flatten('F'), 'open': prices_open.flatten('F')},
    index = pd.MultiIndex.from_tuples(
        list(itertools.product(assetCodes, dates)), names=('assetCode', 'time')
    )
)
spoofed_df['universe'] = 1.0
spoofed_df.head()
spoofed_df = spoofed_df.reset_index().sort_values(['assetCode','time']).set_index(['assetCode', 'time'])
# @marketneutral: make sure we did the open/close transform properly. Looks good.
spoofed_df.loc['MYNBI', ['open', 'close']]['1Q2013'].plot();
#  @marketneutral: Make the "return" based features

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.open/x.open.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: (x.open/x.open.shift(10)) - 1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(10)-1)
     .reset_index(0, drop=True)
)

#  @marketneutral: Make the target variable
spoofed_df = spoofed_df.assign(
    returnsOpenNextMktres10 = spoofed_df.groupby(level='assetCode').
    apply(lambda x: (x.open.shift(-10)/x.open)-1)
    .reset_index(0, drop=True)
)
# @marketneutral: Drop the edges where we don't have data to make returns
spoofed_df = spoofed_df.reset_index().dropna()
spoofed_df.head()
#Derived from Advances in Financial Machine Learning, pp 116-117, Marcos Lopez de Prado

from sklearn.model_selection._split import KFold
import datetime as dt

def featImpMDA_regress(reg,X,y,cv):
    # feat importance based on OOS score reduction
    print('start MDA',dt.datetime.now())
    from sklearn.metrics import mean_squared_error
    cvGen=KFold(n_splits=cv)
    scr0,scr1=pd.Series(),pd.DataFrame(columns=X.columns)
    for i,(train,test) in enumerate(cvGen.split(X=X)):
        print('   Split',i+1)
        X0,y0=X.iloc[train,:],y.iloc[train]
        X1,y1=X.iloc[test,:],y.iloc[test]
        fit=reg.fit(X=X0,y=y0)
        pred=fit.predict(X1)
        scr0.loc[i]=mean_squared_error(y1,pred)
        for j in X.columns:
            X1_=X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single column
            pred=fit.predict(X1_)
            scr1.loc[i,j]=mean_squared_error(y1,pred)
    imp=scr1.subtract(scr0,axis=0) #Increases in error
    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1)
    print('end MDA',dt.datetime.now())
    return imp
# From rfpimp: https://pypi.org/project/rfpimp/

def plot_importances(df_importances, save=None, xrot=0, tickstep=3,
                     label_fontsize=12,
                     figsize=None, scalefig=(1.0, 1.0), show=True):
    """
    Given an array or data frame of importances, plot a horizontal bar chart
    showing the importance values.

    :param df_importances: A data frame with Feature, Importance columns
    :type df_importances: pd.DataFrame
    :param save: A filename identifying where to save the image.
    :param xrot: Degrees to rotate importance (X axis) labels
    :type xrot: int
    :param tickstep: How many ticks to skip in X axis
    :type tickstep: int
    :param label_fontsize:  The font size for the column names and x ticks
    :type label_fontsize:  int
    :param figsize: Specify width and height of image (width,height)
    :type figsize: 2-tuple of floats
    :param scalefig: Scale width and height of image (widthscale,heightscale)
    :type scalefig: 2-tuple of floats
    :param show: Execute plt.show() if true (default is True). Sometimes
                 we want to draw multiple things before calling plt.show()
    :type show: bool
    :return: None

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_test, y_test)
    plot_importances(imp)
    """
    I = df_importances

    if figsize:
        fig = plt.figure(figsize=figsize)
    elif scalefig:
        fig = plt.figure()
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scalefig[0], h * scalefig[1], forward=True)
    else:
        fig = plt.figure()
    ax = plt.gca()
    labels = []
    for col in I.index:
        if isinstance(col, list):
            labels.append('\n'.join(col))
        else:
            labels.append(col)

    for tick in ax.get_xticklabels():
        tick.set_size(label_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_size(label_fontsize)
    ax.barh(np.arange(len(I.index)), I.Importance, height=0.6, tick_label=labels)

    # rotate x-ticks
    if xrot is not None:
        plt.xticks(rotation=xrot)

    # xticks freq
    xticks = ax.get_xticks()
    nticks = len(xticks)
    new_ticks = xticks[np.arange(0, nticks, step=tickstep)]
    ax.set_xticks(new_ticks)

    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show()
def make_regress(df):
    # Label encode the assetCode feature
    X = df[df.universe==1]
    le = preprocessing.LabelEncoder()
    X = X.assign(assetCode = le.fit_transform(X.assetCode))
  
    y = X['returnsOpenNextMktres10']
    X = X.drop(['time', 'returnsOpenNextMktres10'], axis=1)
    
    return X, y
X_r,y_r=make_regress(spoofed_df)
regress = lgb.LGBMModel(objective='regression_l1', learning_rate=0.1, num_leaves=127, max_depth=-1, bagging_fraction=0.75,
    bagging_freq=2, feature_fraction=0.5, lambda_l1=1.0, seed=1015)
mda_imps_r=featImpMDA_regress(regress,X_r,y_r,cv=10)
import matplotlib
%matplotlib inline
m_imps=mda_imps_r.reset_index().rename(index=int,columns={"index":"Feature",'mean':'Importance'}, inplace=False).set_index('Feature')
m_imps=m_imps.sort_values(by='Importance')
plot_importances(m_imps,scalefig=(1.5,1.0))