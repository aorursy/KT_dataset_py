# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/house-prices-advanced-regression-techniques/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

import scipy.stats as st

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objects as go

import plotly.express as px

import plotly.tools as tls

from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,ElasticNetCV

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
train_df = pd.read_csv(os.path.join(dirname, 'train.csv'))

test_df = pd.read_csv(os.path.join(dirname, 'test.csv'))
def describe_df(df):

    for col in df.columns:

        ds = df[col]

        mean = "-"

        var = "-"

        mn = "-"

        mx = "-"

        nulls = str(ds.isna().sum())

        unique = str(ds.nunique())

        if np.issubdtype(ds.dtype, np.number):

            mean = str(np.mean(ds))

            var = str(np.var(ds))

            mn = str(np.min(ds))

            mx = str(np.max(ds))

        print(col+":"+str(ds.dtype)+":"+nulls+":"+unique+":"+mean+":"+var+":"+mn+":"+mx)

        

describe_df(train_df)
#PoolArea = 0 is Pool QC is missing

def handle_missing(df):

    df.PoolQC.fillna("Missing",inplace=True)



    # missing feature

    df.MiscFeature.fillna("Missing",inplace=True)



    # missing feature

    df.Alley.fillna("Missing",inplace=True)

    df.Fence.fillna("Missing",inplace=True)



    # missing feature

    df.FireplaceQu.fillna("Missing",inplace=True)



    df.LotFrontage.fillna(0,inplace=True)

    df.GarageType.fillna("Missing",inplace=True)

    df.GarageQual.fillna("Missing",inplace=True)

    df.GarageCond.fillna("Missing",inplace=True)

    df.GarageFinish.fillna("Missing",inplace=True)

    df.drop(columns='GarageYrBlt', inplace=True) #Colinear with yearblt



    df.BsmtFinType2.fillna("Missing",inplace=True)

    df.BsmtFinType1.fillna("Missing",inplace=True)

    df.BsmtExposure.fillna("Missing",inplace=True)

    df.BsmtCond.fillna("Missing",inplace=True)

    df.BsmtQual.fillna("Missing",inplace=True)



    df.MasVnrType.fillna("Missing",inplace=True)

    df.MasVnrArea.fillna(0,inplace=True)

    df.Electrical.fillna("Missing",inplace=True)

    

handle_missing(train_df)

handle_missing(test_df)



quantitative = [f for f in train_df.columns if train_df.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train_df.columns if train_df.dtypes[f] == 'object']








missing = test_df.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train, test = train_test_split(train_df, test_size = 0.2)
# y = train['SalePrice']

# plt.figure(1); plt.title('Johnson SU')

# sns.distplot(y, kde=False, fit=st.johnsonsu)

# plt.figure(2); plt.title('Normal')

# sns.distplot(y, kde=False, fit=st.norm)

# plt.figure(3); plt.title('Log Normal')

# sns.distplot(y, kde=False, fit=st.lognorm)

# plt.figure(4); plt.title('Johnson transform')

# sns.distplot(st.yeojohnson(y.astype('float64'))[0], kde=False, fit=st.norm)

# plt.figure(5); plt.title('boxcox')

# sns.distplot(st.boxcox(y)[0], kde=False, fit=st.norm)

# gamma, eta, epsilon, lbda = st.johnsonsu.fit(y)

# yt = gamma + eta*np.arcsinh((y-epsilon)/lbda)

# plt.figure(6); plt.title('johnsonsu')

# sns.distplot(yt, kde=False, fit=st.norm)





# test_normality = lambda x: stats.shapiro(x.fillna(0))[1] > 0.05

# print("SalePrice: ",pd.DataFrame(train["SalePrice"]).apply(test_normality)[0])

# print("log(SalePrice): ",pd.DataFrame(np.log(train["SalePrice"])).apply(test_normality)[0])

# print("box(SalePrice): ",pd.DataFrame(st.boxcox(train["SalePrice"])[0]).apply(test_normality)[0])

# print("johnson(SalePrice): ",pd.DataFrame(yt).apply(test_normality)[0])

# def johnson(y, gamma=None, eta=None, epsilon=None, lbda=None):

#     if gamma is None:

#         gamma, eta, epsilon, lbda = stats.johnsonsu.fit(y)

#     yt = gamma + eta*np.arcsinh((y-epsilon)/lbda)

#     return yt, gamma, eta, epsilon, lbda



# train.SalePrice, gamma, eta, epsilon, lbda = johnson(train.SalePrice)

# test.SalePrice, gamma, eta, epsilon, lbda = johnson(test.SalePrice,gamma, eta, epsilon, lbda)



# print("SalePrice: ",pd.DataFrame(train["SalePrice"]).apply(test_normality)[0])

# print("SalePrice: ",pd.DataFrame(test["SalePrice"]).apply(test_normality)[0])
def encode(frame, feature, test):

    ordering = pd.DataFrame()

    ordering['val'] = frame[feature].unique()

    ordering.index = ordering.val

    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']

    ordering = ordering.sort_values('spmean')

    ordering['ordering'] = range(1, ordering.shape[0]+1)

    ordering = ordering['ordering'].to_dict()

    

    for cat, o in ordering.items():

        frame.loc[frame[feature] == cat, feature+'_E'] = o

        test.loc[test[feature] == cat, feature+'_E'] = o

        test[feature+'_E'].fillna(0,inplace=True)

    

def train_cats(train,test):    

    for c in qualitative:

        train[c] = train[c].astype('category')

        test[c] = test[c].astype('category')

        test[c] = test[c].cat.set_categories(train[c].cat.categories)

        if train[c].isnull().any() or test[c].isnull().any():

            train[c] = train[c].cat.add_categories(['MISSING'])

            train[c] = train[c].fillna('MISSING')

            test[c] = test[c].cat.add_categories(['MISSING'])

            test[c] = test[c].fillna('MISSING')



    cat_encoded = []

    for q in qualitative:  

        encode(train, q, test)

        cat_encoded.append(q+'_E')

        

train_cats(train,test)

train_cats(train_df,test_df)



for f in quantitative:

    test_df[f].fillna(0,inplace=True)
#Numerical corelation

def spearman(frame, features, n=10):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]

    spr.spearman = np.abs(spr.spearman)

    spr = spr.sort_values('spearman', ascending= False)

    plt.figure(figsize=(6, 0.25*len(features)))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    return spr[:n].feature.array

    

features = quantitative



top_num = np.array(spearman(train, features,20))
#categorical importance

def anova(frame):

    anv = pd.DataFrame()

    anv['feature'] = qualitative

    pvals = []

    for c in qualitative:

        samples = []

        for cls in frame[c].unique():

            s = frame[frame[c] == cls]['SalePrice'].values

            samples.append(s)

        pval = stats.f_oneway(*samples)[1]

        pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



a = anova(train)

a['disparity'] = np.log(1./a['pval'].values)

sns.barplot(data=a, x='feature', y='disparity')

x=plt.xticks(rotation=90)

a = a.sort_values(by='disparity',ascending=False)

top_cat = np.array(a[:20].feature.array) + "_E"
f = pd.melt(train, value_vars=top_num)

g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)

g = g.map(sns.distplot, "value")
def logTransform(df,f):

    df[f] = np.log1p(df[f])

    

def zeroFeature(df,f):

    df[f+"_present"] = df[f].apply(lambda x: 1 if x>0 else 0)

    



zero_features = ['GarageArea','TotalBsmtSF','OpenPorchSF','MasVnrArea','WoodDeckSF','BsmtFinSF1','2ndFlrSF','LotFrontage']

log_features = ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','LotArea','OpenPorchSF','MasVnrArea','WoodDeckSF','BsmtFinSF1','2ndFlrSF','LotFrontage']



for f in zero_features:

    zeroFeature(train,f)

    zeroFeature(test,f)

    zeroFeature(train_df,f)

    zeroFeature(test_df,f)

    

    

for f in log_features:

    logTransform(train,f)

    logTransform(test,f)

    logTransform(train_df,f)

    logTransform(test_df,f)



logTransform(train,"SalePrice")

logTransform(train_df,"SalePrice")

logTransform(test,"SalePrice")

zero_features = [s + "_present" for s in zero_features]
features = np.append(top_num,top_cat)

for feature in features:

    x = train[feature]

    y = train.SalePrice

    m = LinearRegression()

    m.fit(np.array(x).reshape(-1,1), y)

    err = y - m.predict(np.array(x).reshape(-1,1))

    err_sort = np.sort(err)

    err_sort = (err_sort - np.mean(err_sort))/np.std(err_sort)

    normal = np.sort(np.random.normal(0, 1, err_sort.shape[0]))



    fig, ax = plt.subplots(figsize=(10,10), ncols=2, nrows=2)

    y_title_margin = 1.2

    plt.suptitle(feature)

    ax[0][0].set_title("Scatter")

    ax[0][1].set_title("Error")

    ax[1][0].set_title("Histogram")

    ax[1][1].set_title("Q-Q")

    ax[1][1].set(ylim=(-5, 5))



    sns.regplot(x=x, y=y, ax=ax[0][0])

    sns.scatterplot(x=x, y=err, ax=ax[0][1])

    sns.distplot(err_sort,  kde = False, ax=ax[1][0], fit=st.norm)

    sns.scatterplot(y=err_sort,x=normal, ax=ax[1][1])

    sns.lineplot(y=np.linspace(-5,5,20),x=np.linspace(-5,5,20), ax=ax[1][1])

cat_features = [s + "_E" for s in qualitative]

features = np.append(quantitative,cat_features)

features = np.append(features, zero_features)



X_train = train[features]

y_train = train["SalePrice"]

X_test = test[features]

y_test = test["SalePrice"]



X_train_df = train_df[features]

y_train_df = train_df["SalePrice"]

X_test_df = test_df[features]



scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)


elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10, normalize=True, random_state=43)



elasticNet.fit(X_train, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10, normalize=True, random_state=43)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 50000, cv = 10, normalize=True, random_state=43)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())
y_train_ela = elasticNet.predict(X_train)

y_test_ela = elasticNet.predict(X_test)
# Plot residuals

plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(elasticNet.coef_, index = X_train.columns)

print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the ElasticNet Model")

plt.show()
m = ElasticNet(l1_ratio=ratio, alpha = alpha, normalize=True, random_state=43)

m.fit(X_train_df,y_train_df)

y_pred = np.exp(m.predict(X_test_df))-1
submission = my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_pred})

my_submission.to_csv('submission.csv', index=False)